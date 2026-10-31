"""KataGo ONNX teacher (b6c96-style: 22 spatial + 14/19 global inputs).

Encoding follows KataGo's documented NN inputs (nninputs.cpp fillRow*):
planes 0 on-board, 1/2 own/opp stones, 3/4/5 liberty counts 1/2/3, 6 ko,
7/8 encore (zero, area scoring), 9-13 last five moves, 14-17 ladders,
18/19 area ownership, 20/21 encore stones (zero); globals 0-4 recent
passes, 5 self-komi/20, 6/7 ko-rule, 8 multi-stone suicide, 9 territory
scoring, rest zero for area-scored simple-ko games.

Known limitation, stated rather than hidden: ladder planes 14-17 are left
at zero (full ladder search is not implemented), so KataGo labels are
slightly degraded on ladder-critical positions. Everything else is exact
for the rules this pipeline plays (area scoring, simple ko, komi 7).
Requires the `katago` extra (onnxruntime-gpu / onnxruntime).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import AREA, BOARD_SIZE, KOMI, PASS_MOVE
from .board import EMPTY, GameState, group

PLANES = 22


def _area_owners(board: np.ndarray) -> np.ndarray:
    owners = np.where(board != EMPTY, board, 0).astype(np.int8)
    visited = np.zeros(AREA, dtype=bool)
    for start in range(AREA):
        if board[start] != EMPTY or visited[start]:
            continue
        region, borders, stack = [], set(), [start]
        visited[start] = True
        while stack:
            move = stack.pop()
            region.append(move)
            row, col = divmod(move, BOARD_SIZE)
            for adj in (move - 9 if row else -1, move + 9 if row < 8 else -1,
                        move - 1 if col else -1, move + 1 if col < 8 else -1):
                if adj < 0:
                    continue
                if board[adj] == EMPTY and not visited[adj]:
                    visited[adj] = True
                    stack.append(adj)
                elif board[adj] != EMPTY:
                    borders.add(int(board[adj]))
        if len(borders) == 1:
            owner = borders.pop()
            for move in region:
                owners[move] = owner
    return owners


def encode_katago(state: GameState, global_count: int, komi: float = KOMI) -> tuple[np.ndarray, np.ndarray]:
    spatial = np.zeros((PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    flat = spatial.reshape(PLANES, AREA)
    flat[0, :] = 1.0
    visited: set[int] = set()
    for move in range(AREA):
        color = state.board[move]
        if color == EMPTY:
            continue
        flat[1 if color == state.next_color else 2, move] = 1.0
        if move in visited:
            continue
        stones, liberties = group(state.board, move)
        visited.update(stones)
        if 1 <= len(liberties) <= 3:
            for stone in stones:
                flat[2 + len(liberties), stone] = 1.0
    if state.ko_move >= 0:
        flat[6, state.ko_move] = 1.0
    globals_ = np.zeros(global_count, dtype=np.float32)
    for offset in range(1, 6):
        if len(state.history) >= offset:
            prev = state.history[-offset]
            if prev == PASS_MOVE:
                if offset - 1 < global_count:
                    globals_[offset - 1] = 1.0
            elif 0 <= prev < AREA:
                flat[8 + offset, prev] = 1.0
    owners = _area_owners(state.board)
    for move in range(AREA):
        if owners[move] == state.next_color:
            flat[18, move] = 1.0
        elif owners[move] == -state.next_color:
            flat[19, move] = 1.0
    if global_count > 5:
        self_komi = -komi if state.next_color == 1 else komi
        globals_[5] = float(np.clip(self_komi, -(AREA + 20), AREA + 20)) / 20.0
    if global_count > 8:
        globals_[8] = 1.0  # multi-stone suicide legality flag (KataGo convention)
    return spatial, globals_


class KataGoTeacher:
    """Teacher backed by a KataGo ONNX model via ONNX Runtime.

    Uses CUDAExecutionProvider when available (NVIDIA), else CPU.
    Outputs follow the b6c96 ONNX contract inspected by Moka upstream:
    policy [B,1,9,9] logits, policy_pass [B,1], value [B,3] win/loss/
    no-result logits.
    """

    approximate_planes = (14, 15, 16, 17)  # ladder planes, left at zero

    def __init__(self, model_path: str | Path, use_cuda: bool = True) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("KataGo teacher needs onnxruntime: install the 'katago' extra.") from exc
        available = ort.get_available_providers()
        providers = (["CUDAExecutionProvider"] if use_cuda and "CUDAExecutionProvider" in available else []) + ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.providers = self.session.get_providers()
        inputs = {i.name: i for i in self.session.get_inputs()}
        self.spatial_name = next(n for n in inputs if "spatial" in n)
        self.global_name = next(n for n in inputs if "global" in n)
        self.global_count = int(inputs[self.global_name].shape[-1])
        self.output_names = [o.name for o in self.session.get_outputs()]
        print(f"KataGo teacher on {self.providers} (global inputs: {self.global_count}); "
              f"ladder planes {self.approximate_planes} are zero (documented limitation)")

    def label_state(self, state: GameState) -> tuple[np.ndarray, float]:
        spatial, globals_ = encode_katago(state, self.global_count)
        outputs = self.session.run(None, {self.spatial_name: spatial[None], self.global_name: globals_[None]})
        by_name = dict(zip(self.output_names, outputs))
        policy = np.asarray(by_name["policy"])[0, 0].reshape(AREA)
        policy_pass = np.asarray(by_name["policy_pass"])[0].reshape(-1)[:1]
        logits = np.concatenate([policy, policy_pass]).astype(np.float64)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        value_logits = np.asarray(by_name["value"])[0].astype(np.float64)
        win_loss = np.exp(value_logits[:2] - value_logits[:2].max())
        win_loss /= win_loss.sum()
        return probs.astype(np.float32), float(win_loss[0] - win_loss[1])
