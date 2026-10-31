"""9x9 Go rules and the 12-plane student encoding.

A faithful port of vendor/moka/src/game.ts + constants.ts (simple ko,
suicide rejected, captures, area-scoring helpers omitted: generation only
needs legal play and termination by two passes / move cap).
Plane layout (channel-first, [12, 9, 9]):
  0 current stones, 1 opponent stones, 2/3 groups in atari (cur/opp),
  4/5 groups with 2 liberties (cur/opp), 6 ko point, 7/8 recent moves,
  9/10 recent passes (full-board), 11 komi plane (-komi*color/15).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import AREA, BOARD_SIZE, INPUT_PLANES, KOMI, PASS_MOVE

BLACK, WHITE, EMPTY = 1, -1, 0


def _neighbours(move: int) -> list[int]:
    row, col = divmod(move, BOARD_SIZE)
    out = []
    if row > 0:
        out.append(move - BOARD_SIZE)
    if row < BOARD_SIZE - 1:
        out.append(move + BOARD_SIZE)
    if col > 0:
        out.append(move - 1)
    if col < BOARD_SIZE - 1:
        out.append(move + 1)
    return out


NEIGHBOURS: list[list[int]] = [_neighbours(m) for m in range(AREA)]


def group(board: np.ndarray, start: int) -> tuple[list[int], set[int]]:
    color = board[start]
    if color == EMPTY:
        return [], set()
    stack, visited, liberties, stones = [start], set(), set(), []
    while stack:
        move = stack.pop()
        if move in visited:
            continue
        visited.add(move)
        stones.append(move)
        for adj in NEIGHBOURS[move]:
            if board[adj] == EMPTY:
                liberties.add(adj)
            elif board[adj] == color:
                stack.append(adj)
    return stones, liberties


@dataclass
class GameState:
    board: np.ndarray = field(default_factory=lambda: np.zeros(AREA, dtype=np.int8))
    next_color: int = BLACK
    ko_move: int = -1
    move_count: int = 0
    consecutive_passes: int = 0
    history: list[int] = field(default_factory=list)

    def copy(self) -> "GameState":
        return GameState(self.board.copy(), self.next_color, self.ko_move,
                         self.move_count, self.consecutive_passes, list(self.history))

    def is_game_over(self) -> bool:
        return self.consecutive_passes >= 2


def play_move(state: GameState, move: int) -> GameState | None:
    """Return the next state, or None if illegal (mirrors playMove in game.ts)."""
    if move == PASS_MOVE:
        nxt = state.copy()
        nxt.consecutive_passes += 1
        nxt.ko_move = -1
        nxt.move_count += 1
        nxt.history.append(move)
        nxt.next_color *= -1
        return nxt
    if not (0 <= move < AREA) or state.board[move] != EMPTY or state.ko_move == move:
        return None
    nxt = state.copy()
    nxt.board[move] = state.next_color
    captured: list[int] = []
    for adj in NEIGHBOURS[move]:
        if nxt.board[adj] != -state.next_color:
            continue
        stones, liberties = group(nxt.board, adj)
        if liberties:
            continue
        for stone in stones:
            nxt.board[stone] = EMPTY
            captured.append(stone)
    stones, liberties = group(nxt.board, move)
    if not liberties:
        return None
    nxt.consecutive_passes = 0
    nxt.ko_move = captured[0] if (len(captured) == 1 and len(stones) == 1 and len(liberties) == 1) else -1
    nxt.move_count += 1
    nxt.history.append(move)
    nxt.next_color *= -1
    return nxt


def legal_moves(state: GameState) -> list[int]:
    moves = [m for m in range(AREA) if play_move(state, m) is not None]
    moves.append(PASS_MOVE)
    return moves


def encode_features(state: GameState) -> np.ndarray:
    """[12, 9, 9] float32, current-player perspective (encodeStudentFeatures)."""
    feats = np.zeros((INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    flat = feats.reshape(INPUT_PLANES, AREA)
    visited: set[int] = set()
    for move in range(AREA):
        color = state.board[move]
        if color == EMPTY:
            continue
        current = color == state.next_color
        flat[0 if current else 1, move] = 1.0
        if move in visited:
            continue
        stones, liberties = group(state.board, move)
        visited.update(stones)
        if len(liberties) == 1:
            plane = 2 if current else 3
        elif len(liberties) == 2:
            plane = 4 if current else 5
        else:
            continue
        for stone in stones:
            flat[plane, stone] = 1.0
    if state.ko_move >= 0:
        flat[6, state.ko_move] = 1.0
    for offset, plane in ((1, 7), (2, 8)):
        if len(state.history) >= offset:
            prev = state.history[-offset]
            if prev == PASS_MOVE:
                flat[plane + 2, :] = 1.0
            elif 0 <= prev < AREA:
                flat[plane, prev] = 1.0
    flat[11, :] = (-KOMI * state.next_color) / 15.0
    return feats
