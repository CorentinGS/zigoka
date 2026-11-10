"""Moka's Go rules on an N×N board and the 12-plane student encoding.

A faithful port of vendor/moka/src/game.ts + constants.ts (simple ko,
suicide rejected, captures, area-scoring helpers omitted: generation only
needs legal play and termination by two passes / move cap). The board
size comes from the state (9 by default); the 19×19 move limits are the
repo's (oracle/constants-19.ts).
Plane layout (channel-first, [12, N, N]):
  0 current stones, 1 opponent stones, 2/3 groups in atari (cur/opp),
  4/5 groups with 2 liberties (cur/opp), 6 ko point, 7/8 recent moves,
  9/10 recent passes (full-board), 11 komi plane (-komi*color/15).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from . import BOARD_SIZE, INPUT_PLANES, KOMI, BoardSpec, board_spec

BLACK, WHITE, EMPTY = 1, -1, 0


@lru_cache(maxsize=None)
def neighbours(size: int) -> tuple[tuple[int, ...], ...]:
    """Adjacent points of every point, in Moka's order (up, down, left, right)."""
    out = []
    for move in range(size * size):
        row, col = divmod(move, size)
        adjacent = []
        if row > 0:
            adjacent.append(move - size)
        if row < size - 1:
            adjacent.append(move + size)
        if col > 0:
            adjacent.append(move - 1)
        if col < size - 1:
            adjacent.append(move + 1)
        out.append(tuple(adjacent))
    return tuple(out)


NEIGHBOURS = neighbours(BOARD_SIZE)


def _size_of(board: np.ndarray) -> int:
    size = math.isqrt(len(board))
    if size * size != len(board):
        raise ValueError(f"board of {len(board)} points is not square")
    return size


def group(board: np.ndarray, start: int) -> tuple[list[int], set[int]]:
    color = board[start]
    if color == EMPTY:
        return [], set()
    adjacency = neighbours(_size_of(board))
    stack, visited, liberties, stones = [start], set(), set(), []
    while stack:
        move = stack.pop()
        if move in visited:
            continue
        visited.add(move)
        stones.append(move)
        for adj in adjacency[move]:
            if board[adj] == EMPTY:
                liberties.add(adj)
            elif board[adj] == color:
                stack.append(adj)
    return stones, liberties


@dataclass
class GameState:
    board: np.ndarray | None = None
    next_color: int = BLACK
    ko_move: int = -1
    move_count: int = 0
    consecutive_passes: int = 0
    history: list[int] = field(default_factory=list)
    size: int = BOARD_SIZE

    def __post_init__(self) -> None:
        if self.board is None:
            self.board = np.zeros(self.size * self.size, dtype=np.int8)
        elif len(self.board) != self.size * self.size:
            raise ValueError(f"board has {len(self.board)} points, expected {self.size * self.size}")

    @property
    def spec(self) -> BoardSpec:
        return board_spec(self.size)

    def copy(self) -> "GameState":
        return GameState(self.board.copy(), self.next_color, self.ko_move,
                         self.move_count, self.consecutive_passes, list(self.history), self.size)

    def is_game_over(self) -> bool:
        return self.consecutive_passes >= 2


def play_move(state: GameState, move: int) -> GameState | None:
    """Return the next state, or None if illegal (mirrors playMove in game.ts)."""
    area = state.size * state.size
    if move == area:
        nxt = state.copy()
        nxt.consecutive_passes += 1
        nxt.ko_move = -1
        nxt.move_count += 1
        nxt.history.append(move)
        nxt.next_color *= -1
        return nxt
    if not (0 <= move < area) or state.board[move] != EMPTY or state.ko_move == move:
        return None
    nxt = state.copy()
    nxt.board[move] = state.next_color
    captured: list[int] = []
    for adj in neighbours(state.size)[move]:
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
    """Every legal move, pass included. Tries each point: O(area) plays,
    so rollouts should prefer `sample_legal_move`."""
    area = state.size * state.size
    moves = [m for m in range(area) if play_move(state, m) is not None]
    moves.append(area)
    return moves


def sample_legal_move(state: GameState, probs: np.ndarray, rng: np.random.Generator,
                      temperature: float = 1.0, allow_pass: bool | None = None) -> tuple[int, GameState]:
    """Draw a move from `probs` restricted to legal moves, by rejection:
    draw, try to play, zero the move if illegal, draw again. Returns the
    move and the next state. Pass is allowed from Moka's minimum pass move
    on, unless `allow_pass` says otherwise; it is the fallback when no
    point is legal."""
    area = state.size * state.size
    if allow_pass is None:
        allow_pass = state.move_count >= state.spec.min_pass_move
    weights = np.asarray(probs, dtype=np.float64).copy()
    if temperature != 1.0:
        weights = np.power(np.maximum(weights, 1e-12), 1.0 / temperature)
    if not allow_pass:
        weights[area] = 0.0
    while True:
        total = weights.sum()
        if not total > 0:
            # Every move with mass was illegal: fall back to a uniform legal draw.
            candidates = [m for m in legal_moves(state) if allow_pass or m != area]
            if not candidates:
                return area, play_move(state, area)
            move = int(rng.choice(candidates))
            return move, play_move(state, move)
        move = int(rng.choice(len(weights), p=weights / total))
        nxt = play_move(state, move)
        if nxt is not None:
            return move, nxt
        weights[move] = 0.0


def encode_features(state: GameState) -> np.ndarray:
    """[12, N, N] float32, current-player perspective (encodeStudentFeatures)."""
    size, area = state.size, state.size * state.size
    feats = np.zeros((INPUT_PLANES, size, size), dtype=np.float32)
    flat = feats.reshape(INPUT_PLANES, area)
    visited: set[int] = set()
    for move in range(area):
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
            if prev == area:
                flat[plane + 2, :] = 1.0
            elif 0 <= prev < area:
                flat[plane, prev] = 1.0
    flat[11, :] = (-KOMI * state.next_color) / 15.0
    return feats
