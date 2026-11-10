"""Distillation pipeline for the Moka-format student network, 9×9 and 19×19.

The module-level constants describe the 9×9 board, the historical default
of every stage. Code that handles other sizes takes a `BoardSpec` from
`board_spec(size)` instead.
"""
from __future__ import annotations

from dataclasses import dataclass

INPUT_PLANES = 12
KOMI = 7.0


@dataclass(frozen=True)
class BoardSpec:
    """Moka's rules on an N×N board. The move limits are Moka's for 9×9
    (vendor/moka/src/constants.ts) and the repo's 19×19 constants
    (oracle/constants-19.ts, also used by src/rules/rules.zig)."""

    size: int
    max_moves: int
    min_pass_move: int

    @property
    def area(self) -> int:
        return self.size * self.size

    @property
    def pass_move(self) -> int:
        return self.area

    @property
    def move_count(self) -> int:
        return self.area + 1


BOARD_SPECS = {9: BoardSpec(9, 120, 20), 19: BoardSpec(19, 540, 89)}


def board_spec(size: int) -> BoardSpec:
    if size not in BOARD_SPECS:
        raise ValueError(f"Moka defines move limits for 9×9 and 19×19 only, not {size}×{size}")
    return BOARD_SPECS[size]


_DEFAULT = BOARD_SPECS[9]
BOARD_SIZE = _DEFAULT.size
AREA = _DEFAULT.area
PASS_MOVE = _DEFAULT.pass_move
MOVE_COUNT = _DEFAULT.move_count
MAX_MOVES = _DEFAULT.max_moves
MIN_PASS_MOVE = _DEFAULT.min_pass_move

__all__ = ["AREA", "BOARD_SIZE", "BOARD_SPECS", "BoardSpec", "INPUT_PLANES", "KOMI", "MAX_MOVES",
           "MIN_PASS_MOVE", "MOVE_COUNT", "PASS_MOVE", "board_spec"]
