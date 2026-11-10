"""go-selfplay against a scripted stand-in for `katago analysis` (9×9, no GPU).

The stand-in answers like KataGo (fields of tests/fixtures/katago): a
flat policy over the empty points, best move the first empty point, then
the pass from Moka's minimum pass move on; from move 8 on, Black leads
by 20 points with a winrate of 0.99.
"""
from __future__ import annotations

import json
import pathlib
import stat
import sys

import numpy as np
import pytest

from go_pipeline import board_spec
from go_pipeline.analysis import Label
from go_pipeline.board import GameState, encode_features, play_move
from go_pipeline.selfplay import Game, SelfplayConfig, choose_move, resign_winner, selfplay
from go_pipeline.shards import load_shards, unpack_features

STAND_IN = r'''
import json, sys
COLUMNS = "ABCDEFGHJKLMNOPQRST"
for line in sys.stdin:
    q = json.loads(line)
    size = q["boardXSize"]
    area = size * size
    occupied = set()
    for _, v in q["moves"]:
        if v != "pass":
            occupied.add((size - int(v[1:])) * size + COLUMNS.index(v[0]))
    turn = len(q["moves"])
    empty = [p for p in range(area) if p not in occupied]
    policy = [-1.0 if p in occupied else 1.0 / (len(empty) + 1) for p in range(area)] + [1.0 / (len(empty) + 1)]
    best = "pass" if turn >= int(sys.argv[1]) else f"{COLUMNS[empty[0] % size]}{size - empty[0] // size}"
    second = f"{COLUMNS[empty[1] % size]}{size - empty[1] // size}"
    black_to_move = turn % 2 == 0
    lead = 20.0 if turn >= 8 else 0.0
    winrate = 0.99 if turn >= 8 else 0.5
    if not black_to_move:
        lead, winrate = -lead, 1.0 - winrate
    visits = q["maxVisits"]
    print(json.dumps({
        "id": q["id"], "isDuringSearch": False, "turnNumber": turn, "policy": policy,
        "ownership": [0.0] * area,
        "moveInfos": [{"move": best, "visits": visits - 1, "edgeVisits": visits - 1, "order": 0},
                      {"move": second, "visits": 1, "edgeVisits": 1, "order": 1}],
        "rootInfo": {"winrate": winrate, "rawWinrate": winrate, "scoreLead": lead, "visits": visits},
    }), flush=True)
'''


@pytest.fixture
def stand_in(tmp_path: pathlib.Path) -> pathlib.Path:
    script = tmp_path / "stand_in.py"
    script.write_text(STAND_IN)
    binary = tmp_path / "katago"
    binary.write_text(f"#!/bin/sh\nexec {sys.executable} -I {script} {board_spec(9).min_pass_move}\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


def test_selfplay_writes_consistent_shards(tmp_path: pathlib.Path, stand_in: pathlib.Path) -> None:
    out = tmp_path / "selfplay"
    config = SelfplayConfig(katago=str(stand_in), model="stand-in.bin.gz", out=str(out), board_size=9,
                            positions=300, games_in_flight=4, shard_rows=150, visits=32, fast_visits=4,
                            full_fraction=0.5, resign_fraction=0.5, seed=3, log_every=1e9)
    summary = selfplay(config)
    assert summary["failures"] == 0 and summary["positions"] >= 300
    assert set(summary["ends"]) == {"resign", "passes"}

    data, size = load_shards([str(out)])
    spec = board_spec(size)
    assert size == 9 and len(data["game_id"]) == summary["positions"]
    assert 0.3 < data["has_search"].mean() < 0.7
    records = [json.loads(line) for shard in sorted(out.glob("shard-*")) for line in (shard / "games.jsonl").open()]
    assert sum(r["rows"] for r in records) == summary["positions"]
    # Black leads from move 8 on: every game is a Black win, stored from the side to move.
    colors = np.where(data["turn"] % 2 == 0, 1.0, -1.0)
    np.testing.assert_array_equal(data["outcome"], colors)
    early = data["turn"] < spec.min_pass_move
    assert (data["policy_raw"][early, spec.pass_move] == 0).all()
    assert (data["policy_visits"][early & data["has_search"], spec.pass_move] == 0).all()
    assert (data["policy_visits"][~data["has_search"]] == 0).all()
    # The rows are the positions of the recorded games, in order. A resigned
    # game has a row for the position where it resigned, without its move.
    features = unpack_features(np.asarray(data["planes"]), np.asarray(data["komi_feature"]), size)
    row = 0
    for record in records:
        state = GameState(size=size)
        for i in range(record["rows"]):
            np.testing.assert_array_equal(features[row], encode_features(state))
            assert data["turn"][row] == state.move_count and data["game_id"][row] == data["game_id"][row - i]
            row += 1
            if i < len(record["moves"]):
                state = play_move(state, record["moves"][i])
                assert state is not None
        if record["end"] == "resign":
            assert record["resignEnabled"] and record["rows"] == 8 + config.resign_plies == len(record["moves"]) + 1
        else:
            assert record["moves"][-2:] == [spec.pass_move, spec.pass_move] and record["rows"] == len(record["moves"])
    assert row == len(features)


def _label(size: int, best: int, visits: np.ndarray | None = None) -> Label:
    moves = size * size + 1
    raw = np.full(moves, 1.0 / moves, dtype=np.float32)
    return Label(turn=0, policy_raw=raw, value_raw=0.0, visits=1,
                 policy_visits=np.zeros(moves, dtype=np.float32) if visits is None else visits,
                 value_search=0.0, score_lead=0.0, ownership=None, best_move=best)


def test_choose_move_never_passes_early_or_plays_illegal() -> None:
    config = SelfplayConfig(katago="", model="", out="", board_size=9, opening_moves=0, sample_moves=0,
                            raw_move_prob=0.0)
    rng = np.random.default_rng(0)
    state = GameState(size=9)
    spec = state.spec
    move, _ = choose_move(state, _label(9, spec.pass_move), rng, config)
    assert move != spec.pass_move  # move 0 < min pass move
    occupied = play_move(state, 40)
    visits = np.zeros(spec.move_count, dtype=np.float32)
    visits[40], visits[41] = 0.9, 0.1
    move, nxt = choose_move(occupied, _label(9, 40, visits), rng, config)
    assert move == 41 and nxt is not None  # the best move is taken: the next legal one by visits


def test_resignation_needs_consecutive_lopsided_values() -> None:
    config = SelfplayConfig(katago="", model="", out="", resign_plies=3, resign_threshold=0.9)
    game = Game(0, True, GameState(size=9))
    assert [resign_winner(game, v, config) for v in (-0.95, -0.95, 0.5, -0.95, -0.92)] == [0, 0, 0, 0, 0]
    assert resign_winner(game, -0.99, config) == -1
    assert resign_winner(Game(1, False, GameState(size=9)), -1.0, config) == 0
