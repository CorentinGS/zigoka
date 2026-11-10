"""The KataGo analysis client against real responses of KataGo 1.18.1.

tests/fixtures/katago/queries.jsonl holds the queries, responses.jsonl
what `katago analysis` (b18c384nbt, pipeline/katago/analysis.cfg, ROCm
build) answered. The engine tests replace KataGo with a replayer of those
responses, so they run without a GPU.
"""
from __future__ import annotations

import json
import pathlib
import stat
import sys

import numpy as np
import pytest

from go_pipeline.analysis import MOKA_RULES, AnalysisEngine, build_query, from_vertex, parse_response, to_vertex

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "katago"
SIZE = 19


def _jsonl(name: str) -> list[dict]:
    return [json.loads(line) for line in (FIXTURES / name).read_text().splitlines()]


def _responses(query_id: str) -> list[dict]:
    return [r for r in _jsonl("responses.jsonl") if r.get("id") == query_id]


def test_vertices_round_trip_and_skip_i() -> None:
    assert to_vertex(0, SIZE) == "A19" and to_vertex(SIZE * SIZE - 1, SIZE) == "T1"
    assert to_vertex(8, SIZE) == "J19"  # no I column
    assert to_vertex(SIZE * SIZE, SIZE) == "pass" and from_vertex("pass", SIZE) == SIZE * SIZE
    assert all(from_vertex(to_vertex(m, SIZE), SIZE) == m for m in range(SIZE * SIZE + 1))
    assert from_vertex("C3", 9) == 6 * 9 + 2
    for bad in ("I5", "U1", "A20", "A0", "", "Q"):
        with pytest.raises(ValueError):
            from_vertex(bad, SIZE)


def test_build_query_matches_the_fixture_queries() -> None:
    for fixture in _jsonl("queries.jsonl"):
        if fixture["id"] == "bad-rules":
            continue
        moves = [from_vertex(vertex, SIZE) for _, vertex in fixture["moves"]]
        assert [color for color, _ in fixture["moves"]] == ["B", "W"] * (len(moves) // 2) + ["B"] * (len(moves) % 2)
        query = build_query(fixture["id"], moves, SIZE, max_visits=16, analyze_turns=fixture.get("analyzeTurns"))
        assert query == {**fixture, "komi": 7.0, "rules": MOKA_RULES}


def test_parse_turns_policy_visits_and_perspective() -> None:
    labels = [parse_response(r, SIZE) for r in sorted(_responses("turns"), key=lambda r: r["turnNumber"])]
    assert [label.turn for label in labels] == [0, 1, 3]
    q16, d4, q4 = (from_vertex(v, SIZE) for v in ("Q16", "D4", "Q4"))
    for label, occupied in zip(labels, ([], [q16], [q16, d4, q4])):
        assert label.policy_raw.shape == (SIZE * SIZE + 1,) and label.policy_visits.shape == (SIZE * SIZE + 1,)
        assert abs(float(label.policy_raw.sum()) - 1) < 1e-5 and abs(float(label.policy_visits.sum()) - 1) < 1e-5
        assert (label.policy_raw >= 0).all() and all(label.policy_raw[m] == 0 for m in occupied)
        assert all(label.policy_visits[m] == 0 for m in occupied)
        assert -1 <= label.value_raw <= 1 and -1 <= label.value_search <= 1
        assert label.visits == 16 and label.ownership.shape == (SIZE * SIZE,)
    # Ownership is from the side to move: White to move sees Black's Q16 as the opponent's.
    assert labels[1].ownership[q16] < -0.5 and labels[2].ownership[q16] < -0.5
    assert labels[0].best_move == q16 and labels[2].best_move == from_vertex("D16", SIZE)


def test_parse_lopsided_position_from_the_loser() -> None:
    label = parse_response(_responses("black-ahead-white-to-move")[0], SIZE)
    assert label.turn == 23
    assert label.value_search < -0.99 and label.value_raw < -0.99 and label.score_lead < -100
    black_stones = [from_vertex(v, SIZE) for v in ("D4", "K10", "Q16")]
    assert all(label.ownership[m] < -0.5 for m in black_stones)
    # The search's first move is the one with order 0, not the first listed by chance.
    infos = _responses("black-ahead-white-to-move")[0]["moveInfos"]
    assert label.best_move == from_vertex(min(infos, key=lambda i: i["order"])["move"], SIZE)


def test_parse_after_a_pass_and_error() -> None:
    label = parse_response(_responses("pass")[0], SIZE)
    assert label.turn == 3 and label.score_lead < 0  # White passed: White to move, behind
    with pytest.raises(RuntimeError, match="rules"):
        parse_response(_responses("bad-rules")[0], SIZE)


REPLAYER = r'''
import json, sys
responses = [json.loads(l) for l in open(sys.argv[1])]
print("KataGo replayer: started", file=sys.stderr, flush=True)
print("not json, ignored", flush=True)
print(json.dumps({"warning": "a warning, collected"}), flush=True)
for line in sys.stdin:
    query = json.loads(line)
    if query["id"] == "crash":
        print("replayer: fatal error", file=sys.stderr, flush=True)
        sys.exit(3)
    mine = [r for r in responses if r.get("id") == query["id"]]
    for r in reversed(mine):  # out of turn order, like concurrent searches
        print(json.dumps({**r, "isDuringSearch": False}), flush=True)
'''


@pytest.fixture
def replayer(tmp_path: pathlib.Path) -> pathlib.Path:
    script = tmp_path / "replayer.py"
    script.write_text(REPLAYER)
    binary = tmp_path / "katago"
    binary.write_text(f"#!/bin/sh\nexec {sys.executable} -I {script} {FIXTURES / 'responses.jsonl'}\n")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary


def test_engine_routes_responses_errors_and_exit(replayer: pathlib.Path) -> None:
    queries = {q["id"]: q for q in _jsonl("queries.jsonl")}
    with AnalysisEngine(str(replayer), model="unused") as engine:
        turns = engine.submit(queries["turns"])
        lopsided = engine.submit(queries["black-ahead-white-to-move"])
        bad = engine.submit(queries["bad-rules"])
        assert [r["turnNumber"] for r in turns.result(timeout=30)] == [0, 1, 3]
        assert lopsided.result(timeout=30)[0]["turnNumber"] == 23
        with pytest.raises(RuntimeError, match="rules"):
            bad.result(timeout=30)
        engine.submit(queries["turns"] | {"id": "unanswered"})
        with pytest.raises(ValueError, match="already pending"):
            engine.submit(queries["turns"] | {"id": "unanswered"})
        crash = engine.submit({"id": "crash"})
        with pytest.raises(RuntimeError, match="code 3") as error:
            crash.result(timeout=30)
        assert "fatal error" in str(error.value)
        assert engine.warnings == [{"warning": "a warning, collected"}]


def test_policy_entries_map_to_moka_indices() -> None:
    response = max(_responses("turns"), key=lambda r: r["turnNumber"])
    raw = np.asarray(response["policy"])
    illegal = {int(i) for i in np.flatnonzero(raw < 0)}
    assert illegal == {from_vertex(v, SIZE) for v in ("Q16", "D4", "Q4")}
