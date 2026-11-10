"""KataGo analysis engine client: labels positions with a native KataGo net.

The engine computes its own input features, ladders included, which the
ONNX teacher of katago.py cannot. Protocol: KataGo's
docs/Analysis_Engine.md, one JSON query per line on stdin, one JSON
response per analyzed turn on stdout. tests/fixtures/katago/ holds real
responses of KataGo 1.18.1 that the parsing tests replay.

Moves are Moka indices (row-major from the top-left, pass = N²), sent as
GTP vertices: columns skip I, rows count from the bottom, as in
match/katago.ts. KataGo's policy and ownership arrays use the same
row-major order from the top-left, so they map to Moka indices directly.
The config (pipeline/katago/analysis.cfg) reports every value from the
side to move's perspective, ownership included.

What the fixtures show about KataGo 1.18.1:
* the rules object below is accepted, an unknown key is an error with
  `"field": "rules"`; "pass" is a valid move;
* one response per analyzed turn, each with its `turnNumber`;
* `rootInfo` carries `rawWinrate` (the raw net) next to `winrate`;
* illegal moves have policy -1, the pass is the last entry;
* the children's visits can add up to more than the root's: root
  symmetry pruning searches one move per symmetry class and reports it
  under every equivalent move. The visit distribution is normalized.
"""
from __future__ import annotations

import json
import subprocess
import threading
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COLUMNS = "ABCDEFGHJKLMNOPQRST"
COLORS = ("B", "W")

# Moka's rules (vendor/moka/src/game.ts), the same set as MOKA_RULES in
# match/katago.ts: simple ko, area scoring, no tax, no suicide, no button.
MOKA_RULES = {"ko": "SIMPLE", "scoring": "AREA", "tax": "NONE", "suicide": False,
              "hasButton": False, "whiteHandicapBonus": "0"}
MOKA_KOMI = 7.0

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "katago" / "analysis.cfg"


def to_vertex(move: int, size: int) -> str:
    if move == size * size:
        return "pass"
    row, col = divmod(move, size)
    return f"{COLUMNS[col]}{size - row}"


def from_vertex(vertex: str, size: int) -> int:
    text = vertex.strip().upper()
    if text == "PASS":
        return size * size
    col = COLUMNS.find(text[:1])
    try:
        row = size - int(text[1:])
    except ValueError:
        row = -1
    if col < 0 or col >= size or not 0 <= row < size:
        raise ValueError(f"vertex {vertex!r} is not on a {size}×{size} board")
    return row * size + col


def build_query(query_id: str, moves: list[int], size: int, *, max_visits: int,
                analyze_turns: list[int] | None = None, include_ownership: bool = True,
                komi: float = MOKA_KOMI, override_settings: dict | None = None) -> dict:
    """A query for the game `moves` (Black first, no handicap)."""
    query = {
        "id": query_id,
        "moves": [[COLORS[i % 2], to_vertex(move, size)] for i, move in enumerate(moves)],
        "rules": MOKA_RULES,
        "komi": komi,
        "boardXSize": size,
        "boardYSize": size,
        "maxVisits": max_visits,
        "includePolicy": True,
        "includeOwnership": include_ownership,
    }
    if analyze_turns is not None:
        query["analyzeTurns"] = analyze_turns
    if override_settings:
        query["overrideSettings"] = override_settings
    return query


@dataclass
class Label:
    """Teacher labels of one position, side to move's perspective."""

    turn: int
    policy_raw: np.ndarray        # [N²+1] probabilities, illegal moves at 0
    value_raw: float              # raw net, [-1, 1]
    visits: int                   # root visits of the search
    policy_visits: np.ndarray     # [N²+1] visit distribution of the root children
    value_search: float           # search, [-1, 1]
    score_lead: float             # search, points
    ownership: np.ndarray | None  # [N²] in [-1, 1]
    best_move: int                # move ranked first by the search


def parse_response(response: dict, size: int) -> Label:
    area = size * size
    if "error" in response:
        raise RuntimeError(f"KataGo error: {response['error']} ({response.get('field', '')})")
    policy = np.asarray(response["policy"], dtype=np.float64)
    if policy.shape != (area + 1,):
        raise ValueError(f"policy has {policy.shape} entries, expected {area + 1}")
    policy = np.where(policy > 0, policy, 0.0)
    policy /= policy.sum()
    visits = np.zeros(area + 1, dtype=np.float64)
    best_move, best_order = area, None
    for info in response["moveInfos"]:
        move = from_vertex(info["move"], size)
        visits[move] += info.get("edgeVisits", info["visits"])
        if best_order is None or info["order"] < best_order:
            best_move, best_order = move, info["order"]
    root = response["rootInfo"]
    if visits.sum() > 0:
        visits /= visits.sum()
    ownership = response.get("ownership")
    return Label(
        turn=int(response["turnNumber"]),
        policy_raw=policy.astype(np.float32),
        value_raw=2.0 * float(root["rawWinrate"]) - 1.0,
        visits=int(root["visits"]),
        policy_visits=visits.astype(np.float32),
        value_search=2.0 * float(root["winrate"]) - 1.0,
        score_lead=float(root["scoreLead"]),
        ownership=None if ownership is None else np.asarray(ownership, dtype=np.float32).reshape(area),
        best_move=best_move,
    )


class AnalysisEngine:
    """`katago analysis` as a subprocess. `submit` is thread-safe and returns
    a Future of the responses of every analyzed turn, sorted by turn."""

    def __init__(self, binary: str, model: str, config: str | Path = DEFAULT_CONFIG,
                 overrides: dict | None = None) -> None:
        command = [binary, "analysis", "-model", model, "-config", str(config)]
        if overrides:
            command += ["-override-config", ",".join(f"{k}={v}" for k, v in overrides.items())]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1)
        self._lock = threading.Lock()
        self._pending: dict[str, tuple[int, list[dict], Future]] = {}
        self.warnings: list[dict] = []
        self.log: deque[str] = deque(maxlen=40)  # tail of KataGo's stderr
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._logger = threading.Thread(target=self._read_log, daemon=True)
        self._logger.start()

    def submit(self, query: dict) -> Future:
        turns = query.get("analyzeTurns")
        expected = len(turns) if turns is not None else 1
        future: Future = Future()
        with self._lock:
            if query["id"] in self._pending:
                raise ValueError(f"query id {query['id']!r} is already pending")
            if self.process.poll() is not None:
                raise RuntimeError(self._exit_message())
            self._pending[query["id"]] = (expected, [], future)
            self.process.stdin.write(json.dumps(query) + "\n")
            self.process.stdin.flush()
        return future

    def _read_log(self) -> None:
        for line in self.process.stderr:
            self.log.append(line.rstrip())

    def _exit_message(self) -> str:
        self._logger.join(timeout=5)
        tail = "\n".join(list(self.log)[-10:])
        return f"KataGo exited with code {self.process.poll()}" + (f":\n{tail}" if tail else "")

    def _read(self) -> None:
        for line in self.process.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            response = json.loads(line)
            if "warning" in response:
                self.warnings.append(response)
                continue
            if response.get("isDuringSearch"):
                continue
            with self._lock:
                entry = self._pending.get(response.get("id"))
                if entry is None:
                    if "error" in response:
                        self._fail_all(RuntimeError(f"KataGo error: {response['error']}"))
                    continue
                expected, responses, future = entry
                if "error" in response:
                    del self._pending[response["id"]]
                    future.set_exception(RuntimeError(f"KataGo error on {response['id']}: {response['error']} "
                                                      f"(field {response.get('field', '?')})"))
                    continue
                responses.append(response)
                if len(responses) == expected:
                    del self._pending[response["id"]]
                    future.set_result(sorted(responses, key=lambda r: r["turnNumber"]))
        self.process.wait()
        with self._lock:
            self._fail_all(RuntimeError(self._exit_message()))

    def _fail_all(self, error: Exception) -> None:
        for _, _, future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.close()
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self._reader.join(timeout=5)
        self._logger.join(timeout=5)

    def __enter__(self) -> "AnalysisEngine":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
