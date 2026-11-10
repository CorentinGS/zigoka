"""go-selfplay: KataGo plays itself under Moka's rules and labels every position.

Each game asks the analysis engine about its current position, records
the labels, plays a move and asks again: one query in flight per game,
many games at once, so the engine fills its batches.

Playout cap randomization (KataGo's selfplay recipe): a fraction of the
positions gets a full search (`--visits`), the others a fast one
(`--fast-visits`) that only moves the game on. Every position keeps the
raw policy and raw value of the net; only full searches record the
visit distribution, the search value and the score (`has_search`).

Move choice: the first `--opening-moves` moves sample the raw policy,
the next `--sample-moves` sample the visits at `--temperature`, then the
search's best move, except for a raw-policy sample with probability
`--raw-move-prob`. Moves follow Moka's rules (board.py): no pass before
Moka's minimum pass move, game over after two passes or the move cap.
Before the minimum pass move the pass is also removed from the policy
targets, since Moka cannot play it.

In `--resign-fraction` of the games, a side whose search value stays
below -`--resign-threshold` for `--resign-plies` consecutive positions
resigns. Other games are played out. The result of a played-out game is
the sign of the teacher's score lead at the last position (jigo within
half a point); it goes to the `outcome` field of every row of the game.

Rows go to compact shards (shards.py) of `--shard-rows` rows, written
when a game ends; games.jsonl beside each shard lists its games. No new
game starts while the games played and those in flight (at the mean game
length so far) reach `--positions`; the games in flight finish, and if
they fall short, new games start again.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import board_spec
from .analysis import DEFAULT_CONFIG, AnalysisEngine, Label, build_query, parse_response
from .board import GameState, encode_features, play_move, sample_legal_move
from .shards import ShardWriter, shard_dirs


@dataclass
class SelfplayConfig:
    katago: str
    model: str
    out: str
    config: str = str(DEFAULT_CONFIG)
    board_size: int = 19
    positions: int = 100_000
    games_in_flight: int = 64
    shard_rows: int = 20_000
    visits: int = 64
    fast_visits: int = 8
    full_fraction: float = 0.25
    opening_moves: int | None = None
    sample_moves: int | None = None
    temperature: float = 0.6
    raw_move_prob: float = 0.06
    resign_fraction: float = 0.7
    resign_threshold: float = 0.94
    resign_plies: int = 6
    seed: int = 1
    log_every: float = 30.0

    def resolved(self) -> "SelfplayConfig":
        """Fills the board-size defaults: about 4 % and 15 % of the board."""
        area = self.board_size * self.board_size
        if self.opening_moves is None:
            self.opening_moves = round(area * 0.04)
        if self.sample_moves is None:
            self.sample_moves = round(area * 0.15)
        return self


@dataclass
class Row:
    features: np.ndarray
    label: Label
    full: bool
    color: int  # side to move: 1 Black, -1 White


@dataclass
class Game:
    game_id: int
    resign_enabled: bool
    state: GameState
    rows: list[Row] = field(default_factory=list)
    full: bool = False
    losing: deque = field(default_factory=deque)  # recent values from Black's perspective


def choose_move(state: GameState, label: Label, rng: np.random.Generator, config: SelfplayConfig,
                stats: dict[str, int] | None = None) -> tuple[int, GameState]:
    """The move to play after `label`, legal under Moka's rules, and the next state.
    `stats["fallbacks"]` counts the best moves Moka could not play."""
    turn = state.move_count
    if turn < config.opening_moves:
        return sample_legal_move(state, label.policy_raw, rng)
    if label.policy_visits.sum() > 0 and turn < config.opening_moves + config.sample_moves:
        return sample_legal_move(state, label.policy_visits, rng, temperature=config.temperature)
    if rng.random() < config.raw_move_prob:
        return sample_legal_move(state, label.policy_raw, rng)
    best = label.best_move
    if best != state.spec.pass_move or turn >= state.spec.min_pass_move:
        nxt = play_move(state, best)
        if nxt is not None:
            return best, nxt
    # KataGo's best move is not playable for Moka: the best legal one by visits, then by policy.
    if stats is not None:
        stats["fallbacks"] = stats.get("fallbacks", 0) + 1
    weights = label.policy_visits if label.policy_visits.sum() > 0 else label.policy_raw
    return sample_legal_move(state, weights, rng, temperature=0.05)


def resign_winner(game: Game, value_black: float, config: SelfplayConfig) -> int:
    """+1 / -1 when the game ends by resignation (the winner, Black's sign), 0 otherwise."""
    if not game.resign_enabled:
        return 0
    game.losing.append(value_black)
    while len(game.losing) > config.resign_plies:
        game.losing.popleft()
    if len(game.losing) < config.resign_plies:
        return 0
    if all(v >= config.resign_threshold for v in game.losing):
        return 1
    if all(v <= -config.resign_threshold for v in game.losing):
        return -1
    return 0


def _without_early_pass(label: Label, state: GameState) -> Label:
    """Moka cannot pass before its minimum pass move: drop the pass from the targets."""
    spec = state.spec
    if state.move_count >= spec.min_pass_move:
        return label
    policies = []
    for policy in (label.policy_raw, label.policy_visits):
        policy = policy.copy()
        policy[spec.pass_move] = 0.0
        total = policy.sum()
        policies.append(policy / total if total > 0 else policy)
    if policies[0].sum() == 0:
        return label  # only the pass had mass: keep the teacher's answer
    label.policy_raw, label.policy_visits = policies
    return label


class Selfplay:
    def __init__(self, config: SelfplayConfig, engine: AnalysisEngine) -> None:
        self.config = config.resolved()
        self.spec = board_spec(config.board_size)
        self.engine = engine
        self.rng = np.random.default_rng(config.seed)
        self.out = Path(config.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.shard_index = self._next_shard_index()
        self.writer = ShardWriter(config.board_size)
        self.records: list[dict] = []
        self.next_game = 0
        self.positions = 0
        self.games_done = 0
        self.full_searches = 0
        self.failures = 0
        self.ends: dict[str, int] = {}
        self.stats: dict[str, int] = {"fallbacks": 0}
        self.started = time.monotonic()

    def _next_shard_index(self) -> int:
        try:
            existing = shard_dirs([str(self.out)])
        except ValueError:
            return 0
        return 1 + max(int(p.name.split("-")[-1]) for p in existing)

    def _new_game(self) -> Game:
        game = Game(self.next_game, bool(self.rng.random() < self.config.resign_fraction),
                    GameState(size=self.config.board_size))
        self.next_game += 1
        return game

    def _submit(self, game: Game) -> Future:
        game.full = bool(self.rng.random() < self.config.full_fraction)
        visits = self.config.visits if game.full else self.config.fast_visits
        query = build_query(f"g{game.game_id}-t{game.state.move_count}", game.state.history, self.config.board_size,
                            max_visits=visits)
        return self.engine.submit(query)

    def _finish(self, game: Game, black_result: float, reason: str) -> None:
        for row in game.rows:
            label = row.label
            kwargs = {}
            if row.full:
                kwargs = {"policy_visits": label.policy_visits, "value_search": label.value_search,
                          "score_lead": label.score_lead}
            self.writer.add(features=row.features, policy_raw=label.policy_raw, value_raw=label.value_raw,
                            game_id=game.game_id, turn=label.turn, ownership=label.ownership,
                            outcome=black_result * row.color, **kwargs)
        self.records.append({"gameId": game.game_id, "moves": game.state.history, "blackResult": black_result,
                             "end": reason, "resignEnabled": game.resign_enabled, "rows": len(game.rows)})
        self.positions += len(game.rows)
        self.games_done += 1
        self.ends[reason] = self.ends.get(reason, 0) + 1
        if len(self.writer) >= self.config.shard_rows:
            self._write_shard()

    def _write_shard(self) -> None:
        if not len(self.writer):
            return
        directory = self.out / f"shard-{self.shard_index:05d}"
        meta = {"source": "go-selfplay", "teacher": Path(self.config.model).name, "games": len(self.records),
                "config": asdict(self.config)}
        self.writer.write(directory, meta)
        with open(directory / "games.jsonl", "w") as f:
            for record in self.records:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(f"wrote {directory} ({len(self.writer)} rows, {len(self.records)} games)", flush=True)
        self.shard_index += 1
        self.writer = ShardWriter(self.config.board_size)
        self.records = []

    def _step(self, game: Game, responses: list[dict]) -> bool:
        """Records the position and plays a move; True while the game goes on."""
        state = game.state
        label = _without_early_pass(parse_response(responses[0], self.config.board_size), state)
        if label.turn != state.move_count:
            raise RuntimeError(f"game {game.game_id}: KataGo analyzed turn {label.turn}, expected {state.move_count}")
        self.full_searches += game.full
        game.rows.append(Row(encode_features(state), label, game.full, int(state.next_color)))
        winner = resign_winner(game, label.value_search * state.next_color, self.config)
        if winner:
            self._finish(game, float(winner), "resign")
            return False
        _, game.state = choose_move(state, label, self.rng, self.config, self.stats)
        if game.state.is_game_over() or game.state.move_count >= self.spec.max_moves:
            lead_black = label.score_lead * state.next_color
            result = 0.0 if abs(lead_black) < 0.5 else float(np.sign(lead_black))
            self._finish(game, result, "passes" if game.state.is_game_over() else "move-cap")
            return False
        return True

    def _needs_games(self, in_flight: dict[Future, Game]) -> bool:
        """True while the games played and those in flight, at the mean game
        length so far, fall short of the target. Games in flight always finish."""
        mean_length = self.positions / self.games_done if self.games_done else 0.6 * self.spec.area
        expected = sum(max(len(g.rows), mean_length) for g in in_flight.values())
        return self.positions + expected < self.config.positions

    def _log(self) -> None:
        elapsed = time.monotonic() - self.started
        print(f"{self.positions} positions, {self.games_done} games, {self.positions / max(elapsed, 1e-9):.1f} pos/s, "
              f"ends {self.ends}, failures {self.failures}, fallbacks {self.stats['fallbacks']}", flush=True)

    def run(self) -> dict:
        in_flight: dict[Future, Game] = {}
        last_log = time.monotonic()
        while True:
            while len(in_flight) < self.config.games_in_flight and self._needs_games(in_flight):
                game = self._new_game()
                in_flight[self._submit(game)] = game
            if not in_flight:
                break
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for future in done:
                game = in_flight.pop(future)
                try:
                    alive = self._step(game, future.result())
                except RuntimeError as error:
                    if self.engine.process.poll() is not None:
                        raise
                    self.failures += 1
                    print(f"game {game.game_id} dropped: {error}", flush=True)
                    if self.failures > max(20, self.games_done):
                        raise RuntimeError(f"{self.failures} games dropped for {self.games_done} played") from error
                    alive = False
                if alive:
                    in_flight[self._submit(game)] = game
            if time.monotonic() - last_log >= self.config.log_every:
                self._log()
                last_log = time.monotonic()
        self._write_shard()
        self._log()
        return {"positions": self.positions, "games": self.games_done, "fullSearches": self.full_searches,
                "ends": self.ends, "failures": self.failures, "fallbacks": self.stats["fallbacks"],
                "seconds": time.monotonic() - self.started}


def selfplay(config: SelfplayConfig) -> dict:
    overrides = {"logDir": str(Path(config.out) / "katago_logs"), "numAnalysisThreads": config.games_in_flight}
    with AnalysisEngine(config.katago, config.model, config.config, overrides) as engine:
        summary = Selfplay(config, engine).run()
    if engine.warnings:
        print(f"KataGo warnings: {engine.warnings[:5]}", flush=True)
    return summary


def main() -> None:
    defaults = SelfplayConfig(katago="", model="", out="")
    parser = argparse.ArgumentParser(description="KataGo self-play under Moka's rules, labelled by the analysis engine.")
    parser.add_argument("--katago", required=True, help="katago binary (pipeline/katago/katago-rocm on AMD)")
    parser.add_argument("--model", required=True, help="KataGo network (.bin.gz)")
    parser.add_argument("--out", required=True, help="shard directory; new shards follow the existing ones")
    parser.add_argument("--config", default=defaults.config)
    parser.add_argument("--board-size", type=int, default=defaults.board_size, choices=(9, 19))
    for name in ("positions", "games_in_flight", "shard_rows", "visits", "fast_visits", "opening_moves",
                 "sample_moves", "resign_plies", "seed"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=int, default=getattr(defaults, name))
    for name in ("full_fraction", "temperature", "raw_move_prob", "resign_fraction", "resign_threshold", "log_every"):
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=getattr(defaults, name))
    summary = selfplay(SelfplayConfig(**vars(parser.parse_args())))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
