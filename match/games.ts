import { GO_BLACK_COLOR } from "../vendor/moka/src/constants.ts";
import {
  createGameState,
  encodeStudentFeatures,
  getAreaScore,
  getAutomaticallyDeadMoves,
  isGameOver,
  playMove,
  removeDeadStones,
  selectHighestLegalMove,
} from "../vendor/moka/src/game.ts";
import { createMulberry32 } from "../oracle/mulberry32.ts";
import { getCandidateMoves, search, type Evaluate } from "./search.ts";

const PLAYER_KINDS = [
  "zig-reuse",
  "zig-search",
  "zig-puct",
  "moka-puct",
  "zig-policy",
  "moka-policy",
] as const;
const OPENING_ATTEMPT_LIMIT = 100;
const Z_95 = 1.96;
const ELO_SCALE = 400;

type PlayerKind = (typeof PLAYER_KINDS)[number];

type EngineName = "zig" | "moka";

type SearchInWasm = (
  state: GoGameState,
  options: { timeMs: number; maxVisits?: number; batch: number; reuse: boolean },
) => { move: number; visits: number };

type Engines = Record<EngineName, Evaluate> & { search: SearchInWasm };

type MatchConfig = {
  first: PlayerKind;
  second: PlayerKind;
  firstModel?: string;
  secondModel?: string;
  games: number;
  timeMs: number;
  maxVisits?: number;
  batch: number;
  openingMoves: number;
  parallel: number;
  seed: number;
};

type GameTask = { index: number; opening: number[]; firstPlaysBlack: boolean };

type SideStats = { moves: number; visits: number; thinkingMs: number };

type MoveRecord = { move: number; visits: number; thinkingMs: number };

type GameResult = {
  index: number;
  opening: number[];
  firstPlaysBlack: boolean;
  score: number;
  firstResult: 1 | 0 | -1;
  moveCount: number;
  moves: MoveRecord[];
  first: SideStats;
  second: SideStats;
};

type Player = { kind: PlayerKind; play: (state: GoGameState) => { move: number; visits: number } };

type MoveEvent = { state: GoGameState; move: number; visits: number; thinkingMs: number };

type MatchSummary = ReturnType<typeof summarize>;

const isPlayerKind = (kind: unknown): kind is PlayerKind =>
  PLAYER_KINDS.includes(kind as PlayerKind);

/** Players that keep state in `search.wasm`: two of them need two engine instances. */
const usesSearchWasm = (kind: PlayerKind) => kind === "zig-search" || kind === "zig-reuse";

const createPlayer = (kind: PlayerKind, engines: Engines, config: MatchConfig): Player => {
  const [engine, mode] = kind.split("-") as [EngineName, "reuse" | "search" | "puct" | "policy"];
  const evaluate = engines[engine];

  if (mode === "search" || mode === "reuse") {
    const options = {
      timeMs: config.timeMs,
      maxVisits: config.maxVisits,
      batch: config.batch,
      reuse: mode === "reuse",
    };

    return { kind, play: (state) => engines.search(state, options) };
  }

  if (mode === "puct") {
    return {
      kind,
      play: (state) => search(state, evaluate, { timeMs: config.timeMs, maxVisits: config.maxVisits }),
    };
  }

  return {
    kind,
    play: (state) => ({
      move: selectHighestLegalMove(state, evaluate(encodeStudentFeatures(state)).policyLogits),
      visits: 1,
    }),
  };
};

const playOpening = (moves: number[]) =>
  moves.reduce<GoGameState>((state, move) => {
    const next = playMove(state, move);

    if (!next) {
      throw new Error(`The opening contains an illegal move (${move}).`);
    }

    return next;
  }, createGameState());

const getBlackScore = (state: GoGameState) =>
  getAreaScore(removeDeadStones(state, getAutomaticallyDeadMoves(state)));

const playGame = (
  task: GameTask,
  first: Player,
  second: Player,
  onMove: (event: MoveEvent) => void = () => undefined,
): GameResult => {
  const stats = {
    first: { moves: 0, visits: 0, thinkingMs: 0 },
    second: { moves: 0, visits: 0, thinkingMs: 0 },
  };
  const moves: MoveRecord[] = [];
  let state = playOpening(task.opening);

  while (!isGameOver(state)) {
    const firstToMove = (state.nextColor === GO_BLACK_COLOR) === task.firstPlaysBlack;
    const player = firstToMove ? first : second;
    const side = firstToMove ? stats.first : stats.second;
    const startedAt = performance.now();
    const { move, visits } = player.play(state);
    const thinkingMs = performance.now() - startedAt;

    side.thinkingMs += thinkingMs;
    side.moves += 1;
    side.visits += visits;
    moves.push({ move, visits, thinkingMs });

    const next = playMove(state, move);

    if (!next) {
      throw new Error(`${player.kind} played an illegal move (${move}).`);
    }

    state = next;
    onMove({ state, move, visits, thinkingMs });
  }

  const score = getBlackScore(state);
  const blackResult = Math.sign(score) as 1 | 0 | -1;

  return {
    index: task.index,
    opening: task.opening,
    firstPlaysBlack: task.firstPlaysBlack,
    score,
    firstResult: (task.firstPlaysBlack ? blackResult : -blackResult) as 1 | 0 | -1,
    moveCount: state.moveCount,
    moves,
    ...stats,
  };
};

const sampleMove = (state: GoGameState, evaluate: Evaluate, random: () => number) => {
  const moves = getCandidateMoves(state);
  const logits = evaluate(encodeStudentFeatures(state)).policyLogits;
  const maximum = Math.max(...moves.map((move) => logits[move]));
  const weights = moves.map((move) => Math.exp(logits[move] - maximum));
  let threshold = random() * weights.reduce((sum, weight) => sum + weight, 0);

  for (let index = 0; index < moves.length; index += 1) {
    threshold -= weights[index];

    if (threshold <= 0) {
      return moves[index];
    }
  }

  return moves[moves.length - 1];
};

const createOpenings = (count: number, length: number, evaluate: Evaluate, seed: number) => {
  const random = createMulberry32(seed);
  const openings = new Map<string, number[]>();
  let attempts = 0;

  while (openings.size < count) {
    attempts += 1;

    if (attempts > count * OPENING_ATTEMPT_LIMIT) {
      throw new Error(`Could not draw ${count} distinct openings of ${length} moves.`);
    }

    const moves: number[] = [];
    let state = createGameState();

    while (moves.length < length) {
      const move = sampleMove(state, evaluate, random);
      const next = playMove(state, move);

      if (!next) {
        throw new Error(`The sampled opening move ${move} is illegal.`);
      }

      moves.push(move);
      state = next;
    }

    openings.set(moves.join(","), moves);
  }

  return [...openings.values()];
};

const createTasks = (openings: number[][]): GameTask[] =>
  openings.flatMap((opening, openingIndex) => [
    { index: openingIndex * 2, opening, firstPlaysBlack: true },
    { index: openingIndex * 2 + 1, opening, firstPlaysBlack: false },
  ]);

const summarizeSide = (results: GameResult[], side: "first" | "second") => {
  const moves = results.reduce((sum, result) => sum + result[side].moves, 0);
  const visits = results.reduce((sum, result) => sum + result[side].visits, 0);
  const thinkingMs = results.reduce((sum, result) => sum + result[side].thinkingMs, 0);

  return { visitsPerMove: visits / moves, msPerMove: thinkingMs / moves };
};

const toElo = (score: number) => {
  if (score <= 0) {
    return -Infinity;
  }

  if (score >= 1) {
    return Infinity;
  }

  return -ELO_SCALE * Math.log10(1 / score - 1);
};

const summarize = (config: MatchConfig, results: GameResult[]) => {
  const wins = results.filter((result) => result.firstResult === 1).length;
  const losses = results.filter((result) => result.firstResult === -1).length;
  const draws = results.length - wins - losses;
  const score = (wins + draws / 2) / results.length;
  const margin = Z_95 * Math.sqrt((score * (1 - score)) / results.length);
  const winsAsBlack = results.filter((result) => result.firstPlaysBlack && result.firstResult === 1).length;
  const winsAsWhite = results.filter((result) => !result.firstPlaysBlack && result.firstResult === 1).length;

  return {
    config,
    wins,
    losses,
    draws,
    winsAsBlack,
    winsAsWhite,
    blackWins: results.filter((result) => result.score > 0).length,
    score,
    scoreInterval: [Math.max(0, score - margin), Math.min(1, score + margin)],
    elo: toElo(score),
    eloInterval: [toElo(score - margin), toElo(score + margin)],
    averageGameLength: results.reduce((sum, result) => sum + result.moveCount, 0) / results.length,
    first: summarizeSide(results, "first"),
    second: summarizeSide(results, "second"),
  };
};

const formatElo = (elo: number) =>
  Number.isFinite(elo) ? `${elo >= 0 ? "+" : ""}${elo.toFixed(0)}` : elo > 0 ? "+∞" : "−∞";

export {
  createOpenings,
  createPlayer,
  createTasks,
  formatElo,
  isPlayerKind,
  playGame,
  PLAYER_KINDS,
  summarize,
  usesSearchWasm,
};
export type {
  EngineName,
  Engines,
  GameResult,
  GameTask,
  MatchConfig,
  MatchSummary,
  MoveEvent,
  MoveRecord,
  PlayerKind,
};
