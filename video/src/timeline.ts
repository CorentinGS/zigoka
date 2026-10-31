import { createGameState, playMove } from "../../vendor/moka/src/game.ts";
import { summarize, type GameResult, type MatchConfig } from "../../match/games.ts";
import data from "./match-data.json";

const FPS = 30;
const REAL_TIME_SECONDS = 8;
const RAMP_SECONDS = 3;
const FAST_SPEED = 6;
const END_HOLD_SECONDS = 2;

type RecordedGame = (typeof data.games)[number];

type ReplayGame = RecordedGame & {
  boards: Int8Array[];
  moveTimesMs: number[];
};

type MoveEvent = { atMs: number; isFirst: boolean; visits: number };

const config = data.config as MatchConfig;

const getOpeningState = (opening: number[]) =>
  opening.reduce((state, move) => playMove(state, move) ?? state, createGameState());

const buildGame = (game: RecordedGame): ReplayGame => {
  let state = getOpeningState(game.opening);
  const boards = [state.board];
  const thinkingTotal = game.moves.reduce((sum, [, , thinkingMs]) => sum + thinkingMs, 0);
  const scale = thinkingTotal > 0 ? (game.endMs - game.startMs) / thinkingTotal : 1;
  const moveTimesMs: number[] = [];
  let elapsed = 0;

  for (const [move, , thinkingMs] of game.moves) {
    const next = playMove(state, move);

    if (!next) {
      throw new Error(`Game ${game.index + 1} replays an illegal move (${move}).`);
    }

    state = next;
    elapsed += thinkingMs * scale;
    boards.push(state.board);
    moveTimesMs.push(game.startMs + elapsed);
  }

  return { ...game, boards, moveTimesMs };
};

const games = data.games.map(buildGame);
const matchDurationMs = Math.max(...games.map((game) => game.endMs));
const gamesByEnd = [...games].sort((left, right) => left.endMs - right.endMs);
const slotGames = Array.from({ length: config.parallel }, (_, slot) =>
  games.filter((game) => game.slot === slot).sort((left, right) => left.startMs - right.startMs),
);

const moveEvents: MoveEvent[] = games
  .flatMap((game) =>
    game.moves.map(([, visits], moveIndex) => {
      const moverIsBlack = (game.opening.length + moveIndex) % 2 === 0;

      return {
        atMs: game.moveTimesMs[moveIndex],
        isFirst: moverIsBlack === game.firstPlaysBlack,
        visits,
      };
    }),
  )
  .sort((left, right) => left.atMs - right.atMs);

const countBefore = <Item>(items: Item[], limit: number, getTime: (item: Item) => number) => {
  let low = 0;
  let high = items.length;

  while (low < high) {
    const middle = (low + high) >> 1;

    if (getTime(items[middle]) <= limit) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }

  return low;
};

const rampEndMatchSeconds =
  REAL_TIME_SECONDS + RAMP_SECONDS + ((FAST_SPEED - 1) * RAMP_SECONDS) / 2;

const getSpeed = (videoSeconds: number) => {
  if (videoSeconds < REAL_TIME_SECONDS) {
    return 1;
  }

  if (videoSeconds < REAL_TIME_SECONDS + RAMP_SECONDS) {
    return 1 + ((FAST_SPEED - 1) * (videoSeconds - REAL_TIME_SECONDS)) / RAMP_SECONDS;
  }

  return FAST_SPEED;
};

const getMatchMs = (videoSeconds: number) => {
  let matchSeconds: number;

  if (videoSeconds < REAL_TIME_SECONDS) {
    matchSeconds = videoSeconds;
  } else if (videoSeconds < REAL_TIME_SECONDS + RAMP_SECONDS) {
    const rampSeconds = videoSeconds - REAL_TIME_SECONDS;

    matchSeconds =
      REAL_TIME_SECONDS +
      rampSeconds +
      ((FAST_SPEED - 1) * rampSeconds * rampSeconds) / (2 * RAMP_SECONDS);
  } else {
    matchSeconds =
      rampEndMatchSeconds + (videoSeconds - REAL_TIME_SECONDS - RAMP_SECONDS) * FAST_SPEED;
  }

  return Math.min(matchSeconds * 1000, matchDurationMs);
};

const replayFrameCount = Math.ceil(
  (REAL_TIME_SECONDS +
    RAMP_SECONDS +
    (matchDurationMs / 1000 - rampEndMatchSeconds) / FAST_SPEED +
    END_HOLD_SECONDS) *
    FPS,
);

const getSlotView = (slot: number, matchMs: number) => {
  const candidates = slotGames[slot];
  const startedCount = countBefore(candidates, matchMs, (game) => game.startMs);
  const game = candidates[Math.max(0, startedCount - 1)];
  const playedMoves = countBefore(game.moveTimesMs, matchMs, (time) => time);
  const lastRecord = playedMoves > 0 ? game.moves[playedMoves - 1] : null;
  const finished = candidates
    .slice(0, startedCount)
    .filter((candidate) => candidate.endMs <= matchMs)
    .at(-1);

  return {
    game,
    board: game.boards[playedMoves],
    lastMove: lastRecord ? lastRecord[0] : game.opening.at(-1) ?? null,
    lastVisits: lastRecord ? lastRecord[1] : null,
    moveNumber: game.opening.length + playedMoves,
    isOver: matchMs >= game.endMs,
    lastFinished: finished ?? null,
  };
};

const getStats = (matchMs: number) => {
  const finished = gamesByEnd.slice(0, countBefore(gamesByEnd, matchMs, (game) => game.endMs));
  const events = moveEvents.slice(0, countBefore(moveEvents, matchMs, (event) => event.atMs));
  const sides = { first: { moves: 0, visits: 0 }, second: { moves: 0, visits: 0 } };

  for (const event of events) {
    const side = event.isFirst ? sides.first : sides.second;

    side.moves += 1;
    side.visits += event.visits;
  }

  return {
    finished,
    summary: finished.length > 0 ? summarize(config, finished as unknown as GameResult[]) : null,
    history: finished.map((_, count) => summarize(config, finished.slice(0, count + 1) as unknown as GameResult[])),
    firstVisitsPerMove: sides.first.moves > 0 ? sides.first.visits / sides.first.moves : null,
    secondVisitsPerMove: sides.second.moves > 0 ? sides.second.visits / sides.second.moves : null,
  };
};

const finalStats = getStats(matchDurationMs);

export {
  config,
  finalStats,
  FPS,
  getMatchMs,
  getSlotView,
  getSpeed,
  getStats,
  matchDurationMs,
  replayFrameCount,
};
export type { ReplayGame };
