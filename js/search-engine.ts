import { instantiateWithWeights, type InferExports } from "./engine.ts";

const STATUS_OK = 0;
const STATUS_POOL_FULL = 5;
const STATUS_MESSAGES: Record<number, string> = {
  1: "The model weights are not loaded.",
  2: "The batch size must be between 1 and the engine's maximum batch.",
  3: "Model produced a non-finite output.",
  4: "The search started from a finished game.",
  6: "The search was not started.",
};
const MAX_TARGET = 0xffffffff;
const NO_MOVE = -1;

const POSITION_KO_OFFSET = 0;
const POSITION_MOVE_COUNT_OFFSET = 2;
const POSITION_LAST_MOVE_OFFSET = 4;
const POSITION_PREVIOUS_MOVE_OFFSET = 6;
const POSITION_PASS_COUNT_OFFSET = 8;
const POSITION_NEXT_COLOR_OFFSET = 9;
const POSITION_BOARD_OFFSET = 10;
const MAX_PASS_COUNT = 255;

interface SearchExports extends InferExports {
  position_ptr: () => number;
  moves_ptr: () => number;
  rules_play: (move: number) => number;
  rules_legal_moves: () => number;
  rules_encode: () => void;
  rules_is_game_over: () => number;
  rules_area_score: () => number;
  rules_dead_stones: () => number;
  search_start: () => number;
  search_run: (targetVisits: number, batch: number) => number;
  search_visits: () => number;
  search_nodes: () => number;
  search_node_capacity: () => number;
  search_best_move: () => number;
  search_child_count: () => number;
  search_child_move: (index: number) => number;
  search_child_visits: (index: number) => number;
}

type SearchOptions = { timeMs: number; maxVisits?: number; batch?: number };

type SearchOutcome = { move: number; visits: number; poolFull: boolean };

interface SearchEngine {
  search: (state: GoGameState, options: SearchOptions) => SearchOutcome;
  rootVisits: () => { move: number; visits: number }[];
  exports: SearchExports;
}

/** Copies the part of `state` the rules read into the engine's position: board, ko, counters and the last two moves. */
const writePosition = (exports: SearchExports, state: GoGameState) => {
  const pointer = exports.position_ptr();
  const view = new DataView(exports.memory.buffer, pointer);
  const area = state.board.length;

  view.setInt16(POSITION_KO_OFFSET, state.koMove, true);
  view.setUint16(POSITION_MOVE_COUNT_OFFSET, state.moveCount, true);
  view.setInt16(POSITION_LAST_MOVE_OFFSET, state.moveHistory.at(-1) ?? NO_MOVE, true);
  view.setInt16(POSITION_PREVIOUS_MOVE_OFFSET, state.moveHistory.at(-2) ?? NO_MOVE, true);
  view.setUint8(POSITION_PASS_COUNT_OFFSET, Math.min(state.consecutivePassCount, MAX_PASS_COUNT));
  view.setInt8(POSITION_NEXT_COLOR_OFFSET, state.nextColor);
  new Int8Array(exports.memory.buffer, pointer + POSITION_BOARD_OFFSET, area).set(state.board);
};

/** Reads back what `writePosition` writes, with the board as a copy. */
const readPosition = (exports: SearchExports, area: number) => {
  const pointer = exports.position_ptr();
  const view = new DataView(exports.memory.buffer, pointer);

  return {
    koMove: view.getInt16(POSITION_KO_OFFSET, true),
    moveCount: view.getUint16(POSITION_MOVE_COUNT_OFFSET, true),
    lastMove: view.getInt16(POSITION_LAST_MOVE_OFFSET, true),
    previousMove: view.getInt16(POSITION_PREVIOUS_MOVE_OFFSET, true),
    consecutivePassCount: view.getUint8(POSITION_PASS_COUNT_OFFSET),
    nextColor: view.getInt8(POSITION_NEXT_COLOR_OFFSET),
    board: new Int8Array(exports.memory.buffer, pointer + POSITION_BOARD_OFFSET, area).slice(),
  };
};

const readMoves = (exports: SearchExports, count: number) =>
  Array.from(new Uint16Array(exports.memory.buffer, exports.moves_ptr(), count));

const checkStatus = (status: number) => {
  if (status !== STATUS_OK) {
    throw new Error(STATUS_MESSAGES[status] ?? `The search failed with status ${status}.`);
  }
};

const createSearchEngine = async (
  module: WebAssembly.Module,
  weights: ArrayBuffer,
): Promise<SearchEngine> => {
  const exports = await instantiateWithWeights<SearchExports>(module, weights);

  const search = (state: GoGameState, { timeMs, maxVisits = Infinity, batch = 1 }: SearchOptions) => {
    const startedAt = performance.now();
    const deadline = startedAt + timeMs;

    writePosition(exports, state);
    checkStatus(exports.search_start());

    const rootMs = performance.now() - startedAt;

    let visits = exports.search_visits();
    let chunk = 1;
    let poolFull = false;

    while (visits < maxVisits) {
      const chunkStartedAt = performance.now();

      if (chunkStartedAt >= deadline) {
        break;
      }

      const status = exports.search_run(Math.min(maxVisits, visits + chunk, MAX_TARGET), batch);
      const finishedAt = performance.now();
      const done = exports.search_visits() - visits;

      visits += done;

      if (status === STATUS_POOL_FULL) {
        poolFull = true;
        break;
      }

      checkStatus(status);

      const msPerVisit = Math.max((finishedAt - chunkStartedAt) / Math.max(done, 1), rootMs);

      chunk = Math.max(1, Math.floor((deadline - finishedAt) / msPerVisit / 2));
    }

    return { move: exports.search_best_move(), visits, poolFull };
  };

  const rootVisits = () =>
    Array.from({ length: exports.search_child_count() }, (_, index) => ({
      move: exports.search_child_move(index),
      visits: exports.search_child_visits(index),
    }));

  return { search, rootVisits, exports };
};

export { createSearchEngine, readMoves, readPosition, writePosition };
export type { SearchEngine, SearchExports, SearchOptions, SearchOutcome };
