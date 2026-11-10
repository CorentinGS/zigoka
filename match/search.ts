import {
  GO_BLACK_COLOR,
  GO_BOARD_AREA,
  GO_BOARD_SIZE,
  GO_EMPTY_COLOR,
  GO_MINIMUM_PASS_MOVE_COUNT,
  GO_PASS_MOVE,
} from "../vendor/moka/src/constants.ts";
import {
  encodeStudentFeatures,
  getAreaScore,
  getAutomaticallyDeadMoves,
  getLegalMoves,
  getPassAliveArea,
  isGameOver,
  playMove,
  removeDeadStones,
} from "../vendor/moka/src/game.ts";

const EXPLORATION = 1.5;
const FIRST_PLAY_REDUCTION = 0.2;

type Evaluate = (features: Float32Array) => GoModelInference;

type SearchNode = {
  state: GoGameState;
  value: number;
  moves: number[];
  priors: Float64Array;
  children: (SearchNode | null)[];
  childVisits: Float64Array;
  childValueSums: Float64Array;
  visitCount: number;
};

type SearchLimits = { timeMs: number; maxVisits?: number };

type SearchResult = {
  move: number;
  visits: number;
  reused: number;
  children: { move: number; visits: number }[];
};

/** Keeps the tree of the last search, so that the next one can start from the matching node. */
type SearchTree = { root: SearchNode | null };

const REUSE_DEPTH = 2;

const getCandidateMoves = (state: GoGameState) => {
  const legalMoves = getLegalMoves(state);
  const moves =
    state.moveCount < GO_MINIMUM_PASS_MOVE_COUNT
      ? legalMoves.filter((move) => move !== GO_PASS_MOVE)
      : legalMoves;

  return moves.length > 0 ? moves : [GO_PASS_MOVE];
};

/** Moka's dead stones without the capture aftermath: the stones inside the opponent's pass-alive area. */
const getPassDeadMoves = (state: GoGameState) => {
  const passAliveArea = getPassAliveArea(state.board);
  const moves: number[] = [];

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (state.board[move] !== GO_EMPTY_COLOR && passAliveArea[move] === -state.board[move]) {
      moves.push(move);
    }
  }

  return moves;
};

/**
 * The score the search gives a finished game, from Black's side. At 9×9, Moka's final score. At 19×19,
 * only the pass-dead stones are removed: Moka's capture aftermath also takes every chain left in atari,
 * so the search passed instead of playing captures that the final score then counted.
 */
const getSearchScore = (state: GoGameState) => {
  const deadMoves = GO_BOARD_SIZE === 19 ? getPassDeadMoves(state) : getAutomaticallyDeadMoves(state);

  return getAreaScore(removeDeadStones(state, deadMoves));
};

/** Sign of the final area score, for the player to move. */
const getFinalValue = (state: GoGameState) => {
  const blackResult = Math.sign(getSearchScore(state));

  return state.nextColor === GO_BLACK_COLOR ? blackResult : -blackResult;
};

const getPriors = (moves: number[], logits: Float32Array) => {
  const priors = new Float64Array(moves.length);
  let maximum = -Infinity;

  for (const move of moves) {
    maximum = Math.max(maximum, logits[move]);
  }

  let sum = 0;

  moves.forEach((move, index) => {
    priors[index] = Math.exp(logits[move] - maximum);
    sum += priors[index];
  });

  return priors.map((prior) => prior / sum);
};

const createNode = (state: GoGameState, evaluate: Evaluate): SearchNode => {
  if (isGameOver(state)) {
    return {
      state,
      value: getFinalValue(state),
      moves: [],
      priors: new Float64Array(0),
      children: [],
      childVisits: new Float64Array(0),
      childValueSums: new Float64Array(0),
      visitCount: 0,
    };
  }

  const inference = evaluate(encodeStudentFeatures(state));
  const moves = getCandidateMoves(state);

  return {
    state,
    value: inference.value,
    moves,
    priors: getPriors(moves, inference.policyLogits),
    children: moves.map(() => null),
    childVisits: new Float64Array(moves.length),
    childValueSums: new Float64Array(moves.length),
    visitCount: 0,
  };
};

const selectChild = (node: SearchNode) => {
  const exploration = EXPLORATION * Math.sqrt(node.visitCount + 1);
  const firstPlayValue = node.value - FIRST_PLAY_REDUCTION;
  let selected = 0;
  let selectedScore = -Infinity;

  for (let index = 0; index < node.moves.length; index += 1) {
    const visits = node.childVisits[index];
    const meanValue = visits > 0 ? node.childValueSums[index] / visits : firstPlayValue;
    const score = meanValue + (exploration * node.priors[index]) / (1 + visits);

    if (score > selectedScore) {
      selected = index;
      selectedScore = score;
    }
  }

  return selected;
};

/** Runs one simulation from `root` and returns the value it backed up, for the player to move. */
const simulate = (root: SearchNode, evaluate: Evaluate) => {
  const path: [SearchNode, number][] = [];
  let node = root;
  let value: number;

  while (true) {
    if (node.moves.length === 0) {
      value = node.value;
      break;
    }

    const index = selectChild(node);
    const child = node.children[index];

    path.push([node, index]);

    if (child === null) {
      const nextState = playMove(node.state, node.moves[index]);

      if (!nextState) {
        throw new Error(`The search reached an illegal move (${node.moves[index]}).`);
      }

      const created = createNode(nextState, evaluate);

      node.children[index] = created;
      value = created.value;
      break;
    }

    node = child;
  }

  for (let step = path.length - 1; step >= 0; step -= 1) {
    const [parent, index] = path[step];

    value = -value;
    parent.childVisits[index] += 1;
    parent.childValueSums[index] += value;
    parent.visitCount += 1;
  }

  return value;
};

const createSearchTree = (): SearchTree => ({ root: null });

const isSamePosition = (left: GoGameState, right: GoGameState) =>
  left.koMove === right.koMove &&
  left.moveCount === right.moveCount &&
  left.nextColor === right.nextColor &&
  left.consecutivePassCount === right.consecutivePassCount &&
  left.moveHistory.at(-1) === right.moveHistory.at(-1) &&
  left.moveHistory.at(-2) === right.moveHistory.at(-2) &&
  left.board.every((point, index) => point === right.board[index]);

/** Finds the node of `state` within `REUSE_DEPTH` moves of `node`. */
const findNode = (node: SearchNode, state: GoGameState, depth = 0): SearchNode | null => {
  if (isSamePosition(node.state, state)) {
    return node;
  }

  if (depth === REUSE_DEPTH) {
    return null;
  }

  for (const child of node.children) {
    const found = child === null ? null : findNode(child, state, depth + 1);

    if (found !== null) {
      return found;
    }
  }

  return null;
};

const search = (
  state: GoGameState,
  evaluate: Evaluate,
  limits: SearchLimits,
  tree?: SearchTree,
): SearchResult => {
  const deadline = performance.now() + limits.timeMs;
  const maxVisits = limits.maxVisits ?? Infinity;
  const reusedRoot = tree?.root ? findNode(tree.root, state) : null;
  const root = reusedRoot ?? createNode(state, evaluate);
  const reused = root.visitCount;

  if (root.moves.length === 0) {
    throw new Error("The search started from a finished game.");
  }

  while (root.visitCount < maxVisits && performance.now() < deadline) {
    simulate(root, evaluate);
  }

  if (tree) {
    tree.root = root;
  }

  let best = 0;

  for (let index = 1; index < root.moves.length; index += 1) {
    const visits = root.childVisits[index];
    const bestVisits = root.childVisits[best];

    if (visits > bestVisits || (visits === bestVisits && root.priors[index] > root.priors[best])) {
      best = index;
    }
  }

  return {
    move: root.moves[best],
    visits: root.visitCount,
    reused,
    children: root.moves.map((move, index) => ({ move, visits: root.childVisits[index] })),
  };
};

export { createSearchTree, getCandidateMoves, getSearchScore, search };
export type { Evaluate, SearchLimits, SearchResult, SearchTree };
