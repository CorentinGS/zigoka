import {
  GO_BLACK_COLOR,
  GO_MINIMUM_PASS_MOVE_COUNT,
  GO_PASS_MOVE,
} from "../vendor/moka/src/constants.ts";
import {
  encodeStudentFeatures,
  getAreaScore,
  getAutomaticallyDeadMoves,
  getLegalMoves,
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

type SearchResult = { move: number; visits: number; children: { move: number; visits: number }[] };

const getCandidateMoves = (state: GoGameState) => {
  const legalMoves = getLegalMoves(state);
  const moves =
    state.moveCount < GO_MINIMUM_PASS_MOVE_COUNT
      ? legalMoves.filter((move) => move !== GO_PASS_MOVE)
      : legalMoves;

  return moves.length > 0 ? moves : [GO_PASS_MOVE];
};

const getFinalValue = (state: GoGameState) => {
  const score = getAreaScore(removeDeadStones(state, getAutomaticallyDeadMoves(state)));
  const blackResult = Math.sign(score);

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

const search = (state: GoGameState, evaluate: Evaluate, limits: SearchLimits): SearchResult => {
  const deadline = performance.now() + limits.timeMs;
  const maxVisits = limits.maxVisits ?? Infinity;
  const root = createNode(state, evaluate);

  if (root.moves.length === 0) {
    throw new Error("The search started from a finished game.");
  }

  while (root.visitCount < maxVisits && performance.now() < deadline) {
    simulate(root, evaluate);
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
    children: root.moves.map((move, index) => ({ move, visits: root.childVisits[index] })),
  };
};

export { getCandidateMoves, search };
export type { Evaluate, SearchLimits, SearchResult };
