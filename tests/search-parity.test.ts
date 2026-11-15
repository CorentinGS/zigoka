import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { loadEngines } from "../js/engines.ts";
import type { SearchEngine } from "../js/search-engine.ts";
import { listModels, type RegisteredModel } from "../models/registry.ts";
import { createMulberry32 } from "../oracle/mulberry32.ts";

type MokaRules = typeof import("./moka-rules.ts");

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const PASS_PROBABILITY = 0.05;

const loadRules = async (size: number) =>
  (await import(path.join(rootDirectory, `tests/dist/moka-rules-${size}.js`))) as MokaRules;

type Budget = {
  randomGames: number;
  policyGames: number;
  deadStoneInterval: number;
  searchPositions: number;
  searchVisits: number;
};

// One budget per board size, so that registering a model needs no edit here.
const BUDGETS: Record<number, Budget> = {
  9: { randomGames: 10, policyGames: 5, deadStoneInterval: 5, searchPositions: 8, searchVisits: 150 },
  19: { randomGames: 2, policyGames: 1, deadStoneInterval: 60, searchPositions: 2, searchVisits: 60 },
};

// moka-9, the reference model, plays the most.
const REFERENCE_MODEL_ID = "moka-9";
const REFERENCE_BUDGET: Budget = {
  randomGames: 40,
  policyGames: 10,
  deadStoneInterval: 5,
  searchPositions: 12,
  searchVisits: 150,
};

const getBudget = ({ id, boardSize }: RegisteredModel): Budget => {
  const budget = id === REFERENCE_MODEL_ID ? REFERENCE_BUDGET : BUDGETS[boardSize];

  if (!budget) {
    throw new Error(`No search-parity budget for board size ${boardSize} (model ${id}).`);
  }

  return budget;
};

const loadModelEngines = async (model: RegisteredModel) => {
  const loaded = await loadEngines(model.read, ["search.wasm", "infer.wasm"], { boardSize: model.boardSize });

  return { searchEngine: await loaded.createSearchEngine(), inferEngine: await loaded.createEngine() };
};

const MODELS = listModels().map((model) => ({
  id: model.id,
  loadRules: () => loadRules(model.boardSize),
  loadEngines: () => loadModelEngines(model),
  ...getBudget(model),
}));

/** Plays seeded games to the end: uniform over legal moves, or sampled from the policy. */
const collectGames = (
  rules: MokaRules,
  count: number,
  seed: number,
  policy?: (state: GoGameState) => Float32Array,
) => {
  const random = createMulberry32(seed);
  const games: GoGameState[][] = [];

  for (let game = 0; game < count; game += 1) {
    let state = rules.createGameState();
    const states = [state];

    while (!rules.isGameOver(state)) {
      const legalMoves = rules.getLegalMoves(state);
      const pass = legalMoves[legalMoves.length - 1];
      let move: number;

      if (policy) {
        const logits = policy(state);
        const maximum = Math.max(...legalMoves.map((candidate) => logits[candidate]));
        const weights = legalMoves.map((candidate) => Math.exp(logits[candidate] - maximum));
        let threshold = random() * weights.reduce((sum, weight) => sum + weight, 0);

        move = legalMoves.find((_, index) => (threshold -= weights[index]) <= 0) ?? pass;
      } else {
        move =
          random() < PASS_PROBABILITY || legalMoves.length === 1
            ? pass
            : legalMoves[Math.floor(random() * (legalMoves.length - 1))];
      }

      state = rules.playMove(state, move)!;
      states.push(state);
    }

    games.push(states);
  }

  return games;
};

const toBytes = (values: Float32Array) =>
  Buffer.from(values.buffer, values.byteOffset, values.byteLength);

const checkPosition = (
  rules: MokaRules,
  engine: SearchEngine,
  state: GoGameState,
  checkDeadStones: boolean,
) => {
  const zig = engine.rules;
  const label = `move ${state.moveCount}, history ${state.moveHistory.join(",")}`;
  const features = zig.encodeStudentFeatures(state);

  assert.deepEqual(zig.getLegalMoves(state), rules.getLegalMoves(state), label);
  assert.ok(toBytes(features).equals(toBytes(rules.encodeStudentFeatures(state))), `features, ${label}`);
  assert.equal(zig.isGameOver(state), rules.isGameOver(state), label);
  assert.equal(zig.getAreaScore(state), rules.getAreaScore(state), label);

  if (checkDeadStones || rules.isGameOver(state)) {
    const expected = [...rules.getAutomaticallyDeadMoves(state)].sort((left, right) => left - right);

    assert.deepEqual(zig.getAutomaticallyDeadMoves(state), expected, `dead stones, ${label}`);
    assert.equal(zig.getSearchScore(state), rules.getSearchScore(state), `search score, ${label}`);
  }
};

const checkPlay = (rules: MokaRules, engine: SearchEngine, state: GoGameState, move: number) => {
  const expected = rules.playMove(state, move);
  const played = engine.rules.play(state, move);
  const label = `move ${move} at ${state.moveCount}`;

  if (!expected) {
    assert.equal(played, null, label);
    return;
  }

  assert.deepEqual(
    played,
    {
      koMove: expected.koMove,
      moveCount: expected.moveCount,
      lastMove: expected.moveHistory.at(-1),
      previousMove: expected.moveHistory.at(-2) ?? -1,
      consecutivePassCount: expected.consecutivePassCount,
      nextColor: expected.nextColor,
      board: expected.board,
    },
    label,
  );
};

for (const model of MODELS) {
  test(`${model.id}: rules match Moka on random and policy games`, async () => {
    const rules = await model.loadRules();
    const { searchEngine, inferEngine } = await model.loadEngines();
    const policy = (state: GoGameState) =>
      inferEngine.infer(rules.encodeStudentFeatures(state)).policyLogits;
    const games = [
      ...collectGames(rules, model.randomGames, 1),
      ...collectGames(rules, model.policyGames, 2, policy),
    ];
    const random = createMulberry32(3);
    let deadStoneChecks = 0;

    for (const states of games) {
      states.forEach((state, index) => {
        const checkDeadStones = index % model.deadStoneInterval === 0;

        checkPosition(rules, searchEngine, state, checkDeadStones);
        deadStoneChecks += checkDeadStones ? 1 : 0;

        if (index + 1 < states.length) {
          checkPlay(rules, searchEngine, state, states[index + 1].moveHistory.at(-1)!);
          checkPlay(rules, searchEngine, state, Math.floor(random() * state.board.length));
        }
      });
    }

    assert.ok(deadStoneChecks > 0);
  });

  test(`${model.id}: the WASM search visits like match/search.ts with the Zig network`, async () => {
    const rules = await model.loadRules();
    const { searchEngine, inferEngine } = await model.loadEngines();
    const policy = (state: GoGameState) =>
      inferEngine.infer(rules.encodeStudentFeatures(state)).policyLogits;
    const states = collectGames(rules, 2, 4, policy).flatMap((game) => game.slice(0, -1));
    const step = Math.max(1, Math.floor(states.length / model.searchPositions));

    for (let index = 0; index < states.length; index += step) {
      const state = states[index];
      const expected = rules.search(state, (features) => inferEngine.infer(features), {
        timeMs: Infinity,
        maxVisits: model.searchVisits,
      });
      const actual = searchEngine.search(state, { timeMs: Infinity, maxVisits: model.searchVisits });

      assert.equal(actual.visits, expected.visits, `move ${state.moveCount}`);
      assert.deepEqual(searchEngine.rootVisits(), expected.children, `move ${state.moveCount}`);
      assert.equal(actual.move, expected.move, `move ${state.moveCount}`);
    }
  });

  test(`${model.id}: the WASM search reuses its tree like match/search.ts`, async () => {
    const rules = await model.loadRules();
    const { searchEngine, inferEngine } = await model.loadEngines();
    const evaluate = (features: Float32Array) => inferEngine.infer(features);
    const policy = (state: GoGameState) => evaluate(rules.encodeStudentFeatures(state)).policyLogits;
    const [game] = collectGames(rules, 1, 6, policy);
    const states = game.slice(0, -1);
    let reusedSearches = 0;

    for (const stride of [1, 2]) {
      const tree = rules.createSearchTree();

      for (let index = 0; index < Math.min(states.length, 2 * model.searchPositions); index += stride) {
        const state = states[index];
        const label = `stride ${stride}, move ${state.moveCount}`;
        const limits = { timeMs: Infinity, maxVisits: model.searchVisits };
        const expected = rules.search(state, evaluate, limits, tree);
        const actual = searchEngine.search(state, { ...limits, reuse: true });

        assert.equal(actual.reused, expected.reused, label);
        assert.equal(actual.visits, expected.visits, label);
        assert.deepEqual(searchEngine.rootVisits(), expected.children, label);
        assert.equal(actual.move, expected.move, label);
        reusedSearches += expected.reused > 0 ? 1 : 0;
      }
    }

    const state = states[0];
    const once = searchEngine.search(state, { timeMs: Infinity, maxVisits: model.searchVisits });
    const again = searchEngine.search(state, { timeMs: Infinity, maxVisits: 2 * model.searchVisits, reuse: true });
    const children = searchEngine.rootVisits();
    const fresh = searchEngine.search(state, { timeMs: Infinity, maxVisits: 2 * model.searchVisits });

    assert.equal(again.reused, once.visits);
    assert.deepEqual(children, searchEngine.rootVisits());
    assert.equal(again.move, fresh.move);
    assert.ok(reusedSearches > 0);
  });

  test(`${model.id}: batched search keeps the visit count and plays legal moves`, async () => {
    const rules = await model.loadRules();
    const { searchEngine } = await model.loadEngines();
    const states = collectGames(rules, 1, 5).flat().filter((state) => !rules.isGameOver(state));

    for (const batch of [2, 4, 8]) {
      for (const state of states.filter((_, index) => index % 25 === 0)) {
        const result = searchEngine.search(state, { timeMs: Infinity, maxVisits: model.searchVisits, batch });

        assert.equal(result.visits, model.searchVisits);
        assert.ok(rules.getLegalMoves(state).includes(result.move));
        assert.equal(
          searchEngine.rootVisits().reduce((sum, child) => sum + child.visits, 0),
          model.searchVisits,
        );
      }
    }
  });

  test(`${model.id}: the search refuses a finished game`, async () => {
    const rules = await model.loadRules();
    const { searchEngine } = await model.loadEngines();
    const pass = rules.createGameState().board.length;
    const finished = rules.playMove(rules.playMove(rules.createGameState(), pass)!, pass)!;

    assert.throws(() => searchEngine.search(finished, { timeMs: Infinity, maxVisits: 1 }), {
      message: "The search started from a finished game.",
    });
  });

  test(`${model.id}: the search stops when the tree is full, and the next search starts a new tree`, async () => {
    const rules = await model.loadRules();
    const { searchEngine } = await model.loadEngines();
    const state = rules.createGameState();
    const result = searchEngine.search(state, { timeMs: Infinity, maxVisits: 1_000_000 });

    assert.ok(result.poolFull);
    assert.ok(result.visits > 1000);

    const next = searchEngine.search(state, { timeMs: Infinity, maxVisits: 10 });

    assert.equal(next.poolFull, false);
    assert.equal(next.visits, 10);
    assert.ok(rules.getLegalMoves(state).includes(next.move));
  });

  test(`${model.id}: at 19×19 the search does not count Moka's capture aftermath`, async (context) => {
    const rules = await model.loadRules();
    const state = rules.createGameState();
    const size = Math.sqrt(state.board.length);

    if (size !== 19) {
      context.skip("the search keeps Moka's final score at 9×9");
      return;
    }

    const { searchEngine } = await model.loadEngines();

    // Black wall on column 1, White wall on column 2, and a White stone in atari in the corner of
    // Black's side. White has just passed. Nothing is pass-dead, but the aftermath captures the stone.
    for (let row = 0; row < size; row += 1) {
      state.board[row * size + 1] = 1;
      state.board[row * size + 2] = -1;
    }
    state.board[0] = -1;
    state.moveCount = 200;
    state.moveHistory = [size * 3 + 5, state.board.length];
    state.consecutivePassCount = 1;

    assert.deepEqual(rules.getAutomaticallyDeadMoves(state), [0]);
    assert.equal(rules.getAreaScore(state), 19 - 20 - 16 * 19 - 7);
    assert.equal(rules.getSearchScore(state), rules.getAreaScore(state));

    assert.equal(searchEngine.rules.getSearchScore(state), rules.getSearchScore(state));
  });
}
