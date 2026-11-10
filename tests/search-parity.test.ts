import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { createEngine } from "../js/engine.ts";
import {
  createSearchEngine,
  readMoves,
  readPosition,
  writePosition,
  type SearchEngine,
} from "../js/search-engine.ts";
import { createMulberry32 } from "../oracle/mulberry32.ts";

type MokaRules = typeof import("./moka-rules.ts");

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const PASS_PROBABILITY = 0.05;
const STATUS_GAME_OVER = 4;

const loadRules = async (size: number) =>
  (await import(path.join(rootDirectory, `tests/dist/moka-rules-${size}.js`))) as MokaRules;

const MODELS = [
  {
    id: "moka-9",
    loadRules: () => loadRules(9),
    randomGames: 40,
    policyGames: 10,
    deadStoneInterval: 5,
    searchPositions: 12,
    searchVisits: 150,
  },
  {
    id: "moka-9-v2",
    loadRules: () => loadRules(9),
    randomGames: 10,
    policyGames: 5,
    deadStoneInterval: 5,
    searchPositions: 8,
    searchVisits: 150,
  },
  {
    id: "synthetic-19",
    loadRules: () => loadRules(19),
    randomGames: 4,
    policyGames: 2,
    deadStoneInterval: 60,
    searchPositions: 4,
    searchVisits: 60,
  },
  {
    id: "synthetic-19-v2",
    loadRules: () => loadRules(19),
    randomGames: 2,
    policyGames: 1,
    deadStoneInterval: 60,
    searchPositions: 2,
    searchVisits: 60,
  },
];

const readArrayBuffer = (filePath: string) => {
  const bytes = fs.readFileSync(filePath);

  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
};

const loadEngines = async (id: string) => {
  const weights = readArrayBuffer(path.join(rootDirectory, `zig-out/models/${id}/go-model.bin`));
  const searchModule = new WebAssembly.Module(
    fs.readFileSync(path.join(rootDirectory, `zig-out/${id}/search.wasm`)),
  );
  const inferModule = new WebAssembly.Module(
    fs.readFileSync(path.join(rootDirectory, `zig-out/${id}/infer.wasm`)),
  );

  return {
    searchEngine: await createSearchEngine(searchModule, weights.slice(0)),
    inferEngine: await createEngine(inferModule, weights.slice(0)),
  };
};

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
  const { exports } = engine;
  const label = `move ${state.moveCount}, history ${state.moveHistory.join(",")}`;

  writePosition(exports, state);
  assert.deepEqual(readMoves(exports, exports.rules_legal_moves()), rules.getLegalMoves(state), label);

  exports.rules_encode();
  const features = new Float32Array(exports.memory.buffer, exports.input_ptr(), exports.feature_count());

  assert.ok(toBytes(features).equals(toBytes(rules.encodeStudentFeatures(state))), `features, ${label}`);
  assert.equal(exports.rules_is_game_over() === 1, rules.isGameOver(state), label);
  assert.equal(exports.rules_area_score(), rules.getAreaScore(state), label);

  if (checkDeadStones || rules.isGameOver(state)) {
    const expected = [...rules.getAutomaticallyDeadMoves(state)].sort((left, right) => left - right);

    assert.deepEqual(readMoves(exports, exports.rules_dead_stones()), expected, `dead stones, ${label}`);
  }
};

const checkPlay = (rules: MokaRules, engine: SearchEngine, state: GoGameState, move: number) => {
  const { exports } = engine;
  const area = state.board.length;
  const expected = rules.playMove(state, move);

  writePosition(exports, state);
  const before = readPosition(exports, area);

  assert.equal(exports.rules_play(move), expected ? 1 : 0, `move ${move} at ${state.moveCount}`);

  if (!expected) {
    assert.deepEqual(readPosition(exports, area), before);
    return;
  }

  assert.deepEqual(readPosition(exports, area), {
    koMove: expected.koMove,
    moveCount: expected.moveCount,
    lastMove: expected.moveHistory.at(-1),
    previousMove: expected.moveHistory.at(-2) ?? -1,
    consecutivePassCount: expected.consecutivePassCount,
    nextColor: expected.nextColor,
    board: expected.board,
  });
};

for (const model of MODELS) {
  test(`${model.id}: rules match Moka on random and policy games`, async () => {
    const rules = await model.loadRules();
    const { searchEngine, inferEngine } = await loadEngines(model.id);
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
    const { searchEngine, inferEngine } = await loadEngines(model.id);
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
    const { searchEngine, inferEngine } = await loadEngines(model.id);
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
    const { searchEngine } = await loadEngines(model.id);
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
    const { searchEngine } = await loadEngines(model.id);
    const pass = rules.createGameState().board.length;
    const finished = rules.playMove(rules.playMove(rules.createGameState(), pass)!, pass)!;

    writePosition(searchEngine.exports, finished);
    assert.equal(searchEngine.exports.search_start(0), STATUS_GAME_OVER);
  });

  test(`${model.id}: the search stops when the tree is full`, async () => {
    const rules = await model.loadRules();
    const { searchEngine } = await loadEngines(model.id);
    const result = searchEngine.search(rules.createGameState(), { timeMs: Infinity, maxVisits: 1_000_000 });

    assert.ok(result.poolFull);
    assert.ok(result.visits > 1000);
    assert.ok(searchEngine.exports.search_nodes() <= searchEngine.exports.search_node_capacity());
  });
}
