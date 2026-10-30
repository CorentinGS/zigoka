import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  STATUS_OK,
  findParityMismatch,
  getDimensions,
  instantiate as instantiateWithWeights,
  readFixtureSet,
  readOutputs as readOutputsWithDimensions,
  runFixtureSet as runFixtureSetWithDimensions,
  selectHighestLegalMove,
  sha256Hex,
  writeInputs as writeInputsWithDimensions,
} from "./inference-harness.mjs";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resolve = (relativePath) => path.join(rootDirectory, relativePath);

const MAX_BATCH = 8;
const INPUT_PLANE_COUNT = 12;
const SHA256_BYTE_COUNT = 32;
const MEMORY_CHECK_INFERENCES = 1000;
const SNAPSHOT_MODEL_ID = "moka-9";

const STATUS_WEIGHTS_NOT_LOADED = 1;
const STATUS_BAD_BATCH_COUNT = 2;

const ARTIFACTS = ["infer.wasm", "infer-safe.wasm"];
const FIXTURE_SETS = ["calibration", "validation"];

const listModels = () =>
  fs
    .readdirSync(resolve("zig-out/models"))
    .filter((id) => fs.existsSync(resolve(`zig-out/models/${id}/go-model.json`)))
    .sort((left, right) =>
      left === SNAPSHOT_MODEL_ID ? -1 : right === SNAPSHOT_MODEL_ID ? 1 : left.localeCompare(right),
    );

const readFixtureSets = (id) =>
  FIXTURE_SETS.map((name) => resolve(`oracle/fixtures/${id}/${name}`))
    .filter((directory) => fs.existsSync(path.join(directory, "index.json")))
    .map(readFixtureSet);

const describeArtifact = (model, artifact, fixtureSets) => {
  const instantiate = (wasmModule, options) =>
    instantiateWithWeights(wasmModule, model.weights, options);
  const wasmModule = new WebAssembly.Module(
    fs.readFileSync(resolve(`zig-out/${model.id}/${artifact}`)),
  );
  const engine = instantiate(wasmModule, { loadWeights: true });
  const dimensions = getDimensions(engine);
  const { moveCount } = dimensions;
  const writeInputs = (features, start, count) =>
    writeInputsWithDimensions(engine, dimensions, features, start, count);
  const readOutputs = (count) => readOutputsWithDimensions(engine, dimensions, count);
  const label = `${model.id}/${artifact}`;
  const runs = fixtureSets.map((fixtureSet) => ({
    fixtureSet,
    results: runFixtureSetWithDimensions(engine, dimensions, fixtureSet),
  }));

  for (const { fixtureSet, results } of runs) {
    test(`${label}: ${fixtureSet.name}: every position gives Status ok`, () => {
      assert.deepEqual(results.failedBatches, []);
    });

    test(`${label}: ${fixtureSet.name}: the outputs follow the parity rule against f32Accumulate`, () => {
      const mismatch = findParityMismatch(
        results.logits,
        results.values,
        fixtureSet.logitsF32Accumulate,
        fixtureSet.valueF32Accumulate,
      );

      if (mismatch === null) {
        return;
      }

      const { position, index } = mismatch;
      assert.fail(
        mismatch.kind === "logit"
          ? `position ${position}, logit ${index}: ${results.logits[position * moveCount + index]} ` +
              `vs ${fixtureSet.logitsF32Accumulate[position * moveCount + index]}`
          : `position ${position}, value ${results.values[position]} ` +
              `vs ${fixtureSet.valueF32Accumulate[position]}`,
      );
    });
  }

  if (model.id === SNAPSHOT_MODEL_ID) {
    describeSnapshotChecks(label, artifact, runs, moveCount);
  }

  const validation = runs.find(({ fixtureSet }) => fixtureSet.name === "validation");

  test(`${label}: memory.buffer.byteLength stays constant over 1000 inferences (G5)`, () => {
    const { fixtureSet } = validation;
    const byteLength = engine.memory.buffer.byteLength;

    for (let index = 0; index < MEMORY_CHECK_INFERENCES; index += 1) {
      writeInputs(fixtureSet.features, index % fixtureSet.positions, 1);
      assert.equal(engine.infer(1), STATUS_OK);
    }

    assert.equal(engine.memory.buffer.byteLength, byteLength);
  });

  test(`${label}: infer(8) gives the same results as 8 calls to infer(1)`, () => {
    const { fixtureSet } = validation;
    const single = {
      logits: new Float32Array(MAX_BATCH * moveCount),
      values: new Float64Array(MAX_BATCH),
    };

    for (let k = 0; k < MAX_BATCH; k += 1) {
      writeInputs(fixtureSet.features, k, 1);
      assert.equal(engine.infer(1), STATUS_OK);
      const outputs = readOutputs(1);
      single.logits.set(outputs.logits, k * moveCount);
      single.values.set(outputs.values, k);
    }

    writeInputs(fixtureSet.features, 0, MAX_BATCH);
    assert.equal(engine.infer(MAX_BATCH), STATUS_OK);
    const batch = readOutputs(MAX_BATCH);

    assert.deepEqual(new Uint32Array(batch.logits.buffer), new Uint32Array(single.logits.buffer));
    assert.deepEqual(new BigUint64Array(batch.values.buffer), new BigUint64Array(single.values.buffer));
  });

  test(`${label}: infer(0) and infer(9) return bad_batch_count`, () => {
    assert.equal(engine.infer(0), STATUS_BAD_BATCH_COUNT);
    assert.equal(engine.infer(MAX_BATCH + 1), STATUS_BAD_BATCH_COUNT);
  });

  test(`${label}: infer(1) on a fresh instance before load_weights returns weights_not_loaded`, () => {
    assert.equal(instantiate(wasmModule, { loadWeights: false }).infer(1), STATUS_WEIGHTS_NOT_LOADED);
  });

  test(`${label}: the exported dimensions match the manifest and index.json`, () => {
    const { architecture, weightsBytes } = model.manifest;
    const boardSize = engine.board_size();

    assert.equal(boardSize, architecture.boardSize);
    assert.equal(engine.policy_moves(), architecture.policyMoveCount);

    for (const { fixtureSet } of runs) {
      assert.equal(engine.policy_moves(), fixtureSet.index.moveCount);
      assert.equal(engine.feature_count(), fixtureSet.index.featureCount);
    }

    assert.equal(engine.feature_count(), boardSize * boardSize * INPUT_PLANE_COUNT);
    assert.equal(engine.input_stride(), (boardSize * boardSize + 1) * INPUT_PLANE_COUNT);
    assert.equal(engine.weights_capacity(), weightsBytes);
    assert.equal(engine.max_batch(), MAX_BATCH);
    const expectedSha256 = new Uint8Array(
      engine.memory.buffer,
      engine.expected_sha256_ptr(),
      SHA256_BYTE_COUNT,
    );
    assert.deepEqual(
      Buffer.from(expectedSha256),
      crypto.createHash("sha256").update(model.weights).digest(),
    );
  });
};

const describeSnapshotChecks = (label, artifact, runs, moveCount) => {
  const snapshot = JSON.parse(
    fs.readFileSync(resolve(`oracle/snapshots/${SNAPSHOT_MODEL_ID}.json`), "utf8"),
  );
  const tolerance = JSON.parse(fs.readFileSync(resolve("oracle/tolerance.json"), "utf8"));

  for (const { fixtureSet, results } of runs) {
    test(`${label}: ${fixtureSet.name}: inputs and outputs match the J0 snapshot`, () => {
      const expected = snapshot.sets[fixtureSet.name];

      assert.equal(fixtureSet.positions, expected.positions);
      assert.equal(sha256Hex(fixtureSet.features), expected.featuresSha256);
      assert.equal(sha256Hex(results.logits), expected.outputs[artifact].logitsSha256);
      assert.equal(sha256Hex(results.values), expected.outputs[artifact].valuesSha256);
    });
  }

  const validation = runs.find(({ fixtureSet }) => fixtureSet.name === "validation");

  test(`${label}: validation (G3): the drift from the normal mode stays within tolerance.json`, (t) => {
    const { fixtureSet, results } = validation;
    let maxLogitDifference = 0;
    let maxValueDifference = 0;

    for (let index = 0; index < results.logits.length; index += 1) {
      maxLogitDifference = Math.max(
        maxLogitDifference,
        Math.abs(results.logits[index] - fixtureSet.logitsNormal[index]),
      );
    }

    for (let position = 0; position < fixtureSet.positions; position += 1) {
      maxValueDifference = Math.max(
        maxValueDifference,
        Math.abs(results.values[position] - fixtureSet.valueNormal[position]),
      );
    }

    t.diagnostic(
      `max logit drift ${maxLogitDifference.toExponential(3)} (tolerance ${tolerance.logits}), ` +
        `max value drift ${maxValueDifference.toExponential(3)} (tolerance ${tolerance.value})`,
    );
    assert.ok(maxLogitDifference <= tolerance.logits, `logit drift ${maxLogitDifference}`);
    assert.ok(maxValueDifference <= tolerance.value, `value drift ${maxValueDifference}`);
  });

  test(`${label}: validation (G3): the best legal move is Moka's, except on near ties`, (t) => {
    const { fixtureSet, results } = validation;
    const nearTieLimit = 2 * tolerance.logits;
    let nearTies = 0;

    for (let position = 0; position < fixtureSet.positions; position += 1) {
      const offset = position * moveCount;
      const zigMove = selectHighestLegalMove(
        results.logits.subarray(offset, offset + moveCount),
        fixtureSet.legal.subarray(offset, offset + moveCount),
        moveCount,
      );
      const mokaMove = fixtureSet.chosenMove[position];

      if (zigMove === mokaMove) {
        continue;
      }

      const gap = Math.abs(
        fixtureSet.logitsNormal[offset + mokaMove] - fixtureSet.logitsNormal[offset + zigMove],
      );
      assert.ok(
        gap <= nearTieLimit,
        `position ${position}: Zig plays ${zigMove}, Moka plays ${mokaMove}, logit gap ${gap}`,
      );
      nearTies += 1;
    }

    t.diagnostic(`${nearTies} different moves accepted as near ties`);
  });
};

for (const id of listModels()) {
  const fixtureSets = readFixtureSets(id);

  if (!fixtureSets.some(({ name }) => name === "validation")) {
    test(`${id}: reference data is present`, () => {
      assert.fail(`model ${id} has no reference data; run pnpm oracle`);
    });
    continue;
  }

  const model = {
    id,
    manifest: JSON.parse(fs.readFileSync(resolve(`zig-out/models/${id}/go-model.json`), "utf8")),
    weights: fs.readFileSync(resolve(`zig-out/models/${id}/go-model.bin`)),
  };

  for (const artifact of ARTIFACTS) {
    describeArtifact(model, artifact, fixtureSets);
  }
}
