import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  STATUS_OK,
  getUlpDistance,
  instantiate as instantiateWithWeights,
  readOutputs as readOutputsWithDimensions,
  runFixtureSet as runFixtureSetWithDimensions,
  selectHighestLegalMove as selectHighestLegalMoveWithMoveCount,
  writeInputs as writeInputsWithDimensions,
} from "./inference-harness.mjs";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resolve = (relativePath) => path.join(rootDirectory, relativePath);

const FEATURE_COUNT = 972;
const INPUT_STRIDE = 984;
const MOVE_COUNT = 82;
const MAX_BATCH = 8;
const WEIGHTS_BYTES = 113_648;
const SHA256_BYTE_COUNT = 32;
const VALUE_ULP_LIMIT = 2n;
const MEMORY_CHECK_INFERENCES = 1000;

const STATUS_WEIGHTS_NOT_LOADED = 1;
const STATUS_BAD_BATCH_COUNT = 2;

const ARTIFACTS = ["infer.wasm", "infer-safe.wasm"];

const weights = fs.readFileSync(resolve("vendor/moka/model/go-model.bin"));
const tolerance = JSON.parse(fs.readFileSync(resolve("oracle/tolerance.json"), "utf8"));

const DIMENSIONS = {
  featureCount: FEATURE_COUNT,
  inputStride: INPUT_STRIDE,
  moveCount: MOVE_COUNT,
  maxBatch: MAX_BATCH,
};

const instantiate = (wasmModule, options) => instantiateWithWeights(wasmModule, weights, options);
const writeInputs = (exports, features, start, count) =>
  writeInputsWithDimensions(exports, DIMENSIONS, features, start, count);
const readOutputs = (exports, count) => readOutputsWithDimensions(exports, DIMENSIONS, count);
const runFixtureSet = (exports, fixtureSet) =>
  runFixtureSetWithDimensions(exports, DIMENSIONS, fixtureSet);
const selectHighestLegalMove = (logits, legal) =>
  selectHighestLegalMoveWithMoveCount(logits, legal, MOVE_COUNT);

const copyBytes = (buffer) => new Uint8Array(buffer).buffer;

const readFixtureSet = (name) => {
  const directory = resolve(`oracle/fixtures/${name}`);
  const read = (fileName) => copyBytes(fs.readFileSync(path.join(directory, fileName)));
  const { positions } = JSON.parse(fs.readFileSync(path.join(directory, "index.json"), "utf8"));

  return {
    name,
    positions,
    features: new Float32Array(read("features.f32")),
    logitsNormal: new Float32Array(read("logits-normal.f32")),
    logitsF32Accumulate: new Float32Array(read("logits-f32acc.f32")),
    valueNormal: new Float64Array(read("value-normal.f64")),
    valueF32Accumulate: new Float64Array(read("value-f32acc.f64")),
    legal: new Uint8Array(read("legal.u8")),
    chosenMove: new Uint8Array(read("chosen-move.u8")),
  };
};

const describeArtifact = (artifact) => {
  const wasmModule = new WebAssembly.Module(fs.readFileSync(resolve(`zig-out/bin/${artifact}`)));
  const engine = instantiate(wasmModule, { loadWeights: true });
  const fixtureSets = ["calibration", "validation"].map((name) => {
    const fixtureSet = readFixtureSet(name);

    return { fixtureSet, results: runFixtureSet(engine, fixtureSet) };
  });

  for (const { fixtureSet, results } of fixtureSets) {
    test(`${artifact}: ${fixtureSet.name}: every position gives Status ok`, () => {
      assert.deepEqual(results.failedBatches, []);
    });

    test(`${artifact}: ${fixtureSet.name}: bitwise diagnostic against the f32Accumulate mode`, (t) => {
      const actualBits = new Uint32Array(results.logits.buffer);
      const expectedBits = new Uint32Array(fixtureSet.logitsF32Accumulate.buffer);
      let differingPositions = 0;
      let firstDifference = null;

      for (let position = 0; position < fixtureSet.positions; position += 1) {
        let differingMove = -1;

        for (let move = 0; move < MOVE_COUNT; move += 1) {
          const index = position * MOVE_COUNT + move;

          if (actualBits[index] !== expectedBits[index]) {
            differingMove = move;
            break;
          }
        }

        const valueUlps = getUlpDistance(results.values[position], fixtureSet.valueF32Accumulate[position]);

        if (differingMove >= 0 || valueUlps > VALUE_ULP_LIMIT) {
          differingPositions += 1;
          firstDifference ??=
            differingMove >= 0
              ? `position ${position}, move ${differingMove}: ` +
                `${results.logits[position * MOVE_COUNT + differingMove]} vs ` +
                `${fixtureSet.logitsF32Accumulate[position * MOVE_COUNT + differingMove]}`
              : `position ${position}, value ${results.values[position]} vs ` +
                `${fixtureSet.valueF32Accumulate[position]} (${valueUlps} ulp)`;
        }
      }

      t.diagnostic(
        `${differingPositions} of ${fixtureSet.positions} positions differ` +
          (firstDifference ? `; first: ${firstDifference}` : ""),
      );
    });
  }

  const validation = fixtureSets.find(({ fixtureSet }) => fixtureSet.name === "validation");

  test(`${artifact}: validation (G3): the drift from the normal mode stays within tolerance.json`, (t) => {
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

  test(`${artifact}: validation (G3): the best legal move is Moka's, except on near ties`, (t) => {
    const { fixtureSet, results } = validation;
    const nearTieLimit = 2 * tolerance.logits;
    let nearTies = 0;

    for (let position = 0; position < fixtureSet.positions; position += 1) {
      const offset = position * MOVE_COUNT;
      const zigMove = selectHighestLegalMove(
        results.logits.subarray(offset, offset + MOVE_COUNT),
        fixtureSet.legal.subarray(offset, offset + MOVE_COUNT),
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

  test(`${artifact}: memory.buffer.byteLength stays constant over 1000 inferences (G5)`, () => {
    const { fixtureSet } = validation;
    const byteLength = engine.memory.buffer.byteLength;

    for (let index = 0; index < MEMORY_CHECK_INFERENCES; index += 1) {
      writeInputs(engine, fixtureSet.features, index % fixtureSet.positions, 1);
      assert.equal(engine.infer(1), STATUS_OK);
    }

    assert.equal(engine.memory.buffer.byteLength, byteLength);
  });

  test(`${artifact}: infer(8) gives the same results as 8 calls to infer(1)`, () => {
    const { fixtureSet } = validation;
    const single = { logits: new Float32Array(MAX_BATCH * MOVE_COUNT), values: new Float64Array(MAX_BATCH) };

    for (let k = 0; k < MAX_BATCH; k += 1) {
      writeInputs(engine, fixtureSet.features, k, 1);
      assert.equal(engine.infer(1), STATUS_OK);
      const outputs = readOutputs(engine, 1);
      single.logits.set(outputs.logits, k * MOVE_COUNT);
      single.values.set(outputs.values, k);
    }

    writeInputs(engine, fixtureSet.features, 0, MAX_BATCH);
    assert.equal(engine.infer(MAX_BATCH), STATUS_OK);
    const batch = readOutputs(engine, MAX_BATCH);

    assert.deepEqual(new Uint32Array(batch.logits.buffer), new Uint32Array(single.logits.buffer));
    assert.deepEqual(new BigUint64Array(batch.values.buffer), new BigUint64Array(single.values.buffer));
  });

  test(`${artifact}: infer(0) and infer(9) return bad_batch_count`, () => {
    assert.equal(engine.infer(0), STATUS_BAD_BATCH_COUNT);
    assert.equal(engine.infer(MAX_BATCH + 1), STATUS_BAD_BATCH_COUNT);
  });

  test(`${artifact}: infer(1) on a fresh instance before load_weights returns weights_not_loaded`, () => {
    assert.equal(instantiate(wasmModule, { loadWeights: false }).infer(1), STATUS_WEIGHTS_NOT_LOADED);
  });

  test(`${artifact}: the exported constants match the model`, () => {
    assert.equal(engine.weights_capacity(), WEIGHTS_BYTES);
    assert.equal(engine.max_batch(), MAX_BATCH);
    const expectedSha256 = new Uint8Array(
      engine.memory.buffer,
      engine.expected_sha256_ptr(),
      SHA256_BYTE_COUNT,
    );
    assert.deepEqual(
      Buffer.from(expectedSha256),
      crypto.createHash("sha256").update(weights).digest(),
    );
  });
};

for (const artifact of ARTIFACTS) {
  describeArtifact(artifact);
}
