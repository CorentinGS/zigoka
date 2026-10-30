import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resolve = (relativePath) => path.join(rootDirectory, relativePath);

const FEATURE_COUNT = 972;
const INPUT_STRIDE = 984;
const MOVE_COUNT = 82;
const PASS_MOVE = 81;
const MAX_BATCH = 8;
const WEIGHTS_BYTES = 113_648;
const SHA256_BYTE_COUNT = 32;
const VALUE_ULP_LIMIT = 2n;
const MEMORY_CHECK_INFERENCES = 1000;

const STATUS_OK = 0;
const STATUS_WEIGHTS_NOT_LOADED = 1;
const STATUS_BAD_BATCH_COUNT = 2;

const wasmModule = new WebAssembly.Module(fs.readFileSync(resolve("zig-out/bin/infer.wasm")));
const weights = fs.readFileSync(resolve("vendor/moka/model/go-model.bin"));
const tolerance = JSON.parse(fs.readFileSync(resolve("oracle/tolerance.json"), "utf8"));

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

const instantiate = ({ loadWeights }) => {
  const { exports } = new WebAssembly.Instance(wasmModule, {});

  if (loadWeights) {
    new Uint8Array(exports.memory.buffer, exports.weights_ptr(), exports.weights_capacity()).set(
      weights,
    );
    exports.load_weights();
  }

  return exports;
};

const writeInputs = (exports, features, start, count) => {
  const input = new Float32Array(exports.memory.buffer, exports.input_ptr(), MAX_BATCH * INPUT_STRIDE);

  for (let k = 0; k < count; k += 1) {
    const offset = (start + k) * FEATURE_COUNT;
    input.set(features.subarray(offset, offset + FEATURE_COUNT), k * INPUT_STRIDE);
  }
};

const readOutputs = (exports, count) => ({
  logits: new Float32Array(exports.memory.buffer, exports.policy_ptr(), count * MOVE_COUNT).slice(),
  values: new Float64Array(exports.memory.buffer, exports.value_ptr(), count).slice(),
});

const runFixtureSet = (exports, fixtureSet) => {
  const logits = new Float32Array(fixtureSet.positions * MOVE_COUNT);
  const values = new Float64Array(fixtureSet.positions);
  const failedBatches = [];

  for (let start = 0; start < fixtureSet.positions; start += MAX_BATCH) {
    const count = Math.min(MAX_BATCH, fixtureSet.positions - start);
    writeInputs(exports, fixtureSet.features, start, count);
    const status = exports.infer(count);

    if (status !== STATUS_OK) {
      failedBatches.push({ start, count, status });
    }

    const outputs = readOutputs(exports, count);
    logits.set(outputs.logits, start * MOVE_COUNT);
    values.set(outputs.values, start);
  }

  return { logits, values, failedBatches };
};

const getUlpDistance = (left, right) => {
  const bits = new BigInt64Array(new Float64Array([left, right]).buffer);
  const toOrdered = (value) => (value < 0n ? -(value & 0x7fff_ffff_ffff_ffffn) : value);
  const distance = toOrdered(bits[0]) - toOrdered(bits[1]);

  return distance < 0n ? -distance : distance;
};

const selectHighestLegalMove = (logits, legal) => {
  let selectedMove = legal[PASS_MOVE] === 1 ? PASS_MOVE : legal.indexOf(1);

  if (selectedMove < 0) {
    selectedMove = PASS_MOVE;
  }

  let selectedValue = logits[selectedMove];

  for (let move = 0; move < MOVE_COUNT; move += 1) {
    if (legal[move] === 1 && logits[move] > selectedValue) {
      selectedMove = move;
      selectedValue = logits[move];
    }
  }

  return selectedMove;
};

const engine = instantiate({ loadWeights: true });
const fixtureSets = ["calibration", "validation"].map((name) => {
  const fixtureSet = readFixtureSet(name);

  return { fixtureSet, results: runFixtureSet(engine, fixtureSet) };
});

for (const { fixtureSet, results } of fixtureSets) {
  test(`${fixtureSet.name}: every position gives Status ok`, () => {
    assert.deepEqual(results.failedBatches, []);
  });

  test(`${fixtureSet.name}: bitwise diagnostic against the f32Accumulate mode`, (t) => {
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

test("validation (G3): the drift from the normal mode stays within tolerance.json", (t) => {
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

test("validation (G3): the best legal move is Moka's, except on near ties", (t) => {
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

test("memory.buffer.byteLength stays constant over 1000 inferences (G5)", () => {
  const { fixtureSet } = validation;
  const byteLength = engine.memory.buffer.byteLength;

  for (let index = 0; index < MEMORY_CHECK_INFERENCES; index += 1) {
    writeInputs(engine, fixtureSet.features, index % fixtureSet.positions, 1);
    assert.equal(engine.infer(1), STATUS_OK);
  }

  assert.equal(engine.memory.buffer.byteLength, byteLength);
});

test("infer(8) gives the same results as 8 calls to infer(1)", () => {
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

test("infer(0) and infer(9) return bad_batch_count", () => {
  assert.equal(engine.infer(0), STATUS_BAD_BATCH_COUNT);
  assert.equal(engine.infer(MAX_BATCH + 1), STATUS_BAD_BATCH_COUNT);
});

test("infer(1) on a fresh instance before load_weights returns weights_not_loaded", () => {
  assert.equal(instantiate({ loadWeights: false }).infer(1), STATUS_WEIGHTS_NOT_LOADED);
});

test("the exported constants match the model", () => {
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
