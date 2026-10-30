import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

import { findParityMismatch, getUlpDistance } from "../bench/parity-rule.ts";

export { findParityMismatch, getUlpDistance };

export const STATUS_OK = 0;

export const sha256Hex = (bytes) =>
  crypto
    .createHash("sha256")
    .update(new Uint8Array(bytes.buffer ?? bytes, bytes.byteOffset ?? 0, bytes.byteLength))
    .digest("hex");

export const instantiate = (wasmModule, weights, { loadWeights }) => {
  const { exports } = new WebAssembly.Instance(wasmModule, {});

  if (loadWeights) {
    new Uint8Array(exports.memory.buffer, exports.weights_ptr(), exports.weights_capacity()).set(
      weights,
    );
    exports.load_weights();
  }

  return exports;
};

export const getDimensions = (exports) => ({
  featureCount: exports.feature_count(),
  inputStride: exports.input_stride(),
  moveCount: exports.policy_moves(),
  maxBatch: exports.max_batch(),
});

const copyBytes = (buffer) => new Uint8Array(buffer).buffer;

export const readFixtureSet = (directory) => {
  const read = (fileName) => copyBytes(fs.readFileSync(path.join(directory, fileName)));
  const index = JSON.parse(fs.readFileSync(path.join(directory, "index.json"), "utf8"));

  return {
    name: path.basename(directory),
    index,
    positions: index.positions,
    features: new Float32Array(read("features.f32")),
    logitsNormal: new Float32Array(read("logits-normal.f32")),
    logitsF32Accumulate: new Float32Array(read("logits-f32acc.f32")),
    valueNormal: new Float64Array(read("value-normal.f64")),
    valueF32Accumulate: new Float64Array(read("value-f32acc.f64")),
    legal: new Uint8Array(read("legal.u8")),
    chosenMove: new Uint16Array(read("chosen-move.u16")),
  };
};

export const writeInputs = (exports, dimensions, features, start, count) => {
  const { featureCount, inputStride, maxBatch } = dimensions;
  const input = new Float32Array(exports.memory.buffer, exports.input_ptr(), maxBatch * inputStride);

  for (let k = 0; k < count; k += 1) {
    const offset = (start + k) * featureCount;
    input.set(features.subarray(offset, offset + featureCount), k * inputStride);
  }
};

export const readOutputs = (exports, dimensions, count) => ({
  logits: new Float32Array(
    exports.memory.buffer,
    exports.policy_ptr(),
    count * dimensions.moveCount,
  ).slice(),
  values: new Float64Array(exports.memory.buffer, exports.value_ptr(), count).slice(),
});

export const runFixtureSet = (exports, dimensions, fixtureSet) => {
  const { moveCount, maxBatch } = dimensions;
  const logits = new Float32Array(fixtureSet.positions * moveCount);
  const values = new Float64Array(fixtureSet.positions);
  const failedBatches = [];

  for (let start = 0; start < fixtureSet.positions; start += maxBatch) {
    const count = Math.min(maxBatch, fixtureSet.positions - start);
    writeInputs(exports, dimensions, fixtureSet.features, start, count);
    const status = exports.infer(count);

    if (status !== STATUS_OK) {
      failedBatches.push({ start, count, status });
    }

    const outputs = readOutputs(exports, dimensions, count);
    logits.set(outputs.logits, start * moveCount);
    values.set(outputs.values, start);
  }

  return { logits, values, failedBatches };
};

export const selectHighestLegalMove = (logits, legal, moveCount) => {
  const passMove = moveCount - 1;
  let selectedMove = legal[passMove] === 1 ? passMove : legal.indexOf(1);

  if (selectedMove < 0) {
    selectedMove = passMove;
  }

  let selectedValue = logits[selectedMove];

  for (let move = 0; move < moveCount; move += 1) {
    if (legal[move] === 1 && logits[move] > selectedValue) {
      selectedMove = move;
      selectedValue = logits[move];
    }
  }

  return selectedMove;
};
