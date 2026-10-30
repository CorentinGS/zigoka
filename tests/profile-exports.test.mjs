import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import {
  STATUS_OK,
  findParityMismatch,
  getDimensions,
  instantiate,
  readFixtureSet,
  readOutputs,
  writeInputs,
} from "./inference-harness.mjs";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resolve = (relativePath) => path.join(rootDirectory, relativePath);

const MAX_BATCH = 8;
const STATUS_WEIGHTS_NOT_LOADED = 1;
const STATUS_BAD_BATCH_COUNT = 2;

const modelIds = fs
  .readdirSync(resolve("zig-out/models"))
  .filter((id) => fs.existsSync(resolve(`zig-out/models/${id}/go-model.json`)))
  .sort();

for (const id of modelIds) {
  const label = `${id}/infer-profile.wasm`;
  const wasmModule = new WebAssembly.Module(fs.readFileSync(resolve(`zig-out/${id}/infer-profile.wasm`)));
  const weights = fs.readFileSync(resolve(`zig-out/models/${id}/go-model.bin`));

  test(`${label}: forward_trunk checks the batch count, then the weights`, () => {
    const fresh = instantiate(wasmModule, weights, { loadWeights: false });

    assert.equal(fresh.forward_trunk(0), STATUS_BAD_BATCH_COUNT);
    assert.equal(fresh.forward_trunk(MAX_BATCH + 1), STATUS_BAD_BATCH_COUNT);
    assert.equal(fresh.forward_trunk(1), STATUS_WEIGHTS_NOT_LOADED);

    const engine = instantiate(wasmModule, weights, { loadWeights: true });

    assert.equal(engine.forward_trunk(0), STATUS_BAD_BATCH_COUNT);
    assert.equal(engine.forward_trunk(MAX_BATCH + 1), STATUS_BAD_BATCH_COUNT);
    assert.equal(engine.forward_trunk(1), STATUS_OK);
    assert.equal(engine.forward_trunk(MAX_BATCH), STATUS_OK);
  });

  test(`${label}: infer(1) after forward_trunk(1) follows the parity rule`, (t) => {
    const directory = resolve(`oracle/fixtures/${id}/validation`);

    if (!fs.existsSync(path.join(directory, "index.json"))) {
      assert.fail(`model ${id} has no reference data; run pnpm oracle`);
    }

    const fixtureSet = readFixtureSet(directory);
    const engine = instantiate(wasmModule, weights, { loadWeights: true });
    const dimensions = getDimensions(engine);

    writeInputs(engine, dimensions, fixtureSet.features, 0, 1);
    assert.equal(engine.forward_trunk(1), STATUS_OK);
    assert.equal(engine.infer(1), STATUS_OK);

    const { logits, values } = readOutputs(engine, dimensions, 1);
    const mismatch = findParityMismatch(
      logits,
      values,
      fixtureSet.logitsF32Accumulate.subarray(0, dimensions.moveCount),
      fixtureSet.valueF32Accumulate.subarray(0, 1),
    );

    t.diagnostic(`${dimensions.moveCount} logits compared`);
    assert.equal(mismatch, null);
  });
}
