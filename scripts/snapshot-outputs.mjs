import fs from "node:fs";
import path from "node:path";

import { instantiate, runFixtureSet, sha256Hex } from "../tests/inference-harness.mjs";

const ARTIFACTS = ["infer.wasm", "infer-safe.wasm"];
const FIXTURE_SETS = ["calibration", "validation"];
const MODEL_ID = "moka-9";
const INPUT_PLANE_COUNT = 12;

const readOption = (name) => {
  const index = process.argv.indexOf(name);

  if (index < 0 || index + 1 >= process.argv.length) {
    throw new Error(`Missing option ${name} <value>.`);
  }

  return process.argv[index + 1];
};

const readFixtureSet = (fixturesDirectory, name) => {
  const directory = path.join(fixturesDirectory, name);
  const { positions } = JSON.parse(fs.readFileSync(path.join(directory, "index.json"), "utf8"));
  const featureBytes = fs.readFileSync(path.join(directory, "features.f32"));

  return {
    positions,
    featureBytes,
    features: new Float32Array(new Uint8Array(featureBytes).buffer),
  };
};

const getDimensions = (fixtureSet, exports) => {
  const featureCount = fixtureSet.features.length / fixtureSet.positions;
  const boardArea = featureCount / INPUT_PLANE_COUNT;

  return {
    featureCount,
    inputStride: (boardArea + 1) * INPUT_PLANE_COUNT,
    moveCount: boardArea + 1,
    maxBatch: exports.max_batch(),
  };
};

const main = () => {
  const wasmDirectory = path.resolve(readOption("--wasm"));
  const fixturesDirectory = path.resolve(readOption("--fixtures"));
  const outputPath = path.resolve(readOption("--out"));
  const weights = fs.readFileSync(path.resolve("vendor/moka/model/go-model.bin"));
  const engines = ARTIFACTS.map((artifact) => {
    const wasmModule = new WebAssembly.Module(fs.readFileSync(path.join(wasmDirectory, artifact)));

    return { artifact, exports: instantiate(wasmModule, weights, { loadWeights: true }) };
  });
  const sets = {};

  for (const name of FIXTURE_SETS) {
    const fixtureSet = readFixtureSet(fixturesDirectory, name);
    const outputs = {};

    for (const { artifact, exports } of engines) {
      const results = runFixtureSet(exports, getDimensions(fixtureSet, exports), fixtureSet);

      if (results.failedBatches.length > 0) {
        throw new Error(
          `${artifact} failed on ${name}: ${JSON.stringify(results.failedBatches[0])}`,
        );
      }

      outputs[artifact] = {
        logitsSha256: sha256Hex(results.logits),
        valuesSha256: sha256Hex(results.values),
      };
    }

    sets[name] = {
      positions: fixtureSet.positions,
      featuresSha256: sha256Hex(fixtureSet.featureBytes),
      outputs,
    };
  }

  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify({ modelId: MODEL_ID, sets }, null, 2)}\n`, "utf8");
};

main();
