import { build } from "esbuild";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const resolve = (relativePath) => path.join(rootDirectory, relativePath);

const BROWSER_TARGET = "es2022";
const NODE_TARGET = "node24";

const browserBundles = [
  {
    entryPoints: [resolve("bench/bench.ts")],
    outfile: resolve("bench/dist/bench.js"),
  },
  {
    entryPoints: [resolve("vendor/moka/src/worker.ts")],
    outfile: resolve("bench/dist/moka-worker.js"),
  },
  {
    entryPoints: [resolve("js/worker.ts")],
    outfile: resolve("dist/worker.js"),
  },
  {
    entryPoints: [resolve("tests/e2e/page.ts")],
    outfile: resolve("tests/e2e/dist/page.js"),
  },
];

const nodeBundles = [
  {
    entryPoints: [resolve("oracle/generate-fixtures.ts")],
    outfile: resolve("oracle/dist/generate-9.js"),
  },
  {
    entryPoints: [resolve("oracle/calibrate-tolerance.ts")],
    outfile: resolve("oracle/dist/calibrate.js"),
  },
  {
    entryPoints: [resolve("oracle/generate-synthetic-model.ts")],
    outfile: resolve("oracle/dist/generate-synthetic-model.js"),
  },
];

const buildBundle = async ({ entryPoints, outfile }) => {
  await build({
    bundle: true,
    entryPoints,
    format: "esm",
    logLevel: "warning",
    outfile,
    platform: "browser",
    target: BROWSER_TARGET,
  });
};

const buildNodeBundle = async ({ entryPoints, outfile }) => {
  await build({
    bundle: true,
    entryPoints,
    format: "esm",
    logLevel: "warning",
    outfile,
    platform: "node",
    target: NODE_TARGET,
  });
};

const PUBLISHED_MODEL_ID = "moka-9";
const MODEL_FILES = ["go-model.json", "go-model.bin"];
const BENCHMARK_POSITION_COUNT = 100;
const BENCHMARK_SETS = [
  { fileName: "logits-normal.f32", bytesPerPosition: (index) => index.moveCount * 4 },
  { fileName: "value-normal.f64", bytesPerPosition: () => 8 },
  { fileName: "logits-f32acc.f32", bytesPerPosition: (index) => index.moveCount * 4 },
  { fileName: "value-f32acc.f64", bytesPerPosition: () => 8 },
];

const copyIfPresent = (sourcePath, destinationPath) => {
  if (!fs.existsSync(sourcePath)) {
    return;
  }

  fs.mkdirSync(path.dirname(destinationPath), { recursive: true });
  fs.copyFileSync(sourcePath, destinationPath);
};

const listModels = () => {
  const modelsDirectory = resolve("zig-out/models");

  if (!fs.existsSync(modelsDirectory)) {
    return [];
  }

  return fs
    .readdirSync(modelsDirectory)
    .filter((id) => fs.existsSync(path.join(modelsDirectory, id, "go-model.json")))
    .sort();
};

const copyModelFiles = (id, destinationDirectory) => {
  for (const fileName of MODEL_FILES) {
    copyIfPresent(
      resolve(`zig-out/models/${id}/${fileName}`),
      path.join(destinationDirectory, fileName),
    );
  }
};

const copyPublishedFiles = () => {
  copyModelFiles(PUBLISHED_MODEL_ID, resolve("dist"));
  copyIfPresent(resolve(`zig-out/${PUBLISHED_MODEL_ID}/infer.wasm`), resolve("dist/infer.wasm"));
};

const writeBenchmarkInputs = (id) => {
  const validationDirectory = resolve(`oracle/fixtures/${id}/validation`);

  if (!fs.existsSync(path.join(validationDirectory, "index.json"))) {
    return;
  }

  const index = JSON.parse(fs.readFileSync(path.join(validationDirectory, "index.json"), "utf8"));

  if (index.positions < BENCHMARK_POSITION_COUNT) {
    throw new Error(
      `The validation set of ${id} holds ${index.positions} positions; the bench needs ` +
        `${BENCHMARK_POSITION_COUNT}.`,
    );
  }

  const benchDirectory = resolve(`bench/dist/${id}`);
  const subset = (fileName, bytesPerPosition) =>
    fs
      .readFileSync(path.join(validationDirectory, fileName))
      .subarray(0, BENCHMARK_POSITION_COUNT * bytesPerPosition);

  fs.mkdirSync(benchDirectory, { recursive: true });
  fs.writeFileSync(
    path.join(benchDirectory, "positions.f32"),
    subset("features.f32", index.featureCount * 4),
  );

  for (const { fileName, bytesPerPosition } of BENCHMARK_SETS) {
    fs.writeFileSync(path.join(benchDirectory, fileName), subset(fileName, bytesPerPosition(index)));
  }

  const inputs = {
    modelId: index.modelId,
    modelSha256: index.modelSha256,
    boardSize: index.boardSize,
    featureCount: index.featureCount,
    moveCount: index.moveCount,
    positionCount: BENCHMARK_POSITION_COUNT,
  };

  fs.writeFileSync(
    path.join(benchDirectory, "bench-inputs.json"),
    `${JSON.stringify(inputs, null, 2)}\n`,
    "utf8",
  );
  copyModelFiles(id, benchDirectory);
  copyIfPresent(resolve(`zig-out/${id}/infer.wasm`), path.join(benchDirectory, "infer.wasm"));
  copyIfPresent(resolve("dist/worker.js"), path.join(benchDirectory, "worker.js"));
};

const main = async () => {
  for (const bundle of browserBundles) {
    await buildBundle(bundle);
  }

  for (const bundle of nodeBundles) {
    await buildNodeBundle(bundle);
  }

  copyPublishedFiles();

  for (const id of listModels()) {
    writeBenchmarkInputs(id);
  }
};

await main();
