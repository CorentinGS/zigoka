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
];

const nodeBundles = [
  {
    entryPoints: [resolve("oracle/generate-fixtures.ts")],
    outfile: resolve("oracle/dist/generate.js"),
  },
  {
    entryPoints: [resolve("oracle/calibrate-tolerance.ts")],
    outfile: resolve("oracle/dist/calibrate.js"),
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

const copyModelFiles = () => {
  const modelDirectory = resolve("vendor/moka/model");
  const destinationDirectory = resolve("dist");

  fs.mkdirSync(destinationDirectory, { recursive: true });

  for (const entry of fs.readdirSync(modelDirectory)) {
    fs.copyFileSync(path.join(modelDirectory, entry), path.join(destinationDirectory, entry));
  }
};

const copyWasmArtifact = () => {
  const wasmPath = resolve("zig-out/bin/infer.wasm");

  if (!fs.existsSync(wasmPath)) {
    return;
  }

  fs.mkdirSync(resolve("dist"), { recursive: true });
  fs.copyFileSync(wasmPath, resolve("dist/infer.wasm"));
};

const BENCHMARK_POSITION_COUNT = 100;
const BENCHMARK_PLANE_COUNT = 12;
const BOARD_AREA = 81;
const POLICY_MOVE_COUNT = 82;
const FEATURE_COUNT = BOARD_AREA * BENCHMARK_PLANE_COUNT;

const copyTolerance = () => {
  const tolerancePath = resolve("oracle/tolerance.json");

  if (!fs.existsSync(tolerancePath)) {
    return;
  }

  fs.mkdirSync(resolve("bench/dist"), { recursive: true });
  fs.copyFileSync(tolerancePath, resolve("bench/dist/tolerance.json"));
};

const writeBenchmarkInputs = () => {
  const validationDirectory = resolve("oracle/fixtures/validation");

  if (!fs.existsSync(validationDirectory)) {
    return;
  }

  const index = JSON.parse(fs.readFileSync(path.join(validationDirectory, "index.json"), "utf8"));
  const positionCount = Math.min(BENCHMARK_POSITION_COUNT, index.positions);

  if (positionCount === 0) {
    throw new Error("The validation fixture set holds no position.");
  }

  const features = fs.readFileSync(path.join(validationDirectory, "features.f32"));
  const logits = fs.readFileSync(path.join(validationDirectory, "logits-normal.f32"));
  const values = fs.readFileSync(path.join(validationDirectory, "value-normal.f64"));

  const positions = Buffer.alloc(positionCount * FEATURE_COUNT * 4);
  positions.set(
    features.subarray(0, positionCount * FEATURE_COUNT * 4),
    0,
  );

  const expected = Buffer.alloc(positionCount * (POLICY_MOVE_COUNT + 1) * 4);
  expected.set(logits.subarray(0, positionCount * POLICY_MOVE_COUNT * 4), 0);

  for (let positionIndex = 0; positionIndex < positionCount; positionIndex += 1) {
    const value = values.readDoubleLE(positionIndex * 8);
    expected.writeFloatLE(value, (positionCount * POLICY_MOVE_COUNT + positionIndex) * 4);
  }

  fs.mkdirSync(resolve("bench/dist"), { recursive: true });
  fs.writeFileSync(resolve("bench/dist/positions.f32"), positions);
  fs.writeFileSync(resolve("bench/dist/expected.f32"), expected);
};

const main = async () => {
  for (const bundle of browserBundles) {
    await buildBundle(bundle);
  }

  for (const bundle of nodeBundles) {
    await buildNodeBundle(bundle);
  }

  copyModelFiles();
  copyWasmArtifact();
  copyTolerance();
  writeBenchmarkInputs();
};

await main();
