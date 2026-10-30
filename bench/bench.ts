import { GoModelWorkerClient } from "../vendor/moka/src/client.ts";
import { GoModelRuntime } from "../vendor/moka/src/runtime.ts";
import { createEngine } from "../js/engine.ts";

const POSITION_COUNT = 100;
const FEATURE_COUNT = 972;
const MOVE_COUNT = 82;
const WARMUP_COUNT = 10;
const BURST_COUNT = 5;
const BURST_SIZE = 35;

const MANIFEST_URL = "/dist/go-model.json";
const WEIGHTS_URL = "/dist/go-model.bin";
const MOKA_WORKER_URL = "/bench/dist/moka-worker.js";
const ZIG_WORKER_URL = "/dist/worker.js";
const ZIG_WASM_URL = "/dist/infer.wasm";

const P50_PERCENTILE = 0.5;
const P95_PERCENTILE = 0.95;

type BenchResult = {
  initializationMs: number;
  firstTenMs: number[];
  p50Ms: number;
  p95Ms: number;
  meanMs: number;
  burst35Ms: number[] | null;
  iterations: number;
};

type BenchTolerance = {
  logits: number;
  value: number;
};

type BenchFailure = {
  error: string;
};

type BenchEngine = {
  initializationMs: number;
  infer: (features: Float32Array) => Promise<{ policyLogits: Float32Array; value: number }>;
  dispose: () => void;
};

declare global {
  interface Window {
    __benchResult?: BenchResult | BenchFailure;
  }
}

const getPercentile = (values: number[], percentile: number) => {
  const sortedValues = [...values].sort((leftValue, rightValue) => leftValue - rightValue);
  const index = Math.min(
    sortedValues.length - 1,
    Math.round((sortedValues.length - 1) * percentile),
  );

  return sortedValues[index];
};

const publish = (result: BenchResult | BenchFailure) => {
  window.__benchResult = result;

  const resultsElement = document.querySelector("#results");

  if (resultsElement) {
    resultsElement.textContent = JSON.stringify(result, null, 2);
  }
};

const loadFloat32 = async (url: string): Promise<Float32Array> => {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Unable to load ${url} (${response.status}).`);
  }

  return new Float32Array(await response.arrayBuffer());
};

const loadJson = async <Content>(url: string): Promise<Content> => {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Unable to load ${url} (${response.status}).`);
  }

  return (await response.json()) as Content;
};

const copyFeatures = (positions: Float32Array, positionIndex: number): Float32Array => {
  const start = positionIndex * FEATURE_COUNT;

  return positions.slice(start, start + FEATURE_COUNT);
};

const fetchArrayBuffer = async (url: string): Promise<ArrayBuffer> => {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Unable to load ${url} (${response.status}).`);
  }

  return response.arrayBuffer();
};

const createWorkerEngine = async (workerUrl: string): Promise<BenchEngine> => {
  const client = new GoModelWorkerClient(new Worker(workerUrl, { type: "module" }));
  const initializationStartTime = performance.now();

  await client.initialize({
    manifestUrl: MANIFEST_URL,
    weightsUrl: WEIGHTS_URL,
  });

  return {
    initializationMs: performance.now() - initializationStartTime,
    infer: async (features: Float32Array) => client.infer(features),
    dispose: () => client.dispose(),
  };
};

const createMainThreadEngine = async (): Promise<BenchEngine> => {
  const initializationStartTime = performance.now();
  const manifest = await loadJson<GoModelManifest>(MANIFEST_URL);
  const runtime = GoModelRuntime.create(manifest, await fetchArrayBuffer(WEIGHTS_URL));

  return {
    initializationMs: performance.now() - initializationStartTime,
    infer: async (features: Float32Array) => runtime.infer(features),
    dispose: () => undefined,
  };
};

const createZigMainThreadEngine = async (): Promise<BenchEngine> => {
  const initializationStartTime = performance.now();
  const module = await WebAssembly.compile(await fetchArrayBuffer(ZIG_WASM_URL));
  const engine = await createEngine(module, await fetchArrayBuffer(WEIGHTS_URL));

  return {
    initializationMs: performance.now() - initializationStartTime,
    infer: async (features: Float32Array) => engine.infer(features),
    dispose: () => undefined,
  };
};

const createBenchEngine = (runtime: string, useWorker: boolean): Promise<BenchEngine> => {
  if (runtime === "moka") {
    return useWorker ? createWorkerEngine(MOKA_WORKER_URL) : createMainThreadEngine();
  }

  if (runtime === "zig") {
    return useWorker ? createWorkerEngine(ZIG_WORKER_URL) : createZigMainThreadEngine();
  }

  throw new Error(`Unknown runtime ${runtime}; expected moka or zig.`);
};

const assertFinite = (result: { policyLogits: Float32Array; value: number }): void => {
  for (const logit of result.policyLogits) {
    if (!Number.isFinite(logit)) {
      throw new Error("The engine returned a non-finite policy logit.");
    }
  }

  if (!Number.isFinite(result.value)) {
    throw new Error("The engine returned a non-finite value.");
  }
};

const assertMatchesReference = (
  result: { policyLogits: Float32Array; value: number },
  positionIndex: number,
  expected: Float32Array,
  tolerance: BenchTolerance,
): void => {
  const logitsOffset = positionIndex * MOVE_COUNT;
  const valueOffset = POSITION_COUNT * MOVE_COUNT;

  for (let moveIndex = 0; moveIndex < MOVE_COUNT; moveIndex += 1) {
    const difference = Math.abs(
      (result.policyLogits[moveIndex] as number) - (expected[logitsOffset + moveIndex] as number),
    );

    if (difference > tolerance.logits) {
      throw new Error(
        `Position ${positionIndex}, move ${moveIndex}: logit differs by ${difference}, ` +
          `beyond the tolerance of ${tolerance.logits}.`,
      );
    }
  }

  const valueDifference = Math.abs(result.value - (expected[valueOffset + positionIndex] as number));

  if (valueDifference > tolerance.value) {
    throw new Error(
      `Position ${positionIndex}: value differs by ${valueDifference}, ` +
        `beyond the tolerance of ${tolerance.value}.`,
    );
  }
};

const runBenchmark = async () => {
  const parameters = new URLSearchParams(window.location.search);
  const runtime = parameters.get("runtime") ?? "moka";
  const useWorker = (parameters.get("worker") ?? "1") !== "0";
  const mode = parameters.get("mode") ?? "single";

  if (mode !== "single") {
    throw new Error(`The ${mode} mode is not implemented; only single is.`);
  }

  if (!window.crossOriginIsolated) {
    throw new Error(
      "The page is not cross-origin isolated, so performance.now() is rounded. " +
        "The server must send COOP and COEP headers.",
    );
  }

  const positions = await loadFloat32("/bench/dist/positions.f32");
  const expected = await loadFloat32("/bench/dist/expected.f32");
  const tolerance = await loadJson<BenchTolerance>("/bench/dist/tolerance.json");

  if (positions.length < POSITION_COUNT * FEATURE_COUNT) {
    throw new Error(
      `The bench inputs hold ${positions.length} values, ${POSITION_COUNT * FEATURE_COUNT} expected.`,
    );
  }

  const engine = await createBenchEngine(runtime, useWorker);

  try {
    const firstTenMs: number[] = [];

    for (let warmupIndex = 0; warmupIndex < WARMUP_COUNT; warmupIndex += 1) {
      const features = copyFeatures(positions, warmupIndex);
      const inferenceStartTime = performance.now();

      const result = await engine.infer(features);
      firstTenMs.push(performance.now() - inferenceStartTime);
      assertFinite(result);
    }

    for (let positionIndex = 0; positionIndex < POSITION_COUNT; positionIndex += 1) {
      const result = await engine.infer(copyFeatures(positions, positionIndex));

      assertFinite(result);
      assertMatchesReference(result, positionIndex, expected, tolerance);
    }

    const inferenceDurationsMs: number[] = [];

    for (let positionIndex = 0; positionIndex < POSITION_COUNT; positionIndex += 1) {
      const features = copyFeatures(positions, positionIndex);
      const inferenceStartTime = performance.now();

      const result = await engine.infer(features);
      inferenceDurationsMs.push(performance.now() - inferenceStartTime);
      assertFinite(result);
    }

    let burst35Ms: number[] | null = null;

    if (useWorker) {
      const bursts: number[] = [];

      for (let burstIndex = 0; burstIndex < BURST_COUNT; burstIndex += 1) {
        const burstFeatures = Array.from({ length: BURST_SIZE }, (_, positionIndex) =>
          copyFeatures(positions, positionIndex),
        );
        const burstResults: Awaited<ReturnType<BenchEngine["infer"]>>[] = [];
        const burstStartTime = performance.now();

        for (const features of burstFeatures) {
          burstResults.push(await engine.infer(features));
        }

        bursts.push(performance.now() - burstStartTime);
        burstResults.forEach(assertFinite);
      }

      burst35Ms = bursts;
    }

    const meanMs =
      inferenceDurationsMs.reduce((durationSum, duration) => durationSum + duration, 0) /
      inferenceDurationsMs.length;

    publish({
      initializationMs: engine.initializationMs,
      firstTenMs,
      p50Ms: getPercentile(inferenceDurationsMs, P50_PERCENTILE),
      p95Ms: getPercentile(inferenceDurationsMs, P95_PERCENTILE),
      meanMs,
      burst35Ms,
      iterations: inferenceDurationsMs.length,
    });
  } finally {
    engine.dispose();
  }
};

void runBenchmark().catch((error: unknown) => {
  publish({ error: error instanceof Error ? error.message : String(error) });
});
