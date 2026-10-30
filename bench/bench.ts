import { GoModelWorkerClient } from "../vendor/moka/src/client.ts";
import { GoModelRuntime } from "../vendor/moka/src/runtime.ts";
import { createEngine } from "../js/engine.ts";
import { findParityMismatch } from "./parity-rule.ts";

const DEFAULT_MODEL_ID = "moka-9";
const WARMUP_COUNT = 10;
const BURST_COUNT = 5;
const BURST_SIZE = 35;

const MOKA_WORKER_URL = "/bench/dist/moka-worker.js";

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

type BenchInputs = {
  modelId: string;
  modelSha256: string;
  boardSize: number;
  featureCount: number;
  moveCount: number;
  positionCount: number;
};

type ModelUrls = {
  directory: string;
  manifest: string;
  weights: string;
  zigWasm: string;
  zigWorker: string;
};

type InferenceResult = { policyLogits: Float32Array; value: number };

type BenchFailure = {
  error: string;
};

type BenchEngine = {
  initializationMs: number;
  infer: (features: Float32Array) => Promise<InferenceResult>;
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

const loadFloat32 = async (url: string): Promise<Float32Array> =>
  new Float32Array(await fetchArrayBuffer(url));

const loadFloat64 = async (url: string): Promise<Float64Array> =>
  new Float64Array(await fetchArrayBuffer(url));

const loadJson = async <Content>(url: string): Promise<Content> => {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Unable to load ${url} (${response.status}).`);
  }

  return (await response.json()) as Content;
};

const getModelUrls = (modelId: string): ModelUrls => {
  const directory = `/bench/dist/${modelId}`;

  return {
    directory,
    manifest: `${directory}/go-model.json`,
    weights: `${directory}/go-model.bin`,
    zigWasm: `${directory}/infer.wasm`,
    zigWorker: `${directory}/worker.js`,
  };
};

const copyFeatures = (
  positions: Float32Array,
  featureCount: number,
  positionIndex: number,
): Float32Array => {
  const start = positionIndex * featureCount;

  return positions.slice(start, start + featureCount);
};

const fetchArrayBuffer = async (url: string): Promise<ArrayBuffer> => {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Unable to load ${url} (${response.status}).`);
  }

  return response.arrayBuffer();
};

const createWorkerEngine = async (workerUrl: string, urls: ModelUrls): Promise<BenchEngine> => {
  const client = new GoModelWorkerClient(new Worker(workerUrl, { type: "module" }));
  const initializationStartTime = performance.now();

  await client.initialize({
    manifestUrl: urls.manifest,
    weightsUrl: urls.weights,
  });

  return {
    initializationMs: performance.now() - initializationStartTime,
    infer: async (features: Float32Array) => client.infer(features),
    dispose: () => client.dispose(),
  };
};

const createMainThreadEngine = async (urls: ModelUrls): Promise<BenchEngine> => {
  const initializationStartTime = performance.now();
  const manifest = await loadJson<GoModelManifest>(urls.manifest);
  const runtime = GoModelRuntime.create(manifest, await fetchArrayBuffer(urls.weights));

  return {
    initializationMs: performance.now() - initializationStartTime,
    infer: async (features: Float32Array) => runtime.infer(features),
    dispose: () => undefined,
  };
};

const createZigMainThreadEngine = async (urls: ModelUrls): Promise<BenchEngine> => {
  const initializationStartTime = performance.now();
  const module = await WebAssembly.compile(await fetchArrayBuffer(urls.zigWasm));
  const engine = await createEngine(module, await fetchArrayBuffer(urls.weights));

  return {
    initializationMs: performance.now() - initializationStartTime,
    infer: async (features: Float32Array) => engine.infer(features),
    dispose: () => undefined,
  };
};

const createBenchEngine = (
  runtime: string,
  useWorker: boolean,
  urls: ModelUrls,
): Promise<BenchEngine> => {
  if (runtime === "moka") {
    return useWorker ? createWorkerEngine(MOKA_WORKER_URL, urls) : createMainThreadEngine(urls);
  }

  if (runtime === "zig") {
    return useWorker ? createWorkerEngine(urls.zigWorker, urls) : createZigMainThreadEngine(urls);
  }

  throw new Error(`Unknown runtime ${runtime}; expected moka or zig.`);
};

const assertFinite = (result: InferenceResult): void => {
  for (const logit of result.policyLogits) {
    if (!Number.isFinite(logit)) {
      throw new Error("The engine returned a non-finite policy logit.");
    }
  }

  if (!Number.isFinite(result.value)) {
    throw new Error("The engine returned a non-finite value.");
  }
};

const assertMatchesReference = async (
  runtime: string,
  engine: BenchEngine,
  positions: Float32Array,
  inputs: BenchInputs,
  urls: ModelUrls,
): Promise<void> => {
  const referenceName = runtime === "zig" ? "f32acc" : "normal";
  const expectedLogits = await loadFloat32(`${urls.directory}/logits-${referenceName}.f32`);
  const expectedValues = await loadFloat64(`${urls.directory}/value-${referenceName}.f64`);
  const logits = new Float32Array(inputs.positionCount * inputs.moveCount);
  const values = new Float64Array(inputs.positionCount);

  for (let positionIndex = 0; positionIndex < inputs.positionCount; positionIndex += 1) {
    const result = await engine.infer(copyFeatures(positions, inputs.featureCount, positionIndex));

    assertFinite(result);

    if (result.policyLogits.length !== inputs.moveCount) {
      throw new Error(
        `Position ${positionIndex}: ${result.policyLogits.length} logits, ${inputs.moveCount} expected.`,
      );
    }

    logits.set(result.policyLogits, positionIndex * inputs.moveCount);
    values[positionIndex] = result.value;
  }

  const mismatch = findParityMismatch(logits, values, expectedLogits, expectedValues);

  if (mismatch) {
    throw new Error(
      mismatch.kind === "logit"
        ? `Position ${mismatch.position}, move ${mismatch.index}: the logit differs from ` +
            `the ${referenceName} reference.`
        : `Position ${mismatch.position}: the value differs from the ${referenceName} ` +
            "reference by more than 2 ulp.",
    );
  }
};

const runBenchmark = async () => {
  const parameters = new URLSearchParams(window.location.search);
  const runtime = parameters.get("runtime") ?? "moka";
  const useWorker = (parameters.get("worker") ?? "1") !== "0";
  const mode = parameters.get("mode") ?? "single";
  const urls = getModelUrls(parameters.get("model") ?? DEFAULT_MODEL_ID);

  if (mode !== "single") {
    throw new Error(`The ${mode} mode is not implemented; only single is.`);
  }

  if (!window.crossOriginIsolated) {
    throw new Error(
      "The page is not cross-origin isolated, so performance.now() is rounded. " +
        "The server must send COOP and COEP headers.",
    );
  }

  const inputs = await loadJson<BenchInputs>(`${urls.directory}/bench-inputs.json`);
  const positions = await loadFloat32(`${urls.directory}/positions.f32`);
  const { featureCount, positionCount } = inputs;

  if (positions.length < positionCount * featureCount) {
    throw new Error(
      `The bench inputs hold ${positions.length} values, ${positionCount * featureCount} expected.`,
    );
  }

  const engine = await createBenchEngine(runtime, useWorker, urls);

  try {
    const firstTenMs: number[] = [];

    for (let warmupIndex = 0; warmupIndex < WARMUP_COUNT; warmupIndex += 1) {
      const features = copyFeatures(positions, featureCount, warmupIndex);
      const inferenceStartTime = performance.now();

      const result = await engine.infer(features);
      firstTenMs.push(performance.now() - inferenceStartTime);
      assertFinite(result);
    }

    await assertMatchesReference(runtime, engine, positions, inputs, urls);

    const inferenceDurationsMs: number[] = [];

    for (let positionIndex = 0; positionIndex < positionCount; positionIndex += 1) {
      const features = copyFeatures(positions, featureCount, positionIndex);
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
          copyFeatures(positions, featureCount, positionIndex),
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
