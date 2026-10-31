import { createEngine, type Engine } from "../js/engine.ts";
import { createSearchEngine, type SearchEngine } from "../js/search-engine.ts";
import {
  encodeStudentFeatures,
  selectHighestLegalMove,
} from "../vendor/moka/src/game.ts";
import { LEVEL_BY_ID, type SiteWorkerRequest, type SiteWorkerResponse } from "./protocol.ts";

const WORKER_FILE_NAME = "engine-worker.js";

type LoadedEngine = {
  engine: Engine;
  inferModule: WebAssembly.Module;
  searchModule: WebAssembly.Module;
  weights: ArrayBuffer;
};

const post = (response: SiteWorkerResponse) => self.postMessage(response);

// The page loads this worker as engine-worker.js?v=<hash of the engine files>
// (scripts/build-site.mjs). The files it fetches carry the same version, so no
// cache can pair this worker with the files of another deploy.
const assetUrl = (fileName: string) => {
  const url = new URL(`./${fileName}`, import.meta.url);

  url.search = new URL(import.meta.url).search;

  return url.href;
};

const fetchBytes = async (url: string) => {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`${url} answered ${response.status}.`);
  }

  return response.arrayBuffer();
};

/** Same diagnostics as js/load-error.ts, parameterized by file name. */
const describeWasmLoadError = (
  fileName: string,
  { fetchFailed, simdValid }: { fetchFailed: boolean; simdValid: boolean },
) => {
  if (fetchFailed) {
    return `Failed to load ${fileName}.`;
  }

  if (!simdValid) {
    return "WebAssembly SIMD is not supported in this browser.";
  }

  return (
    `WebAssembly compilation was blocked; the Content Security Policy of ${WORKER_FILE_NAME} ` +
    "must allow 'wasm-unsafe-eval'."
  );
};

const compileWasmModule = async (fileName: string) => {
  let bytes: ArrayBuffer;

  try {
    bytes = await fetchBytes(assetUrl(fileName));
  } catch {
    throw new Error(describeWasmLoadError(fileName, { fetchFailed: true, simdValid: false }));
  }

  try {
    return await WebAssembly.compile(bytes);
  } catch {
    throw new Error(
      describeWasmLoadError(fileName, {
        fetchFailed: false,
        simdValid: WebAssembly.validate(bytes),
      }),
    );
  }
};

const loadEngine = async (): Promise<LoadedEngine> => {
  const manifestUrl = assetUrl("go-model.json");
  const weightsUrl = assetUrl("go-model.bin");

  let manifest: GoModelManifest;
  let weights: ArrayBuffer;

  try {
    const manifestResponse = await fetch(manifestUrl);

    if (!manifestResponse.ok) {
      throw new Error(`${manifestUrl} answered ${manifestResponse.status}.`);
    }

    manifest = (await manifestResponse.json()) as GoModelManifest;
    weights = await fetchBytes(weightsUrl);
  } catch {
    throw new Error("Failed to load the model files.");
  }

  if (weights.byteLength !== manifest.weightsBytes) {
    throw new Error("Model manifest and weights disagree on the weights size.");
  }

  const [inferModule, searchModule] = await Promise.all([
    compileWasmModule("infer.wasm"),
    compileWasmModule("search.wasm"),
  ]);
  const engine = await createEngine(inferModule, weights);

  return { engine, inferModule, searchModule, weights };
};

let loadedEngine: LoadedEngine | null = null;
let searchEngine: SearchEngine | null = null;
let pendingMessages: Promise<void> = Promise.resolve();

const handleInit = async () => {
  const startedAt = performance.now();

  loadedEngine = await loadEngine();
  post({ type: "ready", durationMs: performance.now() - startedAt });
};

const handleNewGame = async () => {
  if (!loadedEngine) {
    throw new Error("The site engine worker is not initialized.");
  }

  // A fresh weights copy gives every game a fresh search tree.
  searchEngine = await createSearchEngine(loadedEngine.searchModule, loadedEngine.weights.slice(0));
};

const handleMove = async (request: Extract<SiteWorkerRequest, { type: "move" }>) => {
  if (!loadedEngine) {
    throw new Error("The site engine worker is not initialized.");
  }

  const level = LEVEL_BY_ID[request.level] ?? LEVEL_BY_ID.beginner;
  const startedAt = performance.now();
  let move: number;
  let visits: number;

  if (level.usesSearch) {
    if (!searchEngine) {
      throw new Error("The site engine worker has no game yet.");
    }

    const outcome = searchEngine.search(request.state, {
      timeMs: level.timeMs,
      maxVisits: level.maxVisits,
      batch: 1,
      reuse: true,
    });

    move = outcome.move;
    visits = outcome.visits;
  } else {
    move = selectHighestLegalMove(
      request.state,
      loadedEngine.engine.infer(encodeStudentFeatures(request.state)).policyLogits,
    );
    visits = 1;
  }

  post({
    type: "moved",
    requestId: request.requestId,
    move,
    visits,
    thinkingMs: performance.now() - startedAt,
  });
};

const handleMessage = async (message: SiteWorkerRequest) => {
  try {
    if (message.type === "init") {
      await handleInit();
      return;
    }

    if (message.type === "new-game") {
      await handleNewGame();
      return;
    }

    await handleMove(message);
  } catch (error) {
    post({ type: "error", error: error instanceof Error ? error.message : String(error) });
  }
};

self.onmessage = (event: MessageEvent<SiteWorkerRequest>) => {
  pendingMessages = pendingMessages.then(() => handleMessage(event.data));
};
