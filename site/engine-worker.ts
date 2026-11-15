import {
  compileEngines,
  fetchFiles,
  type Engine,
  type EngineFile,
  type LoadedEngines,
  type SearchEngine,
} from "../js/engines.ts";
import { GO_BOARD_SIZE } from "../vendor/moka/src/constants.ts";
import {
  encodeStudentFeatures,
  selectHighestLegalMove,
} from "../vendor/moka/src/game.ts";
import { LEVEL_BY_ID, type SiteWorkerRequest, type SiteWorkerResponse } from "./protocol.ts";

const ENGINE_FILES: EngineFile[] = ["infer.wasm", "search.wasm", "go-model.bin"];

type LoadedEngine = {
  engines: LoadedEngines;
  engine: Engine;
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

const loadEngine = async (): Promise<LoadedEngine> => {
  const read = fetchFiles(Object.fromEntries(ENGINE_FILES.map((file) => [file, assetUrl(file)])));
  const [compiled, weights] = await Promise.all([
    compileEngines(read, ["infer.wasm", "search.wasm"]),
    read("go-model.bin"),
  ]);
  const engines = await compiled.load(weights, { boardSize: GO_BOARD_SIZE });

  return { engines, engine: await engines.createEngine() };
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

  // Every game gets a new search engine, so a new search tree.
  searchEngine = await loadedEngine.engines.createSearchEngine();
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
