import { fetchFiles, loadEngines, type Engine, type LoadedEngines, type SearchEngine } from "../js/engines.ts";
import { GO_BOARD_SIZE } from "../vendor/moka/src/constants.ts";
import {
  encodeStudentFeatures,
  selectHighestLegalMove,
} from "../vendor/moka/src/game.ts";
import { WORKER_FILES } from "./engine-files.ts";
import { LEVEL_BY_ID, type SiteWorkerRequest, type SiteWorkerResponse } from "./protocol.ts";

type SiteEngines = {
  /** Creates the search engine of each game. */
  loaded: LoadedEngines<"infer.wasm" | "search.wasm">;
  /** Plays the levels that do not search. */
  inferEngine: Engine;
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

const loadSiteEngines = async (): Promise<SiteEngines> => {
  const read = fetchFiles(Object.fromEntries(WORKER_FILES.map((file) => [file, assetUrl(file)])));
  const loaded = await loadEngines(read, ["infer.wasm", "search.wasm"], { boardSize: GO_BOARD_SIZE });

  return { loaded, inferEngine: await loaded.createEngine() };
};

let siteEngines: SiteEngines | null = null;
let searchEngine: SearchEngine | null = null;
let pendingMessages: Promise<void> = Promise.resolve();

const handleInit = async () => {
  const startedAt = performance.now();

  siteEngines = await loadSiteEngines();
  post({ type: "ready", durationMs: performance.now() - startedAt });
};

const handleNewGame = async () => {
  if (!siteEngines) {
    throw new Error("The site engine worker is not initialized.");
  }

  // Every game gets a new search engine, so a new search tree.
  searchEngine = await siteEngines.loaded.createSearchEngine();
};

const handleMove = async (request: Extract<SiteWorkerRequest, { type: "move" }>) => {
  if (!siteEngines) {
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
      siteEngines.inferEngine.infer(encodeStudentFeatures(request.state)).policyLogits,
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
