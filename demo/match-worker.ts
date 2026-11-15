import { GoModelRuntime } from "../vendor/moka/src/runtime.ts";
import { GO_BOARD_SIZE } from "../vendor/moka/src/constants.ts";
import { fetchFiles, loadEngines, type Engine, type LoadedEngines } from "../js/engines.ts";
import { fetchOk } from "../js/fetch-ok.ts";
import {
  createOpenings,
  createPlayer,
  getTimeMs,
  playGame,
  usesSearchWasm,
  type Engines,
} from "../match/games.ts";
import type { MatchWorkerRequest, MatchWorkerResponse } from "./match-protocol.ts";

const MANIFEST_URL = "/dist/go-model.json";
const read = fetchFiles({
  "go-model.bin": "/dist/go-model.bin",
  "infer.wasm": "/dist/infer.wasm",
  "search.wasm": "/demo/dist/search.wasm",
});

/** What both players share; each has a search engine of its own (createEngines). */
type SharedEngines = {
  loaded: LoadedEngines<"infer.wasm" | "search.wasm">;
  zig: Engine;
  moka: GoModelRuntime;
};

type Players = { first: ReturnType<typeof createPlayer>; second: ReturnType<typeof createPlayer> };

const post = (response: MatchWorkerResponse) => self.postMessage(response);

const loadSharedEngines = async (): Promise<SharedEngines> => {
  const [loaded, manifest] = await Promise.all([
    loadEngines(read, ["infer.wasm", "search.wasm"], { boardSize: GO_BOARD_SIZE }),
    fetchOk(MANIFEST_URL).then(async (response) => (await response.json()) as GoModelManifest),
  ]);
  const zig = await loaded.createEngine();
  const moka = GoModelRuntime.create(manifest, loaded.weights);

  return { loaded, zig, moka };
};

/** Each set of engines has a search engine, and so a search tree, of its own. */
const createEngines = async ({ loaded, zig, moka }: SharedEngines): Promise<Engines> => {
  const searchEngine = await loaded.createSearchEngine();

  return {
    zig: (features) => zig.infer(features),
    moka: (features) => moka.infer(features),
    search: (state, options) => searchEngine.search(state, options),
  };
};

let engines: Engines | null = null;
let players: Players | null = null;

const handleRequest = async (request: MatchWorkerRequest) => {
  if (request.type === "initialize") {
    const startedAt = performance.now();

    const { first, second } = request.config;

    const shared = await loadSharedEngines();

    engines = await createEngines(shared);
    players = {
      first: createPlayer(first, engines, request.config, getTimeMs(request.config, "first")),
      second: createPlayer(
        second,
        usesSearchWasm(first) && usesSearchWasm(second) ? await createEngines(shared) : engines,
        request.config,
        getTimeMs(request.config, "second"),
      ),
    };
    post({ type: "ready", durationMs: performance.now() - startedAt });

    return;
  }

  if (!engines || !players) {
    throw new Error("The match worker is not initialized.");
  }

  if (request.type === "openings") {
    post({
      type: "openings",
      openings: createOpenings(request.count, request.length, engines.zig, request.seed),
    });

    return;
  }

  const { task } = request;
  const result = await playGame(task, players.first, players.second, (event) => {
    post({
      type: "move",
      index: task.index,
      board: event.state.board,
      move: event.move,
      moveCount: event.state.moveCount,
      visits: event.visits,
      thinkingMs: event.thinkingMs,
    });
  });

  post({ type: "result", result });
};

self.addEventListener("message", (event: MessageEvent<MatchWorkerRequest>) => {
  handleRequest(event.data).catch((error: unknown) => {
    post({ type: "error", error: error instanceof Error ? error.message : String(error) });
  });
});
