import { GoModelRuntime } from "../vendor/moka/src/runtime.ts";
import { createEngine } from "../js/engine.ts";
import { createSearchEngine } from "../js/search-engine.ts";
import {
  createOpenings,
  createPlayer,
  playGame,
  type Engines,
} from "../match/games.ts";
import type { MatchWorkerRequest, MatchWorkerResponse } from "./match-protocol.ts";

const MODEL_URLS = {
  manifest: "/dist/go-model.json",
  weights: "/dist/go-model.bin",
  infer: "/dist/infer.wasm",
  search: "/demo/dist/search.wasm",
};

type Players = { first: ReturnType<typeof createPlayer>; second: ReturnType<typeof createPlayer> };

const post = (response: MatchWorkerResponse) => self.postMessage(response);

const fetchChecked = async (url: string) => {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`${url} answered ${response.status}.`);
  }

  return response;
};

const loadEngines = async (): Promise<Engines> => {
  const [manifestResponse, weightsResponse, inferResponse, searchResponse] = await Promise.all([
    fetchChecked(MODEL_URLS.manifest),
    fetchChecked(MODEL_URLS.weights),
    fetchChecked(MODEL_URLS.infer),
    fetchChecked(MODEL_URLS.search),
  ]);
  const manifest = (await manifestResponse.json()) as GoModelManifest;
  const weights = await weightsResponse.arrayBuffer();
  const [inferModule, searchModule] = await Promise.all([
    WebAssembly.compile(await inferResponse.arrayBuffer()),
    WebAssembly.compile(await searchResponse.arrayBuffer()),
  ]);
  const zig = await createEngine(inferModule, weights.slice(0));
  const searchEngine = await createSearchEngine(searchModule, weights.slice(0));
  const moka = GoModelRuntime.create(manifest, weights);

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

    engines = await loadEngines();
    players = {
      first: createPlayer(request.config.first, engines, request.config),
      second: createPlayer(request.config.second, engines, request.config),
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
  const result = playGame(task, players.first, players.second, (event) => {
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
