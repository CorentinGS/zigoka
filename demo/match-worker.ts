import { GoModelRuntime } from "../vendor/moka/src/runtime.ts";
import { createEngine } from "../js/engine.ts";
import {
  createOpenings,
  createPlayer,
  playGame,
  type Evaluators,
} from "../match/games.ts";
import type { MatchWorkerRequest, MatchWorkerResponse } from "./match-protocol.ts";

const MODEL_URLS = {
  manifest: "/dist/go-model.json",
  weights: "/dist/go-model.bin",
  infer: "/dist/infer.wasm",
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

const loadEvaluators = async (): Promise<Evaluators> => {
  const [manifestResponse, weightsResponse, inferResponse] = await Promise.all([
    fetchChecked(MODEL_URLS.manifest),
    fetchChecked(MODEL_URLS.weights),
    fetchChecked(MODEL_URLS.infer),
  ]);
  const manifest = (await manifestResponse.json()) as GoModelManifest;
  const weights = await weightsResponse.arrayBuffer();
  const module = await WebAssembly.compile(await inferResponse.arrayBuffer());
  const zig = await createEngine(module, weights.slice(0));
  const moka = GoModelRuntime.create(manifest, weights);

  return {
    zig: (features) => zig.infer(features),
    moka: (features) => moka.infer(features),
  };
};

let evaluators: Evaluators | null = null;
let players: Players | null = null;

const handleRequest = async (request: MatchWorkerRequest) => {
  if (request.type === "initialize") {
    const startedAt = performance.now();

    evaluators = await loadEvaluators();
    players = {
      first: createPlayer(request.config.first, evaluators, request.config),
      second: createPlayer(request.config.second, evaluators, request.config),
    };
    post({ type: "ready", durationMs: performance.now() - startedAt });

    return;
  }

  if (!evaluators || !players) {
    throw new Error("The match worker is not initialized.");
  }

  if (request.type === "openings") {
    post({
      type: "openings",
      openings: createOpenings(request.count, request.length, evaluators.zig, request.seed),
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
