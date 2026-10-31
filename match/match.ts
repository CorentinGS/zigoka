import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { isMainThread, parentPort, Worker, workerData } from "node:worker_threads";

import { GoModelRuntime } from "../vendor/moka/src/runtime.ts";
import { createEngine } from "../js/engine.ts";
import { createSearchEngine } from "../js/search-engine.ts";
import {
  createOpenings,
  createPlayer,
  createTasks,
  formatElo,
  isPlayerKind,
  playGame,
  PLAYER_KINDS,
  summarize,
  type Engines,
  type GameResult,
  type GameTask,
  type MatchConfig,
  type MatchSummary,
  type PlayerKind,
} from "./games.ts";

const scriptPath = fileURLToPath(import.meta.url);
const rootDirectory = path.resolve(path.dirname(scriptPath), "..", "..");
const resultsDirectory = path.join(rootDirectory, "bench", "results");
const DEFAULT_MODEL_ID = "moka-9";

type GameTiming = { index: number; slot: number; startMs: number; endMs: number };

const loadEngines = async (modelId: string): Promise<Engines> => {
  const modelDirectory = path.join(rootDirectory, "zig-out", "models", modelId);
  const wasmDirectory = path.join(rootDirectory, "zig-out", modelId);

  if (!fs.existsSync(path.join(modelDirectory, "go-model.json"))) {
    throw new Error(`Unknown model "${modelId}": ${path.relative(rootDirectory, modelDirectory)} is missing.`);
  }

  const manifest = JSON.parse(
    fs.readFileSync(path.join(modelDirectory, "go-model.json"), "utf8"),
  ) as GoModelManifest;

  if (manifest.architecture.boardSize !== 9) {
    throw new Error(`The match plays 9x9 games; model "${modelId}" is ${manifest.architecture.boardSize}x${manifest.architecture.boardSize}.`);
  }

  const weights = fs.readFileSync(path.join(modelDirectory, "go-model.bin"));
  const weightsBuffer = weights.buffer.slice(
    weights.byteOffset,
    weights.byteOffset + weights.byteLength,
  );
  const module = await WebAssembly.compile(fs.readFileSync(path.join(wasmDirectory, "infer.wasm")));
  const zig = await createEngine(module, weightsBuffer.slice(0));
  const searchEngine = await createSearchEngine(
    await WebAssembly.compile(fs.readFileSync(path.join(wasmDirectory, "search.wasm"))),
    weightsBuffer.slice(0),
  );
  const moka = GoModelRuntime.create(manifest, weightsBuffer.slice(0));

  return {
    zig: (features) => zig.infer(features),
    moka: (features) => moka.infer(features),
    search: (state, options) => searchEngine.search(state, options),
  };
};

const playerLabel = (config: MatchConfig, side: "first" | "second") => {
  const model = config[side === "first" ? "firstModel" : "secondModel"] ?? DEFAULT_MODEL_ID;

  return model === DEFAULT_MODEL_ID ? config[side] : `${config[side]}@${model}`;
};

const printSummary = (summary: MatchSummary) => {
  const { config } = summary;
  const first = playerLabel(config, "first");
  const second = playerLabel(config, "second");
  const games = summary.wins + summary.losses + summary.draws;
  const [low, high] = summary.scoreInterval;
  const [eloLow, eloHigh] = summary.eloInterval;

  console.log(`${first} vs ${second}, ${games} games, ${config.timeMs} ms per move`);
  console.log(
    `  ${first}: ${summary.wins} wins, ${summary.losses} losses, ${summary.draws} draws ` +
      `(${summary.winsAsBlack} as black, ${summary.winsAsWhite} as white)`,
  );
  console.log(
    `  score ${(summary.score * 100).toFixed(1)} % [${(low * 100).toFixed(1)}, ${(high * 100).toFixed(1)}], ` +
      `Elo ${formatElo(summary.elo)} [${formatElo(eloLow)}, ${formatElo(eloHigh)}]`,
  );
  console.log(`  black won ${summary.blackWins} / ${games}, average game length ${summary.averageGameLength.toFixed(1)} moves`);

  for (const side of ["first", "second"] as const) {
    const stats = summary[side];

    console.log(
      `  ${playerLabel(config, side)}: ${stats.visitsPerMove.toFixed(1)} visits per move, ${stats.msPerMove.toFixed(1)} ms per move`,
    );
  }
};

const parseConfig = (): MatchConfig => {
  const { values } = parseArgs({
    options: {
      first: { type: "string", default: "zig-puct" },
      second: { type: "string", default: "moka-puct" },
      "first-model": { type: "string", default: DEFAULT_MODEL_ID },
      "second-model": { type: "string", default: DEFAULT_MODEL_ID },
      games: { type: "string", default: "100" },
      time: { type: "string", default: "100" },
      "max-visits": { type: "string" },
      batch: { type: "string", default: "1" },
      "opening-moves": { type: "string", default: "4" },
      parallel: { type: "string", default: "4" },
      seed: { type: "string", default: "1" },
    },
  });

  for (const kind of [values.first, values.second]) {
    if (!isPlayerKind(kind)) {
      throw new Error(`Unknown player "${kind}" (expected one of ${PLAYER_KINDS.join(", ")}).`);
    }
  }

  const games = Number(values.games);

  if (!Number.isInteger(games) || games < 2 || games % 2 !== 0) {
    throw new Error("--games must be an even number of at least 2.");
  }

  return {
    first: values.first as PlayerKind,
    second: values.second as PlayerKind,
    firstModel: values["first-model"],
    secondModel: values["second-model"],
    games,
    timeMs: Number(values.time),
    maxVisits: values["max-visits"] === undefined ? undefined : Number(values["max-visits"]),
    batch: Number(values.batch),
    openingMoves: Number(values["opening-moves"]),
    parallel: Math.min(Number(values.parallel), os.availableParallelism()),
    seed: Number(values.seed),
  };
};

const runWorker = async () => {
  const config = workerData as MatchConfig;
  const firstModel = config.firstModel ?? DEFAULT_MODEL_ID;
  const secondModel = config.secondModel ?? DEFAULT_MODEL_ID;
  const firstEngines = await loadEngines(firstModel);
  const secondEngines = await loadEngines(secondModel);
  const first = createPlayer(config.first, firstEngines, config);
  const second = createPlayer(config.second, secondEngines, config);

  parentPort!.on("message", (task: GameTask | null) => {
    if (task === null) {
      process.exit(0);
    }

    parentPort!.postMessage(playGame(task, first, second));
  });
  parentPort!.postMessage(null);
};

const runMain = async () => {
  const config = parseConfig();
  const engines = await loadEngines(DEFAULT_MODEL_ID);
  const tasks = createTasks(
    createOpenings(config.games / 2, config.openingMoves, engines.zig, config.seed),
  );
  const results: GameResult[] = [];
  const timings: GameTiming[] = [];
  const startedAt = performance.now();
  let nextTask = 0;

  await Promise.all(
    Array.from({ length: config.parallel }, (_, slot) => {
      const worker = new Worker(scriptPath, { workerData: config });
      let timing: GameTiming | null = null;

      return new Promise<void>((resolve, reject) => {
        worker.on("error", reject);
        worker.on("exit", () => resolve());
        worker.on("message", (result: GameResult | null) => {
          const now = performance.now() - startedAt;

          if (result !== null && timing !== null) {
            results.push(result);
            timings.push({ ...timing, endMs: now });
            process.stderr.write(`\r${results.length} / ${tasks.length} games`);
          }

          const task = nextTask < tasks.length ? tasks[nextTask++] : null;

          timing = task === null ? null : { index: task.index, slot, startMs: now, endMs: now };
          worker.postMessage(task);
        });
      });
    }),
  );

  process.stderr.write("\n");
  results.sort((left, right) => left.index - right.index);
  timings.sort((left, right) => left.index - right.index);

  const summary = summarize(config, results);
  const outputPath = path.join(
    resultsDirectory,
    `match-${playerLabel(config, "first")}-vs-${playerLabel(config, "second")}-${new Date().toISOString().replaceAll(":", "-")}.json`,
  );

  printSummary(summary);
  console.log(`  wall time ${((performance.now() - startedAt) / 1000).toFixed(0)} s`);
  fs.mkdirSync(resultsDirectory, { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify({ summary, results, timings }, null, 2)}\n`);
  console.log(`Results written to ${path.relative(rootDirectory, outputPath)}`);
};

await (isMainThread ? runMain() : runWorker());
