import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { isMainThread, parentPort, Worker, workerData } from "node:worker_threads";

import { GoModelRuntime } from "../vendor/moka/src/runtime.ts";
import { createEngine } from "../js/engine.ts";
import {
  createOpenings,
  createPlayer,
  createTasks,
  formatElo,
  isPlayerKind,
  playGame,
  PLAYER_KINDS,
  summarize,
  type Evaluators,
  type GameResult,
  type GameTask,
  type MatchConfig,
  type MatchSummary,
  type PlayerKind,
} from "./games.ts";

const scriptPath = fileURLToPath(import.meta.url);
const rootDirectory = path.resolve(path.dirname(scriptPath), "..", "..");
const distDirectory = path.join(rootDirectory, "dist");
const resultsDirectory = path.join(rootDirectory, "bench", "results");

const loadEvaluators = async (): Promise<Evaluators> => {
  const manifest = JSON.parse(
    fs.readFileSync(path.join(distDirectory, "go-model.json"), "utf8"),
  ) as GoModelManifest;
  const weights = fs.readFileSync(path.join(distDirectory, "go-model.bin"));
  const weightsBuffer = weights.buffer.slice(
    weights.byteOffset,
    weights.byteOffset + weights.byteLength,
  );
  const module = await WebAssembly.compile(fs.readFileSync(path.join(distDirectory, "infer.wasm")));
  const zig = await createEngine(module, weightsBuffer.slice(0));
  const moka = GoModelRuntime.create(manifest, weightsBuffer.slice(0));

  return {
    zig: (features) => zig.infer(features),
    moka: (features) => moka.infer(features),
  };
};

const printSummary = (summary: MatchSummary) => {
  const { config } = summary;
  const games = summary.wins + summary.losses + summary.draws;
  const [low, high] = summary.scoreInterval;
  const [eloLow, eloHigh] = summary.eloInterval;

  console.log(`${config.first} vs ${config.second}, ${games} games, ${config.timeMs} ms per move`);
  console.log(
    `  ${config.first}: ${summary.wins} wins, ${summary.losses} losses, ${summary.draws} draws ` +
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
      `  ${config[side]}: ${stats.visitsPerMove.toFixed(1)} visits per move, ${stats.msPerMove.toFixed(1)} ms per move`,
    );
  }
};

const parseConfig = (): MatchConfig => {
  const { values } = parseArgs({
    options: {
      first: { type: "string", default: "zig-puct" },
      second: { type: "string", default: "moka-puct" },
      games: { type: "string", default: "100" },
      time: { type: "string", default: "100" },
      "max-visits": { type: "string" },
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
    games,
    timeMs: Number(values.time),
    maxVisits: values["max-visits"] === undefined ? undefined : Number(values["max-visits"]),
    openingMoves: Number(values["opening-moves"]),
    parallel: Math.min(Number(values.parallel), os.availableParallelism()),
    seed: Number(values.seed),
  };
};

const runWorker = async () => {
  const config = workerData as MatchConfig;
  const evaluators = await loadEvaluators();
  const first = createPlayer(config.first, evaluators, config);
  const second = createPlayer(config.second, evaluators, config);

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
  const evaluators = await loadEvaluators();
  const tasks = createTasks(
    createOpenings(config.games / 2, config.openingMoves, evaluators.zig, config.seed),
  );
  const results: GameResult[] = [];
  const startedAt = performance.now();
  let nextTask = 0;

  await Promise.all(
    Array.from({ length: config.parallel }, () => {
      const worker = new Worker(scriptPath, { workerData: config });

      return new Promise<void>((resolve, reject) => {
        worker.on("error", reject);
        worker.on("exit", () => resolve());
        worker.on("message", (result: GameResult | null) => {
          if (result !== null) {
            results.push(result);
            process.stderr.write(`\r${results.length} / ${tasks.length} games`);
          }

          worker.postMessage(nextTask < tasks.length ? tasks[nextTask++] : null);
        });
      });
    }),
  );

  process.stderr.write("\n");
  results.sort((left, right) => left.index - right.index);

  const summary = summarize(config, results);
  const outputPath = path.join(
    resultsDirectory,
    `match-${config.first}-vs-${config.second}-${new Date().toISOString().replaceAll(":", "-")}.json`,
  );

  printSummary(summary);
  console.log(`  wall time ${((performance.now() - startedAt) / 1000).toFixed(0)} s`);
  fs.mkdirSync(resultsDirectory, { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify({ summary, results }, null, 2)}\n`);
  console.log(`Results written to ${path.relative(rootDirectory, outputPath)}`);
};

await (isMainThread ? runMain() : runWorker());
