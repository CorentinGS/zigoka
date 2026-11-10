import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { isMainThread, parentPort, Worker, workerData } from "node:worker_threads";

import { GO_BOARD_SIZE } from "../vendor/moka/src/constants.ts";
import { createGameState, encodeStudentFeatures, playMove, selectHighestLegalMove } from "../vendor/moka/src/game.ts";
import { GoModelRuntime } from "../vendor/moka/src/runtime.ts";
import { createEngine } from "../js/engine.ts";
import { createSearchEngine } from "../js/search-engine.ts";
import {
  createOpenings,
  createPlayer,
  createTasks,
  formatElo,
  getTimeMs,
  isPlayerKind,
  playGame,
  PLAYER_KINDS,
  summarize,
  type Engines,
  type GameResult,
  type GameTask,
  type MatchConfig,
  type MatchSummary,
  type Player,
  type PlayerKind,
} from "./games.ts";
import { createKataGoPlayer, type KataGoPlayer } from "./katago.ts";
import type { Evaluate } from "./search.ts";

const scriptPath = fileURLToPath(import.meta.url);
const rootDirectory = path.resolve(path.dirname(scriptPath), "..", "..");
const resultsDirectory = path.join(rootDirectory, "bench", "results");
/**
 * The bundle fixes the board: match.js plays 9x9 and defaults to Moka, match-19.js plays 19x19
 * (esbuild swaps in oracle/constants-19.ts) and has no default model.
 */
const DEFAULT_MODEL_ID = GO_BOARD_SIZE === 9 ? "moka-9" : undefined;
/** Zig and Moka must agree on these positions before a match starts; a mismatch would only play badly. */
const CROSS_CHECK_MOVES = 8;
const CROSS_CHECK_LOGIT_TOLERANCE = 0.05;
const CROSS_CHECK_VALUE_TOLERANCE = 0.01;
const KATAGO = "katago";
const KATAGO_CONFIG = path.join(rootDirectory, "match", "katago-gtp.cfg");
/** Visits of the normal model when the human SL model plays: they only decide when to pass. */
const HUMAN_PASS_VISITS = 16;
const KATAGO_HUMAN_CONFIG = path.join(rootDirectory, "match", "katago-human.cfg");

type SideKind = PlayerKind | typeof KATAGO;

type KataGoSettings = { binary?: string; model?: string; visits: number; humanModel?: string; profile?: string };

type CliConfig = Omit<MatchConfig, "first" | "second"> & {
  first: SideKind;
  second: SideKind;
  katago: KataGoSettings;
};

/**
 * When KataGo plays, `score` and `firstResult` come from KataGo's own score of the final position and
 * `mokaScore` keeps Moka's: Zig's search scores with Moka's dead-stone guess and can pass into a
 * position that guess misjudges.
 */
type CliGameResult = GameResult & { mokaScore?: number };

type GameTiming = { index: number; slot: number; startMs: number; endMs: number };

/** Registered models are built into zig-out/, local ones (zig build -Dlocal-model) into zig-out/local/. */
const findBuildDirectory = (modelId: string) => {
  const local = path.join(rootDirectory, "zig-out", "local");

  return fs.existsSync(path.join(local, "models", modelId, "go-model.json")) ? local : path.join(rootDirectory, "zig-out");
};

const loadEngines = async (modelId: string): Promise<Engines> => {
  const buildDirectory = findBuildDirectory(modelId);
  const modelDirectory = path.join(buildDirectory, "models", modelId);
  const wasmDirectory = path.join(buildDirectory, modelId);

  if (!fs.existsSync(path.join(modelDirectory, "go-model.json"))) {
    throw new Error(
      `Unknown model "${modelId}": neither zig-out/models/${modelId} nor zig-out/local/models/${modelId} exists.`,
    );
  }

  const manifest = JSON.parse(
    fs.readFileSync(path.join(modelDirectory, "go-model.json"), "utf8"),
  ) as GoModelManifest;

  if (manifest.architecture.boardSize !== GO_BOARD_SIZE) {
    throw new Error(
      `This match plays ${GO_BOARD_SIZE}x${GO_BOARD_SIZE} games; model "${modelId}" is ` +
        `${manifest.architecture.boardSize}x${manifest.architecture.boardSize}.`,
    );
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

  crossCheck(modelId, (features) => zig.infer(features), (features) => moka.infer(features));

  return {
    zig: (features) => zig.infer(features),
    moka: (features) => moka.infer(features),
    search: (state, options) => searchEngine.search(state, options),
  };
};

/** Compares Zig with Moka's runtime along the greedy line from the empty board. */
const crossCheck = (modelId: string, zig: Evaluate, moka: Evaluate) => {
  let state = createGameState();

  for (let move = 0; move <= CROSS_CHECK_MOVES; move += 1) {
    const features = encodeStudentFeatures(state);
    const expected = moka(features);
    const actual = zig(features);
    let logitError = 0;

    for (let index = 0; index < expected.policyLogits.length; index += 1) {
      logitError = Math.max(logitError, Math.abs(actual.policyLogits[index] - expected.policyLogits[index]));
    }

    if (logitError > CROSS_CHECK_LOGIT_TOLERANCE || Math.abs(actual.value - expected.value) > CROSS_CHECK_VALUE_TOLERANCE) {
      throw new Error(
        `Zig and Moka disagree on model "${modelId}" after ${move} moves ` +
          `(logits differ by ${logitError}, values ${actual.value} and ${expected.value}).`,
      );
    }

    state = playMove(state, selectHighestLegalMove(state, expected.policyLogits)) ?? state;
  }
};

const playerLabel = (config: CliConfig, side: "first" | "second") => {
  const kind = config[side];
  const timeMs = getTimeMs(config, side);
  const time = timeMs === config.timeMs ? "" : `/${timeMs}ms`;

  if (kind === KATAGO) {
    return config.katago.profile === undefined
      ? `${KATAGO}-${config.katago.visits}v`
      : `${KATAGO}-${config.katago.profile}`;
  }

  const model = config[side === "first" ? "firstModel" : "secondModel"];

  return `${model === DEFAULT_MODEL_ID ? kind : `${kind}@${model}`}${time}`;
};

/** Games where KataGo's score and Moka's score name different winners; `undefined` when KataGo did not score. */
type Disagreements = { count: number; games: number };

type CliSummary = MatchSummary<CliConfig> & { disagreements?: Disagreements };

const countDisagreements = (results: CliGameResult[]): Disagreements | undefined => {
  const scored = results.filter((result) => result.mokaScore !== undefined);

  if (scored.length === 0) {
    return undefined;
  }

  return {
    count: scored.filter((result) => Math.sign(result.mokaScore!) !== Math.sign(result.score)).length,
    games: scored.length,
  };
};

const printSummary = (summary: CliSummary) => {
  const { config } = summary;
  const first = playerLabel(config, "first");
  const second = playerLabel(config, "second");
  const games = summary.wins + summary.losses + summary.draws;
  const [low, high] = summary.scoreInterval;
  const [eloLow, eloHigh] = summary.eloInterval;

  console.log(`${first} vs ${second}, ${GO_BOARD_SIZE}x${GO_BOARD_SIZE}, ${games} games, ${config.timeMs} ms per move unless noted`);
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

  if (summary.disagreements) {
    const { count, games: scored } = summary.disagreements;

    console.log(`  scored by KataGo; Moka's score gives another winner in ${count} / ${scored} games`);
  }
};

const isSideKind = (kind: unknown): kind is SideKind => kind === KATAGO || isPlayerKind(kind);

const parseOptionalNumber = (value: string | undefined) => (value === undefined ? undefined : Number(value));

const parseConfig = (): CliConfig => {
  const { values } = parseArgs({
    options: {
      first: { type: "string", default: "zig-puct" },
      second: { type: "string", default: "moka-puct" },
      "first-model": { type: "string", ...(DEFAULT_MODEL_ID === undefined ? {} : { default: DEFAULT_MODEL_ID }) },
      "second-model": { type: "string", ...(DEFAULT_MODEL_ID === undefined ? {} : { default: DEFAULT_MODEL_ID }) },
      games: { type: "string", default: "100" },
      time: { type: "string", default: "100" },
      "first-time": { type: "string" },
      "second-time": { type: "string" },
      "katago-binary": { type: "string", default: process.env.KATAGO_BINARY },
      "katago-model": { type: "string", default: process.env.KATAGO_MODEL },
      "katago-visits": { type: "string" },
      "katago-human-model": { type: "string", default: process.env.KATAGO_HUMAN_MODEL },
      "katago-profile": { type: "string" },
      "max-visits": { type: "string" },
      batch: { type: "string", default: "1" },
      "opening-moves": { type: "string", default: "4" },
      parallel: { type: "string", default: "4" },
      seed: { type: "string", default: "1" },
    },
  });

  for (const kind of [values.first, values.second]) {
    if (!isSideKind(kind)) {
      throw new Error(`Unknown player "${kind}" (expected one of ${[...PLAYER_KINDS, KATAGO].join(", ")}).`);
    }
  }

  if (
    (values.first === KATAGO || values.second === KATAGO) &&
    (values["katago-binary"] === undefined || values["katago-model"] === undefined)
  ) {
    throw new Error("The katago player needs --katago-binary and --katago-model (or KATAGO_BINARY and KATAGO_MODEL).");
  }

  for (const side of ["first", "second"] as const) {
    if (values[side] !== KATAGO && values[`${side}-model`] === undefined) {
      throw new Error(`--${side}-model is required: the ${GO_BOARD_SIZE}x${GO_BOARD_SIZE} match has no default model.`);
    }
  }

  if (values["katago-profile"] !== undefined && values["katago-human-model"] === undefined) {
    throw new Error("--katago-profile needs --katago-human-model (or KATAGO_HUMAN_MODEL).");
  }

  const games = Number(values.games);

  if (!Number.isInteger(games) || games < 2 || games % 2 !== 0) {
    throw new Error("--games must be an even number of at least 2.");
  }

  return {
    first: values.first as SideKind,
    second: values.second as SideKind,
    firstModel: values["first-model"],
    secondModel: values["second-model"],
    games,
    timeMs: Number(values.time),
    firstTimeMs: parseOptionalNumber(values["first-time"]),
    secondTimeMs: parseOptionalNumber(values["second-time"]),
    katago: {
      binary: values["katago-binary"],
      model: values["katago-model"],
      visits: Number(values["katago-visits"] ?? (values["katago-profile"] === undefined ? 1 : HUMAN_PASS_VISITS)),
      humanModel: values["katago-human-model"],
      profile: values["katago-profile"],
    },
    maxVisits: parseOptionalNumber(values["max-visits"]),
    batch: Number(values.batch),
    openingMoves: Number(values["opening-moves"]),
    parallel: Math.min(Number(values.parallel), os.availableParallelism()),
    seed: Number(values.seed),
  };
};

const createSide = async (config: CliConfig, side: "first" | "second"): Promise<Player | KataGoPlayer> => {
  const kind = config[side];

  if (kind === KATAGO) {
    const { binary, model, visits, humanModel, profile } = config.katago;
    const human = profile === undefined ? undefined : { model: humanModel!, profile };

    return createKataGoPlayer({
      binary: binary!,
      model: model!,
      config: human ? KATAGO_HUMAN_CONFIG : KATAGO_CONFIG,
      visits,
      human,
    });
  }

  const engines = await loadEngines(config[side === "first" ? "firstModel" : "secondModel"]!);

  return createPlayer(kind, engines, config, getTimeMs(config, side));
};

const isKataGoPlayer = (player: Player | KataGoPlayer): player is KataGoPlayer => "finalScore" in player;

const scoreWithKataGo = async (result: GameResult, players: (Player | KataGoPlayer)[]): Promise<CliGameResult> => {
  const katago = players.find(isKataGoPlayer);

  if (!katago) {
    return result;
  }

  const score = await katago.finalScore([...result.opening, ...result.moves.map(({ move }) => move)]);
  const blackResult = Math.sign(score) as 1 | 0 | -1;

  return {
    ...result,
    score,
    firstResult: (result.firstPlaysBlack ? blackResult : -blackResult) as 1 | 0 | -1,
    mokaScore: result.score,
  };
};

const runWorker = async () => {
  const config = workerData as CliConfig;
  const first = await createSide(config, "first");
  const second = await createSide(config, "second");

  parentPort!.on("message", async (task: GameTask | null) => {
    if (task === null) {
      for (const player of [first, second].filter(isKataGoPlayer)) {
        player.close();
      }

      process.exit(0);
    }

    parentPort!.postMessage(await scoreWithKataGo(await playGame(task, first, second), [first, second]));
  });
  parentPort!.postMessage(null);
};

const runMain = async () => {
  const config = parseConfig();
  // Openings come from Moka at 9x9, as before, and from the first side that is a model otherwise.
  const openingModel = DEFAULT_MODEL_ID ?? (config.first === KATAGO ? config.secondModel : config.firstModel)!;
  const engines = await loadEngines(openingModel);
  const tasks = createTasks(
    createOpenings(config.games / 2, config.openingMoves, engines.zig, config.seed),
  );
  const results: CliGameResult[] = [];
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
        worker.on("message", (result: CliGameResult | null) => {
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

  const summary: CliSummary = { ...summarize(config, results), disagreements: countDisagreements(results) };
  // A player label carries its own time as `/10ms`, which must not become a directory.
  const fileLabel = (side: "first" | "second") => playerLabel(config, side).replaceAll("/", "-");
  const outputPath = path.join(
    resultsDirectory,
    `match-${fileLabel("first")}-vs-${fileLabel("second")}-${new Date().toISOString().replaceAll(":", "-")}.json`,
  );

  printSummary(summary);
  console.log(`  wall time ${((performance.now() - startedAt) / 1000).toFixed(0)} s`);
  fs.mkdirSync(resultsDirectory, { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify({ summary, results, timings }, null, 2)}\n`);
  console.log(`Results written to ${path.relative(rootDirectory, outputPath)}`);
};

await (isMainThread ? runMain() : runWorker());
