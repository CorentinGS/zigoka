import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  GO_BOARD_AREA,
  GO_BOARD_SIZE,
  GO_MINIMUM_PASS_MOVE_COUNT,
  GO_PASS_MOVE,
  GO_STUDENT_INPUT_PLANE_COUNT,
} from "../vendor/moka/src/constants.ts";
import {
  createGameState,
  encodeStudentFeatures,
  getLegalMoves,
  isGameOver,
  playMove,
  selectHighestLegalMove,
} from "../vendor/moka/src/index.ts";
import { GoModelRuntime as VendorGoModelRuntime } from "../vendor/moka/src/runtime.ts";
import {
  compareDirectories,
  packFloat32,
  packFloat64,
  packUint16,
} from "./fixture-files.ts";
import { assertHealthy, createHealthCounter, type Health, type HealthCounter } from "./health.ts";
import { GoModelRuntime as InstrumentedGoModelRuntime } from "./moka-runtime-instrumented.ts";
import { createMulberry32 } from "./mulberry32.ts";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const rootDirectory = path.resolve(scriptDirectory, "..", "..");
const vendorDirectories = ["moka", "moka-v2"].map((name) => path.join(rootDirectory, "vendor", name));
const fixturesDirectory = path.join(rootDirectory, "oracle", "fixtures");

const PLANE_COUNT = GO_STUDENT_INPUT_PLANE_COUNT;
const FEATURE_COUNT = GO_BOARD_AREA * PLANE_COUNT;
const MOVE_COUNT = GO_BOARD_AREA + 1;

const FORMAT_VERSION = 2;
const TRAINED_MODEL_IDS = new Set(["moka-9", "moka-9-v2"]);
const JOURNAL_PROBE_MODEL_ID = "moka-9";
const JOURNAL_PROBE_BLACK_MOVES = [20, 24, 56, 60];

type ValidationPlan =
  | { kind: "model"; seeds: number[]; randomMoveCount: number }
  | { kind: "random"; seeds: number[]; moveLimit: number };

type FixturePlan = {
  calibrationSeeds: number[];
  validation: ValidationPlan;
  smokeProbeMoves: number[];
};

const FIXTURE_PLANS: Record<number, FixturePlan> = {
  9: {
    calibrationSeeds: Array.from({ length: 50 }, (_, seedIndex) => seedIndex + 1),
    validation: {
      kind: "model",
      seeds: Array.from({ length: 20 }, (_, seedIndex) => seedIndex + 101),
      randomMoveCount: 6,
    },
    smokeProbeMoves: [20, 24, 56, 60],
  },
  19: {
    calibrationSeeds: [],
    validation: {
      kind: "random",
      seeds: Array.from({ length: 4 }, (_, seedIndex) => seedIndex + 101),
      moveLimit: 60,
    },
    smokeProbeMoves: [60, 72, 288, 300],
  },
};

type Position = {
  features: Float32Array;
  logitsNormal: Float32Array;
  logitsF32Accumulate: Float32Array;
  valueNormal: number;
  valueF32Accumulate: number;
  moveCount: number;
  legalMoves: Uint8Array;
  chosenMove: number;
};

type GameEntry = {
  seed: number | null;
  probe: string | null;
  start: number;
  count: number;
};

type Dataset = {
  positions: Position[];
  games: GameEntry[];
};

type Model = {
  id: string;
  manifest: GoModelManifest;
  weightsBuffer: ArrayBuffer;
  sha256: string;
};

type DatasetHeader = {
  modelId: string;
  modelSha256: string;
};

type Runtimes = {
  normal: VendorGoModelRuntime;
  instrumentedNormal: InstrumentedGoModelRuntime;
  f32Accumulate: InstrumentedGoModelRuntime;
  health: { counter: HealthCounter | null };
};

function fail(message: string): never {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

const readValueOption = (name: string): string | null => {
  const index = process.argv.indexOf(name);

  if (index < 0) {
    return null;
  }

  if (index + 1 >= process.argv.length) {
    fail(`Missing value for option ${name}.`);
  }

  return process.argv[index + 1] as string;
};

const readModel = (modelsDirectory: string, id: string): Model => {
  const modelDirectory = path.join(modelsDirectory, id);
  const manifestPath = path.join(modelDirectory, "go-model.json");

  if (!fs.existsSync(manifestPath)) {
    fail(`${manifestPath} is missing. Run "zig build models" first.`);
  }

  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8")) as GoModelManifest;
  const weights = fs.readFileSync(path.join(modelDirectory, "go-model.bin"));
  const weightsBuffer = weights.buffer.slice(
    weights.byteOffset,
    weights.byteOffset + weights.byteLength,
  ) as ArrayBuffer;
  const sha256 = crypto.createHash("sha256").update(weights).digest("hex");

  return { id, manifest, weightsBuffer, sha256 };
};

const assertBoardSize = (model: Model): void => {
  const boardSize = model.manifest.architecture.boardSize;
  const intersectionCount = createGameState().board.length;

  if (intersectionCount !== boardSize * boardSize) {
    fail(
      `This oracle bundle plays on ${intersectionCount} intersections, but model ${model.id} ` +
        `has boardSize ${boardSize} (${boardSize * boardSize} intersections).`,
    );
  }
};

const createRuntimes = ({ manifest, weightsBuffer }: Model): Runtimes => {
  const health: Runtimes["health"] = { counter: null };

  return {
    normal: VendorGoModelRuntime.create(manifest, weightsBuffer),
    instrumentedNormal: InstrumentedGoModelRuntime.create(manifest, weightsBuffer),
    f32Accumulate: InstrumentedGoModelRuntime.create(manifest, weightsBuffer, {
      f32Accumulate: true,
      onTensor: (name, values) => health.counter?.observe(name, values),
    }),
    health,
  };
};

const areFloatArraysEqual = (
  left: Float32Array | Float64Array,
  right: Float32Array | Float64Array,
): boolean => {
  if (left.length !== right.length) {
    return false;
  }

  const leftBytes = Buffer.from(left.buffer, left.byteOffset, left.byteLength);
  const rightBytes = Buffer.from(right.buffer, right.byteOffset, right.byteLength);

  return leftBytes.equals(rightBytes);
};

const recordPosition = (state: GoGameState, runtimes: Runtimes): Position => {
  const features = encodeStudentFeatures(state);
  const normal = runtimes.normal.infer(features);
  const f32Accumulate = runtimes.f32Accumulate.infer(features);
  const legalMoves = new Uint8Array(MOVE_COUNT);

  for (const move of getLegalMoves(state)) {
    if (move === GO_PASS_MOVE) {
      if (state.moveCount >= GO_MINIMUM_PASS_MOVE_COUNT) {
        legalMoves[GO_PASS_MOVE] = 1;
      }

      continue;
    }

    legalMoves[move] = 1;
  }

  runtimes.health.counter?.observeOutputs(
    f32Accumulate.policyLogits,
    f32Accumulate.value,
    legalMoves,
  );

  return {
    features,
    logitsNormal: normal.policyLogits,
    logitsF32Accumulate: f32Accumulate.policyLogits,
    valueNormal: normal.value,
    valueF32Accumulate: f32Accumulate.value,
    moveCount: state.moveCount,
    legalMoves,
    chosenMove: selectHighestLegalMove(state, normal.policyLogits),
  };
};

/**
 * The instrumented runtime must stay a faithful copy of Moka in normal mode.
 * Every probe position is compared bit for bit against the vendored runtime.
 */
const assertNormalParity = (state: GoGameState, runtimes: Runtimes): void => {
  const features = encodeStudentFeatures(state);
  const vendorResult = runtimes.normal.infer(features);
  const instrumentedResult = runtimes.instrumentedNormal.infer(features);

  if (!areFloatArraysEqual(vendorResult.policyLogits, instrumentedResult.policyLogits)) {
    fail(
      "The instrumented runtime does not reproduce Moka bit for bit in normal mode. " +
        "The oracle cannot be trusted.",
    );
  }

  if (
    !areFloatArraysEqual(
      Float64Array.of(vendorResult.value),
      Float64Array.of(instrumentedResult.value),
    )
  ) {
    fail(
      "The instrumented runtime does not reproduce the Moka value bit for bit in normal " +
        "mode. The oracle cannot be trusted.",
    );
  }
};

const drawRandomMove = (state: GoGameState, random: () => number): number => {
  const legalMoves = getLegalMoves(state);
  const candidates =
    state.moveCount < GO_MINIMUM_PASS_MOVE_COUNT
      ? legalMoves.filter((move) => move !== GO_PASS_MOVE)
      : legalMoves;

  if (candidates.length === 0) {
    return GO_PASS_MOVE;
  }

  return candidates[Math.floor(random() * candidates.length)];
};

const playRandomGame = (
  seed: number,
  moveLimit: number,
  runtimes: Runtimes,
  dataset: Dataset,
): void => {
  const random = createMulberry32(seed);
  const start = dataset.positions.length;
  let state = createGameState();

  while (!isGameOver(state) && dataset.positions.length - start < moveLimit) {
    dataset.positions.push(recordPosition(state, runtimes));

    const move = drawRandomMove(state, random);
    const nextState = playMove(state, move);

    if (!nextState) {
      fail(`Random game ${seed}: move ${move} was drawn but is illegal.`);
    }

    state = nextState;
  }

  dataset.games.push({
    seed,
    probe: null,
    start,
    count: dataset.positions.length - start,
  });
};

const playMokaGame = (
  seed: number,
  randomMoveCount: number,
  runtimes: Runtimes,
  dataset: Dataset,
): void => {
  const random = createMulberry32(seed);
  const start = dataset.positions.length;
  let state = createGameState();
  let moveCount = 0;

  while (!isGameOver(state)) {
    const position = recordPosition(state, runtimes);
    dataset.positions.push(position);

    const move =
      moveCount < randomMoveCount ? drawRandomMove(state, random) : position.chosenMove;
    const nextState = playMove(state, move);

    if (!nextState) {
      fail(`Validation game ${seed}: move ${move} is illegal.`);
    }

    state = nextState;
    moveCount += 1;
  }

  dataset.games.push({
    seed,
    probe: null,
    start,
    count: dataset.positions.length - start,
  });
};

const playJournalProbe = (runtimes: Runtimes, dataset: Dataset): void => {
  const start = dataset.positions.length;
  let state = createGameState();

  for (const blackMove of JOURNAL_PROBE_BLACK_MOVES) {
    const stateAfterBlackMove = playMove(state, blackMove);

    if (!stateAfterBlackMove) {
      fail(
        `Journal probe: Black move ${blackMove} is illegal. Moka answered on that ` +
          "intersection. The probe must be redefined.",
      );
    }

    state = stateAfterBlackMove;
    assertNormalParity(state, runtimes);

    const position = recordPosition(state, runtimes);
    dataset.positions.push(position);

    const whiteMove = position.chosenMove;
    const stateAfterWhiteMove = playMove(state, whiteMove);

    if (!stateAfterWhiteMove) {
      fail(`Journal probe: Moka answered ${whiteMove}, which is illegal.`);
    }

    state = stateAfterWhiteMove;
  }

  dataset.games.push({
    seed: null,
    probe: "journal",
    start,
    count: dataset.positions.length - start,
  });
};

const playSmokeProbe = (plan: FixturePlan, runtimes: Runtimes, dataset: Dataset): void => {
  const start = dataset.positions.length;
  let state = createGameState();

  for (const move of plan.smokeProbeMoves) {
    const nextState = playMove(state, move);

    if (!nextState) {
      fail(`Smoke probe: move ${move} is illegal.`);
    }

    state = nextState;
  }

  assertNormalParity(state, runtimes);
  dataset.positions.push(recordPosition(state, runtimes));
  dataset.games.push({
    seed: null,
    probe: "smoke",
    start,
    count: dataset.positions.length - start,
  });
};

const generateCalibrationDataset = (plan: FixturePlan, runtimes: Runtimes): Dataset => {
  const dataset: Dataset = { positions: [], games: [] };

  for (const seed of plan.calibrationSeeds) {
    playRandomGame(seed, Infinity, runtimes, dataset);
  }

  return dataset;
};

const generateValidationDataset = (
  model: Model,
  plan: FixturePlan,
  runtimes: Runtimes,
): Dataset => {
  const dataset: Dataset = { positions: [], games: [] };

  const { validation } = plan;

  for (const seed of validation.seeds) {
    if (validation.kind === "model") {
      playMokaGame(seed, validation.randomMoveCount, runtimes, dataset);
    } else {
      playRandomGame(seed, validation.moveLimit, runtimes, dataset);
    }
  }

  if (model.id === JOURNAL_PROBE_MODEL_ID) {
    playJournalProbe(runtimes, dataset);
  }

  playSmokeProbe(plan, runtimes, dataset);

  return dataset;
};

const datasetFiles = (
  header: DatasetHeader,
  dataset: Dataset,
  health: Health,
): Map<string, Buffer> => {
  const positionCount = dataset.positions.length;
  const features = new Float32Array(positionCount * FEATURE_COUNT);
  const logitsNormal = new Float32Array(positionCount * MOVE_COUNT);
  const logitsF32Accumulate = new Float32Array(positionCount * MOVE_COUNT);
  const valueNormal = new Float64Array(positionCount);
  const valueF32Accumulate = new Float64Array(positionCount);
  const moveCounts = new Uint16Array(positionCount);
  const legalMoves = new Uint8Array(positionCount * MOVE_COUNT);
  const chosenMoves = new Uint16Array(positionCount);

  for (let positionIndex = 0; positionIndex < positionCount; positionIndex += 1) {
    const position = dataset.positions[positionIndex] as Position;

    features.set(position.features, positionIndex * FEATURE_COUNT);
    logitsNormal.set(position.logitsNormal, positionIndex * MOVE_COUNT);
    logitsF32Accumulate.set(position.logitsF32Accumulate, positionIndex * MOVE_COUNT);
    valueNormal[positionIndex] = position.valueNormal;
    valueF32Accumulate[positionIndex] = position.valueF32Accumulate;
    moveCounts[positionIndex] = position.moveCount;
    legalMoves.set(position.legalMoves, positionIndex * MOVE_COUNT);
    chosenMoves[positionIndex] = position.chosenMove;
  }

  const files = new Map<string, Buffer>();

  files.set("features.f32", packFloat32(features));
  files.set("logits-normal.f32", packFloat32(logitsNormal));
  files.set("logits-f32acc.f32", packFloat32(logitsF32Accumulate));
  files.set("value-normal.f64", packFloat64(Array.from(valueNormal)));
  files.set("value-f32acc.f64", packFloat64(Array.from(valueF32Accumulate)));
  files.set("move-count.u16", packUint16(Array.from(moveCounts)));
  files.set("legal.u8", Buffer.from(legalMoves));
  files.set("chosen-move.u16", packUint16(Array.from(chosenMoves)));
  files.set(
    "index.json",
    Buffer.from(
      `${JSON.stringify(
        {
          formatVersion: FORMAT_VERSION,
          modelId: header.modelId,
          modelSha256: header.modelSha256,
          boardSize: GO_BOARD_SIZE,
          featureCount: FEATURE_COUNT,
          moveCount: MOVE_COUNT,
          health,
          positions: positionCount,
          games: dataset.games,
        },
        null,
        2,
      )}\n`,
      "utf8",
    ),
  );

  return files;
};

const writeFiles = (directory: string, files: Map<string, Buffer>): void => {
  fs.mkdirSync(directory, { recursive: true });

  for (const [name, content] of files) {
    fs.writeFileSync(path.join(directory, name), content);
  }
};

const tensorFiles = (runtimes: Runtimes): Map<string, Buffer> => {
  const tensors = runtimes.f32Accumulate.getDequantizedTensors();
  const files = new Map<string, Buffer>();
  const shapes: Record<string, number[]> = {};

  for (const name of Object.keys(tensors).sort()) {
    const tensor = tensors[name] as GoModelFloatTensor;

    files.set(`${name}.f32`, packFloat32(tensor.data));
    shapes[name] = tensor.shape;
  }

  files.set("index.json", Buffer.from(`${JSON.stringify({ tensors: shapes }, null, 2)}\n`, "utf8"));

  return files;
};

const readVendoredDigests = (vendorDirectory: string, label: string): Map<string, string> => {
  const vendoredMarkdown = fs.readFileSync(path.join(vendorDirectory, "VENDORED.md"), "utf8");
  const block = vendoredMarkdown.match(/```text\n([\s\S]*?)```/);

  if (!block) {
    fail(`${label}/VENDORED.md holds no sha256sum block.`);
  }

  const digests = new Map<string, string>();

  for (const line of (block[1] as string).split("\n")) {
    if (line.trim() === "") {
      continue;
    }

    const match = line.match(/^([0-9a-f]{64})\s+(.+)$/);

    if (!match) {
      fail(`${label}/VENDORED.md holds an unreadable digest line: ${line}`);
    }

    digests.set(match[2] as string, match[1] as string);
  }

  return digests;
};

const verifyVendoredFiles = (vendorDirectory: string): void => {
  const label = path.relative(rootDirectory, vendorDirectory);
  const digests = readVendoredDigests(vendorDirectory, label);
  const presentFiles = fs
    .readdirSync(vendorDirectory, { recursive: true })
    .map((entry) => entry.toString())
    .filter((entry) => fs.statSync(path.join(vendorDirectory, entry)).isFile())
    .filter((entry) => entry !== "VENDORED.md")
    .sort();

  for (const [filePath, expectedDigest] of digests) {
    const absolutePath = path.join(vendorDirectory, filePath);

    if (!fs.existsSync(absolutePath)) {
      fail(`${label}/${filePath} is listed in VENDORED.md but is missing.`);
    }

    const digest = crypto.createHash("sha256").update(fs.readFileSync(absolutePath)).digest("hex");

    if (digest !== expectedDigest) {
      fail(
        `${label}/${filePath} does not match its digest in VENDORED.md ` +
          `(expected ${expectedDigest}, got ${digest}).`,
      );
    }
  }

  for (const filePath of presentFiles) {
    if (!digests.has(filePath)) {
      fail(`${label}/${filePath} is not listed in VENDORED.md.`);
    }
  }
};

const main = (): void => {
  const shouldCheck = process.argv.includes("--check");
  const modelId = readValueOption("--model") ?? fail("Missing option --model <id>.");
  const modelsDirectory = path.resolve(
    readValueOption("--models-dir") ?? path.join(rootDirectory, "zig-out", "models"),
  );
  const plan = FIXTURE_PLANS[GO_BOARD_SIZE] ?? fail(`No fixture plan for board size ${GO_BOARD_SIZE}.`);

  vendorDirectories.forEach(verifyVendoredFiles);
  process.stdout.write("Vendored files match their digests.\n");

  const model = readModel(modelsDirectory, modelId);
  assertBoardSize(model);

  const runtimes = createRuntimes(model);

  if (model.id === JOURNAL_PROBE_MODEL_ID) {
    playJournalProbe(runtimes, { positions: [], games: [] });
  }

  playSmokeProbe(plan, runtimes, { positions: [], games: [] });
  process.stdout.write("The instrumented runtime reproduces Moka bit for bit on the probes.\n");

  const header: DatasetHeader = { modelId: model.id, modelSha256: model.sha256 };
  const files = new Map<string, Map<string, Buffer>>();
  const generateSet = (setName: string, gameCount: number, generate: () => Dataset) => {
    process.stdout.write(`Generating the ${model.id} ${setName} set (${gameCount} games)...\n`);
    runtimes.health.counter = createHealthCounter();
    const dataset = generate();
    const health = runtimes.health.counter.toJSON();
    runtimes.health.counter = null;

    if (!TRAINED_MODEL_IDS.has(model.id)) {
      try {
        assertHealthy(model.id, health);
      } catch (error) {
        fail(error instanceof Error ? error.message : String(error));
      }
    }

    files.set(setName, datasetFiles(header, dataset, health));
  };

  if (plan.calibrationSeeds.length > 0) {
    generateSet("calibration", plan.calibrationSeeds.length, () =>
      generateCalibrationDataset(plan, runtimes),
    );
  }

  generateSet("validation", plan.validation.seeds.length, () =>
    generateValidationDataset(model, plan, runtimes),
  );
  files.set("tensors", tensorFiles(runtimes));

  const modelDirectory = path.join(fixturesDirectory, model.id);
  const summary = [...files]
    .filter(([setName]) => setName !== "tensors")
    .map(([setName, setFiles]) => {
      const index = JSON.parse((setFiles.get("index.json") as Buffer).toString("utf8"));
      return `${index.positions} ${setName} positions`;
    })
    .join(" and ");

  if (!shouldCheck) {
    fs.rmSync(modelDirectory, { recursive: true, force: true });

    for (const [setName, setFiles] of files) {
      writeFiles(path.join(modelDirectory, setName), setFiles);
    }

    process.stdout.write(`Wrote ${summary} for ${model.id}.\n`);

    return;
  }

  const temporaryDirectory = path.join(fixturesDirectory, ".check", model.id);

  fs.rmSync(temporaryDirectory, { recursive: true, force: true });

  for (const [setName, setFiles] of files) {
    writeFiles(path.join(temporaryDirectory, setName), setFiles);
  }

  const referenceSets = fs.existsSync(modelDirectory) ? fs.readdirSync(modelDirectory).sort() : [];
  const candidateSets = [...files.keys()].sort();

  if (referenceSets.join(",") !== candidateSets.join(",")) {
    fail(
      `oracle/fixtures/${model.id} holds ${referenceSets.join(", ") || "nothing"}, ` +
        `expected ${candidateSets.join(", ")}. Run "pnpm oracle" before "pnpm oracle:check".`,
    );
  }

  for (const setName of candidateSets) {
    const difference = compareDirectories(
      path.join(modelDirectory, setName),
      path.join(temporaryDirectory, setName),
    );

    if (difference) {
      fail(`The reference data of ${model.id} is not reproducible: ${setName}/${difference}.`);
    }
  }

  fs.rmSync(temporaryDirectory, { recursive: true, force: true });

  process.stdout.write(
    `The reference data of ${model.id} is reproducible: ${summary}, byte for byte.\n`,
  );
};

main();
