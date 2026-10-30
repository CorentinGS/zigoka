import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import {
  GO_BOARD_AREA,
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
import { GoModelRuntime as InstrumentedGoModelRuntime } from "./moka-runtime-instrumented.ts";
import { createMulberry32 } from "./mulberry32.ts";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const rootDirectory = path.resolve(scriptDirectory, "..", "..");
const vendorDirectory = path.join(rootDirectory, "vendor", "moka");
const fixturesDirectory = path.join(rootDirectory, "oracle", "fixtures");

const PLANE_COUNT = GO_STUDENT_INPUT_PLANE_COUNT;
const FEATURE_COUNT = GO_BOARD_AREA * PLANE_COUNT;
const MOVE_COUNT = GO_BOARD_AREA + 1;

const CALIBRATION_SEEDS = Array.from({ length: 50 }, (_, seedIndex) => seedIndex + 1);
const VALIDATION_SEEDS = Array.from({ length: 20 }, (_, seedIndex) => seedIndex + 101);
const VALIDATION_RANDOM_MOVE_COUNT = 6;

const JOURNAL_PROBE_BLACK_MOVES = [20, 24, 56, 60];
const SMOKE_PROBE_MOVES = [20, 24, 56, 60];

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

type Runtimes = {
  normal: VendorGoModelRuntime;
  instrumentedNormal: InstrumentedGoModelRuntime;
  f32Accumulate: InstrumentedGoModelRuntime;
};

function fail(message: string): never {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

const readModel = () => {
  const manifest = JSON.parse(
    fs.readFileSync(path.join(vendorDirectory, "model", "go-model.json"), "utf8"),
  ) as GoModelManifest;
  const weights = fs.readFileSync(path.join(vendorDirectory, "model", "go-model.bin"));
  const weightsBuffer = weights.buffer.slice(
    weights.byteOffset,
    weights.byteOffset + weights.byteLength,
  ) as ArrayBuffer;

  return { manifest, weightsBuffer };
};

const createRuntimes = (): Runtimes => {
  const { manifest, weightsBuffer } = readModel();

  return {
    normal: VendorGoModelRuntime.create(manifest, weightsBuffer),
    instrumentedNormal: InstrumentedGoModelRuntime.create(manifest, weightsBuffer),
    f32Accumulate: InstrumentedGoModelRuntime.create(manifest, weightsBuffer, {
      f32Accumulate: true,
    }),
  };
};

const areFloat32ArraysEqual = (left: Float32Array, right: Float32Array): boolean => {
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

  if (!areFloat32ArraysEqual(vendorResult.policyLogits, instrumentedResult.policyLogits)) {
    fail(
      "The instrumented runtime does not reproduce Moka bit for bit in normal mode. " +
        "The oracle cannot be trusted.",
    );
  }

  if (vendorResult.value !== instrumentedResult.value) {
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

const playRandomGame = (seed: number, runtimes: Runtimes, dataset: Dataset): void => {
  const random = createMulberry32(seed);
  const start = dataset.positions.length;
  let state = createGameState();

  while (!isGameOver(state)) {
    dataset.positions.push(recordPosition(state, runtimes));

    const move = drawRandomMove(state, random);
    const nextState = playMove(state, move);

    if (!nextState) {
      fail(`Calibration game ${seed}: move ${move} was drawn but is illegal.`);
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

const playMokaGame = (seed: number, runtimes: Runtimes, dataset: Dataset): void => {
  const random = createMulberry32(seed);
  const start = dataset.positions.length;
  let state = createGameState();
  let moveCount = 0;

  while (!isGameOver(state)) {
    const position = recordPosition(state, runtimes);
    dataset.positions.push(position);

    const move =
      moveCount < VALIDATION_RANDOM_MOVE_COUNT ? drawRandomMove(state, random) : position.chosenMove;
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

const playSmokeProbe = (runtimes: Runtimes, dataset: Dataset): void => {
  const start = dataset.positions.length;
  let state = createGameState();

  for (const move of SMOKE_PROBE_MOVES) {
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

const generateCalibrationDataset = (runtimes: Runtimes): Dataset => {
  const dataset: Dataset = { positions: [], games: [] };

  for (const seed of CALIBRATION_SEEDS) {
    playRandomGame(seed, runtimes, dataset);
  }

  return dataset;
};

const generateValidationDataset = (runtimes: Runtimes): Dataset => {
  const dataset: Dataset = { positions: [], games: [] };

  for (const seed of VALIDATION_SEEDS) {
    playMokaGame(seed, runtimes, dataset);
  }

  playJournalProbe(runtimes, dataset);
  playSmokeProbe(runtimes, dataset);

  return dataset;
};

const packFloat32 = (values: Float32Array): Buffer => {
  const buffer = Buffer.alloc(values.length * 4);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  for (let valueIndex = 0; valueIndex < values.length; valueIndex += 1) {
    view.setFloat32(valueIndex * 4, values[valueIndex] as number, true);
  }

  return buffer;
};

const packFloat64 = (values: number[]): Buffer => {
  const buffer = Buffer.alloc(values.length * 8);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  for (let valueIndex = 0; valueIndex < values.length; valueIndex += 1) {
    view.setFloat64(valueIndex * 8, values[valueIndex] as number, true);
  }

  return buffer;
};

const packUint16 = (values: number[]): Buffer => {
  const buffer = Buffer.alloc(values.length * 2);
  const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);

  for (let valueIndex = 0; valueIndex < values.length; valueIndex += 1) {
    view.setUint16(valueIndex * 2, values[valueIndex] as number, true);
  }

  return buffer;
};

const datasetFiles = (dataset: Dataset): Map<string, Buffer> => {
  const positionCount = dataset.positions.length;
  const features = new Float32Array(positionCount * FEATURE_COUNT);
  const logitsNormal = new Float32Array(positionCount * MOVE_COUNT);
  const logitsF32Accumulate = new Float32Array(positionCount * MOVE_COUNT);
  const valueNormal = new Float64Array(positionCount);
  const valueF32Accumulate = new Float64Array(positionCount);
  const moveCounts = new Uint16Array(positionCount);
  const legalMoves = new Uint8Array(positionCount * MOVE_COUNT);
  const chosenMoves = new Uint8Array(positionCount);

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
  files.set("chosen-move.u8", Buffer.from(chosenMoves));
  files.set(
    "index.json",
    Buffer.from(
      `${JSON.stringify({ positions: positionCount, games: dataset.games }, null, 2)}\n`,
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

const readVendoredDigests = (): Map<string, string> => {
  const vendoredMarkdown = fs.readFileSync(path.join(vendorDirectory, "VENDORED.md"), "utf8");
  const block = vendoredMarkdown.match(/```text\n([\s\S]*?)```/);

  if (!block) {
    fail("vendor/moka/VENDORED.md holds no sha256sum block.");
  }

  const digests = new Map<string, string>();

  for (const line of (block[1] as string).split("\n")) {
    if (line.trim() === "") {
      continue;
    }

    const match = line.match(/^([0-9a-f]{64})\s+(.+)$/);

    if (!match) {
      fail(`vendor/moka/VENDORED.md holds an unreadable digest line: ${line}`);
    }

    digests.set(match[2] as string, match[1] as string);
  }

  return digests;
};

const verifyVendoredFiles = (): void => {
  const digests = readVendoredDigests();
  const presentFiles = fs
    .readdirSync(vendorDirectory, { recursive: true })
    .map((entry) => entry.toString())
    .filter((entry) => fs.statSync(path.join(vendorDirectory, entry)).isFile())
    .filter((entry) => entry !== "VENDORED.md")
    .sort();

  for (const [filePath, expectedDigest] of digests) {
    const absolutePath = path.join(vendorDirectory, filePath);

    if (!fs.existsSync(absolutePath)) {
      fail(`vendor/moka/${filePath} is listed in VENDORED.md but is missing.`);
    }

    const digest = crypto.createHash("sha256").update(fs.readFileSync(absolutePath)).digest("hex");

    if (digest !== expectedDigest) {
      fail(
        `vendor/moka/${filePath} does not match its digest in VENDORED.md ` +
          `(expected ${expectedDigest}, got ${digest}).`,
      );
    }
  }

  for (const filePath of presentFiles) {
    if (!digests.has(filePath)) {
      fail(`vendor/moka/${filePath} is not listed in VENDORED.md.`);
    }
  }
};

const compareDirectories = (reference: string, candidate: string): string | null => {
  const referenceFiles = fs.readdirSync(reference).sort();
  const candidateFiles = fs.readdirSync(candidate).sort();

  if (referenceFiles.join(",") !== candidateFiles.join(",")) {
    return `file lists differ: ${referenceFiles.join(", ")} against ${candidateFiles.join(", ")}`;
  }

  for (const fileName of referenceFiles) {
    const referenceContent = fs.readFileSync(path.join(reference, fileName));
    const candidateContent = fs.readFileSync(path.join(candidate, fileName));

    if (!referenceContent.equals(candidateContent)) {
      return `${fileName} differs`;
    }
  }

  return null;
};

const main = (): void => {
  const shouldCheck = process.argv.includes("--check");

  verifyVendoredFiles();
  process.stdout.write("Vendored files match their digests.\n");

  const runtimes = createRuntimes();

  process.stdout.write("Generating the calibration set (50 games)...\n");
  const calibration = generateCalibrationDataset(runtimes);

  process.stdout.write("Generating the validation set (20 games and 2 probes)...\n");
  const validation = generateValidationDataset(runtimes);

  const tensorFilesByName = tensorFiles(runtimes);

  if (!shouldCheck) {
    writeFiles(path.join(fixturesDirectory, "calibration"), datasetFiles(calibration));
    writeFiles(path.join(fixturesDirectory, "validation"), datasetFiles(validation));
    writeFiles(path.join(fixturesDirectory, "tensors"), tensorFilesByName);

    process.stdout.write(
      `Wrote ${calibration.positions.length} calibration positions and ` +
        `${validation.positions.length} validation positions.\n`,
    );

    return;
  }

  const temporaryDirectory = path.join(fixturesDirectory, ".check");

  fs.rmSync(temporaryDirectory, { recursive: true, force: true });
  writeFiles(path.join(temporaryDirectory, "calibration"), datasetFiles(calibration));
  writeFiles(path.join(temporaryDirectory, "validation"), datasetFiles(validation));
  writeFiles(path.join(temporaryDirectory, "tensors"), tensorFilesByName);

  for (const setName of ["calibration", "validation", "tensors"]) {
    const referenceDirectory = path.join(fixturesDirectory, setName);

    if (!fs.existsSync(referenceDirectory)) {
      fail(
        `oracle/fixtures/${setName} is missing. Run "pnpm oracle" before "pnpm oracle:check".`,
      );
    }

    const difference = compareDirectories(referenceDirectory, path.join(temporaryDirectory, setName));

    if (difference) {
      fail(`The reference data is not reproducible: ${setName}/${difference}.`);
    }
  }

  fs.rmSync(temporaryDirectory, { recursive: true, force: true });

  process.stdout.write(
    `The reference data is reproducible: ${calibration.positions.length} calibration positions ` +
      `and ${validation.positions.length} validation positions, byte for byte.\n`,
  );
};

main();
