import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const scriptDirectory = path.dirname(fileURLToPath(import.meta.url));
const rootDirectory = path.resolve(scriptDirectory, "..", "..");
const calibrationDirectory = path.join(rootDirectory, "oracle", "fixtures", "calibration");
const tolerancePath = path.join(rootDirectory, "oracle", "tolerance.json");

const MOVE_COUNT = 82;
const LOGITS_CAP = 2e-3;
const VALUE_CAP = 1e-3;
const TOLERANCE_FACTOR = 4;

function fail(message: string): never {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

const readFloat32 = (filePath: string): Float32Array => {
  const content = fs.readFileSync(filePath);
  return new Float32Array(content.buffer, content.byteOffset, content.byteLength / 4);
};

const readFloat64 = (filePath: string): Float64Array => {
  const content = fs.readFileSync(filePath);
  return new Float64Array(content.buffer, content.byteOffset, content.byteLength / 8);
};

const main = (): void => {
  const indexPath = path.join(calibrationDirectory, "index.json");

  if (!fs.existsSync(indexPath)) {
    fail("oracle/fixtures/calibration is missing. Run \"pnpm oracle\" first.");
  }

  const index = JSON.parse(fs.readFileSync(indexPath, "utf8")) as { positions: number };
  const positionCount = index.positions;

  const logitsNormal = readFloat32(path.join(calibrationDirectory, "logits-normal.f32"));
  const logitsF32Accumulate = readFloat32(path.join(calibrationDirectory, "logits-f32acc.f32"));
  const valueNormal = readFloat64(path.join(calibrationDirectory, "value-normal.f64"));
  const valueF32Accumulate = readFloat64(path.join(calibrationDirectory, "value-f32acc.f64"));

  if (
    logitsNormal.length !== positionCount * MOVE_COUNT ||
    logitsF32Accumulate.length !== positionCount * MOVE_COUNT ||
    valueNormal.length !== positionCount ||
    valueF32Accumulate.length !== positionCount
  ) {
    fail("The calibration fixtures are inconsistent with their index.json.");
  }

  let maxLogitDifference = 0;
  let maxValueDifference = 0;

  for (let valueIndex = 0; valueIndex < logitsNormal.length; valueIndex += 1) {
    const difference = Math.abs(
      (logitsNormal[valueIndex] as number) - (logitsF32Accumulate[valueIndex] as number),
    );

    if (difference > maxLogitDifference) {
      maxLogitDifference = difference;
    }
  }

  for (let valueIndex = 0; valueIndex < valueNormal.length; valueIndex += 1) {
    const difference = Math.abs(
      (valueNormal[valueIndex] as number) - (valueF32Accumulate[valueIndex] as number),
    );

    if (difference > maxValueDifference) {
      maxValueDifference = difference;
    }
  }

  const logitsTolerance = TOLERANCE_FACTOR * maxLogitDifference;
  const valueTolerance = TOLERANCE_FACTOR * maxValueDifference;

  const tolerance = {
    logits: logitsTolerance,
    value: valueTolerance,
    measured: {
      logits: maxLogitDifference,
      value: maxValueDifference,
      positions: positionCount,
      factor: TOLERANCE_FACTOR,
      capLogits: LOGITS_CAP,
      capValue: VALUE_CAP,
    },
  };

  fs.writeFileSync(tolerancePath, `${JSON.stringify(tolerance, null, 2)}\n`, "utf8");

  process.stdout.write(
    `Measured drift over ${positionCount} positions: logits ${maxLogitDifference.toExponential(3)}, ` +
      `value ${maxValueDifference.toExponential(3)}.\n` +
      `G3 tolerance: logits ${logitsTolerance.toExponential(3)} (cap ${LOGITS_CAP}), ` +
      `value ${valueTolerance.toExponential(3)} (cap ${VALUE_CAP}).\n` +
      `Written to oracle/tolerance.json.\n`,
  );

  if (logitsTolerance > LOGITS_CAP || valueTolerance > VALUE_CAP) {
    fail(
      "The calibrated tolerance exceeds the caps. The project stops here, before any Zig " +
        "code is written, and the design has to be revisited.",
    );
  }
};

main();
