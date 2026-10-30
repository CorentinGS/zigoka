import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { chromium } from "@playwright/test";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resultsDirectory = path.join(rootDirectory, "bench", "results");

const ROUND_COUNT = 5;
const WORKER_VALUES = [1, 0];
const DEFAULT_PORT = 4173;
const G4_RATIO_THRESHOLD = 2;
const BASELINE_DRIFT_PERCENT = 20;
const CONTROL_MODEL_ID = "moka-9";
const TRUNK_STAGE = "trunk";

const readOption = (name) => process.argv.includes(name);

const readValueOption = (name) => {
  const index = process.argv.indexOf(name);

  if (index === -1) {
    return null;
  }

  if (index + 1 >= process.argv.length) {
    throw new Error(`Missing value for option ${name}.`);
  }

  return process.argv[index + 1];
};

const readBenchInputs = (modelId) => {
  const inputsPath = path.join(rootDirectory, "bench", "dist", modelId, "bench-inputs.json");

  if (!fs.existsSync(inputsPath)) {
    throw new Error(
      `${path.relative(rootDirectory, inputsPath)} is missing. Run "pnpm oracle" and ` +
        '"pnpm build" first.',
    );
  }

  return JSON.parse(fs.readFileSync(inputsPath, "utf8"));
};

const readPort = () => {
  const portIndex = process.argv.indexOf("--port");

  if (portIndex === -1) {
    return DEFAULT_PORT;
  }

  const port = Number(process.argv[portIndex + 1]);

  if (!Number.isInteger(port) || port <= 0 || port > 65535) {
    throw new Error(`Invalid port: ${process.argv[portIndex + 1]}`);
  }

  return port;
};

const waitForServer = async (port) => {
  const deadline = Date.now() + 30_000;

  while (Date.now() < deadline) {
    try {
      const response = await fetch(`http://127.0.0.1:${port}/bench/index.html`);

      if (response.ok) {
        return;
      }
    } catch {
      // The server is not listening yet.
    }

    await new Promise((resolve) => setTimeout(resolve, 100));
  }

  throw new Error("The bench server did not start in time.");
};

const median = (values) => {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);

  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
};

const measure = async (page, port, modelId, runtime, worker, stage) => {
  const url =
    `http://127.0.0.1:${port}/bench/index.html` +
    `?model=${modelId}&runtime=${runtime}&worker=${worker}&mode=single&stage=${stage}`;

  await page.goto(url);
  await page.waitForFunction(() => window.__benchResult !== undefined, undefined, {
    timeout: 300_000,
  });

  const result = await page.evaluate(() => window.__benchResult);

  if (!result) {
    throw new Error("The bench page published no result.");
  }

  if (result.error) {
    throw new Error(result.error);
  }

  return result;
};

const buildSummary = (runs) => {
  const summary = {};

  for (const run of runs) {
    summary[run.runtime] ??= {};
    summary[run.runtime][run.worker] ??= { medianP50Ms: [] };
    summary[run.runtime][run.worker].medianP50Ms.push(run.p50Ms);
  }

  for (const runtime of Object.keys(summary)) {
    for (const worker of Object.keys(summary[runtime])) {
      summary[runtime][worker] = {
        medianP50Ms: median(summary[runtime][worker].medianP50Ms),
      };
    }
  }

  return summary;
};

const printTable = (runs) => {
  const header = ["model", "round", "runtime", "worker", "init ms", "p50 ms", "p95 ms", "mean ms"];
  const rows = runs.map((run) => [
    run.model,
    String(run.round),
    run.runtime,
    String(run.worker),
    run.initializationMs.toFixed(2),
    run.p50Ms.toFixed(3),
    run.p95Ms.toFixed(3),
    run.meanMs.toFixed(3),
  ]);
  const widths = header.map((cell, columnIndex) =>
    Math.max(cell.length, ...rows.map((row) => (row[columnIndex] ?? "").length)),
  );
  const format = (cells) =>
    cells.map((cell, columnIndex) => cell.padEnd(widths[columnIndex])).join("  ");

  process.stdout.write(`${format(header)}\n`);
  process.stdout.write(`${format(widths.map((width) => "-".repeat(width)))}\n`);

  for (const row of rows) {
    process.stdout.write(`${format(row)}\n`);
  }
};

const readBaselineMedian = (baselinePath) => {
  if (!fs.existsSync(baselinePath)) {
    return null;
  }

  const baseline = JSON.parse(fs.readFileSync(baselinePath, "utf8"));

  return baseline.summary?.moka?.["1"]?.medianP50Ms ?? null;
};

const getBaselinePath = (modelId) => path.join(resultsDirectory, `baseline-${modelId}.json`);

const buildModelReport = (modelId, benchInputs, runs) => {
  const summary = buildSummary(runs.filter((run) => run.model === modelId));
  const mokaMedian = summary.moka?.["1"]?.medianP50Ms ?? null;
  const zigWorkerMedian = summary.zig?.["1"]?.medianP50Ms ?? null;
  const zigMainThreadMedian = summary.zig?.["0"]?.medianP50Ms ?? null;
  const trunkMedian = summary.zig?.[TRUNK_STAGE]?.medianP50Ms ?? null;
  const baselineMedian = readBaselineMedian(getBaselinePath(modelId));

  return {
    modelSha256: benchInputs.modelSha256,
    boardSize: benchInputs.boardSize,
    summary,
    ratio: mokaMedian !== null && zigWorkerMedian !== null ? mokaMedian / zigWorkerMedian : null,
    headShare:
      zigMainThreadMedian !== null && trunkMedian !== null
        ? (zigMainThreadMedian - trunkMedian) / zigMainThreadMedian
        : null,
    baselineDriftPercent:
      baselineMedian === null || mokaMedian === null
        ? null
        : ((mokaMedian - baselineMedian) / baselineMedian) * 100,
  };
};

const writeResults = (report) => {
  const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
  const resultsPath = path.join(resultsDirectory, `${timestamp}.json`);

  fs.writeFileSync(resultsPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  process.stdout.write(`Results written to ${path.relative(rootDirectory, resultsPath)}.\n`);
};

const main = async () => {
  const isBaseline = readOption("--baseline");
  const isVerdict = readOption("--verdict");
  const port = readPort();
  const modelId = readValueOption("--model") ?? CONTROL_MODEL_ID;
  const modelIds =
    isVerdict && modelId !== CONTROL_MODEL_ID ? [modelId, CONTROL_MODEL_ID] : [modelId];
  const benchInputs = Object.fromEntries(modelIds.map((id) => [id, readBenchInputs(id)]));
  const runtimes = isBaseline ? ["moka"] : ["moka", "zig"];

  fs.mkdirSync(resultsDirectory, { recursive: true });

  const serverProcess = spawn(
    process.execPath,
    [path.join(rootDirectory, "bench", "serve.mjs"), "--port", String(port)],
    { stdio: ["ignore", "ignore", "inherit"] },
  );

  let browser;

  try {
    await waitForServer(port);

    browser = await chromium.launch({ headless: true });
    const page = await browser.newPage();
    const runs = [];

    const record = async (round, model, runtime, worker, stage) => {
      const result = await measure(page, port, model, runtime, worker, stage);

      runs.push({
        model,
        round,
        runtime,
        worker: stage === TRUNK_STAGE ? TRUNK_STAGE : worker,
        initializationMs: result.initializationMs,
        firstTenMs: result.firstTenMs,
        p50Ms: result.p50Ms,
        p95Ms: result.p95Ms,
        meanMs: result.meanMs,
        burst35Ms: result.burst35Ms,
        iterations: result.iterations,
      });
    };

    for (let round = 1; round <= ROUND_COUNT; round += 1) {
      const isOddRound = round % 2 === 1;
      const orderedModels = isOddRound ? modelIds : [...modelIds].reverse();
      const orderedRuntimes = isOddRound ? runtimes : [...runtimes].reverse();

      for (const model of orderedModels) {
        for (const worker of WORKER_VALUES) {
          for (const runtime of orderedRuntimes) {
            await record(round, model, runtime, worker, "full");

            if (runtime === "zig" && worker === 0) {
              await record(round, model, runtime, worker, TRUNK_STAGE);
            }
          }
        }
      }
    }

    printTable(runs);

    const date = new Date().toISOString();
    const machine = {
      cpu: os.cpus()[0]?.model ?? "unknown",
      browserVersion: browser.version(),
      node: process.version,
    };

    if (isBaseline) {
      const baselinePath = getBaselinePath(modelId);
      const baseline = {
        modelId,
        modelSha256: benchInputs[modelId].modelSha256,
        boardSize: benchInputs[modelId].boardSize,
        date,
        machine,
        runs,
        summary: buildSummary(runs),
      };

      fs.writeFileSync(baselinePath, `${JSON.stringify(baseline, null, 2)}\n`, "utf8");
      process.stdout.write(`Baseline written to ${path.relative(rootDirectory, baselinePath)}.\n`);

      return;
    }

    const models = Object.fromEntries(
      modelIds.map((id) => [id, buildModelReport(id, benchInputs[id], runs)]),
    );

    for (const [id, model] of Object.entries(models)) {
      if (
        model.baselineDriftPercent !== null &&
        Math.abs(model.baselineDriftPercent) > BASELINE_DRIFT_PERCENT
      ) {
        process.stdout.write(
          `Warning: Moka drifted by ${model.baselineDriftPercent.toFixed(1)}% on ${id} since ` +
            `${path.relative(rootDirectory, getBaselinePath(id))}. This run does not compare ` +
            "with that baseline.\n",
        );
      }
    }

    const report = { date, machine, runs, models };

    if (!isVerdict) {
      writeResults(report);

      return;
    }

    const { ratio } = models[modelId];

    if (ratio === null) {
      throw new Error("The verdict needs both engines measured with a Worker.");
    }

    const passed = ratio >= G4_RATIO_THRESHOLD;

    report.verdict = { model: modelId, threshold: G4_RATIO_THRESHOLD, ratio, passed };
    writeResults(report);

    for (const [id, model] of Object.entries(models)) {
      process.stdout.write(
        `${id}: ratio ${model.ratio.toFixed(2)}, head share ${(model.headShare * 100).toFixed(1)}%.\n`,
      );
    }

    process.stdout.write(
      `G4 ratio on ${modelId}: ${ratio.toFixed(2)} (${passed ? "passed" : "failed"}).\n`,
    );

    if (!passed) {
      process.exitCode = 1;
    }
  } finally {
    await browser?.close();
    serverProcess.kill("SIGTERM");
  }
};

try {
  await main();
} catch (error) {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exit(2);
}
