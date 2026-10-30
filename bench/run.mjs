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
const DEFAULT_MODEL_ID = "moka-9";

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

const measure = async (page, port, modelId, runtime, worker) => {
  const url =
    `http://127.0.0.1:${port}/bench/index.html` +
    `?model=${modelId}&runtime=${runtime}&worker=${worker}&mode=single`;

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
  const header = ["round", "runtime", "worker", "init ms", "p50 ms", "p95 ms", "mean ms"];
  const rows = runs.map((run) => [
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

const main = async () => {
  const isBaseline = readOption("--baseline");
  const isVerdict = readOption("--verdict");
  const port = readPort();
  const modelId = readValueOption("--model") ?? DEFAULT_MODEL_ID;
  const benchInputs = readBenchInputs(modelId);
  const baselinePath = path.join(resultsDirectory, `baseline-${modelId}.json`);
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

    for (let round = 1; round <= ROUND_COUNT; round += 1) {
      const orderedRuntimes =
        round % 2 === 1 ? runtimes : [...runtimes].reverse();

      for (const worker of WORKER_VALUES) {
        for (const runtime of orderedRuntimes) {
          const result = await measure(page, port, modelId, runtime, worker);

          runs.push({
            round,
            runtime,
            worker,
            initializationMs: result.initializationMs,
            firstTenMs: result.firstTenMs,
            p50Ms: result.p50Ms,
            p95Ms: result.p95Ms,
            meanMs: result.meanMs,
            burst35Ms: result.burst35Ms,
            iterations: result.iterations,
          });
        }
      }
    }

    printTable(runs);

    const report = {
      modelId,
      modelSha256: benchInputs.modelSha256,
      boardSize: benchInputs.boardSize,
      date: new Date().toISOString(),
      machine: {
        cpu: os.cpus()[0]?.model ?? "unknown",
        browserVersion: browser.version(),
        node: process.version,
      },
      runs,
      summary: buildSummary(runs),
    };

    if (isVerdict) {
      const mokaMedian = report.summary.moka?.["1"]?.medianP50Ms;
      const zigMedian = report.summary.zig?.["1"]?.medianP50Ms;

      if (!mokaMedian || !zigMedian) {
        throw new Error("The verdict needs both engines measured with a Worker.");
      }

      const ratio = mokaMedian / zigMedian;
      const passed = ratio >= G4_RATIO_THRESHOLD;
      const baselineMedian = readBaselineMedian(baselinePath);
      const baselineDriftPercent =
        baselineMedian === null
          ? null
          : ((mokaMedian - baselineMedian) / baselineMedian) * 100;

      report.verdict = { ratio, passed, baselineDriftPercent };

      if (baselineDriftPercent !== null && Math.abs(baselineDriftPercent) > BASELINE_DRIFT_PERCENT) {
        process.stdout.write(
          `Warning: Moka drifted by ${baselineDriftPercent.toFixed(1)}% since the S0 baseline ` +
            `(${baselineMedian.toFixed(3)} ms). This run does not compare with S0.\n`,
        );
      }

      const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
      fs.writeFileSync(
        path.join(resultsDirectory, `${timestamp}.json`),
        `${JSON.stringify(report, null, 2)}\n`,
        "utf8",
      );

      process.stdout.write(`G4 ratio: ${ratio.toFixed(2)} (${passed ? "passed" : "failed"}).\n`);

      if (!passed) {
        process.exitCode = 1;
      }

      return;
    }

    if (isBaseline) {
      fs.writeFileSync(baselinePath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
      process.stdout.write(`Baseline written to ${path.relative(rootDirectory, baselinePath)}.\n`);

      return;
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    fs.writeFileSync(
      path.join(resultsDirectory, `${timestamp}.json`),
      `${JSON.stringify(report, null, 2)}\n`,
      "utf8",
    );
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
