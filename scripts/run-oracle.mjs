import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const resolve = (relativePath) => path.join(rootDirectory, relativePath);

const FIRST_MODEL_ID = "moka-9";
const FORMAT_V1_DIRECTORIES = ["calibration", "validation", "tensors"];

const fail = (message) => {
  process.stderr.write(`${message}\n`);
  process.exit(1);
};

const readValueOption = (name) => {
  const index = process.argv.indexOf(name);

  if (index < 0) {
    return null;
  }

  if (index + 1 >= process.argv.length) {
    fail(`Missing value for option ${name}.`);
  }

  return process.argv[index + 1];
};

const removeFormatV1Directories = () => {
  for (const name of FORMAT_V1_DIRECTORIES) {
    fs.rmSync(resolve(`oracle/fixtures/${name}`), { recursive: true, force: true });
  }
};

const listModels = (modelsDirectory, selected) => {
  if (!fs.existsSync(modelsDirectory)) {
    fail(`${modelsDirectory} is missing. Run "zig build models" first.`);
  }

  const ids = fs
    .readdirSync(modelsDirectory)
    .filter((id) => fs.existsSync(path.join(modelsDirectory, id, "go-model.json")))
    .sort((left, right) =>
      left === FIRST_MODEL_ID ? -1 : right === FIRST_MODEL_ID ? 1 : left.localeCompare(right),
    );

  if (selected === null) {
    return ids;
  }

  if (!ids.includes(selected)) {
    fail(`unknown model ${selected}; known models: ${ids.join(", ")}`);
  }

  return [selected];
};

/** Runs one generator, its stdout and stderr kept together until it exits. */
const runModel = (args) =>
  new Promise((resolvePromise) => {
    const child = spawn(process.execPath, args, { stdio: ["ignore", "pipe", "pipe"] });
    const chunks = [];

    child.stdout.on("data", (chunk) => chunks.push(chunk));
    child.stderr.on("data", (chunk) => chunks.push(chunk));
    child.on("close", (status) => {
      process.stdout.write(Buffer.concat(chunks));
      resolvePromise({ status });
    });
  });

/** Runs the tasks, at most `limit` at a time, and returns their results in task order. */
const runInParallel = async (tasks, limit) => {
  const results = new Array(tasks.length);
  let next = 0;
  const worker = async () => {
    while (next < tasks.length) {
      const index = next;
      next += 1;
      results[index] = await tasks[index]();
    }
  };

  await Promise.all(Array.from({ length: limit }, worker));

  return results;
};

const main = async () => {
  const shouldCheck = process.argv.includes("--check");
  const modelsDirectory = path.resolve(readValueOption("--models-dir") ?? resolve("zig-out/models"));
  const models = listModels(modelsDirectory, readValueOption("--model")).map((id) => {
    const manifest = JSON.parse(
      fs.readFileSync(path.join(modelsDirectory, id, "go-model.json"), "utf8"),
    );
    const boardSize = manifest.architecture.boardSize;
    const bundle = resolve(`oracle/dist/generate-${boardSize}.js`);

    if (!fs.existsSync(bundle)) {
      fail(`no oracle bundle for board size ${boardSize} (model ${id})`);
    }

    return { id, bundle };
  });

  removeFormatV1Directories();

  // Each model writes its own oracle/fixtures/<id>/, so the models run in parallel; each
  // one's output is printed in one block once it finishes, in the order they finish.
  const results = await runInParallel(
    models.map(({ id, bundle }) => () => {
      const args = [bundle, "--model", id, "--models-dir", modelsDirectory];

      if (shouldCheck) {
        args.push("--check");
      }

      return runModel(args);
    }),
    Math.min(models.length, os.availableParallelism()),
  );
  const failed = results.find(({ status }) => status !== 0);

  if (failed) {
    process.exit(failed.status ?? 1);
  }
};

await main();
