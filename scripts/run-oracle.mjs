import { spawnSync } from "node:child_process";
import fs from "node:fs";
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

const main = () => {
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

  for (const { id, bundle } of models) {
    const args = [bundle, "--model", id, "--models-dir", modelsDirectory];

    if (shouldCheck) {
      args.push("--check");
    }

    const { status } = spawnSync(process.execPath, args, { stdio: "inherit" });

    if (status !== 0) {
      process.exit(status ?? 1);
    }
  }
};

main();
