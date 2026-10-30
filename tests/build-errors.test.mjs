import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const run = (command, args) =>
  spawnSync(command, args, { cwd: rootDirectory, encoding: "utf8" });

test("zig build refuses an unknown model and lists the known ones", () => {
  const { status, stderr } = run("zig", ["build", "wasm", "-Dmodel=nope"]);

  assert.notEqual(status, 0);
  assert.match(stderr, /unknown model nope/);
  assert.match(stderr, /moka-9/);
});

test("run-oracle refuses a board size without an oracle bundle", (t) => {
  const modelsDirectory = fs.mkdtempSync(path.join(os.tmpdir(), "run-oracle-"));
  t.after(() => fs.rmSync(modelsDirectory, { recursive: true, force: true }));

  const manifest = JSON.parse(
    fs.readFileSync(path.join(rootDirectory, "vendor/moka/model/go-model.json"), "utf8"),
  );
  manifest.architecture.boardSize = 13;
  fs.mkdirSync(path.join(modelsDirectory, "bad"));
  fs.writeFileSync(path.join(modelsDirectory, "bad", "go-model.json"), JSON.stringify(manifest));

  const { status, stderr } = run(process.execPath, [
    "scripts/run-oracle.mjs",
    "--models-dir",
    modelsDirectory,
  ]);

  assert.notEqual(status, 0);
  assert.match(stderr, /no oracle bundle for board size 13 \(model bad\)/);
});
