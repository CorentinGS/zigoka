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

/** Writes `<tmp>/models/registry.json`, the repository's registry after `edit`; `<tmp>` stands for the root. */
const writeRegistry = (t, edit) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "model-registry-"));
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));

  const registry = JSON.parse(fs.readFileSync(path.join(rootDirectory, "models/registry.json"), "utf8"));
  edit(registry.models);
  fs.mkdirSync(path.join(directory, "models"));
  fs.writeFileSync(path.join(directory, "models", "registry.json"), JSON.stringify(registry, null, 2));

  return { root: directory, registry: path.join(directory, "models", "registry.json") };
};

const REGISTRY_REFUSALS = [
  {
    name: "a model listed twice",
    edit: (models) => models.push({ ...models[0] }),
    error: /model moka-9 is listed twice/,
  },
  {
    name: "an unknown kind",
    edit: (models) => {
      models[0].kind = "distilled";
    },
    error: /registry\.json:\d+:\d+: InvalidEnumTag/,
  },
  {
    name: "a trunkFrom that is not registered",
    edit: (models) => {
      models.find(({ kind }) => kind === "synthetic").trunkFrom = "nope";
    },
    error: /takes its trunk from nope, which is not registered/,
  },
  {
    name: "a trunkFrom that is not a trained model",
    edit: (models) => {
      const synthetic = models.filter(({ kind }) => kind === "synthetic");
      synthetic[0].trunkFrom = synthetic[1].id;
    },
    error: /which is not a trained model/,
  },
  {
    name: "an unknown key",
    edit: (models) => {
      models[0].boardSize = 9;
    },
    error: /registry\.json:\d+:\d+: UnknownField/,
  },
  {
    name: "a missing field",
    edit: (models) => {
      delete models[0].memoryPages;
    },
    error: /registry\.json:\d+:\d+: MissingField/,
  },
];

for (const { name, edit, error } of REGISTRY_REFUSALS) {
  test(`zig build refuses a registry with ${name}`, (t) => {
    const { registry } = writeRegistry(t, edit);
    const { status, stderr } = run("zig", ["build", "wasm", `-Dregistry=${registry}`]);

    assert.notEqual(status, 0);
    assert.match(stderr, error);
  });
}

test("zig build refuses an unknown model and lists the known ones", () => {
  const { status, stderr } = run("zig", ["build", "wasm", "-Dmodel=nope"]);

  assert.notEqual(status, 0);
  assert.match(stderr, /unknown model nope/);
  assert.match(stderr, /moka-9/);
});

test("run-oracle refuses a board size without an oracle bundle", (t) => {
  const { root, registry } = writeRegistry(t, (models) => {
    models.splice(0, models.length, { ...models[0], id: "bad", directory: "models/bad" });
  });
  const manifest = JSON.parse(
    fs.readFileSync(path.join(rootDirectory, "vendor/moka/model/go-model.json"), "utf8"),
  );
  manifest.architecture.boardSize = 13;
  fs.mkdirSync(path.join(root, "models", "bad"));
  fs.writeFileSync(path.join(root, "models", "bad", "go-model.json"), JSON.stringify(manifest));

  const { status, stderr } = run(process.execPath, ["scripts/run-oracle.mjs", "--registry", registry]);

  assert.notEqual(status, 0);
  assert.match(stderr, /no oracle bundle for board size 13 \(model bad\)/);
});
