import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test, type TestContext } from "node:test";
import { fileURLToPath } from "node:url";

import { findModel, listModels } from "../models/registry.ts";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const writeFile = (filePath: string, content: string) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, content);
};

const writeManifest = (directory: string, boardSize: number) =>
  writeFile(path.join(directory, "go-model.json"), JSON.stringify({ architecture: { boardSize } }));

/**
 * A repository of its own in a temporary folder: `zeta` and `alpha` registered in that order,
 * `draft` exported locally, and only `zeta/infer.wasm` built.
 */
const createRepository = (t: TestContext) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "model-registry-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));

  const search = { nodes: 1, edges: 1, memoryPages: 2 };
  const registry = path.join(root, "models", "registry.json");

  writeFile(
    registry,
    JSON.stringify({
      models: [
        { id: "zeta", directory: "vendor/zeta/model", kind: "trained", memoryPages: 1, search },
        {
          id: "alpha",
          directory: "models/alpha",
          kind: "synthetic",
          trunkFrom: "zeta",
          memoryPages: 1,
          search,
        },
      ],
    }),
  );
  writeManifest(path.join(root, "vendor", "zeta", "model"), 9);
  writeManifest(path.join(root, "models", "alpha"), 19);
  writeManifest(path.join(root, "models", "local", "draft"), 19);
  writeFile(path.join(root, "zig-out", "zeta", "infer.wasm"), "");

  return { root, registry };
};

test("listModels keeps the registry's order and never lists a local model", (t) => {
  const { root, registry } = createRepository(t);
  const models = listModels({ registry });

  assert.deepEqual(
    models.map(({ id }) => id),
    ["zeta", "alpha"],
  );
  assert.deepEqual(
    models.map(({ directory }) => directory),
    [path.join(root, "vendor", "zeta", "model"), path.join(root, "models", "alpha")],
  );
});

test("a model's board size comes from its manifest", (t) => {
  const { registry } = createRepository(t);

  assert.deepEqual(
    listModels({ registry }).map(({ boardSize }) => boardSize),
    [9, 19],
  );
  assert.equal(findModel("moka-9").boardSize, 9);
  assert.equal(findModel("moka-19").boardSize, 19);
});

test("findModel gives a synthetic model the trained model whose trunk it keeps", (t) => {
  const { registry } = createRepository(t);
  const model = findModel("alpha", { registry });

  assert.ok(!model.local && model.kind === "synthetic");
  assert.equal(model.trunkFrom, "zeta");
});

test("findModel finds a local model, which has no kind", (t) => {
  const { root, registry } = createRepository(t);
  const model = findModel("draft", { registry });

  assert.equal(model.local, true);
  assert.equal("kind" in model, false);
  assert.equal(model.directory, path.join(root, "models", "local", "draft"));
  assert.equal(model.boardSize, 19);
});

test("findModel refuses an unknown model and lists the registered ones", (t) => {
  const { registry } = createRepository(t);

  assert.throws(
    () => findModel("nope", { registry }),
    /Unknown model "nope": registered models are zeta, alpha, and models\/local\/nope\/ does not exist\./,
  );
});

test("wasm names the build command of a missing file; wasmIfBuilt gives null", (t) => {
  const { root, registry } = createRepository(t);
  const zeta = findModel("zeta", { registry });
  const draft = findModel("draft", { registry });

  assert.throws(() => zeta.wasm("search.wasm"), /zeta\/search\.wasm is not built: run "zig build search"\./);
  assert.throws(
    () => draft.wasm("infer.wasm"),
    /draft\/infer\.wasm is not built: run "zig build wasm -Dlocal-model=draft:<registered id>"\./,
  );
  assert.equal(zeta.wasmIfBuilt("search.wasm"), null);

  const built = path.join(root, "zig-out", "zeta", "infer.wasm");

  assert.equal(zeta.wasm("infer.wasm"), built);
  assert.equal(zeta.wasmIfBuilt("infer.wasm"), built);

  writeFile(path.join(root, "zig-out", "local", "draft", "infer.wasm"), "");
  assert.equal(draft.wasm("infer.wasm"), path.join(root, "zig-out", "local", "draft", "infer.wasm"));
});

test("every versioned model folder is registered, and every registered folder holds a model", () => {
  const subdirectories = (directory: string) =>
    fs
      .readdirSync(path.join(rootDirectory, directory), { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map(({ name }) => name);
  const versionedFolders = [
    ...subdirectories("models")
      .filter((name) => name !== "local")
      .map((name) => path.join(rootDirectory, "models", name)),
    ...subdirectories("vendor")
      .map((name) => path.join(rootDirectory, "vendor", name, "model"))
      .filter((directory) => fs.existsSync(directory)),
  ];
  const registeredFolders = listModels().map(({ directory }) => directory);

  assert.deepEqual(
    versionedFolders.filter((directory) => !registeredFolders.includes(directory)),
    [],
    "folders missing from models/registry.json",
  );

  for (const model of listModels()) {
    assert.ok(fs.existsSync(model.manifestPath), `${model.id}: ${model.manifestPath} is missing`);
    assert.ok(fs.existsSync(model.weightsPath), `${model.id}: ${model.weightsPath} is missing`);
  }
});
