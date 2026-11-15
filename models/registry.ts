import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import type { ReadFile } from "../js/engines.ts";

/** The wasm files `zig build` makes for each model, with the step that builds each one. */
const WASM_STEPS = {
  "infer.wasm": "wasm",
  "infer-safe.wasm": "wasm-safe",
  "infer-profile.wasm": "wasm-profile",
  "search.wasm": "search",
} as const;

const REGISTRY_FILE_NAME = "registry.json";

type WasmName = keyof typeof WASM_STEPS;

type RegistryOptions = {
  /** Another registry file. The repository root is the parent of its folder. */
  registry?: string;
};

type ModelFiles = {
  id: string;
  /** The model's source folder, which holds its manifest and weights. */
  directory: string;
  manifestPath: string;
  weightsPath: string;
  /** Read on first use, so that a model can be listed before its files exist. */
  manifest: GoModelManifest;
  /** The manifest's board size, which the registry never restates. */
  boardSize: number;
  /** The path of a built wasm file; throws, naming the build command, if it is not built. */
  wasm: (name: WasmName) => string;
  /** The path of a wasm file if it is built, for the callers to whom it is optional. */
  wasmIfBuilt: (name: WasmName) => string | null;
  /** Reads a built wasm file or the weights: the Node adapter of js/engines.ts. */
  read: ReadFile;
};

type Registration =
  | { kind: "trained" }
  | {
      kind: "synthetic";
      /** The trained model whose trunk this model keeps. */
      trunkFrom: string;
    };

type Capacity = {
  memoryPages: number;
  search: { nodes: number; edges: number; memoryPages: number };
};

type RegisteredModel = ModelFiles & Registration & Capacity & { local: false };

/** An unversioned export of a pipeline checkpoint, built with `zig build -Dlocal-model`. */
type LocalModel = ModelFiles & { local: true };

/** One entry of the registry file. `build.zig` validates the file; this module trusts it. */
type RegistryEntry = Registration & Capacity & { id: string; directory: string };

/** The first `models/registry.json` up the tree: next to this module, or above the bundle it is in. */
const findDefaultRegistry = () => {
  let directory = path.dirname(fileURLToPath(import.meta.url));

  for (;;) {
    const candidate = path.join(directory, "models", REGISTRY_FILE_NAME);

    if (fs.existsSync(candidate)) {
      return candidate;
    }

    const parent = path.dirname(directory);

    if (parent === directory) {
      throw new Error(`No models/${REGISTRY_FILE_NAME} above ${fileURLToPath(import.meta.url)}.`);
    }

    directory = parent;
  }
};

const readRegistry = ({ registry }: RegistryOptions = {}) => {
  const registryPath = path.resolve(registry ?? findDefaultRegistry());
  const { models } = JSON.parse(fs.readFileSync(registryPath, "utf8")) as { models: RegistryEntry[] };

  return { rootDirectory: path.dirname(path.dirname(registryPath)), entries: models };
};

/** The files of a model, with `fields` copied in. Never spread the result: that reads the manifest. */
const describeModel = <Fields extends object>(
  id: string,
  directory: string,
  wasmDirectory: string,
  buildCommand: string,
  fields: Fields,
): Fields & ModelFiles => {
  const manifestPath = path.join(directory, "go-model.json");
  let manifest: GoModelManifest | undefined;
  const readManifest = () =>
    (manifest ??= JSON.parse(fs.readFileSync(manifestPath, "utf8")) as GoModelManifest);
  const wasmIfBuilt = (name: WasmName) => {
    const wasmPath = path.join(wasmDirectory, name);

    return fs.existsSync(wasmPath) ? wasmPath : null;
  };
  const wasm = (name: WasmName) => {
    const wasmPath = wasmIfBuilt(name);

    if (wasmPath === null) {
      throw new Error(`${id}/${name} is not built: run "zig build ${WASM_STEPS[name]}${buildCommand}".`);
    }

    return wasmPath;
  };
  const weightsPath = path.join(directory, "go-model.bin");

  return {
    ...fields,
    id,
    directory,
    manifestPath,
    weightsPath,
    get manifest() {
      return readManifest();
    },
    get boardSize() {
      return readManifest().architecture.boardSize;
    },
    wasm,
    wasmIfBuilt,
    read: async (file) => {
      const bytes = await fs.promises.readFile(file === "go-model.bin" ? weightsPath : wasm(file));

      return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    },
  };
};

const toRegisteredModel = (rootDirectory: string, { directory, ...entry }: RegistryEntry): RegisteredModel =>
  describeModel(
    entry.id,
    path.join(rootDirectory, directory),
    path.join(rootDirectory, "zig-out", entry.id),
    "",
    { ...entry, local: false as const },
  );

/** The registered models, in the registry's order. Local models are never listed. */
const listModels = (options?: RegistryOptions): RegisteredModel[] => {
  const { rootDirectory, entries } = readRegistry(options);

  return entries.map((entry) => toRegisteredModel(rootDirectory, entry));
};

/** A registered model, or else a local model in `models/local/<id>/`. */
const findModel = (id: string, options?: RegistryOptions): RegisteredModel | LocalModel => {
  const { rootDirectory, entries } = readRegistry(options);
  const entry = entries.find((candidate) => candidate.id === id);

  if (entry) {
    return toRegisteredModel(rootDirectory, entry);
  }

  const localDirectory = path.join(rootDirectory, "models", "local", id);

  if (!fs.existsSync(path.join(localDirectory, "go-model.json"))) {
    throw new Error(
      `Unknown model "${id}": registered models are ${entries.map((candidate) => candidate.id).join(", ")}, ` +
        `and models/local/${id}/ does not exist.`,
    );
  }

  return describeModel(
    id,
    localDirectory,
    path.join(rootDirectory, "zig-out", "local", id),
    ` -Dlocal-model=${id}:<registered id>`,
    { local: true as const },
  );
};

export { findModel, listModels };
export type { LocalModel, RegisteredModel, RegistryOptions, WasmName };
