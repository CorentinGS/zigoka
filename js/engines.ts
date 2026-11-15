import { getBufferSha256 } from "../vendor/moka/src/utils/get-buffer-sha256.ts";
import {
  toEngine,
  toProfileEngine,
  type Engine,
  type InferExports,
  type ProfileEngine,
  type ProfileExports,
} from "./engine.ts";
import { toSearchEngine, type SearchEngine, type SearchExports } from "./search-engine.ts";

const SHA256_BYTE_COUNT = 32;
const SHA256_PREFIX_LENGTH = 8;
const SIMD_ERROR = "WebAssembly SIMD is not supported in this browser.";

type WasmFile = "infer.wasm" | "search.wasm" | "infer-profile.wasm";
type EngineFile = WasmFile | "go-model.bin";

/** Reads one file of a model. Its errors are the adapter's, which names the file. */
type ReadFile = (file: EngineFile) => Promise<ArrayBuffer>;

type LoadedEngines = {
  /** The weights given to `load`: every engine copies them, and nothing may change them. */
  weights: ArrayBuffer;
  createEngine: () => Promise<Engine>;
  /** A new instance, so a new search tree, on each call. */
  createSearchEngine: () => Promise<SearchEngine>;
  createProfileEngine: () => Promise<ProfileEngine>;
};

type CompiledEngines = {
  /** `boardSize` is the board the host plays; every engine created must play it too. */
  load: (weights: ArrayBuffer, options: { boardSize: number }) => Promise<LoadedEngines>;
};

const toHex = (bytes: Uint8Array) =>
  Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");

/** Names the script whose Content Security Policy applies: the worker's, or the page's. */
const describeBlockedCompilation = (location: { pathname: string }) =>
  "WebAssembly compilation was blocked; the Content Security Policy of " +
  `${location.pathname.split("/").pop() || "this page"} must allow 'wasm-unsafe-eval'.`;

const compileWasm = async (bytes: ArrayBuffer) => {
  try {
    return await WebAssembly.compile(bytes);
  } catch (error) {
    if (!WebAssembly.validate(bytes)) {
      throw new Error(SIMD_ERROR);
    }

    // Valid bytes that do not compile: in a browser, its Content Security Policy blocked them.
    const location = (globalThis as { location?: { pathname: string } }).location;

    throw location ? new Error(describeBlockedCompilation(location)) : error;
  }
};

/** Reads and compiles the wasm files of a model that the host needs. */
const compileEngines = async (read: ReadFile, files: WasmFile[]): Promise<CompiledEngines> => {
  const modules = new Map(
    await Promise.all(
      files.map(async (file) => [file, await compileWasm(await read(file))] as const),
    ),
  );

  return {
    load: async (weights, { boardSize }) => {
      const sha256 = await getBufferSha256(weights);

      /** A new instance of `file`, with the weights checked and loaded. */
      const instantiate = async <Exports extends InferExports>(file: WasmFile) => {
        const module = modules.get(file);

        if (!module) {
          throw new Error(`${file} was not compiled: pass it to compileEngines.`);
        }

        const instance = await WebAssembly.instantiate(module, {});
        const exports = instance.exports as unknown as Exports;
        const engineBoardSize = exports.board_size();

        if (engineBoardSize !== boardSize) {
          throw new Error(
            `${file} plays ${engineBoardSize}×${engineBoardSize} games; ` +
              `this host plays ${boardSize}×${boardSize}.`,
          );
        }

        const memory = exports.memory.buffer;
        const weightsCapacity = exports.weights_capacity();

        if (weights.byteLength !== weightsCapacity) {
          throw new Error(`Model weights have the wrong length (expected ${weightsCapacity} bytes).`);
        }

        const expectedSha256 = toHex(
          new Uint8Array(memory, exports.expected_sha256_ptr(), SHA256_BYTE_COUNT),
        );

        if (sha256 !== expectedSha256) {
          throw new Error(
            "Model weights do not match this engine build " +
              `(expected sha256 ${expectedSha256.slice(0, SHA256_PREFIX_LENGTH)}…).`,
          );
        }

        new Uint8Array(memory, exports.weights_ptr(), weightsCapacity).set(new Uint8Array(weights));
        exports.load_weights();

        return exports;
      };

      return {
        weights,
        createEngine: async () => toEngine(await instantiate("infer.wasm")),
        createSearchEngine: async () => toSearchEngine(await instantiate<SearchExports>("search.wasm")),
        createProfileEngine: async () =>
          toProfileEngine(await instantiate<ProfileExports>("infer-profile.wasm")),
      };
    },
  };
};

/** The browser adapter: fetches each file from its URL. A failure names the file. */
const fetchFiles =
  (urls: Partial<Record<EngineFile, string | URL>>): ReadFile =>
  async (file) => {
    const url = urls[file];

    if (url === undefined) {
      throw new Error(`No URL given for ${file}.`);
    }

    try {
      const response = await fetch(url);

      if (!response.ok) {
        throw new Error(`${url} answered ${response.status}.`);
      }

      return await response.arrayBuffer();
    } catch {
      throw new Error(`Failed to load ${file}.`);
    }
  };

export { compileEngines, fetchFiles };
export type {
  CompiledEngines,
  Engine,
  EngineFile,
  LoadedEngines,
  ProfileEngine,
  ReadFile,
  SearchEngine,
  WasmFile,
};
