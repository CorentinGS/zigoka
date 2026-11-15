import { getBufferSha256 } from "../vendor/moka/src/utils/get-buffer-sha256.ts";
import {
  toEngine,
  toProfileEngine,
  type Engine,
  type InferExports,
  type ProfileEngine,
  type ProfileExports,
} from "./engine.ts";
import { fetchOk } from "./fetch-ok.ts";
import { toSearchEngine, type SearchEngine, type SearchExports } from "./search-engine.ts";

const SHA256_BYTE_COUNT = 32;
const SHA256_PREFIX_LENGTH = 8;
const SIMD_ERROR = "WebAssembly SIMD is not supported in this browser.";

/** Each wasm file a host can compile: the method that creates its engine, and that engine. */
const ENGINE_KINDS = {
  "infer.wasm": {
    create: "createEngine",
    toEngine: (exports: InferExports): Engine => toEngine(exports),
  },
  "search.wasm": {
    create: "createSearchEngine",
    toEngine: (exports: InferExports): SearchEngine => toSearchEngine(exports as SearchExports),
  },
  "infer-profile.wasm": {
    create: "createProfileEngine",
    toEngine: (exports: InferExports): ProfileEngine => toProfileEngine(exports as ProfileExports),
  },
} as const;

type WasmFile = keyof typeof ENGINE_KINDS;
type EngineFile = WasmFile | "go-model.bin";

/** Reads one file of a model. Its errors are the adapter's, which names the file. */
type ReadFile = (file: EngineFile) => Promise<ArrayBuffer>;

type LoadOptions = {
  /** The board the host plays: every engine must play it too. */
  boardSize: number;
};

/** The engines of the files given to compileEngines: one create method for each file. */
type LoadedEngines<Files extends WasmFile = WasmFile> = {
  /** The weights given to `load`: every engine copies them, and nothing may change them. */
  readonly weights: ArrayBuffer;
} & {
  /** A new instance, so a new search tree, on each call. */
  readonly [File in Files as (typeof ENGINE_KINDS)[File]["create"]]: () => Promise<
    ReturnType<(typeof ENGINE_KINDS)[File]["toEngine"]>
  >;
};

type CompiledEngines<Files extends WasmFile> = {
  /** Instantiates and checks each compiled file: a wrong board, length or sha256 fails here. */
  load: (weights: ArrayBuffer, options: LoadOptions) => Promise<LoadedEngines<Files>>;
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

/** A new instance of `file`, after checking that it plays the host's board and takes these weights. */
const instantiate = async (
  file: WasmFile,
  module: WebAssembly.Module,
  weights: ArrayBuffer,
  sha256: string,
  boardSize: number,
) => {
  const instance = await WebAssembly.instantiate(module, {});
  const exports = instance.exports as unknown as InferExports;
  const engineBoardSize = exports.board_size();

  // Before the length: weights for another board also have another length.
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

  const expectedSha256 = toHex(new Uint8Array(memory, exports.expected_sha256_ptr(), SHA256_BYTE_COUNT));

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

/** Reads and compiles the wasm files of a model that the host needs. */
const compileEngines = async <Files extends WasmFile>(
  read: ReadFile,
  files: readonly Files[],
): Promise<CompiledEngines<Files>> => {
  const modules = await Promise.all(
    files.map(async (file) => [file, await compileWasm(await read(file))] as const),
  );

  return {
    load: async (weights, { boardSize }) => {
      const sha256 = await getBufferSha256(weights);
      const creators = await Promise.all(
        modules.map(async ([file, module]) => {
          const create = () => instantiate(file, module, weights, sha256, boardSize);
          // The instance that checked the file goes to the first engine; the others are new.
          let checked: InferExports | null = await create();

          const createEngine = async () => {
            const exports = checked ?? (await create());

            checked = null;

            return ENGINE_KINDS[file].toEngine(exports);
          };

          return [ENGINE_KINDS[file].create, createEngine] as const;
        }),
      );

      return { weights, ...Object.fromEntries(creators) } as LoadedEngines<Files>;
    },
  };
};

/** compileEngines, then `load` with the weights that `read` gives: for a host that loads once. */
const loadEngines = async <Files extends WasmFile>(
  read: ReadFile,
  files: readonly Files[],
  options: LoadOptions,
) => {
  const [compiled, weights] = await Promise.all([compileEngines(read, files), read("go-model.bin")]);

  return compiled.load(weights, options);
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
      return await (await fetchOk(url)).arrayBuffer();
    } catch {
      throw new Error(`Failed to load ${file}.`);
    }
  };

export { compileEngines, fetchFiles, loadEngines };
export type {
  CompiledEngines,
  Engine,
  EngineFile,
  LoadedEngines,
  LoadOptions,
  ProfileEngine,
  ReadFile,
  SearchEngine,
  WasmFile,
};
