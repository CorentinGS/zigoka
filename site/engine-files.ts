import type { EngineFile } from "../js/engines.ts";

/**
 * The files site/engine-worker.ts fetches from its own folder. scripts/build-site.mjs hashes them,
 * with the worker, into the worker's `?v=`.
 */
const WORKER_FILES: readonly EngineFile[] = ["infer.wasm", "search.wasm", "go-model.bin"];

export { WORKER_FILES };
