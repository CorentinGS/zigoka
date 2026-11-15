import assert from "node:assert/strict";
import http from "node:http";
import type { AddressInfo } from "node:net";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { compileEngines, fetchFiles, loadEngines, type ReadFile } from "../js/engines.ts";
import { findModel } from "../models/registry.ts";

type MokaRules = typeof import("./moka-rules.ts");

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const model = findModel("moka-9");

const loadRules = async () =>
  (await import(path.join(rootDirectory, "tests/dist/moka-rules-9.js"))) as MokaRules;

const loadWith = async (weights: ArrayBuffer, boardSize = model.boardSize) =>
  (await compileEngines(model.read, ["infer.wasm", "search.wasm"])).load(weights, { boardSize });

test("load refuses a host that plays another board, and names both sizes", async () => {
  await assert.rejects(loadWith(await model.read("go-model.bin"), 19), {
    message: /^(infer|search)\.wasm plays 9×9 games; this host plays 19×19\.$/,
  });
});

test("load refuses weights of the wrong length", async () => {
  const weights = await model.read("go-model.bin");

  await assert.rejects(loadWith(weights.slice(0, weights.byteLength - 1)), {
    message: `Model weights have the wrong length (expected ${weights.byteLength} bytes).`,
  });
});

test("load refuses weights of another build", async () => {
  const weights = await model.read("go-model.bin");

  new Uint8Array(weights)[0] ^= 1;

  await assert.rejects(loadWith(weights), {
    message: /^Model weights do not match this engine build \(expected sha256 [0-9a-f]{8}…\)\.$/,
  });
});

test("only the compiled files have a create method", async () => {
  const loaded = await loadEngines(model.read, ["infer.wasm"], { boardSize: model.boardSize });

  // @ts-expect-error search.wasm was not compiled.
  assert.equal(loaded.createSearchEngine, undefined);
  assert.equal(typeof loaded.createEngine, "function");
});

test("each search engine has a tree of its own, and the weights stay the host's", async () => {
  const { createGameState } = await loadRules();
  const weights = await model.read("go-model.bin");
  const copy = weights.slice(0);
  const loaded = await loadWith(weights);
  const first = await loaded.createSearchEngine();
  const second = await loaded.createSearchEngine();
  const state = createGameState();
  const limits = { timeMs: Infinity, maxVisits: 20 };

  const searched = first.search(state, limits);

  assert.equal(first.search(state, { ...limits, maxVisits: 40, reuse: true }).reused, searched.visits);
  assert.equal(second.search(state, { ...limits, reuse: true }).reused, 0);
  assert.equal(loaded.weights, weights);
  assert.deepEqual(new Uint8Array(weights), new Uint8Array(copy));
});

test("bytes that do not validate are blamed on SIMD", async () => {
  const read: ReadFile = async (file) => (file === "infer.wasm" ? new ArrayBuffer(8) : model.read(file));

  await assert.rejects(compileEngines(read, ["infer.wasm"]), {
    message: "WebAssembly SIMD is not supported in this browser.",
  });
});

test("fetchFiles names the file it could not load", async () => {
  const read = fetchFiles({ "infer.wasm": "http://127.0.0.1:1/infer.wasm" });

  await assert.rejects(compileEngines(read, ["infer.wasm"]), { message: "Failed to load infer.wasm." });
});

test("fetchFiles takes an error status for a failed load, not for wasm bytes", async (t) => {
  const server = http.createServer((_request, response) => {
    response.statusCode = 404;
    response.end("not found");
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(() => server.close());

  const { port } = server.address() as AddressInfo;
  const read = fetchFiles({ "infer.wasm": `http://127.0.0.1:${port}/infer.wasm` });

  await assert.rejects(compileEngines(read, ["infer.wasm"]), { message: "Failed to load infer.wasm." });
});
