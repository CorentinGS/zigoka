import assert from "node:assert/strict";
import { test } from "node:test";
import { describeLoadError } from "../js/load-error.ts";

test("a failed download gives the download error, whatever validate says", () => {
  assert.equal(describeLoadError({ fetchFailed: true, simdValid: false }), "Failed to load infer.wasm.");
  assert.equal(describeLoadError({ fetchFailed: true, simdValid: true }), "Failed to load infer.wasm.");
});

test("bytes that do not validate mean the browser has no simd128", () => {
  assert.equal(
    describeLoadError({ fetchFailed: false, simdValid: false }),
    "WebAssembly SIMD is not supported in this browser.",
  );
});

test("valid bytes that do not compile mean the Content Security Policy blocked them", () => {
  assert.equal(
    describeLoadError({ fetchFailed: false, simdValid: true }),
    "WebAssembly compilation was blocked; the Content Security Policy of worker.js must allow " +
      "'wasm-unsafe-eval'.",
  );
});
