/**
 * The messages the published Worker gives moka-9, as the e2e tests expect them. Each one is a row
 * of the error table in docs/designs/zig-wasm-moka.md, which tests/error-table.test.ts checks.
 */
const WEIGHTS_MISMATCH_ERROR =
  "Model weights do not match this engine build (expected sha256 35c1e51c…).";
const WEIGHTS_LENGTH_ERROR = "Model weights have the wrong length (expected 113648 bytes).";
const BOARD_SIZE_ERROR = "infer.wasm plays 9×9 games; this host plays 19×19.";
const NOT_INITIALIZED_ERROR = "Go model worker is not initialized.";
const INPUT_LENGTH_ERROR = "Input features must have 972 values.";
const NON_FINITE_ERROR = "Model produced a non-finite output.";
const CSP_ERROR =
  "WebAssembly compilation was blocked; the Content Security Policy of worker.js must allow " +
  "'wasm-unsafe-eval'.";

export {
  BOARD_SIZE_ERROR,
  CSP_ERROR,
  INPUT_LENGTH_ERROR,
  NON_FINITE_ERROR,
  NOT_INITIALIZED_ERROR,
  WEIGHTS_LENGTH_ERROR,
  WEIGHTS_MISMATCH_ERROR,
};
