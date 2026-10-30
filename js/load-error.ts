const DOWNLOAD_ERROR = "Failed to load infer.wasm.";
const SIMD_ERROR = "WebAssembly SIMD is not supported in this browser.";
const CSP_ERROR =
  "WebAssembly compilation was blocked; the Content Security Policy of worker.js " +
  "must allow 'wasm-unsafe-eval'.";

interface LoadFailure {
  fetchFailed: boolean;
  simdValid: boolean;
}

const describeLoadError = ({ fetchFailed, simdValid }: LoadFailure) => {
  if (fetchFailed) {
    return DOWNLOAD_ERROR;
  }

  return simdValid ? CSP_ERROR : SIMD_ERROR;
};

export { describeLoadError };
export type { LoadFailure };
