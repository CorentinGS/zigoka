import { getBufferSha256 } from "../vendor/moka/src/utils/get-buffer-sha256.ts";

const SHA256_BYTE_COUNT = 32;
const SHA256_PREFIX_LENGTH = 8;

const STATUS_OK = 0;
const STATUS_NON_FINITE_OUTPUT = 3;

interface InferExports {
  memory: WebAssembly.Memory;
  weights_ptr: () => number;
  weights_capacity: () => number;
  expected_sha256_ptr: () => number;
  load_weights: () => void;
  input_ptr: () => number;
  policy_ptr: () => number;
  value_ptr: () => number;
  max_batch: () => number;
  board_size: () => number;
  feature_count: () => number;
  policy_moves: () => number;
  input_stride: () => number;
  infer: (count: number) => number;
}

interface Engine {
  infer: (features: Float32Array) => GoModelInference;
}

const toHex = (bytes: Uint8Array) =>
  Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");

const createEngine = async (
  module: WebAssembly.Module,
  weights: ArrayBuffer,
): Promise<Engine> => {
  const instance = await WebAssembly.instantiate(module, {});
  const exports = instance.exports as unknown as InferExports;
  const memory = exports.memory.buffer;
  const weightsCapacity = exports.weights_capacity();

  if (weights.byteLength !== weightsCapacity) {
    throw new Error(`Model weights have the wrong length (expected ${weightsCapacity} bytes).`);
  }

  const expectedSha256 = toHex(
    new Uint8Array(memory, exports.expected_sha256_ptr(), SHA256_BYTE_COUNT),
  );

  if ((await getBufferSha256(weights)) !== expectedSha256) {
    throw new Error(
      "Model weights do not match this engine build " +
        `(expected sha256 ${expectedSha256.slice(0, SHA256_PREFIX_LENGTH)}…).`,
    );
  }

  new Uint8Array(memory, exports.weights_ptr(), weightsCapacity).set(new Uint8Array(weights));
  exports.load_weights();

  const featureCount = exports.feature_count();
  const input = new Float32Array(memory, exports.input_ptr(), featureCount);
  const policy = new Float32Array(memory, exports.policy_ptr(), exports.policy_moves());
  const value = new Float64Array(memory, exports.value_ptr(), 1);

  return {
    infer: (features: Float32Array) => {
      if (features.length !== featureCount) {
        throw new Error(`Input features must have ${featureCount} values.`);
      }

      input.set(features);
      const status = exports.infer(1);

      if (status === STATUS_NON_FINITE_OUTPUT) {
        throw new Error("Model produced a non-finite output.");
      }

      if (status !== STATUS_OK) {
        throw new Error("Go model inference failed.");
      }

      return { policyLogits: policy.slice(), value: value[0] };
    },
  };
};

export { createEngine };
export type { Engine };
