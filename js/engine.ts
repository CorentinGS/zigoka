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

/** infer-profile.wasm: the exports of infer.wasm, plus the trunk alone. */
interface ProfileExports extends InferExports {
  forward_trunk: (count: number) => number;
}

interface Engine {
  infer: (features: Float32Array) => GoModelInference;
}

/** The bench's engine: `forwardTrunk` runs the trunk without the heads. */
interface ProfileEngine extends Engine {
  forwardTrunk: (features: Float32Array) => void;
}

const checkStatus = (status: number) => {
  if (status === STATUS_NON_FINITE_OUTPUT) {
    throw new Error("Model produced a non-finite output.");
  }

  if (status !== STATUS_OK) {
    throw new Error("Go model inference failed.");
  }
};

/** Writes one position's features to the input, after checking their count. */
const createInputWriter = (exports: InferExports) => {
  const featureCount = exports.feature_count();
  const input = new Float32Array(exports.memory.buffer, exports.input_ptr(), featureCount);

  return (features: Float32Array) => {
    if (features.length !== featureCount) {
      throw new Error(`Input features must have ${featureCount} values.`);
    }

    input.set(features);
  };
};

/** An engine on an instance whose weights js/engines.ts has checked and loaded. */
const toEngine = (exports: InferExports): Engine => {
  const writeInput = createInputWriter(exports);
  const policy = new Float32Array(exports.memory.buffer, exports.policy_ptr(), exports.policy_moves());
  const value = new Float64Array(exports.memory.buffer, exports.value_ptr(), 1);

  return {
    infer: (features: Float32Array) => {
      writeInput(features);
      checkStatus(exports.infer(1));

      return { policyLogits: policy.slice(), value: value[0] as number };
    },
  };
};

const toProfileEngine = (exports: ProfileExports): ProfileEngine => {
  const writeInput = createInputWriter(exports);

  return {
    ...toEngine(exports),
    forwardTrunk: (features: Float32Array) => {
      writeInput(features);
      checkStatus(exports.forward_trunk(1));
    },
  };
};

export { toEngine, toProfileEngine };
export type { Engine, InferExports, ProfileEngine, ProfileExports };
