const SMALLEST_NORMAL_FLOAT32 = 2 ** -126;
const SATURATED_VALUE = 0.999;
const SATURATED_VALUE_FRACTION_LIMIT = 0.1;
const HEAD_TENSOR_PATTERN = /^(policy|value)\./;
const VALUE_HIDDEN_TENSOR = "value.hidden";

type Health = {
  activationCount: number;
  nonFiniteCount: number;
  trunkSubnormalCount: number;
  headSubnormalCount: number;
  deadValueHiddenPositionCount: number;
  saturatedValueFraction: number;
  medianLegalLogitRange: number;
};

type HealthCounter = {
  observe: (name: string, values: Float32Array) => void;
  observeOutputs: (logits: Float32Array, value: number, legalMoves: Uint8Array) => void;
  toJSON: () => Health;
};

const isSubnormal = (value: number) => value !== 0 && Math.abs(value) < SMALLEST_NORMAL_FLOAT32;

const getMedian = (values: number[]): number => {
  if (values.length === 0) {
    return 0;
  }

  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);

  return sorted.length % 2 === 0
    ? ((sorted[middle - 1] as number) + (sorted[middle] as number)) / 2
    : (sorted[middle] as number);
};

const createHealthCounter = (): HealthCounter => {
  let activationCount = 0;
  let nonFiniteCount = 0;
  let trunkSubnormalCount = 0;
  let headSubnormalCount = 0;
  let deadValueHiddenPositionCount = 0;
  let positionCount = 0;
  let saturatedValueCount = 0;
  const legalLogitRanges: number[] = [];

  return {
    observe: (name, values) => {
      const isHead = HEAD_TENSOR_PATTERN.test(name);
      let hasActiveUnit = false;

      activationCount += values.length;

      for (const value of values) {
        if (!Number.isFinite(value)) {
          nonFiniteCount += 1;
        } else if (isSubnormal(value)) {
          if (isHead) {
            headSubnormalCount += 1;
          } else {
            trunkSubnormalCount += 1;
          }
        }

        if (value > 0) {
          hasActiveUnit = true;
        }
      }

      if (name === VALUE_HIDDEN_TENSOR && !hasActiveUnit) {
        deadValueHiddenPositionCount += 1;
      }
    },
    observeOutputs: (logits, value, legalMoves) => {
      let minimum = Infinity;
      let maximum = -Infinity;

      positionCount += 1;

      if (Math.abs(value) >= SATURATED_VALUE) {
        saturatedValueCount += 1;
      }

      for (let move = 0; move < logits.length; move += 1) {
        if (legalMoves[move] === 1) {
          minimum = Math.min(minimum, logits[move] as number);
          maximum = Math.max(maximum, logits[move] as number);
        }
      }

      legalLogitRanges.push(maximum >= minimum ? maximum - minimum : 0);
    },
    toJSON: () => ({
      activationCount,
      nonFiniteCount,
      trunkSubnormalCount,
      headSubnormalCount,
      deadValueHiddenPositionCount,
      saturatedValueFraction: positionCount === 0 ? 0 : saturatedValueCount / positionCount,
      medianLegalLogitRange: getMedian(legalLogitRanges),
    }),
  };
};

const assertHealthy = (modelId: string, health: Health): void => {
  const failures: [keyof Health, boolean, string][] = [
    ["nonFiniteCount", health.nonFiniteCount > 0, "0"],
    ["headSubnormalCount", health.headSubnormalCount > 0, "0"],
    ["deadValueHiddenPositionCount", health.deadValueHiddenPositionCount > 0, "0"],
    [
      "saturatedValueFraction",
      health.saturatedValueFraction >= SATURATED_VALUE_FRACTION_LIMIT,
      `< ${SATURATED_VALUE_FRACTION_LIMIT}`,
    ],
    ["medianLegalLogitRange", !(health.medianLegalLogitRange > 0), "> 0"],
  ];

  for (const [key, isFailed, limit] of failures) {
    if (isFailed) {
      throw new Error(
        `model ${modelId} failed the health check: health.${key} = ${health[key]}, limit ${limit}`,
      );
    }
  }
};

export { assertHealthy, createHealthCounter };
export type { Health, HealthCounter };
