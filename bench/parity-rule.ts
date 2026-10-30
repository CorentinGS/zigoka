const VALUE_ULP_LIMIT = 2n;

type ParityMismatch = {
  position: number;
  index: number | null;
  kind: "logit" | "value";
};

const getUlpDistance = (left: number, right: number): bigint => {
  const bits = new BigInt64Array(new Float64Array([left, right]).buffer);
  const toOrdered = (value: bigint) => (value < 0n ? -(value & 0x7fff_ffff_ffff_ffffn) : value);
  const distance = toOrdered(bits[0] as bigint) - toOrdered(bits[1] as bigint);

  return distance < 0n ? -distance : distance;
};

/**
 * Returns the first output that breaks the common parity rule: logits identical bit for bit,
 * value within 2 ulp of f64. Returns null when every output conforms.
 */
const findParityMismatch = (
  logits: Float32Array,
  values: Float64Array,
  expectedLogits: Float32Array,
  expectedValues: Float64Array,
): ParityMismatch | null => {
  const positionCount = expectedValues.length;
  const moveCount = expectedLogits.length / positionCount;

  if (values.length !== positionCount || logits.length !== expectedLogits.length) {
    throw new Error(
      `Expected ${positionCount} values and ${expectedLogits.length} logits, ` +
        `got ${values.length} and ${logits.length}.`,
    );
  }

  const actualBits = new Uint32Array(logits.buffer, logits.byteOffset, logits.length);
  const expectedBits = new Uint32Array(
    expectedLogits.buffer,
    expectedLogits.byteOffset,
    expectedLogits.length,
  );

  for (let position = 0; position < positionCount; position += 1) {
    for (let move = 0; move < moveCount; move += 1) {
      const index = position * moveCount + move;

      if (actualBits[index] !== expectedBits[index]) {
        return { position, index: move, kind: "logit" };
      }
    }

    if (getUlpDistance(values[position] as number, expectedValues[position] as number) > VALUE_ULP_LIMIT) {
      return { position, index: null, kind: "value" };
    }
  }

  return null;
};

export { VALUE_ULP_LIMIT, findParityMismatch, getUlpDistance };
export type { ParityMismatch };
