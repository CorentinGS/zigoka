import assert from "node:assert/strict";
import { test } from "node:test";
import { assertHealthy, createHealthCounter } from "../oracle/health.ts";

const SUBNORMAL = 2 ** -130;
const LEGAL_MOVES = Uint8Array.of(1, 1, 1);

const observeHealthyPosition = (
  counter: ReturnType<typeof createHealthCounter>,
  overrides: { trunk?: number; head?: number; hidden?: number; value?: number; logits?: number[] },
) => {
  counter.observe("stem", Float32Array.of(overrides.trunk ?? 1, 0));
  counter.observe("policy.linear", Float32Array.of(overrides.head ?? 1, 2, 3));
  counter.observe("value.hidden", Float32Array.of(overrides.hidden ?? 0.5, 0));
  counter.observeOutputs(
    Float32Array.from(overrides.logits ?? [1, 2, 3]),
    overrides.value ?? 0.2,
    LEGAL_MOVES,
  );
};

const getHealth = (
  overrides: Parameters<typeof observeHealthyPosition>[1],
  positionCount = 10,
) => {
  const counter = createHealthCounter();

  observeHealthyPosition(counter, overrides);

  for (let position = 1; position < positionCount; position += 1) {
    observeHealthyPosition(counter, {});
  }

  return counter.toJSON();
};

const expectRejected = (health: ReturnType<typeof getHealth>, key: string) => {
  assert.throws(
    () => assertHealthy("synthetic-19", health),
    new RegExp(`^Error: model synthetic-19 failed the health check: health\\.${key} = .+, limit .+$`),
  );
};

test("a healthy set is accepted", () => {
  assert.doesNotThrow(() => assertHealthy("synthetic-19", getHealth({})));
});

test("a NaN output is refused", () => {
  expectRejected(getHealth({ head: Number.NaN }), "nonFiniteCount");
});

test("a subnormal in the heads is refused", () => {
  expectRejected(getHealth({ head: SUBNORMAL }), "headSubnormalCount");
});

test("a position where every value.hidden unit is zero is refused", () => {
  expectRejected(getHealth({ hidden: 0 }), "deadValueHiddenPositionCount");
});

test("10% of saturated values is refused", () => {
  expectRejected(getHealth({ value: -0.9995 }), "saturatedValueFraction");
});

test("a zero median legal logit range is refused", () => {
  const counter = createHealthCounter();

  for (let position = 0; position < 3; position += 1) {
    observeHealthyPosition(counter, { logits: [2, 2, 2] });
  }

  expectRejected(counter.toJSON(), "medianLegalLogitRange");
});

test("subnormals in the trunk are accepted and counted", () => {
  const health = getHealth({ trunk: SUBNORMAL });

  assert.equal(health.trunkSubnormalCount, 1);
  assert.doesNotThrow(() => assertHealthy("synthetic-19", health));
});
