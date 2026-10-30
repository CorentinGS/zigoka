import { expect, test, type Page } from "@playwright/test";
import type { WorkerReply } from "./page.ts";

const FEATURE_COUNT = 972;
const MOVE_COUNT = 82;
const NOMINAL_POSITION_COUNT = 10;
const SHORT_INPUT_LENGTH = 810;

const WEIGHTS_MISMATCH_ERROR =
  "Model weights do not match this engine build (expected sha256 35c1e51c…).";
const WEIGHTS_LENGTH_ERROR = "Model weights have the wrong length (expected 113648 bytes).";
const NOT_INITIALIZED_ERROR = "Go model worker is not initialized.";
const INPUT_LENGTH_ERROR = "Input features must have 972 values.";
const NON_FINITE_ERROR = "Model produced a non-finite output.";
const CSP_ERROR =
  "WebAssembly compilation was blocked; the Content Security Policy of worker.js must allow " +
  "'wasm-unsafe-eval'.";

type WeightsChange = "none" | "flip-byte" | "truncate";

const openPage = async (page: Page, query = "") => {
  await page.goto(`/e2e/index.html${query}`);
  await page.waitForFunction(() => window.e2e !== undefined);
};

const initializeThroughClient = (page: Page) =>
  page.evaluate(async () => {
    const e2e = window.e2e!;

    try {
      await e2e.client.initialize({ manifestUrl: e2e.manifestUrl, weightsUrl: e2e.weightsUrl });

      return "ready";
    } catch (error) {
      return error instanceof Error ? error.message : String(error);
    }
  });

const initializeDirectly = (page: Page, change: WeightsChange): Promise<WorkerReply> =>
  page.evaluate(async (weightsChange) => {
    const e2e = window.e2e!;
    const model = await e2e.fetchModel();
    let weights = model.weights;

    if (weightsChange === "flip-byte") {
      new Uint8Array(weights)[1000] ^= 1;
    } else if (weightsChange === "truncate") {
      weights = weights.slice(0, weights.byteLength - 1);
    }

    return e2e.post({ type: "initialize", manifest: model.manifest, weightsBuffer: weights }, [weights]);
  }, change);

const inferDirectly = (page: Page, length: number, hasNaN: boolean): Promise<WorkerReply> =>
  page.evaluate(
    async ({ featureLength, withNaN }) => {
      const features = new Float32Array(featureLength);

      if (withNaN) {
        features[0] = Number.NaN;
      }

      return window.e2e!.post({ type: "infer", features }, [features.buffer]);
    },
    { featureLength: length, withNaN: hasNaN },
  );

const expectError = (reply: WorkerReply, message: string) => {
  expect(reply.type).toBe("error");
  expect(reply.type === "error" ? reply.error : null).toBe(message);
};

test("the nominal case matches Moka within the G3 tolerance on 10 positions", async ({ page }) => {
  await openPage(page);
  expect(await initializeThroughClient(page)).toBe("ready");

  const results = await page.evaluate(
    async ({ positionCount, featureCount, moveCount }) => {
      const e2e = window.e2e!;
      const positions = await e2e.fetchFloat32("/bench/dist/positions.f32");
      const expected = await e2e.fetchFloat32("/bench/dist/expected.f32");
      const tolerance = (await (await fetch("/bench/dist/tolerance.json")).json()) as {
        logits: number;
        value: number;
      };
      const benchPositionCount = positions.length / featureCount;
      const outcomes = [];

      for (let position = 0; position < positionCount; position += 1) {
        const features = positions.slice(position * featureCount, (position + 1) * featureCount);
        const result = await e2e.client.infer(features);
        let maxLogitDifference = 0;

        for (let move = 0; move < moveCount; move += 1) {
          maxLogitDifference = Math.max(
            maxLogitDifference,
            Math.abs(result.policyLogits[move] - expected[position * moveCount + move]),
          );
        }

        outcomes.push({
          isFloat32Array: result.policyLogits instanceof Float32Array,
          logitCount: result.policyLogits.length,
          isValueFinite: Number.isFinite(result.value),
          isWithinLogitTolerance: maxLogitDifference <= tolerance.logits,
          isWithinValueTolerance:
            Math.abs(result.value - expected[benchPositionCount * moveCount + position]) <=
            tolerance.value,
        });
      }

      return outcomes;
    },
    { positionCount: NOMINAL_POSITION_COUNT, featureCount: FEATURE_COUNT, moveCount: MOVE_COUNT },
  );

  expect(results).toHaveLength(NOMINAL_POSITION_COUNT);

  for (const result of results) {
    expect(result).toEqual({
      isFloat32Array: true,
      logitCount: MOVE_COUNT,
      isValueFinite: true,
      isWithinLogitTolerance: true,
      isWithinValueTolerance: true,
    });
  }
});

test("weights modified by one byte are refused", async ({ page }) => {
  await openPage(page);
  expectError(await initializeDirectly(page, "flip-byte"), WEIGHTS_MISMATCH_ERROR);
});

test("truncated weights are refused with the length error", async ({ page }) => {
  await openPage(page);
  expectError(await initializeDirectly(page, "truncate"), WEIGHTS_LENGTH_ERROR);
});

test("a failed initialize after a successful one leaves no active weights (CUJ 3)", async ({
  page,
}) => {
  await openPage(page);
  expect((await initializeDirectly(page, "none")).type).toBe("ready");
  expectError(await initializeDirectly(page, "truncate"), WEIGHTS_LENGTH_ERROR);
  expectError(await inferDirectly(page, FEATURE_COUNT, false), NOT_INITIALIZED_ERROR);
});

test("an input of 810 values is refused", async ({ page }) => {
  await openPage(page);
  expect((await initializeDirectly(page, "none")).type).toBe("ready");
  expectError(await inferDirectly(page, SHORT_INPUT_LENGTH, false), INPUT_LENGTH_ERROR);
});

test("an input that holds a NaN gives the non-finite output error", async ({ page }) => {
  await openPage(page);
  expect((await initializeDirectly(page, "none")).type).toBe("ready");
  expectError(await inferDirectly(page, FEATURE_COUNT, true), NON_FINITE_ERROR);
});

test("a strict CSP on worker.js blocks the compilation", async ({ page }) => {
  await openPage(page, "?worker-csp=strict");
  expect(await initializeThroughClient(page)).toBe(CSP_ERROR);
});

test("records whether a strict CSP on the page alone blocks the compilation", async ({ page }) => {
  await openPage(page, "?csp=strict");
  const pageCompilation = await page.evaluate(async () => {
    try {
      await WebAssembly.compile(await (await fetch("/dist/infer.wasm")).arrayBuffer());

      return "compiled";
    } catch {
      return "blocked";
    }
  });

  expect(pageCompilation, "the strict CSP must apply to the page itself").toBe("blocked");

  const outcome = await initializeThroughClient(page);

  test.info().annotations.push({
    type: "page-csp",
    description: outcome === "ready" ? "not blocked" : `blocked: ${outcome}`,
  });
  expect(["ready", CSP_ERROR]).toContain(outcome);
});
