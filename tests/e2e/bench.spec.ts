import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { listModels } from "../../models/registry.ts";

const benchDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "bench", "dist");

/** The Zig engines bench/run.mjs times. Moka's are vendored code, which the bench only measures. */
const ZIG_ENGINES = [
  { name: "Worker", query: "worker=1&stage=full" },
  { name: "main-thread", query: "worker=0&stage=full" },
  { name: "trunk", query: "worker=0&stage=trunk" },
];
const RUN_TIMEOUT_MS = 60_000;

// Every model, not only the published one: scripts/build-js.mjs copies the 9×9 dist/worker.js
// next to each model's infer.wasm, so the bench is the one host that runs worker.js on 19×19.
for (const { id } of listModels()) {
  for (const engine of ZIG_ENGINES) {
    test(`the bench runs the Zig ${engine.name} engine of ${id} and matches its reference`, async ({ page }) => {
      test.setTimeout(RUN_TIMEOUT_MS);

      expect(
        fs.existsSync(path.join(benchDirectory, id, "bench-inputs.json")),
        `bench/dist/${id}/bench-inputs.json is missing: run "pnpm oracle" and "pnpm build" first.`,
      ).toBe(true);

      await page.goto(`/bench/index.html?model=${id}&runtime=zig&mode=single&${engine.query}`);
      await page.waitForFunction(() => window.__benchResult !== undefined, undefined, {
        timeout: RUN_TIMEOUT_MS,
      });

      const result = await page.evaluate(() => window.__benchResult);

      expect(result !== undefined && "error" in result ? result.error : undefined).toBeUndefined();
    });
  }
}
