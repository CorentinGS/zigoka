import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { loadEngines } from "../js/engines.ts";
import type { SearchEngine } from "../js/search-engine.ts";
import { findModel } from "../models/registry.ts";

type MokaRules = typeof import("./moka-rules.ts");

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const MODEL_ID = "moka-9";
const TIME_MS = 50;

/** One new search engine for each search: the visit clock reads the engine's tree. */
const loadSearchEngines = async () => {
  const model = findModel(MODEL_ID);

  return loadEngines(model.read, ["search.wasm"], { boardSize: model.boardSize });
};

const loadRules = async () =>
  (await import(path.join(rootDirectory, "tests/dist/moka-rules-9.js"))) as MokaRules;

const countVisits = (engine: SearchEngine) =>
  engine.rootVisits().reduce((sum, child) => sum + child.visits, 0);

/**
 * A simulated clock on which each visit costs `msPerVisit` and nothing else costs time, so the
 * search plays the same visits on every machine. With a resolution, it reads the time rounded
 * down to a tick, `phaseMs` into the first tick. It reads the visits of the engine's tree, so the
 * engine must be new.
 */
const createVisitClock = (engine: SearchEngine, msPerVisit: number, resolutionMs = 0, phaseMs = 0) => {
  const elapsed = () => countVisits(engine) * msPerVisit;
  const now = () =>
    resolutionMs > 0 ? Math.floor((elapsed() + phaseMs) / resolutionMs) * resolutionMs : elapsed();

  return { now, elapsed };
};

test("a finite budget ends the search at its deadline", async () => {
  const rules = await loadRules();
  const engine = await (await loadSearchEngines()).createSearchEngine();
  const msPerVisit = 0.25;
  const clock = createVisitClock(engine, msPerVisit);
  const outcome = engine.search(rules.createGameState(), { timeMs: TIME_MS, now: clock.now });

  assert.equal(outcome.poolFull, false);
  assert.ok(clock.elapsed() >= TIME_MS, `stopped at ${clock.elapsed()} ms`);
  assert.ok(clock.elapsed() <= TIME_MS + msPerVisit, `stopped at ${clock.elapsed()} ms`);
});

// A clock that cannot time a chunk reads 0 ms for it, at the start of the search as near its
// deadline: the search must not take that for free visits. Browsers coarsen the clock of a page
// that is not cross-origin isolated, such as the site.
test("a clock coarser than a visit still ends the search near its deadline", async () => {
  const rules = await loadRules();
  const engines = await loadSearchEngines();
  const resolutionMs = 1;

  for (const msPerVisit of [0.25, 0.4]) {
    for (const phaseMs of [0, 0.5, 0.9]) {
      const engine = await engines.createSearchEngine();
      const clock = createVisitClock(engine, msPerVisit, resolutionMs, phaseMs);
      const outcome = engine.search(rules.createGameState(), { timeMs: TIME_MS, now: clock.now });
      const label = `${msPerVisit} ms a visit, phase ${phaseMs} ms: stopped at ${clock.elapsed()} ms`;

      assert.equal(outcome.poolFull, false, label);
      assert.ok(clock.elapsed() >= TIME_MS - resolutionMs, label);
      assert.ok(clock.elapsed() <= TIME_MS * 1.1, label);
    }
  }
});
