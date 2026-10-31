import assert from "node:assert/strict";
import { test } from "node:test";

import { fromVertex, parseScore, toVertex } from "../match/katago.ts";

const PASS = 81;

test("GTP vertices skip the letter I and count rows from the bottom", () => {
  assert.equal(toVertex(0), "A9");
  assert.equal(toVertex(8), "J9");
  assert.equal(toVertex(72), "A1");
  assert.equal(toVertex(80), "J1");
  assert.equal(toVertex(40), "E5");
  assert.equal(toVertex(PASS), "pass");
});

test("every point and the pass survive a round trip through GTP", () => {
  for (let move = 0; move <= PASS; move += 1) {
    assert.equal(fromVertex(toVertex(move)), move);
  }

  assert.equal(fromVertex("PASS"), PASS);
  assert.equal(fromVertex(" e5 "), 40);
});

test("vertices outside the 9x9 board are rejected", () => {
  for (const vertex of ["I5", "K1", "A10", "A0", "Z", ""]) {
    assert.throws(() => fromVertex(vertex), /unknown vertex/);
  }
});

test("GTP scores become Black's margin", () => {
  assert.equal(parseScore("B+3.5"), 3.5);
  assert.equal(parseScore("W+0.5"), -0.5);
  assert.equal(parseScore("0"), 0);
  assert.throws(() => parseScore("Jigo"), /unknown score/);
});
