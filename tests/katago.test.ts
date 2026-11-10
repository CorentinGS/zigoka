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

test("on 19x19, vertices reach column T and row 19, and the pass is move 361", () => {
  assert.equal(toVertex(0, 19), "A19");
  assert.equal(toVertex(18, 19), "T19");
  assert.equal(toVertex(342, 19), "A1");
  assert.equal(toVertex(360, 19), "T1");
  assert.equal(toVertex(180, 19), "K10");
  assert.equal(toVertex(361, 19), "pass");

  for (let move = 0; move <= 361; move += 1) {
    assert.equal(fromVertex(toVertex(move, 19), 19), move);
  }

  for (const vertex of ["I5", "U1", "A20", "A0"]) {
    assert.throws(() => fromVertex(vertex, 19), /unknown vertex/);
  }
});
