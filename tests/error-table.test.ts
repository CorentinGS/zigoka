import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import * as workerErrors from "./e2e/worker-errors.ts";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DESIGN_PATH = "docs/designs/zig-wasm-moka.md";
const TABLE_HEADER = "| Cause | Message |";

/** The messages of the Worker's error table, each between « and ». */
const readTableMessages = () => {
  const lines = fs.readFileSync(path.join(rootDirectory, DESIGN_PATH), "utf8").split("\n");
  const header = lines.indexOf(TABLE_HEADER);

  assert.notEqual(header, -1, `${DESIGN_PATH} has no line "${TABLE_HEADER}".`);

  const rows = lines.slice(header + 2);
  const end = rows.findIndex((line) => !line.startsWith("|"));

  return (end === -1 ? rows : rows.slice(0, end)).map((row) => /«\s*(.+?)\s*»/.exec(row)?.[1]);
};

test("each Worker error the e2e tests expect is a row of the design's error table", () => {
  const messages = readTableMessages();
  const missing = Object.entries(workerErrors)
    .filter(([, message]) => !messages.includes(message))
    .map(([name, message]) => `${name}: ${message}`);

  assert.ok(Object.keys(workerErrors).length > 0, "tests/e2e/worker-errors.ts exports no error.");
  assert.deepEqual(missing, [], `Missing from the error table of ${DESIGN_PATH}.`);
});
