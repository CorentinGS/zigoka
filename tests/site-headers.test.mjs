import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const HEADERS_PATH = path.join(rootDirectory, "site", "_headers");
const PAGE_PATHS = ["/", "/index.html", "/about", "/about.html"];
const WORKER_PATH = "/engine-worker.js";

/** Parses a Cloudflare Pages `_headers` file into rules of a path and its headers. */
const parseRules = (text) => {
  const rules = [];

  for (const line of text.split("\n")) {
    if (line.trim() === "") {
      continue;
    }

    if (!line.startsWith(" ")) {
      rules.push({ path: line.trim(), headers: [] });
      continue;
    }

    const separator = line.indexOf(":");

    rules.at(-1).headers.push([line.slice(0, separator).trim().toLowerCase(), line.slice(separator + 1).trim()]);
  }

  return rules;
};

const matches = (rulePath, urlPath) => {
  const pattern = rulePath.replace(/[.+?^${}()|[\]\\]/g, "\\$&").replaceAll("*", ".*");

  return new RegExp(`^${pattern}$`).test(urlPath);
};

/** Like Pages, every rule that matches `urlPath` adds its headers. */
const headersFor = (rules, urlPath, name) =>
  rules
    .filter((rule) => matches(rule.path, urlPath))
    .flatMap((rule) => rule.headers.filter(([header]) => header === name).map(([, value]) => value));

const rules = parseRules(fs.readFileSync(HEADERS_PATH, "utf8"));

test("site/_headers: every rule sets at least one header", () => {
  // Pages applies a header block to the path line just above it only.
  for (const rule of rules) {
    assert.ok(rule.headers.length > 0, `${rule.path} has no header`);
  }
});

test("site/_headers: the engine worker gets one CSP, which allows wasm compilation", () => {
  const policies = headersFor(rules, WORKER_PATH, "content-security-policy");

  assert.equal(policies.length, 1, `${WORKER_PATH} receives ${policies.length} policies`);
  assert.match(policies[0], /script-src [^;]*'wasm-unsafe-eval'/);
});

test("site/_headers: each page gets one CSP, which allows the worker", () => {
  for (const pagePath of PAGE_PATHS) {
    const policies = headersFor(rules, pagePath, "content-security-policy");

    assert.equal(policies.length, 1, `${pagePath} receives ${policies.length} policies`);
    assert.match(policies[0], /worker-src 'self'/);
  }
});
