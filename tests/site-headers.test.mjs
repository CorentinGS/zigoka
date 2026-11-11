import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const HEADERS_PATH = path.join(rootDirectory, "site", "_headers");
const PAGE_PATHS = ["/", "/index.html", "/about", "/about.html", "/19x19/", "/19x19/index.html"];
// One worker per edition: the 9×9 one at the root, the 19×19 one in its directory.
const WORKER_PATHS = ["/engine-worker.js", "/19x19/engine-worker.js"];

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

test("site/_headers: each engine worker gets one CSP, which allows wasm compilation", () => {
  for (const workerPath of WORKER_PATHS) {
    const policies = headersFor(rules, workerPath, "content-security-policy");

    assert.equal(policies.length, 1, `${workerPath} receives ${policies.length} policies`);
    assert.match(policies[0], /script-src [^;]*'wasm-unsafe-eval'/);
  }
});

test("site/_headers: every file is revalidated, with a single Cache-Control", () => {
  const enginePaths = ["/play.js", "/infer.wasm", "/search.wasm", "/go-model.bin", "/19x19/play.js", "/19x19/go-model.bin"];

  for (const urlPath of [...PAGE_PATHS, ...WORKER_PATHS, ...enginePaths]) {
    assert.deepEqual(headersFor(rules, urlPath, "cache-control"), ["no-cache"], urlPath);
  }
});

test("site/_headers: each page gets one CSP, which allows the worker and Cloudflare Web Analytics", () => {
  for (const pagePath of PAGE_PATHS) {
    const policies = headersFor(rules, pagePath, "content-security-policy");

    assert.equal(policies.length, 1, `${pagePath} receives ${policies.length} policies`);
    assert.match(policies[0], /worker-src 'self'/);
    // The custom domain injects the beacon, which then reports to cloudflareinsights.com.
    assert.match(policies[0], /script-src [^;]*https:\/\/static\.cloudflareinsights\.com/);
    assert.match(policies[0], /connect-src [^;]*https:\/\/cloudflareinsights\.com/);
  }
});
