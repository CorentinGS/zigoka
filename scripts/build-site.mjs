import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resolve = (relativePath) => path.join(rootDirectory, relativePath);
const publicDirectory = resolve("site/public");

// site/_headers is Cloudflare Pages' header file. Pages joins the headers of every
// rule that matches a path, and a browser enforces each Content-Security-Policy it
// receives: a CSP on /* would also reach engine-worker.js and block its
// wasm-unsafe-eval. The page CSP is therefore set path by path.
// A browser caches headers with a file: every file is revalidated (`no-cache`, a
// 304 when unchanged). A Cloudflare zone in front of the custom domain can still
// override that and cache for an hour, so every reference also carries a content
// hash (`?v=`): a new deploy changes the URLs and never meets a stale copy.
const PUBLISHED_FILES = [
  ["site/index.html", "index.html"],
  ["site/about.html", "about.html"],
  ["site/site.css", "site.css"],
  ["site/_headers", "_headers"],
  ["site/dist/play.js", "play.js"],
  ["site/dist/engine-worker.js", "engine-worker.js"],
  ["demo/dist/search.wasm", "search.wasm"],
  ["dist/infer.wasm", "infer.wasm"],
  ["dist/go-model.json", "go-model.json"],
  ["dist/go-model.bin", "go-model.bin"],
  ["vendor/moka/LICENSE", "moka-license.txt"],
  ["vendor/moka/THIRD_PARTY_NOTICES.md", "moka-notices.txt"],
];

// The worker gives its own `?v=` to the files it fetches (site/engine-worker.ts).
const ENGINE_FILES = ["engine-worker.js", "infer.wasm", "search.wasm", "go-model.json", "go-model.bin"];
const PAGES = ["index.html", "about.html"];

const publicPath = (fileName) => path.join(publicDirectory, fileName);

const versionOf = (fileNames) => {
  const hash = crypto.createHash("sha256");

  for (const fileName of fileNames) {
    hash.update(fs.readFileSync(publicPath(fileName)));
  }

  return hash.digest("hex").slice(0, 12);
};

/** Replaces `reference` in a published file, which must contain it exactly once. */
const versionReference = (fileName, reference, version) => {
  const text = fs.readFileSync(publicPath(fileName), "utf8");
  const count = text.split(reference).length - 1;

  if (count !== 1) {
    throw new Error(`${fileName} references ${reference} ${count} times, expected once.`);
  }

  fs.writeFileSync(publicPath(fileName), text.replace(reference, `${reference.slice(0, -1)}?v=${version}"`));
};

const main = () => {
  const missingFiles = PUBLISHED_FILES.filter(([source]) => !fs.existsSync(resolve(source)));

  if (missingFiles.length > 0) {
    const listed = missingFiles.map(([source]) => source).join(", ");

    throw new Error(
      `Missing site source files: ${listed}. Run \`pnpm build\` first to produce the ` +
        "engine and model artifacts, then rebuild the site.",
    );
  }

  fs.rmSync(publicDirectory, { recursive: true, force: true });
  fs.mkdirSync(publicDirectory, { recursive: true });

  for (const [source, destination] of PUBLISHED_FILES) {
    fs.copyFileSync(resolve(source), path.join(publicDirectory, destination));
  }

  versionReference("play.js", '"./engine-worker.js"', versionOf(ENGINE_FILES));

  for (const page of PAGES) {
    versionReference(page, '"./site.css"', versionOf(["site.css"]));
  }

  versionReference("index.html", '"./play.js"', versionOf(["play.js"]));
};

main();
