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
};

main();
