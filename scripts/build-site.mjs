import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resolve = (relativePath) => path.join(rootDirectory, relativePath);
const publicDirectory = resolve("site/public");

// Cloudflare Pages headers for the published site; wasm-unsafe-eval is required
// to compile infer.wasm and search.wasm inside engine-worker.js.
const HEADERS_CONTENT = `/*
  X-Content-Type-Options: nosniff
  Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; connect-src 'self'; worker-src 'self'; base-uri 'none'; frame-ancestors 'none'
  Referrer-Policy: no-referrer

/engine-worker.js
  Content-Security-Policy: script-src 'self' 'wasm-unsafe-eval'
  Cache-Control: public, max-age=3600

/infer.wasm
/search.wasm
  Cache-Control: public, max-age=86400

/go-model.json
/go-model.bin
  Cache-Control: public, max-age=86400
`;

const PUBLISHED_FILES = [
  ["site/index.html", "index.html"],
  ["site/about.html", "about.html"],
  ["site/site.css", "site.css"],
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

  fs.writeFileSync(path.join(publicDirectory, "_headers"), HEADERS_CONTENT);
};

main();
