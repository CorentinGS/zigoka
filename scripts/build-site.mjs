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

// Files shared by every page, at the root of the site.
const SHARED_FILES = [
  ["site/about.html", "about.html"],
  ["site/site.css", "site.css"],
  ["site/_headers", "_headers"],
  ["vendor/moka-v2/LICENSE", "moka-license.txt"],
  ["vendor/moka-v2/THIRD_PARTY_NOTICES.md", "moka-notices.txt"],
  ["models/moka-19/KATAGO-NETWORK-LICENSE.txt", "katago-network-license.txt"],
];

// One edition per board size, each in a directory of its own: its page, the bundles built
// with its constants (scripts/build-js.mjs) and the engine of its model, whose wasm modules
// are built for that model's layout. The 9×9 edition plays Moka v2, at the root.
const EDITIONS = [
  { directory: ".", page: "site/index.html", dist: "site/dist", modelId: "moka-9-v2", css: '"./site.css"' },
  {
    directory: "19x19",
    page: "site/19x19.html",
    dist: "site/dist/19x19",
    modelId: "moka-19",
    css: '"../site.css"',
  },
];

const editionFiles = ({ page, dist, modelId }) => [
  [page, "index.html"],
  [`${dist}/play.js`, "play.js"],
  [`${dist}/engine-worker.js`, "engine-worker.js"],
  [`zig-out/${modelId}/search.wasm`, "search.wasm"],
  [`zig-out/${modelId}/infer.wasm`, "infer.wasm"],
  [`zig-out/models/${modelId}/go-model.json`, "go-model.json"],
  [`zig-out/models/${modelId}/go-model.bin`, "go-model.bin"],
];

const PUBLISHED_FILES = [
  ...SHARED_FILES,
  ...EDITIONS.flatMap((edition) =>
    editionFiles(edition).map(([source, destination]) => [source, path.join(edition.directory, destination)]),
  ),
];

// The worker gives its own `?v=` to the files it fetches (site/engine-worker.ts).
const ENGINE_FILES = ["engine-worker.js", "infer.wasm", "search.wasm", "go-model.json", "go-model.bin"];

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

  for (const [source, destination] of PUBLISHED_FILES) {
    fs.mkdirSync(path.dirname(publicPath(destination)), { recursive: true });
    fs.copyFileSync(resolve(source), publicPath(destination));
  }

  const cssVersion = versionOf(["site.css"]);

  for (const { directory, css } of EDITIONS) {
    const at = (fileName) => path.join(directory, fileName);

    versionReference(at("play.js"), '"./engine-worker.js"', versionOf(ENGINE_FILES.map(at)));
    versionReference(at("index.html"), css, cssVersion);
    versionReference(at("index.html"), '"./play.js"', versionOf([at("play.js")]));
  }

  versionReference("about.html", '"./site.css"', cssVersion);
};

main();
