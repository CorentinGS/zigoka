import fs from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_PORT = 4173;

const MIME_TYPES = new Map([
  [".bin", "application/octet-stream"],
  [".css", "text/css; charset=utf-8"],
  [".f32", "application/octet-stream"],
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".json", "application/json; charset=utf-8"],
  [".map", "application/json; charset=utf-8"],
  [".wasm", "application/wasm"],
]);

const readPortArgument = () => {
  const portIndex = process.argv.indexOf("--port");

  if (portIndex === -1) {
    return DEFAULT_PORT;
  }

  const port = Number(process.argv[portIndex + 1]);

  if (!Number.isInteger(port) || port <= 0 || port > 65535) {
    throw new Error(`Invalid port: ${process.argv[portIndex + 1]}`);
  }

  return port;
};

const SERVED_DIRECTORIES = new Map([
  ["bench", "bench"],
  ["demo", "demo"],
  ["dist", "dist"],
  ["e2e", path.join("tests", "e2e")],
  ["site", path.join("site", "public")],
]);
const STRICT_CSP = "script-src 'self'";

const parseRequestUrl = (requestUrl) => {
  try {
    const url = new URL(requestUrl, "http://localhost");

    return {
      urlPath: decodeURIComponent(url.pathname),
      isStrictCsp: url.searchParams.get("csp") === "strict",
    };
  } catch {
    return null;
  }
};

const resolveRequestPath = (urlPath) => {
  if (urlPath === "/") {
    return path.join(rootDirectory, "bench", "index.html");
  }

  for (const [prefix, directory] of SERVED_DIRECTORIES) {
    if (!urlPath.startsWith(`/${prefix}/`)) {
      continue;
    }

    const servedDirectory = path.join(rootDirectory, directory);
    // A directory serves its index.html, as Cloudflare Pages does (/site/19x19/).
    const relativePath = urlPath.slice(prefix.length + 2);
    const resolved = path.resolve(servedDirectory, relativePath.endsWith("/") ? `${relativePath}index.html` : relativePath);

    return resolved.startsWith(`${servedDirectory}${path.sep}`) ? resolved : null;
  }

  return null;
};

const send = (response, statusCode, contentType, content, isStrictCsp = false) => {
  response.writeHead(statusCode, {
    "Content-Type": contentType,
    "Content-Length": content.length,
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Resource-Policy": "same-origin",
    ...(isStrictCsp ? { "Content-Security-Policy": STRICT_CSP } : {}),
  });
  response.end(content);
};

const serve = (request, response) => {
  if (request.method !== "GET" && request.method !== "HEAD") {
    send(response, 405, "text/plain; charset=utf-8", Buffer.from("Method not allowed."));
    return;
  }

  const requestUrl = parseRequestUrl(request.url ?? "/");
  const filePath = requestUrl ? resolveRequestPath(requestUrl.urlPath) : null;

  if (!filePath || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
    send(response, 404, "text/plain; charset=utf-8", Buffer.from("Not found."));
    return;
  }

  const contentType = MIME_TYPES.get(path.extname(filePath)) ?? "application/octet-stream";
  const content = fs.readFileSync(filePath);

  send(response, 200, contentType, content, requestUrl.isStrictCsp);
};

const port = readPortArgument();

const server = http.createServer(serve);

server.listen(port, "127.0.0.1", () => {
  process.stdout.write(
    `Serving the bench on http://127.0.0.1:${port}/bench/index.html ` +
      `and the demo on http://127.0.0.1:${port}/demo/index.html\n`,
  );
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    server.close(() => process.exit(0));
  });
}
