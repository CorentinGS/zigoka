# zigoka

A zero-allocation Zig/WebAssembly engine for the 9×9 Go network of
[Moka](https://github.com/millionco/moka). It is a drop-in replacement for
Moka's Web Worker: same API, same rules and encoding, only the network runs in
Zig.

The repository also has a PUCT search compiled to WebAssembly, with Moka's
rules ported to Zig, and a match runner to compare engines.

## Quick start

[mise](https://mise.jdx.dev/) installs the pinned Zig, Node and pnpm versions.

```sh
mise install
pnpm install
pnpm setup:browser   # Chromium for the browser tests
pnpm oracle          # reference data, generated with Moka's code
pnpm build
pnpm demo            # http://127.0.0.1:4173/demo/index.html
```

Run `pnpm oracle` before the first `pnpm build`: the build copies its
reference data for the tests.

## Commands

| Command | What it does |
| --- | --- |
| `pnpm build` | builds the WebAssembly modules and the JS bundles |
| `pnpm test` | Zig and Node tests |
| `pnpm test:e2e` | browser tests in Chromium |
| `pnpm check:all` | everything CI runs |
| `pnpm demo` | play against the engine in the browser |
| `pnpm match` | engine-vs-engine matches in Node |

## Using the worker

Serve `dist/worker.js` and `dist/infer.wasm` from the same folder, then give
Moka's client this worker instead of Moka's:

```js
const worker = new Worker("/zigoka/worker.js", { type: "module" });
const moka = new GoModelWorkerClient(worker);

await moka.initialize({
  manifestUrl: "/moka/go-model.json",
  weightsUrl: "/moka/go-model.bin",
});
```

The rest is Moka's API, unchanged. The engine needs WebAssembly SIMD, and a
Content Security Policy sent with `worker.js` must allow `'wasm-unsafe-eval'`.

## Layout

| Path | Contents |
| --- | --- |
| `src/` | the Zig engine: inference, rules, search |
| `js/` | the worker and its JS wrappers |
| `site/` | Zigoka, the play site: page, worker, static build |
| `demo/` | demo pages |
| `match/` | match runner |
| `oracle/`, `tests/` | reference data and tests |
| `pipeline/` | Python distillation pipeline |
| `video/` | Remotion replay of a match |
| `docs/designs/` | design documents |
| `vendor/` | copies of Moka v1 and of the Moka v2 model |

## License

MIT, see [`LICENSE`](LICENSE). The model and reference code come from
[Moka](https://github.com/millionco/moka) by Million Software, Inc., under the
MIT license; see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
`video/` uses [Remotion](https://www.remotion.dev/), which has its own
license.
