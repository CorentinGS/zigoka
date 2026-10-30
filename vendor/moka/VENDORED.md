# Vendored sources

This directory holds a read-only copy of parts of [Moka](https://github.com/millionco/moka),
the Go 9×9 neural network published by Million Software, Inc. under the MIT
license.

*   **Upstream URL** : <https://github.com/millionco/moka>
*   **Commit** : `a418039e9d823779e3e6d0ea17a244230abc8379`
*   **License** : MIT, see `LICENSE` in this directory
*   **Policy** : the files in this directory are never modified. Every
    instrumented or derived copy lives elsewhere in the repository (see
    `oracle/moka-runtime-instrumented.ts`).

## File digests

The block below has the format of `sha256sum`: `<digest>  <path relative to
vendor/moka/>`, one line per copied file. `pnpm oracle:check` reads this block
and verifies every digest.

```text
2fbf5f0cbec0bd4f0e775679b44658cd3e2aa9c68555923c9e25e58959b64014  experiment-log.md
37ac151e3959a8121fbab61b1f66fb5487d66feb6c6d5c0cc9f5e72b390bcb29  LICENSE
35c1e51cc9518a21e2f81cded7f69f41d22c80ca05522ed5e4056cb26d9ed053  model/go-model.bin
00927862de6c29749124b3257b6618156c07b930ba624e010601b90434c1340e  model/go-model.json
2fae6776554d23e17a412366e1e9282b38a514e940bbbec058774c4eb9046d2e  README.md
ec29ea9cc563142842e036baffb3765799298f0135da987762a565858932f923  src/client.ts
ea5e519983e706b2766a2d38d21223720dcb53f9b6de10da2a801a6b44174800  src/constants.ts
1698d9e34e3bbfbd986582386ba85b318a8f9bae4b4a0bcef39c57a87351bf97  src/game.ts
cc20bbd370e8869c5abff8c968128dd3a3c8da627566fabc51e73f30ce0daa9b  src/index.ts
5549af2630e7ba6b5442df5ce73586b14cb9ae06f1687789630a4be7c2f6f05f  src/runtime-constants.ts
87faf2a8c318aa11c66e12a4a44baff7174446b797493a5830e042d7fdd0ec6e  src/runtime.ts
d9532b8f9983ce58e83d8f3374a49cbcda7b354aa92c0df6015b687ce0df4460  src/types.ts
413b48eba990125b5b7f79581395fbea77f8a3a6d6570bd7a6cc52879317e9f7  src/utils/decompress-gzip-response.ts
89a309d3a34f86eb50de6bb16f648f03df5579cc5de15fe35562f221ff880de1  src/utils/get-buffer-sha256.ts
055f6cb3d16b31492917a9219d33e22e712bd1af4af0ee041722adddc4f7620e  src/utils/read-response-buffer.ts
915d047173fd8c0890f1b4be316b60e950ace1924060d797475f41deb3b1269e  src/worker.ts
1696b47de137b09cb4fd264bf2fd772d2bd4e2787ebc517730f8db381a509f80  THIRD_PARTY_NOTICES.md
1985b70dd20e32000720b9bc73f206a0338ca08dc2d3f2ba4f827527d332ff6f  wasm/src/lib.rs
1bf39746c9939f1834e58c2a00362f459d5c2d22e3a2054a68585456c95e9f41  web/benchmark.ts
cb60cb98ed645642dd9e60773d07be41bb46cc5c7d1aca49bcc433259aab29fd  web/constants.ts
```

## Type checking

`vendor/moka/src/` is included in the `tsconfig.json` program so that the
global types of Moka (`GoGameState`, `GoModelManifest`, and the Worker message
types) are available to the oracle and the benchmark. Moka's own code is not
held to this repository's indexing rules: `noUncheckedIndexedAccess` is
disabled in `tsconfig.json`, rather than patching the vendored files. The
same applies to `oracle/moka-runtime-instrumented.ts`, which is a copy of
Moka's runtime.

## Contents

*   `src/` — the TypeScript runtime, rules and Worker protocol of Moka, in
    full. This is the numerical reference used by the oracle.
*   `model/go-model.json`, `model/go-model.bin` — the model manifest and its
    INT8 weights. `go-model.bin` has SHA-256
    `35c1e51cc9518a21e2f81cded7f69f41d22c80ca05522ed5e4056cb26d9ed053`, the
    same value as `sha256` in the manifest.
*   `web/benchmark.ts`, `web/constants.ts` — Moka's benchmark page and its
    constants, kept for reference only. The benchmark of this repository is a
    corrected rewrite.
*   `wasm/src/lib.rs` — Moka's earlier Rust/WASM experiment, kept for
    reference only.
*   `README.md`, `experiment-log.md` — Moka's documentation and research log.
    `experiment-log.md` is the source of the fixed probe of the validation
    set.
*   `LICENSE`, `THIRD_PARTY_NOTICES.md` — Moka's license and notices.
