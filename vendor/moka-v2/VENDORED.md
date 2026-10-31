# Vendored sources

This directory holds a read-only copy of the Moka v2 model, proposed upstream
in [millionco/moka#3](https://github.com/millionco/moka/pull/3) ("Moka v2:
300 KB b18-distilled model"). That pull request was still open when this copy
was made, so these files are not on Moka's main branch.

*   **Upstream URL** : <https://github.com/millionco/moka>
*   **Commit** : `1ad518a3300cf14885293f1c8cfe4d840b975136` (head of the
    `moka-v2` branch)
*   **License** : MIT, see `LICENSE` in this directory
*   **Policy** : the files in this directory are never modified.

The pull request does not change `src/`: at this commit, every file of
`src/` has the same digest as in `vendor/moka/VENDORED.md`. The runtime, the
rules and the input encoding of `vendor/moka/src/` therefore also serve this
model, and they are not copied again.

## File digests

The block below has the format of `sha256sum`: `<digest>  <path relative to
vendor/moka-v2/>`, one line per copied file. `pnpm oracle:check` reads this
block and verifies every digest.

```text
aefa84ea395da009c4cd04c23ffe44b34753f0ce30d3c6d07da245b39b6e30bd  experiment-log.md
37ac151e3959a8121fbab61b1f66fb5487d66feb6c6d5c0cc9f5e72b390bcb29  LICENSE
cedecffaf54d47ebf432ae94b7892fc12701766672e19df6710b75c320f9ce3b  model/go-model.bin
1e1107d83dfbfb69b611f594eeea6a294fd706761ef0171df7dec37e881dff69  model/go-model.json
9aa21d1c58347c487ea4e993b87eca17c351aa4097876e4314d4249a2e3e36a5  README.md
1696b47de137b09cb4fd264bf2fd772d2bd4e2787ebc517730f8db381a509f80  THIRD_PARTY_NOTICES.md
```

## Contents

*   `model/go-model.json`, `model/go-model.bin` — the manifest and INT8
    weights of Moka v2: trunk of 48 channels, bottleneck of 24, 16 residual
    blocks. `go-model.bin` has SHA-256
    `cedecffaf54d47ebf432ae94b7892fc12701766672e19df6710b75c320f9ce3b`, the
    same value as `sha256` in the manifest.
*   `README.md`, `experiment-log.md` — Moka's documentation and research log
    at this commit.
*   `LICENSE`, `THIRD_PARTY_NOTICES.md` — Moka's license and notices, the same
    files as in `vendor/moka/`.
