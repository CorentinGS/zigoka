# Coding standards

The rules a review applies. The mechanical ones are checked, so a review skips them:

- `tests/conventions.test.mjs`: arrow functions, never the `function` keyword; each module ends on one export list, with no declaration exported inline.
- `tests/build-commands.test.mjs`: each `zig build` step and `pnpm` script that a script or README calls exists.
- `tests/error-table.test.ts`: each Worker error the e2e tests expect is a row of the error table in `docs/designs/zig-wasm-moka.md`.
- `scripts/check-commit-subjects.mjs`, run by CI: commit subjects read `Area : subject`, in ASCII.

The rules below take judgement.

## Names

Code names a domain concept with its `GLOSSARY.md` term, and steers clear of the terms the glossary lists under _Avoid_. A change that brings a new concept adds its glossary entry in the same change. A name that fits no glossary term is a finding: `LoadedModel`, for a model's engines plus Moka's runtime, is neither a Model nor an Engine.

## Designs

A design in `docs/designs/` records what was approved. A later change to it goes in a "Ce qui a changé depuis" list at the top of that design (see `docs/designs/zig-wasm-moka.md`), which names the module or file that changed. The body and the numbered plan steps stay as written: they are the history of the plan.

## Languages

Designs, commit subjects and commit bodies are in French, written without accents in commits. Code, comments, error messages, `README.md` and `GLOSSARY.md` are in English.
