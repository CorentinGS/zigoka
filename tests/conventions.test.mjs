import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

// TypeScript 7 ships no parser in JavaScript, only its scanner. The scanner leaves two choices to
// its parser, which this file makes itself: a `}` that closes a template substitution, and a `/`
// that starts a regular expression.
import { LanguageVariant, SyntaxKind } from "typescript/unstable/ast";
import { createScanner } from "typescript/unstable/ast/scanner";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/**
 * The folders whose modules follow the conventions. oracle/ is not one yet: its three
 * `function fail(): never` let TypeScript narrow after a call, which an arrow function only does
 * with its type written out.
 */
const CHECKED_PATHS = ["bench", "demo", "js", "match", "models", "scripts", "site", "tests", "playwright.config.ts"];

/** A `/` after one of these divides; after any other token it starts a regular expression. */
const OPERAND_ENDS = new Set([
  SyntaxKind.Identifier,
  SyntaxKind.CloseParenToken,
  SyntaxKind.CloseBracketToken,
  SyntaxKind.CloseBraceToken,
  SyntaxKind.NumericLiteral,
  SyntaxKind.BigIntLiteral,
  SyntaxKind.StringLiteral,
  SyntaxKind.NoSubstitutionTemplateLiteral,
  SyntaxKind.TemplateTail,
  SyntaxKind.RegularExpressionLiteral,
  SyntaxKind.ThisKeyword,
  SyntaxKind.SuperKeyword,
  SyntaxKind.TrueKeyword,
  SyntaxKind.FalseKeyword,
  SyntaxKind.NullKeyword,
]);
const OPENERS = new Set([
  SyntaxKind.OpenBraceToken,
  SyntaxKind.OpenParenToken,
  SyntaxKind.OpenBracketToken,
  SyntaxKind.TemplateHead,
  SyntaxKind.TemplateMiddle,
]);
const CLOSERS = new Set([
  SyntaxKind.CloseBraceToken,
  SyntaxKind.CloseParenToken,
  SyntaxKind.CloseBracketToken,
  SyntaxKind.TemplateMiddle,
  SyntaxKind.TemplateTail,
]);

/** Each token's kind, start and nesting depth: 0 for the tokens of top-level statements. */
const scanTokens = (text) => {
  const scanner = createScanner(true, LanguageVariant.Standard, text);
  const braces = [];
  const tokens = [];
  let depth = 0;
  let previous;

  for (let kind = scanner.scan(); kind !== SyntaxKind.EndOfFile; kind = scanner.scan()) {
    if (kind === SyntaxKind.CloseBraceToken && braces.pop() === "template") {
      kind = scanner.reScanTemplateToken(false);
    } else if ((kind === SyntaxKind.SlashToken || kind === SyntaxKind.SlashEqualsToken) && !OPERAND_ENDS.has(previous)) {
      kind = scanner.reScanSlashToken();
    }

    if (kind === SyntaxKind.OpenBraceToken) {
      braces.push("block");
    } else if (kind === SyntaxKind.TemplateHead || kind === SyntaxKind.TemplateMiddle) {
      braces.push("template");
    }

    if (CLOSERS.has(kind)) {
      depth -= 1;
    }

    tokens.push({ kind, start: scanner.getTokenStart(), depth });

    if (OPENERS.has(kind)) {
      depth += 1;
    }

    previous = kind;
  }

  // A misread `}` or `/` leaves a bracket open or swallows the rest of the file as one token.
  return { tokens, isBalanced: depth === 0 && braces.length === 0 };
};

const indexAfterSemicolon = (tokens, index) => {
  let next = index;

  while (next < tokens.length && !(tokens[next].depth === 0 && tokens[next].kind === SyntaxKind.SemicolonToken)) {
    next += 1;
  }

  return next + 1;
};

/** Where the export at `index` ends, and whether it is a re-export, which sits with the imports. */
const readExport = (tokens, index) => {
  const kindAt = (offset) => tokens[index + offset]?.kind;
  const listOffset = kindAt(1) === SyntaxKind.TypeKeyword ? 2 : 1;

  if (kindAt(listOffset) === SyntaxKind.OpenBraceToken) {
    let next = index + listOffset + 1;

    while (tokens[next].depth !== 0) {
      next += 1;
    }

    const isReexport = tokens[next + 1]?.kind === SyntaxKind.FromKeyword;

    next += isReexport ? 3 : 1;

    return { kind: isReexport ? "re-export" : "list", end: tokens[next]?.kind === SyntaxKind.SemicolonToken ? next + 1 : next };
  }

  if (kindAt(1) === SyntaxKind.AsteriskToken) {
    return { kind: "re-export", end: indexAfterSemicolon(tokens, index) };
  }

  if (kindAt(1) === SyntaxKind.DefaultKeyword) {
    return { kind: "list", end: indexAfterSemicolon(tokens, index) };
  }

  return { kind: "inline", end: index + 1 };
};

/**
 * The conventions of a module: arrow functions, never the `function` keyword; and one export
 * list at the end, with no declaration exported inline. A re-export may sit with the imports, and
 * a config file may end on `export default`.
 */
const findViolations = (text) => {
  const { tokens, isBalanced } = scanTokens(text);
  const lineOf = (token) => text.slice(0, token.start).split("\n").length;
  const violations = isBalanced ? [] : ["the scan ends inside a bracket or a template, so it misread the file"];
  let exportsBegan = false;
  let codeAfterExports = false;
  let index = 0;

  for (const token of tokens) {
    if (token.kind === SyntaxKind.FunctionKeyword) {
      violations.push(`${lineOf(token)}: the function keyword; write an arrow function`);
    }
  }

  while (index < tokens.length) {
    const token = tokens[index];

    if (token.depth !== 0) {
      index += 1;
    } else if (token.kind === SyntaxKind.ExportKeyword) {
      const { kind, end } = readExport(tokens, index);

      if (kind === "inline") {
        violations.push(`${lineOf(token)}: a declaration exported inline; list it in the export at the end`);
      }

      exportsBegan ||= kind === "list";
      index = end;
    } else {
      if (exportsBegan && !codeAfterExports) {
        violations.push(`${lineOf(token)}: code after the exports, which end the module`);
        codeAfterExports = true;
      }

      index += 1;
    }
  }

  return violations;
};

const listCheckedFiles = () => {
  const result = spawnSync("git", ["ls-files", "--", ...CHECKED_PATHS], { cwd: rootDirectory, encoding: "utf8" });

  assert.equal(result.status, 0, result.stderr);

  return result.stdout.split("\n").filter((file) => /\.(ts|mts|mjs|js)$/.test(file));
};

test("the conventions check reads past template substitutions and regular expressions", () => {
  const source = [
    'import { a } from "./a.ts";',
    'export * from "./b.ts";',
    "const quote = `a ${`b ${{ c: 1 }.c}`} \"`;",
    "const pattern = /[\"'`]\\/{/g;",
    "function late() {}",
    "export const inline = 1;",
    "export { a, quote };",
    'export type { Pattern } from "./c.ts";',
    "const after = pattern;",
  ].join("\n");

  assert.deepEqual(findViolations(source), [
    "5: the function keyword; write an arrow function",
    "6: a declaration exported inline; list it in the export at the end",
    "9: code after the exports, which end the module",
  ]);
  assert.deepEqual(findViolations("const open = (1;"), [
    "the scan ends inside a bracket or a template, so it misread the file",
  ]);
});

test("modules write arrow functions and end on one export list", () => {
  const files = listCheckedFiles();
  const violations = files.flatMap((file) =>
    findViolations(fs.readFileSync(path.join(rootDirectory, file), "utf8")).map((violation) => `${file}:${violation}`),
  );

  assert.ok(files.length > 0, "no file to check");
  assert.deepEqual(violations, []);
});
