import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

const rootDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

/** pnpm's own commands: what follows them is not a script of package.json. */
const PNPM_COMMANDS = new Set(["add", "dlx", "exec", "i", "install", "remove", "update"]);
const STEP_PATTERN = /^[a-z][a-z0-9-]*$/;
const SCRIPT_PATTERN = /^[a-z][a-z0-9:-]*$/;
const COMMAND_PATTERN = /\b(zig build|pnpm)\b([^\n`"'&|;)#]*)/g;

const run = (command, args) => {
  const result = spawnSync(command, args, { cwd: rootDirectory, encoding: "utf8" });

  assert.equal(result.status, 0, result.stderr);

  return result.stdout;
};

const readZigSteps = () =>
  new Set(
    run("zig", ["build", "-l"])
      .split("\n")
      .map((line) => /^\s+(\S+)/.exec(line)?.[1])
      .filter((step) => step !== undefined),
  );

const readScripts = (packagePath) =>
  JSON.parse(fs.readFileSync(path.join(rootDirectory, packagePath), "utf8")).scripts ?? {};

const listFiles = (...patterns) => run("git", ["ls-files", ...patterns]).split("\n").filter(Boolean);

/** Markdown runs only what sits in code: fenced blocks and inline code spans. */
const readMarkdownCode = (text) => [
  ...[...text.matchAll(/^```[^\n]*\n([\s\S]*?)^```/gm)].map((match) => match[1]),
  ...[...text.replace(/^```[\s\S]*?^```/gm, "").matchAll(/`([^`\n]+)`/g)].map((match) => match[1]),
];

/**
 * The text each file runs or tells a reader to run. Design docs stay out: their commands are the
 * history of a plan, written for the steps of their day.
 */
const readSources = () => [
  ...listFiles("package.json", "video/package.json").map((file) => ({
    file,
    texts: Object.values(readScripts(file)),
  })),
  ...listFiles("*.sh", ".github/workflows/*.yml").map((file) => ({
    file,
    texts: [fs.readFileSync(path.join(rootDirectory, file), "utf8")],
  })),
  ...listFiles("*.md")
    .filter((file) => !file.startsWith("docs/designs/") && !file.startsWith("vendor/"))
    .map((file) => ({ file, texts: readMarkdownCode(fs.readFileSync(path.join(rootDirectory, file), "utf8")) })),
];

/** The scripts a file can call with pnpm: those of the package it belongs to, and the root's. */
const scriptsFor = (file) =>
  new Set([
    ...Object.keys(readScripts("package.json")),
    ...(file.startsWith("video/") ? Object.keys(readScripts("video/package.json")) : []),
  ]);

/** A placeholder such as `$name` or `<id>` names no step. */
const isPlaceholder = (word) => /[$<>={}]/.test(word);

const findCommands = ({ file, texts }) =>
  texts.flatMap((text) =>
    [...text.matchAll(COMMAND_PATTERN)].map(([command, tool, rest]) => ({
      file,
      command: command.trim(),
      tool,
      words: rest.trim().split(/\s+/).filter(Boolean),
    })),
  );

const findUnknownNames = ({ tool, words, file }, zigSteps) => {
  if (tool === "zig build") {
    return words
      .filter((word) => !word.startsWith("-") && !isPlaceholder(word) && STEP_PATTERN.test(word))
      .filter((step) => !zigSteps.has(step));
  }

  const [first, second] = words;
  const script = first === "run" ? second : first;

  if (script === undefined || PNPM_COMMANDS.has(script) || isPlaceholder(script) || !SCRIPT_PATTERN.test(script)) {
    return [];
  }

  return scriptsFor(file).has(script) ? [] : [script];
};

test("each zig build step and pnpm script the scripts and READMEs call exists", () => {
  const zigSteps = readZigSteps();
  const commands = readSources().flatMap(findCommands);
  const unknown = commands.flatMap((command) =>
    findUnknownNames(command, zigSteps).map((name) => `${command.file}: "${command.command}" calls ${name}`),
  );

  assert.ok(commands.length > 0, "the scan found no command");
  assert.deepEqual(unknown, []);
});
