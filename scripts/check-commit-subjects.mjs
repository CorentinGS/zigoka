// Checks the subjects of the commits a push or a pull request brings, as CI runs it:
//   node scripts/check-commit-subjects.mjs <base> <head> <fallback base>
// <base> is the commit before the push, or the pull request's base. A new branch has none (all
// zeros), and a force push leaves one this clone may lack: the range then starts at <fallback base>,
// the default branch.
import { spawnSync } from "node:child_process";

/** "Area : subject", in ASCII: the commits are in French, written without accents. */
const SUBJECT_PATTERN = /^[A-Z][A-Za-z0-9 -]* : [ -~]+$/;

const git = (args) => spawnSync("git", args, { encoding: "utf8" });

const isCommit = (revision) =>
  revision !== undefined && revision !== "" && !/^0+$/.test(revision) && git(["cat-file", "-e", `${revision}^{commit}`]).status === 0;

const [base, head = "HEAD", fallback = "origin/main"] = process.argv.slice(2);
const start = isCommit(base) ? base : fallback;

if (!isCommit(start)) {
  console.error(`Neither ${base || "(no base)"} nor ${fallback} is a commit of this clone.`);
  process.exit(2);
}

const log = git(["log", "--no-merges", "--format=%h %s", `${start}..${head}`]);

if (log.status !== 0) {
  console.error(log.stderr);
  process.exit(2);
}

const commits = log.stdout.split("\n").filter(Boolean);
const wrong = commits.filter((line) => !SUBJECT_PATTERN.test(line.slice(line.indexOf(" ") + 1)));

if (wrong.length > 0) {
  console.error(
    `${wrong.length} of ${commits.length} commits in ${start}..${head} do not read "Area : subject" in ASCII ` +
      '(for example "Moteurs : load controle chaque wasm compile"):',
  );

  for (const line of wrong) {
    console.error(`  ${line}`);
  }

  process.exit(1);
}

console.log(`${commits.length} commit subjects in ${start}..${head} read "Area : subject".`);
