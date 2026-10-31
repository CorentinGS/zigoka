import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const videoDirectory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const resultsDirectory = path.resolve(videoDirectory, "..", "bench", "results");
const outputPath = path.join(videoDirectory, "src", "match-data.json");

const findLatestRecording = () => {
  const candidates = fs
    .readdirSync(resultsDirectory)
    .filter((name) => name.startsWith("match-") && name.endsWith(".json"))
    .map((name) => path.join(resultsDirectory, name))
    .filter((file) => "timings" in JSON.parse(fs.readFileSync(file, "utf8")))
    .sort();

  if (candidates.length === 0) {
    throw new Error("No match recording with timings in bench/results (run pnpm match first).");
  }

  return candidates.at(-1);
};

const inputPath = process.argv[2] ? path.resolve(process.argv[2]) : findLatestRecording();
const { summary, results, timings } = JSON.parse(fs.readFileSync(inputPath, "utf8"));
const timingByIndex = new Map(timings.map((timing) => [timing.index, timing]));
const round = (value) => Math.round(value * 10) / 10;

const games = results.map((result) => {
  const timing = timingByIndex.get(result.index);

  return {
    index: result.index,
    slot: timing.slot,
    startMs: round(timing.startMs),
    endMs: round(timing.endMs),
    opening: result.opening,
    firstPlaysBlack: result.firstPlaysBlack,
    score: result.score,
    firstResult: result.firstResult,
    moveCount: result.moveCount,
    first: result.first,
    second: result.second,
    moves: result.moves.map(({ move, visits, thinkingMs }) => [move, visits, round(thinkingMs)]),
  };
});

fs.writeFileSync(outputPath, `${JSON.stringify({ config: summary.config, games })}\n`);
console.log(
  `Wrote ${games.length} games from ${path.relative(videoDirectory, inputPath)} to ${path.relative(videoDirectory, outputPath)} ` +
    `(${(fs.statSync(outputPath).size / 1024).toFixed(0)} KB).`,
);
