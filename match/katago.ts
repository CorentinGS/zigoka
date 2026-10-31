import { spawn } from "node:child_process";
import os from "node:os";
import path from "node:path";
import readline from "node:readline";

import { GO_BOARD_SIZE, GO_KOMI_POINTS, GO_PASS_MOVE } from "../vendor/moka/src/constants.ts";
import type { MoveChoice, Player } from "./games.ts";

const COLUMNS = "ABCDEFGHJKLMNOPQRST";
const COLORS = ["B", "W"] as const;

/** Moka's rules: simple ko, area scoring, no suicide, no button or tax. */
const MOKA_RULES = [
  "koRule=SIMPLE",
  "scoringRule=AREA",
  "taxRule=NONE",
  "multiStoneSuicideLegal=false",
  "hasButton=false",
  "whiteHandicapBonus=0",
];

/** With `human`, moves come from the human SL model imitating `human.profile`, such as "rank_5k". */
type KataGoOptions = {
  binary: string;
  model: string;
  config: string;
  visits: number;
  human?: { model: string; profile: string };
};

type KataGoPlayer = Player & {
  finalScore: (moves: number[]) => Promise<number>;
  close: () => void;
};

const toVertex = (move: number) => {
  if (move === GO_PASS_MOVE) {
    return "pass";
  }

  const row = Math.floor(move / GO_BOARD_SIZE);
  const column = move % GO_BOARD_SIZE;

  return `${COLUMNS[column]}${GO_BOARD_SIZE - row}`;
};

const fromVertex = (vertex: string) => {
  const upper = vertex.trim().toUpperCase();

  if (upper === "PASS") {
    return GO_PASS_MOVE;
  }

  const column = COLUMNS.indexOf(upper[0] ?? "");
  const row = GO_BOARD_SIZE - Number(upper.slice(1));

  if (column < 0 || column >= GO_BOARD_SIZE || !Number.isInteger(row) || row < 0 || row >= GO_BOARD_SIZE) {
    throw new Error(`KataGo answered an unknown vertex "${vertex}".`);
  }

  return row * GO_BOARD_SIZE + column;
};

/** Parses a GTP score such as "B+3.5", "W+0.5" or "0" into Black's margin. */
const parseScore = (score: string) => {
  const text = score.trim().toUpperCase();

  if (text === "0") {
    return 0;
  }

  const margin = Number(text.slice(2));

  if (!/^[BW]\+/.test(text) || !Number.isFinite(margin)) {
    throw new Error(`KataGo answered an unknown score "${score}".`);
  }

  return text[0] === "B" ? margin : -margin;
};

/** Runs KataGo in GTP mode with a fixed number of visits per move, under Moka's rules. */
const createKataGoPlayer = async ({ binary, model, config, visits, human }: KataGoOptions): Promise<KataGoPlayer> => {
  const overrides = [
    ...MOKA_RULES,
    `maxVisits=${visits}`,
    `logDir=${path.join(os.tmpdir(), "go-network-katago-logs")}`,
    ...(human ? [`humanSLProfile=${human.profile}`] : []),
  ].join(",");
  const models = human ? ["-model", model, "-human-model", human.model] : ["-model", model];
  const child = spawn(binary, ["gtp", ...models, "-config", config, "-override-config", overrides], {
    stdio: ["pipe", "pipe", "ignore"],
  });
  const lines = readline.createInterface({ input: child.stdout });
  const pending: { resolve: (answer: string) => void; reject: (error: Error) => void }[] = [];
  let answer: string[] = [];

  lines.on("line", (line) => {
    if (line.trim() !== "") {
      answer.push(line);
      return;
    }

    if (answer.length === 0) {
      return;
    }

    const [first, ...rest] = answer;
    const request = pending.shift();

    answer = [];

    if (!request) {
      return;
    }

    if (first.startsWith("=")) {
      request.resolve([first.slice(1), ...rest].join("\n").trim());
    } else {
      request.reject(new Error(`KataGo refused a command: ${[first, ...rest].join(" ")}`));
    }
  });

  child.on("exit", (code) => {
    for (const request of pending.splice(0)) {
      request.reject(new Error(`KataGo exited with code ${code}.`));
    }
  });

  const send = (command: string) =>
    new Promise<string>((resolve, reject) => {
      pending.push({ resolve, reject });
      child.stdin.write(`${command}\n`);
    });

  let known: number[] = [];

  /** Brings KataGo's board to `moves`, replaying from an empty board unless `moves` extends it. */
  const sync = async (moves: number[]) => {
    const extends_ = known.length <= moves.length && known.every((move, index) => move === moves[index]);

    if (!extends_) {
      await send("clear_board");
      known = [];
    }

    for (let index = known.length; index < moves.length; index += 1) {
      await send(`play ${COLORS[index % 2]} ${toVertex(moves[index])}`);
    }

    known = [...moves];
  };

  await send(`boardsize ${GO_BOARD_SIZE}`);
  await send(`komi ${GO_KOMI_POINTS}`);

  const play = async (state: GoGameState): Promise<MoveChoice> => {
    await sync(state.moveHistory);

    const vertex = await send(`genmove ${COLORS[state.moveHistory.length % 2]}`);

    if (vertex.toUpperCase() === "RESIGN") {
      throw new Error("KataGo resigned although resignation is disabled.");
    }

    const move = fromVertex(vertex);

    known = [...state.moveHistory, move];

    return { move, visits };
  };

  const finalScore = async (moves: number[]) => {
    await sync(moves);

    return parseScore(await send("final_score"));
  };

  return {
    kind: human ? `katago-${human.profile}` : `katago-${visits}`,
    play,
    finalScore,
    close: () => {
      child.stdin.end("quit\n");
    },
  };
};

export { createKataGoPlayer, fromVertex, parseScore, toVertex };
export type { KataGoOptions, KataGoPlayer };
