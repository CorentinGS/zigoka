import type { GameResult, GameTask, MatchConfig } from "../match/games.ts";

type MatchWorkerRequest =
  | { type: "initialize"; config: MatchConfig }
  | { type: "openings"; count: number; length: number; seed: number }
  | { type: "game"; task: GameTask };

type MatchWorkerResponse =
  | { type: "ready"; durationMs: number }
  | { type: "openings"; openings: number[][] }
  | {
      type: "move";
      index: number;
      board: Int8Array;
      move: number;
      moveCount: number;
      visits: number;
      thinkingMs: number;
    }
  | { type: "result"; result: GameResult }
  | { type: "error"; error: string };

export type { MatchWorkerRequest, MatchWorkerResponse };
