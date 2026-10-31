type LevelId = "beginner" | "easy" | "medium" | "strong";

type SiteWorkerRequest =
  | { type: "init" }
  | { type: "new-game" }
  | { type: "move"; requestId: number; state: GoGameState; level: LevelId };

type SiteWorkerResponse =
  | { type: "ready"; durationMs: number }
  | { type: "moved"; requestId: number; move: number; visits: number; thinkingMs: number }
  | { type: "error"; error: string };

type LevelSpec = {
  id: LevelId;
  label: string;
  usesSearch: boolean;
  maxVisits?: number;
  timeMs: number;
};

const LEVELS: LevelSpec[] = [
  { id: "beginner", label: "Beginner", usesSearch: false, timeMs: 0 },
  { id: "easy", label: "Easy", usesSearch: true, maxVisits: 16, timeMs: 500 },
  { id: "medium", label: "Medium", usesSearch: true, maxVisits: 80, timeMs: 1000 },
  { id: "strong", label: "Strong", usesSearch: true, maxVisits: 320, timeMs: 2000 },
];

const LEVEL_BY_ID = Object.fromEntries(LEVELS.map((level) => [level.id, level])) as Record<
  LevelId,
  LevelSpec
>;

export { LEVELS, LEVEL_BY_ID };
export type { LevelId, LevelSpec, SiteWorkerRequest, SiteWorkerResponse };
