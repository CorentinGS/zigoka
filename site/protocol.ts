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

// Medium and Strong think for a fixed time, without a visit cap; no level
// thinks longer than 1 s. Moka v2 runs about 1,400 visits per second at 9×9;
// moka-19 runs far fewer at 19×19, with a larger network on a larger board.
const LEVELS: LevelSpec[] = [
  { id: "beginner", label: "Beginner", usesSearch: false, timeMs: 0 },
  { id: "easy", label: "Easy", usesSearch: true, maxVisits: 16, timeMs: 500 },
  { id: "medium", label: "Medium (0.3 s)", usesSearch: true, timeMs: 300 },
  { id: "strong", label: "Strong (1 s)", usesSearch: true, timeMs: 1000 },
];

const LEVEL_BY_ID = Object.fromEntries(LEVELS.map((level) => [level.id, level])) as Record<
  LevelId,
  LevelSpec
>;

export { LEVELS, LEVEL_BY_ID };
export type { LevelId, LevelSpec, SiteWorkerRequest, SiteWorkerResponse };
