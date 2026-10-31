import type { MatchSummary } from "../../match/games.ts";
import { theme } from "./theme.ts";

const PADDING = { top: 14, right: 16, bottom: 30, left: 52 };
const Y_TICKS = [0, 0.5, 1];

const ScoreChart = ({
  history,
  totalGames,
  width,
  height,
  color,
}: {
  history: MatchSummary[];
  totalGames: number;
  width: number;
  height: number;
  color: string;
}) => {
  const plotWidth = width - PADDING.left - PADDING.right;
  const plotHeight = height - PADDING.top - PADDING.bottom;
  const x = (games: number) => PADDING.left + (games / totalGames) * plotWidth;
  const y = (score: number) => PADDING.top + (1 - score) * plotHeight;
  const points = history.map((summary, index) => [x(index + 1), y(summary.score)] as const);
  const line = points.map(([px, py], index) => `${index === 0 ? "M" : "L"}${px},${py}`).join("");
  const band =
    history.length > 1
      ? [
          ...history.map((summary, index) => `${index === 0 ? "M" : "L"}${x(index + 1)},${y(summary.scoreInterval[1])}`),
          ...history
            .map((summary, index) => `L${x(index + 1)},${y(summary.scoreInterval[0])}`)
            .reverse(),
          "Z",
        ].join("")
      : "";
  const last = points.at(-1);

  return (
    <svg width={width} height={height}>
      {Y_TICKS.map((tick) => (
        <g key={tick}>
          <line
            x1={PADDING.left}
            x2={width - PADDING.right}
            y1={y(tick)}
            y2={y(tick)}
            stroke={theme.border}
            strokeWidth={1}
            strokeDasharray={tick === 0.5 ? "6 5" : undefined}
          />
          <text
            x={PADDING.left - 10}
            y={y(tick)}
            fill={theme.muted}
            fontSize={16}
            textAnchor="end"
            dominantBaseline="central"
          >
            {`${tick * 100} %`}
          </text>
        </g>
      ))}
      <text x={x(0)} y={height - 6} fill={theme.muted} fontSize={16}>
        0
      </text>
      <text x={x(totalGames)} y={height - 6} fill={theme.muted} fontSize={16} textAnchor="end">
        {`${totalGames} games`}
      </text>
      {band ? <path d={band} fill={color} opacity={0.18} /> : null}
      {points.length > 1 ? <path d={line} fill="none" stroke={color} strokeWidth={2.5} /> : null}
      {last ? (
        <circle cx={last[0]} cy={last[1]} r={5} fill={color} stroke={theme.panel} strokeWidth={2} />
      ) : null}
    </svg>
  );
};

export { ScoreChart };
