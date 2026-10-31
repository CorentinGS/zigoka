import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { formatElo } from "../../match/games.ts";
import { getEngineColor, PLAYER_NAMES, theme } from "./theme.ts";
import { config, finalStats } from "./timeline.ts";

const CONTROL_VISITS = 16;
const CONTROL_SCORE = "50.0 %";

const fadeIn = (frame: number, start: number) => ({
  opacity: interpolate(frame, [start, start + 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  }),
});

const formatPercent = (value: number) => `${(value * 100).toFixed(1)} %`;

const Outro = () => {
  const frame = useCurrentFrame();
  const summary = finalStats.summary!;
  const firstName = PLAYER_NAMES[config.first];
  const secondName = PLAYER_NAMES[config.second];

  return (
    <AbsoluteFill
      style={{
        background: theme.background,
        color: theme.text,
        fontFamily: theme.font,
        fontVariantNumeric: "tabular-nums",
        justifyContent: "center",
        alignItems: "center",
        gap: 36,
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: 34, color: theme.muted, ...fadeIn(frame, 0) }}>
        {config.games} games · {config.parallel} in parallel · {config.timeMs} ms per move
      </div>
      <div style={{ display: "flex", gap: 48, alignItems: "center", ...fadeIn(frame, 8) }}>
        <div style={{ fontSize: 44, fontWeight: 700, color: getEngineColor(config.first) }}>{firstName}</div>
        <div style={{ fontSize: 120, fontWeight: 800 }}>
          {summary.wins} – {summary.losses}
        </div>
        <div style={{ fontSize: 44, fontWeight: 700, color: getEngineColor(config.second) }}>{secondName}</div>
      </div>
      <div style={{ fontSize: 32, color: theme.muted, ...fadeIn(frame, 16) }}>
        {summary.draws} draws · score {formatPercent(summary.score)} · Elo {formatElo(summary.elo)} (95 %:{" "}
        {formatElo(summary.eloInterval[0])} to {formatElo(summary.eloInterval[1])})
      </div>
      <div style={{ fontSize: 36, ...fadeIn(frame, 40) }}>
        {finalStats.firstVisitsPerMove!.toFixed(0)} positions searched per move against{" "}
        {finalStats.secondVisitsPerMove!.toFixed(1)}, in the same 100 ms.
      </div>
      <div style={{ fontSize: 26, color: theme.muted, lineHeight: 1.5, ...fadeIn(frame, 70) }}>
        <div>
          Control: a 100-game match with both sides capped at {CONTROL_VISITS} visits per move ends
          at {CONTROL_SCORE}.
        </div>
        <div>The gap comes from speed, not from the harness.</div>
        <div style={{ marginTop: 18, fontSize: 22 }}>
          Replay of a match recorded with Node 24 worker threads on an AMD Ryzen 7 8845HS · PUCT
          search in TypeScript on Moka's rules
        </div>
      </div>
    </AbsoluteFill>
  );
};

export { Outro };
