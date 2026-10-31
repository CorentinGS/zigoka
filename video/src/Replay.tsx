import { AbsoluteFill, interpolate, useCurrentFrame } from "remotion";
import { formatElo } from "../../match/games.ts";
import { Goban } from "./Goban.tsx";
import { ScoreChart } from "./ScoreChart.tsx";
import { getEngineColor, PLAYER_NAMES, theme } from "./theme.ts";
import { config, FPS, getMatchMs, getSlotView, getSpeed, getStats } from "./timeline.ts";

const BOARD_SIZE = 230;
const COLUMNS = 5;
const FLASH_MS = 700;
const ROUNDED_SPEED = 5.95;

const firstName = PLAYER_NAMES[config.first];
const secondName = PLAYER_NAMES[config.second];
const firstShortName = firstName.split(/[ ,]/)[0];
const secondShortName = secondName.split(/[ ,]/)[0];
const firstColor = getEngineColor(config.first);
const secondColor = getEngineColor(config.second);

const formatPercent = (value: number) => `${(value * 100).toFixed(1)} %`;

const formatClock = (matchMs: number) => {
  const seconds = Math.floor(matchMs / 1000);

  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
};

const Dot = ({ color, size = 14 }: { color: string; size?: number }) => (
  <span
    style={{
      display: "inline-block",
      width: size,
      height: size,
      borderRadius: size / 2,
      background: color,
      marginRight: 8,
      verticalAlign: "middle",
    }}
  />
);

const Tile = ({ label, value, detail }: { label: string; value: string; detail?: string }) => (
  <div
    style={{
      flex: "1 1 0",
      padding: "14px 18px",
      background: theme.panel,
      border: `1px solid ${theme.border}`,
      borderRadius: 14,
      display: "flex",
      flexDirection: "column",
      gap: 6,
    }}
  >
    <span style={{ color: theme.muted, fontSize: 20 }}>{label}</span>
    <strong style={{ fontSize: 42, fontWeight: 700 }}>{value}</strong>
    {detail ? <span style={{ color: theme.muted, fontSize: 18 }}>{detail}</span> : null}
  </div>
);

const VisitsBar = ({
  name,
  color,
  value,
  maximum,
}: {
  name: string;
  color: string;
  value: number | null;
  maximum: number;
}) => (
  <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 21 }}>
      <span>
        <Dot color={color} size={12} />
        {name}
      </span>
      <strong>{value === null ? "–" : value.toFixed(value < 100 ? 1 : 0)}</strong>
    </div>
    <div style={{ height: 14, background: theme.background, borderRadius: 7 }}>
      <div
        style={{
          width: `${value === null ? 0 : Math.max(1, (value / maximum) * 100)}%`,
          height: "100%",
          background: color,
          borderRadius: 7,
        }}
      />
    </div>
  </div>
);

const SlotCard = ({ slot, matchMs, speed }: { slot: number; matchMs: number; speed: number }) => {
  const view = getSlotView(slot, matchMs);
  const [blackKind, whiteKind] = view.game.firstPlaysBlack
    ? [config.first, config.second]
    : [config.second, config.first];
  const flash = view.lastFinished
    ? interpolate(matchMs - view.lastFinished.endMs, [0, FLASH_MS * speed * 0.6, FLASH_MS * speed], [1, 1, 0], {
        extrapolateRight: "clamp",
      })
    : 0;
  const winner = view.lastFinished?.firstResult;
  const flashColor = winner === 1 ? firstColor : winner === -1 ? secondColor : theme.draw;
  const flashText =
    winner === 1 ? `${firstShortName} wins` : winner === -1 ? `${secondShortName} wins` : "Draw";
  const moverIsBlack = view.moveNumber % 2 === 1;
  const moverKind = moverIsBlack ? blackKind : whiteKind;

  return (
    <div
      style={{
        padding: 12,
        background: theme.panel,
        border: `2px solid ${flash > 0 ? flashColor : theme.border}`,
        borderRadius: 16,
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: 8,
      }}
    >
      <div style={{ position: "relative", width: BOARD_SIZE, height: BOARD_SIZE }}>
        <Goban board={view.board} lastMove={view.lastMove} size={BOARD_SIZE} />
        {flash > 0 && view.lastFinished ? (
          <div
            style={{
              position: "absolute",
              left: 12,
              right: 12,
              bottom: 12,
              padding: "8px 10px",
              borderRadius: 10,
              background: theme.background,
              border: `2px solid ${flashColor}`,
              opacity: flash,
              fontSize: 19,
              fontWeight: 700,
              textAlign: "center",
            }}
          >
            {flashText} · {view.lastFinished.score === 0 ? "Jigo" : `${view.lastFinished.score > 0 ? "B" : "W"}+${Math.abs(view.lastFinished.score)}`}
          </div>
        ) : null}
      </div>
      <div style={{ width: BOARD_SIZE, fontSize: 18, display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ display: "flex", justifyContent: "space-between", color: theme.muted }}>
          <span>Game {view.game.index + 1}</span>
          <span>move {view.moveNumber}</span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span>
            <Dot color={getEngineColor(blackKind)} size={11} />B · <Dot color={getEngineColor(whiteKind)} size={11} />W
          </span>
          <span style={{ color: theme.muted }}>
            {view.lastVisits === null ? "opening" : (
              <>
                <Dot color={getEngineColor(moverKind)} size={11} />
                {view.lastVisits} visits
              </>
            )}
          </span>
        </div>
      </div>
    </div>
  );
};

const Replay = () => {
  const frame = useCurrentFrame();
  const videoSeconds = frame / FPS;
  const matchMs = getMatchMs(videoSeconds);
  const speed = getSpeed(videoSeconds);
  const stats = getStats(matchMs);
  const summary = stats.summary;
  const opacity = interpolate(frame, [0, 12], [0, 1], { extrapolateRight: "clamp" });
  const visitsMaximum = Math.max(stats.firstVisitsPerMove ?? 1, stats.secondVisitsPerMove ?? 1);
  const visitsRatio =
    stats.firstVisitsPerMove && stats.secondVisitsPerMove
      ? stats.firstVisitsPerMove / stats.secondVisitsPerMove
      : null;

  return (
    <AbsoluteFill
      style={{
        background: theme.background,
        color: theme.text,
        fontFamily: theme.font,
        fontVariantNumeric: "tabular-nums",
        padding: "28px 40px",
        gap: 18,
        opacity,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", height: 80 }}>
        <div>
          <div style={{ fontSize: 40, fontWeight: 700 }}>
            <Dot color={firstColor} size={20} />
            {firstName}
            <span style={{ color: theme.muted, fontWeight: 400 }}> vs </span>
            <Dot color={secondColor} size={20} />
            {secondName}
          </div>
          <div style={{ fontSize: 22, color: theme.muted, marginTop: 6 }}>
            {config.parallel} games in parallel · {config.timeMs} ms per move · same network, same
            search · replay of a recorded match
          </div>
        </div>
        <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
          <span style={{ fontSize: 30, color: theme.muted }}>{formatClock(matchMs)}</span>
          <span
            style={{
              fontSize: 28,
              fontWeight: 700,
              padding: "10px 18px",
              borderRadius: 12,
              border: `2px solid ${theme.border}`,
              background: theme.panel,
              minWidth: 150,
              textAlign: "center",
            }}
          >
            {speed < 1.05 ? "Real time" : `×${speed.toFixed(speed < ROUNDED_SPEED ? 1 : 0)}`}
          </span>
        </div>
      </div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${COLUMNS}, 1fr)`,
          gap: 20,
        }}
      >
        {Array.from({ length: config.parallel }, (_, slot) => (
          <SlotCard key={slot} slot={slot} matchMs={matchMs} speed={speed} />
        ))}
      </div>
      <div style={{ display: "flex", gap: 16, flex: 1, minHeight: 0 }}>
        <Tile label="Games played" value={`${stats.finished.length} / ${config.games}`} />
        <Tile
          label={`${firstShortName}: W / L / D`}
          value={summary ? `${summary.wins} / ${summary.losses} / ${summary.draws}` : "–"}
        />
        <Tile
          label={`${firstShortName} score`}
          value={summary ? formatPercent(summary.score) : "–"}
          detail={summary ? `95 %: ${(summary.scoreInterval[0] * 100).toFixed(1)}–${formatPercent(summary.scoreInterval[1])}` : undefined}
        />
        <Tile
          label="Elo difference"
          value={summary ? formatElo(summary.elo) : "–"}
          detail={summary ? `95 %: ${formatElo(summary.eloInterval[0])} to ${formatElo(summary.eloInterval[1])}` : undefined}
        />
        <div
          style={{
            flex: "1.7 1 0",
            padding: "14px 22px",
            background: theme.panel,
            border: `1px solid ${theme.border}`,
            borderRadius: 14,
            display: "flex",
            flexDirection: "column",
            gap: 12,
          }}
        >
          <span style={{ color: theme.muted, fontSize: 20 }}>
            Visits per move{visitsRatio ? ` · ${visitsRatio.toFixed(0)}× more for ${firstShortName}` : ""}
          </span>
          <VisitsBar name={firstName} color={firstColor} value={stats.firstVisitsPerMove} maximum={visitsMaximum} />
          <VisitsBar name={secondName} color={secondColor} value={stats.secondVisitsPerMove} maximum={visitsMaximum} />
        </div>
        <div
          style={{
            flex: "2 1 0",
            padding: "14px 16px 8px",
            background: theme.panel,
            border: `1px solid ${theme.border}`,
            borderRadius: 14,
            display: "flex",
            flexDirection: "column",
            gap: 4,
          }}
        >
          <span style={{ color: theme.muted, fontSize: 20 }}>{firstName} score after each game</span>
          <ScoreChart
            history={stats.history}
            totalGames={config.games}
            width={480}
            height={160}
            color={firstColor}
          />
        </div>
      </div>
    </AbsoluteFill>
  );
};

export { Replay };
