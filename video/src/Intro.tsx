import { AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig } from "remotion";
import { theme } from "./theme.ts";

const ZIG_INFERENCE_MS = 0.265;
const MOKA_INFERENCE_MS = 5.66;

const fadeUp = (frame: number, start: number) => ({
  opacity: interpolate(frame, [start, start + 15], [0, 1], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  }),
  transform: `translateY(${interpolate(frame, [start, start + 15], [24, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  })}px)`,
});

const EngineTiming = ({
  name,
  detail,
  milliseconds,
  color,
  progress,
}: {
  name: string;
  detail: string;
  milliseconds: number;
  color: string;
  progress: number;
}) => (
  <div
    style={{
      width: 560,
      padding: "28px 34px",
      background: theme.panel,
      border: `1px solid ${theme.border}`,
      borderRadius: 18,
      display: "flex",
      flexDirection: "column",
      gap: 12,
    }}
  >
    <div style={{ fontSize: 30, fontWeight: 700 }}>
      <span
        style={{
          display: "inline-block",
          width: 18,
          height: 18,
          borderRadius: 9,
          background: color,
          marginRight: 12,
        }}
      />
      {name}
    </div>
    <div style={{ fontSize: 22, color: theme.muted }}>{detail}</div>
    <div style={{ fontSize: 76, fontWeight: 800 }}>
      {(milliseconds * progress).toFixed(milliseconds < 1 ? 3 : 2)} ms
    </div>
    <div style={{ height: 16, background: theme.background, borderRadius: 8 }}>
      <div
        style={{
          width: `${(milliseconds / MOKA_INFERENCE_MS) * 100 * progress}%`,
          minWidth: 6,
          height: "100%",
          background: color,
          borderRadius: 8,
        }}
      />
    </div>
  </div>
);

const Intro = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();
  const progress = spring({ frame: frame - 30, fps, config: { damping: 200 } });
  const exit = interpolate(frame, [durationInFrames - 12, durationInFrames], [1, 0], {
    extrapolateLeft: "clamp",
  });

  return (
    <AbsoluteFill
      style={{
        background: theme.background,
        color: theme.text,
        fontFamily: theme.font,
        fontVariantNumeric: "tabular-nums",
        justifyContent: "center",
        alignItems: "center",
        gap: 44,
        opacity: exit,
      }}
    >
      <div style={{ textAlign: "center", ...fadeUp(frame, 0) }}>
        <div style={{ fontSize: 68, fontWeight: 800 }}>Moka's 9×9 Go network, rewritten in Zig</div>
        <div style={{ fontSize: 30, color: theme.muted, marginTop: 14 }}>
          WebAssembly SIMD · no allocation · same outputs as Moka, parity-tested
        </div>
      </div>
      <div style={{ display: "flex", gap: 40, ...fadeUp(frame, 20) }}>
        <EngineTiming
          name="Zig / WebAssembly"
          detail="one position, Chromium Web Worker"
          milliseconds={ZIG_INFERENCE_MS}
          color={theme.zig}
          progress={progress}
        />
        <EngineTiming
          name="Moka (TypeScript)"
          detail="one position, Chromium Web Worker"
          milliseconds={MOKA_INFERENCE_MS}
          color={theme.moka}
          progress={progress}
        />
      </div>
      <div style={{ fontSize: 36, textAlign: "center", ...fadeUp(frame, 75) }}>
        <strong>{(MOKA_INFERENCE_MS / ZIG_INFERENCE_MS).toFixed(0)}× faster.</strong>{" "}
        <span style={{ color: theme.muted }}>
          What does it buy? Same search, 100 ms per move, 10 games at once.
        </span>
      </div>
    </AbsoluteFill>
  );
};

export { Intro };
