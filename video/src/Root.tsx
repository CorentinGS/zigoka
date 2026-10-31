import { Composition, Series } from "remotion";
import { Intro } from "./Intro.tsx";
import { Outro } from "./Outro.tsx";
import { Replay } from "./Replay.tsx";
import { FPS, replayFrameCount } from "./timeline.ts";

const INTRO_FRAMES = 5 * FPS;
const OUTRO_FRAMES = 8 * FPS;

const ZigMatch = () => (
  <Series>
    <Series.Sequence durationInFrames={INTRO_FRAMES}>
      <Intro />
    </Series.Sequence>
    <Series.Sequence durationInFrames={replayFrameCount}>
      <Replay />
    </Series.Sequence>
    <Series.Sequence durationInFrames={OUTRO_FRAMES}>
      <Outro />
    </Series.Sequence>
  </Series>
);

const Root = () => (
  <Composition
    id="ZigMatch"
    component={ZigMatch}
    durationInFrames={INTRO_FRAMES + replayFrameCount + OUTRO_FRAMES}
    fps={FPS}
    width={1920}
    height={1080}
  />
);

export { Root };
