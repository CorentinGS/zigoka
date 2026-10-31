import { theme } from "./theme.ts";

const BOARD_SIZE = 9;
const CELL = 40;
const MARGIN = 24;
const STONE_RADIUS = 18.5;
const STAR_POINTS = [20, 24, 40, 56, 60];
const SIZE = 2 * MARGIN + (BOARD_SIZE - 1) * CELL;
const LINE_END = MARGIN + (BOARD_SIZE - 1) * CELL;

const getCenter = (point: number) => ({
  x: MARGIN + (point % BOARD_SIZE) * CELL,
  y: MARGIN + Math.floor(point / BOARD_SIZE) * CELL,
});

const Goban = ({
  board,
  lastMove,
  size,
}: {
  board: ArrayLike<number>;
  lastMove: number | null;
  size: number;
}) => {
  const lines = Array.from({ length: BOARD_SIZE }, (_, index) => MARGIN + index * CELL);
  const stones = Array.from({ length: BOARD_SIZE * BOARD_SIZE }, (_, point) => point).filter(
    (point) => board[point] !== 0,
  );

  return (
    <svg width={size} height={size} viewBox={`0 0 ${SIZE} ${SIZE}`}>
      <rect width={SIZE} height={SIZE} rx={8} fill={theme.board} />
      {lines.map((offset) => (
        <g key={offset} stroke={theme.line} strokeWidth={1.2}>
          <line x1={MARGIN} y1={offset} x2={LINE_END} y2={offset} />
          <line x1={offset} y1={MARGIN} x2={offset} y2={LINE_END} />
        </g>
      ))}
      {STAR_POINTS.map((point) => {
        const { x, y } = getCenter(point);

        return <circle key={point} cx={x} cy={y} r={3.5} fill={theme.line} />;
      })}
      {stones.map((point) => {
        const { x, y } = getCenter(point);
        const isBlack = board[point] === 1;

        return (
          <g key={point}>
            <circle
              cx={x}
              cy={y}
              r={STONE_RADIUS}
              fill={isBlack ? "#141414" : "#f2f1ec"}
              stroke="#111"
              strokeWidth={1}
            />
            {point === lastMove ? (
              <circle
                cx={x}
                cy={y}
                r={7}
                fill="none"
                stroke={isBlack ? "#f2f1ec" : "#141414"}
                strokeWidth={2.5}
              />
            ) : null}
          </g>
        );
      })}
    </svg>
  );
};

export { Goban };
