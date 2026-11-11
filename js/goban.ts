import {
  GO_BLACK_COLOR,
  GO_BOARD_AREA,
  GO_BOARD_SIZE,
  GO_EMPTY_COLOR,
  GO_PASS_MOVE,
} from "../vendor/moka/src/constants.ts";

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const CELL_SIZE = 40;
const COORDINATE_MARGIN = 30;
const PLAIN_MARGIN = 24;
const STONE_RADIUS = 18;
// Star points of the two board sizes the bundles are built for (the 3-3 / 4-4 points and the center).
const STAR_POINTS =
  (GO_BOARD_SIZE as number) === 19 ? [60, 66, 72, 174, 180, 186, 288, 294, 300] : [20, 24, 40, 56, 60];
// Go columns skip the letter I.
const COLUMN_LABELS = "ABCDEFGHJKLMNOPQRST";

type GobanView = {
  board: ArrayLike<number>;
  lastMove: number | null;
  deadMoves?: number[];
  showCoordinates?: boolean;
  isInteractive?: boolean;
  hoverMove?: number | null;
  hoverColor?: number;
};

const getColorName = (color: number) => (color === GO_BLACK_COLOR ? "black" : "white");

const formatMove = (move: number) =>
  move === GO_PASS_MOVE
    ? "pass"
    : `${COLUMN_LABELS[move % GO_BOARD_SIZE]}${GO_BOARD_SIZE - Math.floor(move / GO_BOARD_SIZE)}`;

const formatScore = (score: number) => {
  if (score === 0) {
    return "Jigo";
  }

  return `${score > 0 ? "B" : "W"}+${Math.abs(score)}`;
};

const createSvgElement = (name: string, attributes: Record<string, string | number>) => {
  const element = document.createElementNS(SVG_NAMESPACE, name);

  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }

  return element;
};

const renderGoban = (svg: SVGSVGElement, view: GobanView) => {
  const margin = view.showCoordinates ? COORDINATE_MARGIN : PLAIN_MARGIN;
  const size = 2 * margin + (GO_BOARD_SIZE - 1) * CELL_SIZE;
  const lineEnd = margin + (GO_BOARD_SIZE - 1) * CELL_SIZE;
  const deadMoves = view.deadMoves ?? [];
  const getPointCenter = (move: number) => ({
    x: margin + (move % GO_BOARD_SIZE) * CELL_SIZE,
    y: margin + Math.floor(move / GO_BOARD_SIZE) * CELL_SIZE,
  });

  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
  svg.append(createSvgElement("rect", { width: size, height: size, rx: 6, fill: "var(--board)" }));

  for (let index = 0; index < GO_BOARD_SIZE; index += 1) {
    const offset = margin + index * CELL_SIZE;
    const lineStyle = { stroke: "var(--line)", "stroke-width": 1 };

    svg.append(
      createSvgElement("line", { x1: margin, y1: offset, x2: lineEnd, y2: offset, ...lineStyle }),
      createSvgElement("line", { x1: offset, y1: margin, x2: offset, y2: lineEnd, ...lineStyle }),
    );
  }

  if (view.showCoordinates) {
    for (let index = 0; index < GO_BOARD_SIZE; index += 1) {
      const offset = margin + index * CELL_SIZE;
      const labelStyle = {
        fill: "var(--line)",
        "font-size": 12,
        "text-anchor": "middle",
        "dominant-baseline": "central",
      };
      const columnLabel = COLUMN_LABELS[index];
      const rowLabel = String(GO_BOARD_SIZE - index);

      for (const [x, y, text] of [
        [offset, margin / 2, columnLabel],
        [offset, size - margin / 2, columnLabel],
        [margin / 2, offset, rowLabel],
        [size - margin / 2, offset, rowLabel],
      ] as const) {
        const label = createSvgElement("text", { x, y, ...labelStyle });

        label.textContent = text;
        svg.append(label);
      }
    }
  }

  for (const move of STAR_POINTS) {
    const { x, y } = getPointCenter(move);

    svg.append(createSvgElement("circle", { cx: x, cy: y, r: 3, fill: "var(--line)" }));
  }

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    const color = view.board[move];
    const { x, y } = getPointCenter(move);

    if (color !== GO_EMPTY_COLOR) {
      const isBlack = color === GO_BLACK_COLOR;
      const isDead = deadMoves.includes(move);

      svg.append(
        createSvgElement("circle", {
          cx: x,
          cy: y,
          r: STONE_RADIUS,
          fill: isBlack ? "#111" : "#f4f4f0",
          stroke: "#111",
          "stroke-width": 1,
          opacity: isDead ? 0.35 : 1,
          "data-stone": getColorName(color),
          ...(isDead ? { "data-dead": "true" } : {}),
        }),
      );

      if (move === view.lastMove) {
        svg.append(
          createSvgElement("circle", {
            cx: x,
            cy: y,
            r: 6,
            fill: "none",
            stroke: isBlack ? "#f4f4f0" : "#111",
            "stroke-width": 2,
          }),
        );
      }
    }
    if (view.isInteractive && view.hoverMove === move && color === GO_EMPTY_COLOR) {
      const hoverColor = view.hoverColor ?? GO_BLACK_COLOR;
      const isHoverBlack = hoverColor === GO_BLACK_COLOR;

      svg.append(
        createSvgElement("circle", {
          cx: x,
          cy: y,
          r: STONE_RADIUS,
          fill: isHoverBlack ? "#111" : "#f4f4f0",
          stroke: "#111",
          "stroke-width": 1,
          opacity: 0.5,
          "data-ghost": getColorName(hoverColor),
        }),
      );
    }

    if (view.isInteractive) {
      svg.append(
        createSvgElement("circle", {
          cx: x,
          cy: y,
          r: CELL_SIZE / 2,
          fill: "transparent",
          "data-move": move,
        }),
      );
    }
  }
};

export { formatMove, formatScore, getColorName, renderGoban };
export type { GobanView };
