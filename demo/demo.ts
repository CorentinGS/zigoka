import { GoModelWorkerClient } from "../vendor/moka/src/client.ts";
import {
  GO_BLACK_COLOR,
  GO_BOARD_AREA,
  GO_BOARD_SIZE,
  GO_EMPTY_COLOR,
  GO_PASS_MOVE,
  GO_WHITE_COLOR,
} from "../vendor/moka/src/constants.ts";
import {
  createGameState,
  encodeStudentFeatures,
  getAreaScore,
  getAutomaticallyDeadMoves,
  isGameOver,
  playMove,
  removeDeadStones,
  selectHighestLegalMove,
} from "../vendor/moka/src/game.ts";

const MODEL = { manifestUrl: "/dist/go-model.json", weightsUrl: "/dist/go-model.bin" };
const ZIG_WORKER_URL = "/dist/worker.js";
const MOKA_WORKER_URL = "/bench/dist/moka-worker.js";

const HUMAN_COLOR = GO_BLACK_COLOR;
const DEFAULT_AUTOPLAY_DELAY_MS = 150;
const DISPLAYED_TIMING_COUNT = 12;

const SVG_NAMESPACE = "http://www.w3.org/2000/svg";
const CELL_SIZE = 40;
const BOARD_MARGIN = 30;
const STONE_RADIUS = 18;
const STAR_POINTS = [20, 24, 40, 56, 60];
const COLUMN_LABELS = "ABCDEFGHJ";

type PageState = "loading" | "human" | "thinking" | "over" | "error";

type MoveTiming = {
  moveNumber: number;
  move: number;
  zigMs: number;
  mokaMs: number;
  isSameMove: boolean;
};

type Engines = { zig: GoModelWorkerClient; moka: GoModelWorkerClient };

const getElement = <Element extends HTMLElement | SVGElement>(id: string) => {
  const element = document.getElementById(id);

  if (!element) {
    throw new Error(`Missing element #${id}.`);
  }

  return element as Element;
};

const board = getElement<SVGSVGElement>("board");
const passButton = getElement<HTMLButtonElement>("pass");
const newGameButton = getElement<HTMLButtonElement>("new-game");
const autoplayInput = getElement<HTMLInputElement>("autoplay");
const statusElement = getElement<HTMLElement>("status");
const timingsElement = getElement<HTMLElement>("timings");
const winProbabilityElement = getElement<HTMLElement>("win-probability");
const initializationElement = getElement<HTMLElement>("initialization");

const readAutoplayDelay = () => {
  const parameter = new URLSearchParams(window.location.search).get("delay");
  const delay = Number(parameter);

  return parameter !== null && Number.isFinite(delay) && delay >= 0
    ? delay
    : DEFAULT_AUTOPLAY_DELAY_MS;
};

const autoplayDelayMs = readAutoplayDelay();

let gameState = createGameState();
let lastMove: number | null = null;
let deadMoves: number[] = [];
let timings: MoveTiming[] = [];
let pageState: PageState = "loading";
let gameNumber = 0;

const getColorName = (color: number) => (color === GO_BLACK_COLOR ? "black" : "white");

const formatMove = (move: number) =>
  move === GO_PASS_MOVE
    ? "pass"
    : `${COLUMN_LABELS[move % GO_BOARD_SIZE]}${GO_BOARD_SIZE - Math.floor(move / GO_BOARD_SIZE)}`;

const formatMs = (value: number) => value.toFixed(2);

const getMedian = (values: number[]) => {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);

  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
};

const createSvgElement = (name: string, attributes: Record<string, string | number>) => {
  const element = document.createElementNS(SVG_NAMESPACE, name);

  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }

  return element;
};

const getPointCenter = (move: number) => ({
  x: BOARD_MARGIN + (move % GO_BOARD_SIZE) * CELL_SIZE,
  y: BOARD_MARGIN + Math.floor(move / GO_BOARD_SIZE) * CELL_SIZE,
});

const renderBoard = () => {
  const size = 2 * BOARD_MARGIN + (GO_BOARD_SIZE - 1) * CELL_SIZE;
  const lineEnd = BOARD_MARGIN + (GO_BOARD_SIZE - 1) * CELL_SIZE;

  board.replaceChildren();
  board.setAttribute("viewBox", `0 0 ${size} ${size}`);
  board.append(
    createSvgElement("rect", { width: size, height: size, rx: 6, fill: "var(--board)" }),
  );

  for (let index = 0; index < GO_BOARD_SIZE; index += 1) {
    const offset = BOARD_MARGIN + index * CELL_SIZE;
    const lineStyle = { stroke: "var(--line)", "stroke-width": 1 };

    board.append(
      createSvgElement("line", { x1: BOARD_MARGIN, y1: offset, x2: lineEnd, y2: offset, ...lineStyle }),
      createSvgElement("line", { x1: offset, y1: BOARD_MARGIN, x2: offset, y2: lineEnd, ...lineStyle }),
    );
  }

  for (let index = 0; index < GO_BOARD_SIZE; index += 1) {
    const offset = BOARD_MARGIN + index * CELL_SIZE;
    const labelStyle = {
      fill: "var(--line)",
      "font-size": 12,
      "text-anchor": "middle",
      "dominant-baseline": "central",
    };
    const columnLabel = COLUMN_LABELS[index];
    const rowLabel = String(GO_BOARD_SIZE - index);

    for (const [x, y, text] of [
      [offset, BOARD_MARGIN / 2, columnLabel],
      [offset, size - BOARD_MARGIN / 2, columnLabel],
      [BOARD_MARGIN / 2, offset, rowLabel],
      [size - BOARD_MARGIN / 2, offset, rowLabel],
    ] as const) {
      const label = createSvgElement("text", { x, y, ...labelStyle });

      label.textContent = text;
      board.append(label);
    }
  }

  for (const move of STAR_POINTS) {
    const { x, y } = getPointCenter(move);

    board.append(createSvgElement("circle", { cx: x, cy: y, r: 3, fill: "var(--line)" }));
  }

  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    const color = gameState.board[move];
    const { x, y } = getPointCenter(move);

    if (color !== GO_EMPTY_COLOR) {
      const isBlack = color === GO_BLACK_COLOR;

      board.append(
        createSvgElement("circle", {
          cx: x,
          cy: y,
          r: STONE_RADIUS,
          fill: isBlack ? "#111" : "#f4f4f0",
          stroke: "#111",
          "stroke-width": 1,
          opacity: deadMoves.includes(move) ? 0.35 : 1,
          "data-stone": getColorName(color),
        }),
      );

      if (move === lastMove) {
        board.append(
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

    board.append(
      createSvgElement("circle", {
        cx: x,
        cy: y,
        r: CELL_SIZE / 2,
        fill: "transparent",
        "data-move": move,
      }),
    );
  }
};

const renderTimings = () => {
  const zigTimes = timings.map((timing) => timing.zigMs);
  const mokaTimes = timings.map((timing) => timing.mokaMs);
  const sameCount = timings.filter((timing) => timing.isSameMove).length;

  getElement("zig-median").textContent = timings.length ? `${formatMs(getMedian(zigTimes))} ms` : "–";
  getElement("moka-median").textContent = timings.length ? `${formatMs(getMedian(mokaTimes))} ms` : "–";
  getElement("ratio").textContent = timings.length
    ? `${(getMedian(mokaTimes) / getMedian(zigTimes)).toFixed(1)}×`
    : "–";
  getElement("same-count").textContent = timings.length ? `${sameCount} / ${timings.length}` : "–";

  timingsElement.replaceChildren(
    ...timings
      .slice(-DISPLAYED_TIMING_COUNT)
      .reverse()
      .map((timing) => {
        const row = document.createElement("tr");
        const cells = [
          String(timing.moveNumber),
          formatMove(timing.move),
          formatMs(timing.zigMs),
          formatMs(timing.mokaMs),
          timing.isSameMove ? "yes" : "no",
        ];

        for (const text of cells) {
          const cell = document.createElement("td");

          cell.textContent = text;
          row.append(cell);
        }

        row.lastElementChild?.setAttribute("class", timing.isSameMove ? "same" : "different");

        return row;
      }),
  );
};

const setPageState = (state: PageState, status: string) => {
  pageState = state;
  document.body.dataset.state = state;
  document.body.dataset.moveCount = String(gameState.moveCount);
  statusElement.textContent = status;

  const isIdle = state === "human" || state === "over";

  passButton.disabled = state !== "human";
  newGameButton.disabled = !isIdle;
  autoplayInput.disabled = state === "loading" || state === "error" || state === "over";
};

const formatScore = (score: number) => {
  if (score === 0) {
    return "Jigo";
  }

  return `${score > 0 ? "B" : "W"}+${Math.abs(score)}`;
};

const finishGame = () => {
  deadMoves = getAutomaticallyDeadMoves(gameState);
  const score = getAreaScore(removeDeadStones(gameState, deadMoves));

  renderBoard();
  setPageState(
    "over",
    `Game over: ${formatScore(score)} (${deadMoves.length} dead stones removed, komi 7).`,
  );
};

const timeInference = async (client: GoModelWorkerClient, features: Float32Array) => {
  const startTime = performance.now();
  const inference = await client.infer(features);

  return { inference, durationMs: performance.now() - startTime };
};

const waitForTurn = (message = "") => {
  if (isGameOver(gameState)) {
    finishGame();
  } else if (autoplayInput.checked || gameState.nextColor !== HUMAN_COLOR) {
    const currentGame = gameNumber;

    window.setTimeout(() => {
      if (currentGame === gameNumber) {
        void playEngineMove();
      }
    }, autoplayInput.checked ? autoplayDelayMs : 0);
  } else {
    setPageState("human", `${message}Your move (black).`);
  }
};

let engines: Engines | null = null;
let isEngineThinking = false;

const playEngineMove = async () => {
  if (!engines || isEngineThinking) {
    return;
  }

  const currentGame = gameNumber;
  const color = gameState.nextColor;

  isEngineThinking = true;
  setPageState("thinking", `Zig is thinking (${getColorName(color)})…`);

  try {
    const zig = await timeInference(engines.zig, encodeStudentFeatures(gameState));
    const moka = await timeInference(engines.moka, encodeStudentFeatures(gameState));

    if (currentGame !== gameNumber) {
      return;
    }

    const move = selectHighestLegalMove(gameState, zig.inference.policyLogits);
    const mokaMove = selectHighestLegalMove(gameState, moka.inference.policyLogits);
    const nextState = playMove(gameState, move);

    if (!nextState) {
      throw new Error(`The engine chose an illegal move (${formatMove(move)}).`);
    }

    timings.push({
      moveNumber: gameState.moveCount + 1,
      move,
      zigMs: zig.durationMs,
      mokaMs: moka.durationMs,
      isSameMove: move === mokaMove,
    });
    gameState = nextState;
    lastMove = move === GO_PASS_MOVE ? null : move;
    winProbabilityElement.textContent =
      `Zig's estimate after its last move: ${getColorName(color)} wins with probability ` +
      `${(((zig.inference.value + 1) / 2) * 100).toFixed(0)} %.`;
    renderBoard();
    renderTimings();
    waitForTurn(`Zig played ${formatMove(move)}. `);
  } catch (error) {
    setPageState("error", error instanceof Error ? error.message : String(error));
  } finally {
    isEngineThinking = false;
  }
};

const playHumanMove = (move: number) => {
  if (pageState !== "human" || autoplayInput.checked) {
    return;
  }

  const nextState = playMove(gameState, move);

  if (!nextState) {
    setPageState("human", `Illegal move (${formatMove(move)}). Your move (black).`);

    return;
  }

  gameState = nextState;
  lastMove = move === GO_PASS_MOVE ? null : move;
  renderBoard();
  waitForTurn();
};

const startNewGame = () => {
  gameNumber += 1;
  gameState = createGameState();
  lastMove = null;
  deadMoves = [];
  timings = [];
  winProbabilityElement.textContent = "";
  renderBoard();
  renderTimings();
  waitForTurn();
};

const initializeEngine = async (workerUrl: string) => {
  const client = new GoModelWorkerClient(new Worker(workerUrl, { type: "module" }));
  const startTime = performance.now();

  try {
    await client.initialize(MODEL);
  } catch (error) {
    client.dispose();
    throw error;
  }

  return { client, durationMs: performance.now() - startTime };
};

board.addEventListener("click", (event) => {
  const target = event.target;

  if (target instanceof SVGElement && target.dataset.move !== undefined) {
    playHumanMove(Number(target.dataset.move));
  }
});
passButton.addEventListener("click", () => playHumanMove(GO_PASS_MOVE));
newGameButton.addEventListener("click", startNewGame);
autoplayInput.addEventListener("change", () => {
  if (autoplayInput.checked && pageState === "human") {
    waitForTurn();
  }
});

const start = async () => {
  renderBoard();
  setPageState("loading", "Loading the engines…");

  try {
    const zig = await initializeEngine(ZIG_WORKER_URL);
    const moka = await initializeEngine(MOKA_WORKER_URL);

    engines = { zig: zig.client, moka: moka.client };
    initializationElement.textContent =
      `Initialization: Zig ${formatMs(zig.durationMs)} ms, Moka ${formatMs(moka.durationMs)} ms ` +
      "(download, digest check and weight loading).";
  } catch (error) {
    setPageState(
      "error",
      `The engines could not start: ${error instanceof Error ? error.message : String(error)}`,
    );

    return;
  }

  waitForTurn();
};

void start();
