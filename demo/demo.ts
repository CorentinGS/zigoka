import { GoModelWorkerClient } from "../vendor/moka/src/client.ts";
import { GO_BLACK_COLOR, GO_PASS_MOVE } from "../vendor/moka/src/constants.ts";
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
import { formatMove, formatScore, getColorName, renderGoban } from "../js/goban.ts";

const MODEL = { manifestUrl: "/dist/go-model.json", weightsUrl: "/dist/go-model.bin" };
const ZIG_WORKER_URL = "/dist/worker.js";
const MOKA_WORKER_URL = "/bench/dist/moka-worker.js";

const HUMAN_COLOR = GO_BLACK_COLOR;
const DEFAULT_AUTOPLAY_DELAY_MS = 150;
const DISPLAYED_TIMING_COUNT = 12;

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

const formatMs = (value: number) => value.toFixed(2);

const getMedian = (values: number[]) => {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);

  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
};

const renderBoard = () =>
  renderGoban(board, {
    board: gameState.board,
    lastMove,
    deadMoves,
    showCoordinates: true,
    isInteractive: true,
  });

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
