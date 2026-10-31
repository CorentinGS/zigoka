import { formatMove, formatScore, renderGoban } from "../js/goban.ts";
import { GO_BLACK_COLOR, GO_PASS_MOVE, GO_WHITE_COLOR } from "../vendor/moka/src/constants.ts";
import {
  createGameState,
  getAreaScore,
  getAutomaticallyDeadMoves,
  isGameOver,
  playMove,
  removeDeadStones,
} from "../vendor/moka/src/game.ts";
import { LEVEL_BY_ID, type LevelId, type LevelSpec, type SiteWorkerResponse } from "./protocol.ts";

type PageState = "loading" | "human" | "thinking" | "over" | "error";

type PlayedMove = { move: number; color: number };

const getElement = <Element extends HTMLElement | SVGElement>(id: string) => {
  const element = document.getElementById(id);

  if (!element) {
    throw new Error(`Missing element #${id}.`);
  }

  return element as Element;
};

const board = getElement<SVGSVGElement>("board");
const statusElement = getElement<HTMLElement>("status");
const movesElement = getElement<HTMLOListElement>("moves");
const levelSelect = getElement<HTMLSelectElement>("level");
const colorSelect = getElement<HTMLSelectElement>("color");
const passButton = getElement<HTMLButtonElement>("pass");
const resignButton = getElement<HTMLButtonElement>("resign");
const undoButton = getElement<HTMLButtonElement>("undo");
const newGameButton = getElement<HTMLButtonElement>("new-game");

/** Labels for the two stone colors, as used in the status line. */
const getColorLabel = (color: number) => (color === GO_BLACK_COLOR ? "black" : "white");

// history[k] is the state before move k (history[0] is the empty board), so the
// current state is always the last snapshot and moves.length === history.length - 1.
let history: GoGameState[] = [createGameState()];
let moves: PlayedMove[] = [];
let deadMoves: number[] = [];
let humanColor = GO_BLACK_COLOR;
let pageState: PageState = "loading";
let hoverMove: number | null = null;
let pendingRequestId: number | null = null;
let nextRequestId = 1;

const currentState = () => history[history.length - 1];

const readLevel = (): LevelSpec => {
  const level = LEVEL_BY_ID[levelSelect.value as LevelId];

  return level ?? LEVEL_BY_ID.beginner;
};

/** Latest snapshot where the human is to play and at least one move was played after it. */
const findUndoIndex = () => {
  for (let index = history.length - 2; index >= 0; index -= 1) {
    if (history[index].nextColor === humanColor) {
      return index;
    }
  }

  return null;
};

const setPageState = (state: PageState, status: string) => {
  pageState = state;
  document.body.dataset.state = state;
  statusElement.textContent = status;
  // The board's hit targets depend on pageState ("human" only), so every
  // transition must re-render; otherwise the board stays inert after loading
  // or after the engine replies.
  renderBoard();

  const isHumanTurn = state === "human";

  passButton.disabled = !isHumanTurn;
  resignButton.disabled = !isHumanTurn;
  undoButton.disabled = !isHumanTurn || findUndoIndex() === null;
  newGameButton.disabled = state === "loading" || state === "error";
};

const renderBoard = () =>
  renderGoban(board, {
    board: currentState().board,
    lastMove:
      moves.length > 0 && moves[moves.length - 1].move !== GO_PASS_MOVE
        ? moves[moves.length - 1].move
        : null,
    deadMoves,
    showCoordinates: true,
    isInteractive: pageState === "human",
    hoverMove,
    hoverColor: currentState().nextColor,
  });

const renderMoves = () => {
  movesElement.replaceChildren(
    ...moves.map((played) => {
      const item = document.createElement("li");

      item.textContent = `${getColorLabel(played.color)} ${
        played.move === GO_PASS_MOVE ? "pass" : formatMove(played.move)
      }`;

      return item;
    }),
  );
  movesElement.scrollTop = movesElement.scrollHeight;
};

const render = () => {
  renderBoard();
  renderMoves();
};

const finishGame = () => {
  const current = currentState();

  deadMoves = getAutomaticallyDeadMoves(current);
  const score = getAreaScore(removeDeadStones(current, deadMoves));
  setPageState("over", `Game over: ${formatScore(score)} (komi 7)`);
};

const requestEngineMove = () => {
  const current = currentState();
  const requestId = nextRequestId;

  nextRequestId += 1;
  pendingRequestId = requestId;
  setPageState("thinking", `Thinking (${getColorLabel(current.nextColor)})…`);
  worker.postMessage({
    type: "move",
    requestId,
    // postMessage structured-clones, so the worker can never mutate our snapshot.
    state: current,
    level: levelSelect.value as LevelId,
  });
};

/** Dispatches the turn after every applied move; `suffix` carries the visits note. */
const beginTurn = (suffix = "") => {
  const current = currentState();

  if (isGameOver(current)) {
    finishGame();

    return;
  }

  if (current.nextColor !== humanColor) {
    requestEngineMove();

    return;
  }

  setPageState("human", `Your move (${getColorLabel(current.nextColor)}).${suffix}`);
};

const recordMove = (move: number, color: number, nextState: GoGameState) => {
  moves.push({ move, color });
  history.push(nextState);
};

const applyEngineMove = (move: number, visits: number) => {
  const current = currentState();
  const nextState = playMove(current, move);

  if (!nextState) {
    setPageState("error", `The engine chose an illegal move (${formatMove(move)}).`);

    return;
  }

  recordMove(move, current.nextColor, nextState);
  render();

  const suffix = readLevel().usesSearch ? ` (${visits} visits)` : "";

  beginTurn(suffix);
};

const playHumanMove = (move: number) => {
  if (pageState !== "human") {
    return;
  }

  const current = currentState();
  const nextState = playMove(current, move);

  if (!nextState) {
    // Illegal clicks (occupied point, ko, suicide) are ignored; the status keeps
    // the exact "Your move (…)." text the page contract promises.
    return;
  }

  recordMove(move, current.nextColor, nextState);
  render();
  beginTurn();
};

const resign = () => {
  if (pageState !== "human") {
    return;
  }

  // Standard notation credits the winner: the opponent of whoever resigned.
  const winnerLetter = humanColor === GO_BLACK_COLOR ? "W" : "B";

  pendingRequestId = null;
  setPageState("over", `Game over: ${winnerLetter}+R (resignation)`);
};

const undo = () => {
  if (pageState !== "human") {
    return;
  }

  const undoIndex = findUndoIndex();

  if (undoIndex === null) {
    return;
  }

  history = history.slice(0, undoIndex + 1);
  moves = moves.slice(0, undoIndex);
  deadMoves = [];
  hoverMove = null;
  pendingRequestId = null;
  render();
  beginTurn();
};

const startGame = () => {
  history = [createGameState()];
  moves = [];
  deadMoves = [];
  hoverMove = null;
  pendingRequestId = null;
  humanColor = colorSelect.value === "white" ? GO_WHITE_COLOR : GO_BLACK_COLOR;
  render();
  worker.postMessage({ type: "new-game" });
  beginTurn();
};

const worker = new Worker(new URL("./engine-worker.js", import.meta.url), { type: "module" });

worker.addEventListener("message", (event: MessageEvent<SiteWorkerResponse>) => {
  const response = event.data;

  if (response.type === "ready") {
    startGame();

    return;
  }

  if (response.type === "error") {
    pendingRequestId = null;
    setPageState("error", `Engine error: ${response.error}`);

    return;
  }

  // A reply from a previous game (or before an undo) must not move a stone.
  if (pendingRequestId !== response.requestId) {
    return;
  }

  pendingRequestId = null;
  applyEngineMove(response.move, response.visits);
});

worker.addEventListener("error", () => {
  pendingRequestId = null;
  setPageState("error", "Could not load the engine (engine-worker.js).");
});

board.addEventListener("click", (event) => {
  const target = event.target;

  if (target instanceof SVGElement && target.dataset.move !== undefined) {
    playHumanMove(Number(target.dataset.move));
  }
});

board.addEventListener("pointermove", (event) => {
  if (event.pointerType !== "mouse") {
    return;
  }

  const target = event.target;
  const move =
    target instanceof SVGElement && target.dataset.move !== undefined
      ? Number(target.dataset.move)
      : null;

  if (move === hoverMove) {
    return;
  }

  hoverMove = move;

  if (pageState === "human") {
    renderBoard();
  }
});

board.addEventListener("pointerleave", () => {
  if (hoverMove === null) {
    return;
  }

  hoverMove = null;

  if (pageState === "human") {
    renderBoard();
  }
});

passButton.addEventListener("click", () => playHumanMove(GO_PASS_MOVE));
resignButton.addEventListener("click", resign);
undoButton.addEventListener("click", undo);
newGameButton.addEventListener("click", startGame);

render();
setPageState("loading", "Loading the engine…");
worker.postMessage({ type: "init" });
