import { GO_BLACK_COLOR } from "../vendor/moka/src/constants.ts";
import { createGameState, playMove } from "../vendor/moka/src/game.ts";
import {
  createTasks,
  formatElo,
  isPlayerKind,
  PLAYER_KINDS,
  summarize,
  type GameResult,
  type GameTask,
  type MatchConfig,
  type PlayerKind,
} from "../match/games.ts";
import { formatMove, formatScore, renderGoban } from "./goban.ts";
import type { MatchWorkerRequest, MatchWorkerResponse } from "./match-protocol.ts";

const WORKER_URL = "/demo/dist/match-worker.js";
const OPENING_MOVES = 4;
const SEED = 1;
const DEFAULT_PARALLEL = 4;
const BATCH = 1;

const PLAYER_LABELS: Record<PlayerKind, string> = {
  "zig-reuse": "Zig + search in WASM, tree reuse",
  "zig-search": "Zig + search in WASM",
  "zig-puct": "Zig + search",
  "moka-puct": "Moka + search",
  "zig-policy": "Zig, top move",
  "moka-policy": "Moka, top move",
};

type PageState = "idle" | "loading" | "running" | "done" | "stopped" | "error";

type SideTotals = { moves: number; visits: number; thinkingMs: number };

const createTotals = (): Record<"first" | "second", SideTotals> => ({
  first: { moves: 0, visits: 0, thinkingMs: 0 },
  second: { moves: 0, visits: 0, thinkingMs: 0 },
});

type Slot = {
  worker: Worker;
  svg: SVGSVGElement;
  title: HTMLElement;
  detail: HTMLElement;
  previous: HTMLElement;
  isReady: boolean;
  task: GameTask | null;
};

const getElement = <Element extends HTMLElement>(id: string) => {
  const element = document.getElementById(id);

  if (!element) {
    throw new Error(`Missing element #${id}.`);
  }

  return element as Element;
};

const form = getElement<HTMLFormElement>("match-form");
const firstSelect = getElement<HTMLSelectElement>("first");
const secondSelect = getElement<HTMLSelectElement>("second");
const timeInput = getElement<HTMLInputElement>("time");
const gamesInput = getElement<HTMLInputElement>("games");
const parallelInput = getElement<HTMLInputElement>("parallel");
const startButton = getElement<HTMLButtonElement>("start");
const stopButton = getElement<HTMLButtonElement>("stop");
const statusElement = getElement("status");
const boardsElement = getElement("boards");
const resultsElement = getElement("results");

let config: MatchConfig | null = null;
let slots: Slot[] = [];
let tasks: GameTask[] = [];
let nextTask = 0;
let results: GameResult[] = [];
let totals = createTotals();
let startedAt = 0;

const getLabel = (kind: PlayerKind) => PLAYER_LABELS[kind];

const setPageState = (state: PageState, status: string) => {
  const isBusy = state === "loading" || state === "running";

  document.body.dataset.state = state;
  document.body.dataset.gamesDone = String(results.length);
  statusElement.textContent = status;
  startButton.disabled = isBusy;
  stopButton.disabled = !isBusy;

  for (const input of [firstSelect, secondSelect, timeInput, gamesInput, parallelInput]) {
    input.disabled = isBusy;
  }
};

const setText = (id: string, text: string) => {
  getElement(id).textContent = text;
};

const formatPercent = (value: number) => `${(value * 100).toFixed(1)} %`;

const renderSide = (name: "first" | "second") => {
  const side = totals[name];

  setText(`stat-${name}-visits`, side.moves === 0 ? "–" : (side.visits / side.moves).toFixed(1));
  setText(
    `stat-${name}-ms`,
    side.moves === 0 ? "" : `${(side.thinkingMs / side.moves).toFixed(0)} ms per move`,
  );
};

const renderStats = () => {
  if (!config) {
    return;
  }

  setText("label-record", `${getLabel(config.first)}: wins / losses / draws`);
  setText("label-first-visits", `${getLabel(config.first)}, visits per move`);
  setText("label-second-visits", `${getLabel(config.second)}, visits per move`);
  setText("label-color", `${getLabel(config.first)} plays`);
  setText("stat-games", `${results.length} / ${config.games}`);
  renderSide("first");
  renderSide("second");

  if (results.length === 0) {
    for (const id of ["stat-record", "stat-score", "stat-elo", "stat-black", "stat-length"]) {
      setText(id, "–");
    }

    setText("stat-score-range", "");
    setText("stat-elo-range", "");

    return;
  }

  const summary = summarize(config, results);
  const [low, high] = summary.scoreInterval;
  const [eloLow, eloHigh] = summary.eloInterval;

  setText("stat-record", `${summary.wins} / ${summary.losses} / ${summary.draws}`);
  setText("stat-score", formatPercent(summary.score));
  setText("stat-score-range", `95 %: ${formatPercent(low)} to ${formatPercent(high)}`);
  setText("stat-elo", formatElo(summary.elo));
  setText("stat-elo-range", `95 %: ${formatElo(eloLow)} to ${formatElo(eloHigh)}`);
  setText("stat-black", `${summary.blackWins} / ${results.length}`);
  setText("stat-length", `${summary.averageGameLength.toFixed(1)} moves`);
};

const describeResult = (result: GameResult) =>
  result.firstResult === 1 ? "win" : result.firstResult === -1 ? "loss" : "draw";

const renderResult = (result: GameResult) => {
  const row = document.createElement("tr");
  const cells = [
    String(result.index + 1),
    result.firstPlaysBlack ? "black" : "white",
    result.opening.map(formatMove).join(" "),
    String(result.moveCount),
    formatScore(result.score),
    describeResult(result),
  ];

  for (const text of cells) {
    const cell = document.createElement("td");

    cell.textContent = text;
    row.append(cell);
  }

  row.lastElementChild?.setAttribute(
    "class",
    result.firstResult === 1 ? "same" : result.firstResult === -1 ? "different" : "",
  );
  resultsElement.prepend(row);
};

const getOpeningState = (opening: number[]) =>
  opening.reduce<GoGameState>((state, move) => playMove(state, move) ?? state, createGameState());

const describePlayers = (task: GameTask) => {
  if (!config) {
    return "";
  }

  const [black, white] = task.firstPlaysBlack
    ? [config.first, config.second]
    : [config.second, config.first];

  return `Game ${task.index + 1}: ${getLabel(black)} (B) vs ${getLabel(white)} (W)`;
};

const createSlot = (matchConfig: MatchConfig): Slot => {
  const figure = document.createElement("figure");
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  const caption = document.createElement("figcaption");
  const title = document.createElement("strong");
  const detail = document.createElement("span");
  const previous = document.createElement("span");
  const worker = new Worker(WORKER_URL, { type: "module" });
  const slot: Slot = { worker, svg, title, detail, previous, isReady: false, task: null };

  figure.className = "game";
  svg.setAttribute("role", "img");
  svg.setAttribute("aria-label", "Game board");
  title.textContent = "Loading the engines…";
  caption.append(title, detail, previous);
  figure.append(svg, caption);
  boardsElement.append(figure);
  renderGoban(svg, { board: createGameState().board, lastMove: null });

  worker.addEventListener("message", (event: MessageEvent<MatchWorkerResponse>) =>
    handleResponse(slot, event.data),
  );
  worker.addEventListener("error", (event) => failMatch(event.message || "A match worker failed."));
  send(slot, { type: "initialize", config: matchConfig });

  return slot;
};

const send = (slot: Slot, request: MatchWorkerRequest) => slot.worker.postMessage(request);

const finishIfIdle = () => {
  if (slots.every((slot) => slot.task === null)) {
    const seconds = ((performance.now() - startedAt) / 1000).toFixed(0);

    terminateWorkers();
    setPageState("done", `Match over: ${results.length} games in ${seconds} s.`);
  }
};

const dispatchNext = (slot: Slot) => {
  if (nextTask >= tasks.length) {
    slot.task = null;
    slot.title.textContent = "No more games";
    slot.detail.textContent = "";
    finishIfIdle();

    return;
  }

  const task = tasks[nextTask];

  nextTask += 1;
  slot.task = task;
  slot.title.textContent = describePlayers(task);
  slot.detail.textContent = `Opening ${task.opening.map(formatMove).join(" ")}`;
  renderGoban(slot.svg, {
    board: getOpeningState(task.opening).board,
    lastMove: task.opening.at(-1) ?? null,
  });
  send(slot, { type: "game", task });
};

const recordMove = (slot: Slot, response: Extract<MatchWorkerResponse, { type: "move" }>) => {
  if (!slot.task) {
    return;
  }

  const moverIsBlack = response.moveCount % 2 === 1;
  const side = moverIsBlack === slot.task.firstPlaysBlack ? totals.first : totals.second;

  side.moves += 1;
  side.visits += response.visits;
  side.thinkingMs += response.thinkingMs;
  renderSide(moverIsBlack === slot.task.firstPlaysBlack ? "first" : "second");
  slot.detail.textContent =
    `Move ${response.moveCount}: ${formatMove(response.move)}, ` +
    `${response.visits} visits in ${response.thinkingMs.toFixed(0)} ms`;
  renderGoban(slot.svg, { board: response.board, lastMove: response.move });
};

const recordResult = (slot: Slot, result: GameResult) => {
  if (!config) {
    return;
  }

  results.push(result);
  renderResult(result);
  renderStats();
  slot.previous.textContent =
    `Previous: game ${result.index + 1}, ${formatScore(result.score)} ` +
    `(${describeResult(result)} for ${getLabel(config.first)})`;
  setPageState("running", `${results.length} of ${config.games} games played.`);
  dispatchNext(slot);
};

const handleResponse = (slot: Slot, response: MatchWorkerResponse) => {
  if (!slots.includes(slot) || !config) {
    return;
  }

  if (response.type === "error") {
    failMatch(response.error);
  } else if (response.type === "ready") {
    slot.isReady = true;
    slot.title.textContent = `Engines loaded in ${response.durationMs.toFixed(0)} ms`;

    if (slots.every((candidate) => candidate.isReady)) {
      send(slots[0], {
        type: "openings",
        count: config.games / 2,
        length: OPENING_MOVES,
        seed: SEED,
      });
    }
  } else if (response.type === "openings") {
    tasks = createTasks(response.openings);
    startedAt = performance.now();
    setPageState("running", `0 of ${config.games} games played.`);

    for (const candidate of slots) {
      dispatchNext(candidate);
    }
  } else if (response.type === "move") {
    recordMove(slot, response);
  } else {
    recordResult(slot, response.result);
  }
};

const terminateWorkers = () => {
  for (const slot of slots) {
    slot.worker.terminate();
  }
};

const failMatch = (message: string) => {
  terminateWorkers();
  slots = [];
  setPageState("error", `The match stopped: ${message}`);
};

const readConfig = (): MatchConfig | string => {
  const games = Number(gamesInput.value);
  const timeMs = Number(timeInput.value);
  const parallel = Number(parallelInput.value);

  if (!isPlayerKind(firstSelect.value) || !isPlayerKind(secondSelect.value)) {
    return "Choose both players.";
  }

  if (!Number.isInteger(games) || games < 2 || games % 2 !== 0) {
    return "The number of games must be even and at least 2.";
  }

  if (!(timeMs > 0)) {
    return "The time per move must be positive.";
  }

  if (!Number.isInteger(parallel) || parallel < 1) {
    return "The number of games at once must be at least 1.";
  }

  return {
    first: firstSelect.value,
    second: secondSelect.value,
    games,
    timeMs,
    batch: BATCH,
    openingMoves: OPENING_MOVES,
    parallel: Math.min(parallel, games),
    seed: SEED,
  };
};

const startMatch = () => {
  const nextConfig = readConfig();

  if (typeof nextConfig === "string") {
    setPageState("idle", nextConfig);

    return;
  }

  terminateWorkers();
  config = nextConfig;
  tasks = [];
  nextTask = 0;
  results = [];
  totals = createTotals();
  boardsElement.replaceChildren();
  resultsElement.replaceChildren();
  setPageState("loading", `Loading the engines in ${config.parallel} workers…`);
  slots = Array.from({ length: config.parallel }, () => createSlot(nextConfig));
  renderStats();
};

const stopMatch = () => {
  terminateWorkers();
  slots = [];
  setPageState("stopped", `Match stopped after ${results.length} games.`);
};

const fillPlayerSelect = (select: HTMLSelectElement, selected: PlayerKind) => {
  for (const kind of PLAYER_KINDS) {
    const option = document.createElement("option");

    option.value = kind;
    option.textContent = getLabel(kind);
    option.selected = kind === selected;
    select.append(option);
  }
};

const applyQuery = () => {
  const parameters = new URLSearchParams(window.location.search);
  const first = parameters.get("first");
  const second = parameters.get("second");

  fillPlayerSelect(firstSelect, isPlayerKind(first) ? first : "zig-reuse");
  fillPlayerSelect(secondSelect, isPlayerKind(second) ? second : "moka-puct");
  parallelInput.value = String(Math.min(DEFAULT_PARALLEL, navigator.hardwareConcurrency || 1));

  for (const [name, input] of [
    ["time", timeInput],
    ["games", gamesInput],
    ["parallel", parallelInput],
  ] as const) {
    const value = parameters.get(name);

    if (value !== null) {
      input.value = value;
    }
  }
};

form.addEventListener("submit", (event) => {
  event.preventDefault();
  startMatch();
});
stopButton.addEventListener("click", stopMatch);

applyQuery();
setPageState("idle", "Choose the players and start the match.");
