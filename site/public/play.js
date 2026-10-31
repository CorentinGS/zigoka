// vendor/moka/src/constants.ts
var GO_BLACK_COLOR = 1;
var GO_BOARD_AREA = 81;
var GO_BOARD_SIZE = 9;
var GO_EMPTY_COLOR = 0;
var GO_GAME_OVER_PASS_COUNT = 2;
var GO_KOMI_POINTS = 7;
var GO_MAXIMUM_MOVE_COUNT = 120;
var GO_NO_KO_MOVE = -1;
var GO_PASS_ALIVE_MAXIMUM_INTERNAL_POINT_COUNT = 1;
var GO_PASS_ALIVE_REQUIRED_VITAL_REGION_COUNT = 2;
var GO_PASS_MOVE = 81;
var GO_WHITE_COLOR = -1;

// js/goban.ts
var SVG_NAMESPACE = "http://www.w3.org/2000/svg";
var CELL_SIZE = 40;
var COORDINATE_MARGIN = 30;
var PLAIN_MARGIN = 24;
var STONE_RADIUS = 18;
var STAR_POINTS = [20, 24, 40, 56, 60];
var COLUMN_LABELS = "ABCDEFGHJ";
var getColorName = (color) => color === GO_BLACK_COLOR ? "black" : "white";
var formatMove = (move) => move === GO_PASS_MOVE ? "pass" : `${COLUMN_LABELS[move % GO_BOARD_SIZE]}${GO_BOARD_SIZE - Math.floor(move / GO_BOARD_SIZE)}`;
var formatScore = (score) => {
  if (score === 0) {
    return "Jigo";
  }
  return `${score > 0 ? "B" : "W"}+${Math.abs(score)}`;
};
var createSvgElement = (name, attributes) => {
  const element = document.createElementNS(SVG_NAMESPACE, name);
  for (const [key, value] of Object.entries(attributes)) {
    element.setAttribute(key, String(value));
  }
  return element;
};
var renderGoban = (svg, view) => {
  const margin = view.showCoordinates ? COORDINATE_MARGIN : PLAIN_MARGIN;
  const size = 2 * margin + (GO_BOARD_SIZE - 1) * CELL_SIZE;
  const lineEnd = margin + (GO_BOARD_SIZE - 1) * CELL_SIZE;
  const deadMoves2 = view.deadMoves ?? [];
  const getPointCenter = (move) => ({
    x: margin + move % GO_BOARD_SIZE * CELL_SIZE,
    y: margin + Math.floor(move / GO_BOARD_SIZE) * CELL_SIZE
  });
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
  svg.append(createSvgElement("rect", { width: size, height: size, rx: 6, fill: "var(--board)" }));
  for (let index = 0; index < GO_BOARD_SIZE; index += 1) {
    const offset = margin + index * CELL_SIZE;
    const lineStyle = { stroke: "var(--line)", "stroke-width": 1 };
    svg.append(
      createSvgElement("line", { x1: margin, y1: offset, x2: lineEnd, y2: offset, ...lineStyle }),
      createSvgElement("line", { x1: offset, y1: margin, x2: offset, y2: lineEnd, ...lineStyle })
    );
  }
  if (view.showCoordinates) {
    for (let index = 0; index < GO_BOARD_SIZE; index += 1) {
      const offset = margin + index * CELL_SIZE;
      const labelStyle = {
        fill: "var(--line)",
        "font-size": 12,
        "text-anchor": "middle",
        "dominant-baseline": "central"
      };
      const columnLabel = COLUMN_LABELS[index];
      const rowLabel = String(GO_BOARD_SIZE - index);
      for (const [x, y, text] of [
        [offset, margin / 2, columnLabel],
        [offset, size - margin / 2, columnLabel],
        [margin / 2, offset, rowLabel],
        [size - margin / 2, offset, rowLabel]
      ]) {
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
      svg.append(
        createSvgElement("circle", {
          cx: x,
          cy: y,
          r: STONE_RADIUS,
          fill: isBlack ? "#111" : "#f4f4f0",
          stroke: "#111",
          "stroke-width": 1,
          opacity: deadMoves2.includes(move) ? 0.35 : 1,
          "data-stone": getColorName(color)
        })
      );
      if (move === view.lastMove) {
        svg.append(
          createSvgElement("circle", {
            cx: x,
            cy: y,
            r: 6,
            fill: "none",
            stroke: isBlack ? "#f4f4f0" : "#111",
            "stroke-width": 2
          })
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
          "data-ghost": getColorName(hoverColor)
        })
      );
    }
    if (view.isInteractive) {
      svg.append(
        createSvgElement("circle", {
          cx: x,
          cy: y,
          r: CELL_SIZE / 2,
          fill: "transparent",
          "data-move": move
        })
      );
    }
  }
};

// vendor/moka/src/game.ts
var createGameState = () => ({
  board: new Int8Array(GO_BOARD_AREA),
  consecutivePassCount: 0,
  koMove: GO_NO_KO_MOVE,
  moveCount: 0,
  moveHistory: [],
  nextColor: GO_BLACK_COLOR
});
var copyGameState = (gameState) => ({
  board: gameState.board.slice(),
  consecutivePassCount: gameState.consecutivePassCount,
  koMove: gameState.koMove,
  moveCount: gameState.moveCount,
  moveHistory: [...gameState.moveHistory],
  nextColor: gameState.nextColor
});
var getAdjacentMoves = (move) => {
  const row = Math.floor(move / GO_BOARD_SIZE);
  const column = move % GO_BOARD_SIZE;
  const adjacentMoves = [];
  if (row > 0) {
    adjacentMoves.push(move - GO_BOARD_SIZE);
  }
  if (row < GO_BOARD_SIZE - 1) {
    adjacentMoves.push(move + GO_BOARD_SIZE);
  }
  if (column > 0) {
    adjacentMoves.push(move - 1);
  }
  if (column < GO_BOARD_SIZE - 1) {
    adjacentMoves.push(move + 1);
  }
  return adjacentMoves;
};
var getGroup = (board2, startingMove) => {
  const color = board2[startingMove];
  if (color === GO_EMPTY_COLOR) {
    return { liberties: /* @__PURE__ */ new Set(), stones: [] };
  }
  const pendingMoves = [startingMove];
  const visitedMoves = /* @__PURE__ */ new Set();
  const liberties = /* @__PURE__ */ new Set();
  const stones = [];
  while (pendingMoves.length > 0) {
    const move = pendingMoves.pop();
    if (move === void 0 || visitedMoves.has(move)) {
      continue;
    }
    visitedMoves.add(move);
    stones.push(move);
    for (const adjacentMove of getAdjacentMoves(move)) {
      const adjacentColor = board2[adjacentMove];
      if (adjacentColor === GO_EMPTY_COLOR) {
        liberties.add(adjacentMove);
      } else if (adjacentColor === color) {
        pendingMoves.push(adjacentMove);
      }
    }
  }
  return { liberties, stones };
};
var playMove = (gameState, move) => {
  if (move === GO_PASS_MOVE) {
    const nextState2 = copyGameState(gameState);
    nextState2.consecutivePassCount += 1;
    nextState2.koMove = GO_NO_KO_MOVE;
    nextState2.moveCount += 1;
    nextState2.moveHistory.push(move);
    nextState2.nextColor *= -1;
    return nextState2;
  }
  if (gameState.board[move] !== GO_EMPTY_COLOR || gameState.koMove === move) {
    return null;
  }
  const nextState = copyGameState(gameState);
  nextState.board[move] = gameState.nextColor;
  const capturedMoves = [];
  for (const adjacentMove of getAdjacentMoves(move)) {
    if (nextState.board[adjacentMove] !== -gameState.nextColor) {
      continue;
    }
    const opponentGroup = getGroup(nextState.board, adjacentMove);
    if (opponentGroup.liberties.size > 0) {
      continue;
    }
    for (const capturedMove of opponentGroup.stones) {
      nextState.board[capturedMove] = GO_EMPTY_COLOR;
      capturedMoves.push(capturedMove);
    }
  }
  const playedGroup = getGroup(nextState.board, move);
  if (playedGroup.liberties.size === 0) {
    return null;
  }
  nextState.consecutivePassCount = 0;
  nextState.koMove = capturedMoves.length === 1 && playedGroup.stones.length === 1 && playedGroup.liberties.size === 1 ? capturedMoves[0] : GO_NO_KO_MOVE;
  nextState.moveCount += 1;
  nextState.moveHistory.push(move);
  nextState.nextColor *= -1;
  return nextState;
};
var getColorGroups = (board2, color) => {
  const groupKeyByMove = new Int16Array(GO_BOARD_AREA);
  groupKeyByMove.fill(GO_NO_KO_MOVE);
  const groups = /* @__PURE__ */ new Map();
  const visitedMoves = /* @__PURE__ */ new Set();
  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (board2[move] !== color || visitedMoves.has(move)) {
      continue;
    }
    const stones = getGroup(board2, move).stones;
    const groupKey = stones[0];
    groups.set(groupKey, stones);
    for (const stoneMove of stones) {
      visitedMoves.add(stoneMove);
      groupKeyByMove[stoneMove] = groupKey;
    }
  }
  return { groupKeyByMove, groups };
};
var getNonColorRegions = (board2, color, groupKeyByMove) => {
  const regions = [];
  const visitedMoves = /* @__PURE__ */ new Set();
  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (board2[move] === color || visitedMoves.has(move)) {
      continue;
    }
    const pendingMoves = [move];
    const regionMoves = [];
    const borderingGroupKeys = /* @__PURE__ */ new Set();
    let containsOpponent = false;
    let internalPointCount = 0;
    let vitalGroupKeys = null;
    while (pendingMoves.length > 0) {
      const regionMove = pendingMoves.pop();
      if (regionMove === void 0 || visitedMoves.has(regionMove)) {
        continue;
      }
      visitedMoves.add(regionMove);
      regionMoves.push(regionMove);
      containsOpponent ||= board2[regionMove] === -color;
      const adjacentGroupKeys = /* @__PURE__ */ new Set();
      for (const adjacentMove of getAdjacentMoves(regionMove)) {
        if (board2[adjacentMove] === color) {
          const groupKey = groupKeyByMove[adjacentMove];
          adjacentGroupKeys.add(groupKey);
          borderingGroupKeys.add(groupKey);
        } else if (!visitedMoves.has(adjacentMove)) {
          pendingMoves.push(adjacentMove);
        }
      }
      if (adjacentGroupKeys.size === 0) {
        internalPointCount += 1;
      }
      if (board2[regionMove] === GO_EMPTY_COLOR) {
        if (vitalGroupKeys === null) {
          vitalGroupKeys = adjacentGroupKeys;
        } else {
          for (const vitalGroupKey of vitalGroupKeys) {
            if (!adjacentGroupKeys.has(vitalGroupKey)) {
              vitalGroupKeys.delete(vitalGroupKey);
            }
          }
        }
      }
    }
    regions.push({
      borderingGroupKeys,
      containsOpponent,
      internalPointCount,
      moves: regionMoves,
      vitalGroupKeys: vitalGroupKeys ?? /* @__PURE__ */ new Set()
    });
  }
  return regions;
};
var getPassAliveAnalysis = (board2, color) => {
  const { groupKeyByMove, groups } = getColorGroups(board2, color);
  const regions = getNonColorRegions(board2, color, groupKeyByMove);
  const passAliveGroupKeys = new Set(groups.keys());
  const passAliveRegionIndexes = new Set(regions.map((_, regionIndex) => regionIndex));
  while (true) {
    const removedGroupKeys = /* @__PURE__ */ new Set();
    for (const groupKey of passAliveGroupKeys) {
      let vitalRegionCount = 0;
      for (const regionIndex of passAliveRegionIndexes) {
        if (regions[regionIndex].vitalGroupKeys.has(groupKey)) {
          vitalRegionCount += 1;
        }
      }
      if (vitalRegionCount < GO_PASS_ALIVE_REQUIRED_VITAL_REGION_COUNT) {
        removedGroupKeys.add(groupKey);
      }
    }
    if (removedGroupKeys.size === 0) {
      break;
    }
    for (const removedGroupKey of removedGroupKeys) {
      passAliveGroupKeys.delete(removedGroupKey);
    }
    for (const regionIndex of passAliveRegionIndexes) {
      const doesBorderRemovedGroup = [...removedGroupKeys].some(
        (removedGroupKey) => regions[regionIndex].borderingGroupKeys.has(removedGroupKey)
      );
      if (doesBorderRemovedGroup) {
        passAliveRegionIndexes.delete(regionIndex);
      }
    }
  }
  return { groups, passAliveGroupKeys, regions };
};
var markPassAliveAreaForColor = (board2, color, area) => {
  const { groups, passAliveGroupKeys, regions } = getPassAliveAnalysis(board2, color);
  for (const groupKey of passAliveGroupKeys) {
    for (const stoneMove of groups.get(groupKey) ?? []) {
      area[stoneMove] = color;
    }
  }
  for (const region of regions) {
    const doesBorderNonPassAliveGroup = [...region.borderingGroupKeys].some(
      (groupKey) => !passAliveGroupKeys.has(groupKey)
    );
    const isPassAliveRegion = region.internalPointCount <= GO_PASS_ALIVE_MAXIMUM_INTERNAL_POINT_COUNT && !doesBorderNonPassAliveGroup && groups.size > 0;
    const isSafeTerritory = !region.containsOpponent && !doesBorderNonPassAliveGroup && groups.size > 0;
    if (isPassAliveRegion || isSafeTerritory) {
      for (const regionMove of region.moves) {
        area[regionMove] = color;
      }
    } else if (!region.containsOpponent && groups.size > 0) {
      for (const regionMove of region.moves) {
        if (area[regionMove] === GO_EMPTY_COLOR) {
          area[regionMove] = color;
        }
      }
    }
  }
};
var getPassAliveArea = (board2) => {
  const area = new Int8Array(GO_BOARD_AREA);
  markPassAliveAreaForColor(board2, GO_BLACK_COLOR, area);
  markPassAliveAreaForColor(board2, GO_WHITE_COLOR, area);
  return area;
};
var getAreaScore = (gameState) => {
  let score = -GO_KOMI_POINTS;
  const visitedMoves = /* @__PURE__ */ new Set();
  for (const color of gameState.board) {
    score += color;
  }
  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (gameState.board[move] !== GO_EMPTY_COLOR || visitedMoves.has(move)) {
      continue;
    }
    const pendingMoves = [move];
    const territoryMoves = [];
    const borderingColors = /* @__PURE__ */ new Set();
    while (pendingMoves.length > 0) {
      const territoryMove = pendingMoves.pop();
      if (territoryMove === void 0 || visitedMoves.has(territoryMove)) {
        continue;
      }
      visitedMoves.add(territoryMove);
      territoryMoves.push(territoryMove);
      for (const adjacentMove of getAdjacentMoves(territoryMove)) {
        const adjacentColor = gameState.board[adjacentMove];
        if (adjacentColor === GO_EMPTY_COLOR && !visitedMoves.has(adjacentMove)) {
          pendingMoves.push(adjacentMove);
        } else if (adjacentColor !== GO_EMPTY_COLOR) {
          borderingColors.add(adjacentColor);
        }
      }
    }
    if (borderingColors.size === 1 && borderingColors.has(GO_BLACK_COLOR)) {
      score += territoryMoves.length;
    } else if (borderingColors.size === 1 && borderingColors.has(GO_WHITE_COLOR)) {
      score -= territoryMoves.length;
    }
  }
  return score;
};
var removeDeadStones = (gameState, deadMoves2) => {
  const nextState = copyGameState(gameState);
  for (const deadMove of deadMoves2) {
    if (deadMove < 0 || deadMove >= GO_BOARD_AREA) {
      throw new Error(`Dead move must be between 0 and ${GO_BOARD_AREA - 1}.`);
    }
    nextState.board[deadMove] = GO_EMPTY_COLOR;
  }
  return nextState;
};
var getBestImmediateCapture = (gameState) => {
  const opponentColor = -gameState.nextColor;
  const opponentStoneCount = gameState.board.filter((color) => color === opponentColor).length;
  let bestCaptureCount = 0;
  let bestNextGameState = null;
  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    const nextGameState = playMove(gameState, move);
    if (!nextGameState) {
      continue;
    }
    const nextOpponentStoneCount = nextGameState.board.filter(
      (color) => color === opponentColor
    ).length;
    const captureCount = opponentStoneCount - nextOpponentStoneCount;
    if (captureCount > bestCaptureCount) {
      bestCaptureCount = captureCount;
      bestNextGameState = nextGameState;
    }
  }
  return bestNextGameState;
};
var getAutomaticallyDeadMoves = (gameState) => {
  const passAliveArea = getPassAliveArea(gameState.board);
  const passDeadMoves = [];
  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (gameState.board[move] !== GO_EMPTY_COLOR && passAliveArea[move] === -gameState.board[move]) {
      passDeadMoves.push(move);
    }
  }
  let aftermathGameState = removeDeadStones(gameState, passDeadMoves);
  aftermathGameState.consecutivePassCount = 0;
  for (let aftermathMoveCount = 0; aftermathMoveCount < GO_BOARD_AREA && aftermathGameState.consecutivePassCount < GO_GAME_OVER_PASS_COUNT; aftermathMoveCount += 1) {
    const captureGameState = getBestImmediateCapture(aftermathGameState);
    const nextGameState = captureGameState ?? playMove(aftermathGameState, GO_PASS_MOVE);
    if (!nextGameState) {
      break;
    }
    aftermathGameState = nextGameState;
  }
  const deadMoves2 = [...passDeadMoves];
  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (gameState.board[move] !== GO_EMPTY_COLOR && aftermathGameState.board[move] === GO_EMPTY_COLOR) {
      deadMoves2.push(move);
    }
  }
  return [...new Set(deadMoves2)];
};
var isGameOver = (gameState) => gameState.consecutivePassCount >= GO_GAME_OVER_PASS_COUNT || gameState.moveCount >= GO_MAXIMUM_MOVE_COUNT;

// site/protocol.ts
var LEVELS = [
  { id: "beginner", label: "Beginner", usesSearch: false, timeMs: 0 },
  { id: "easy", label: "Easy", usesSearch: true, maxVisits: 16, timeMs: 500 },
  { id: "medium", label: "Medium (0.3 s)", usesSearch: true, timeMs: 300 },
  { id: "strong", label: "Strong (1 s)", usesSearch: true, timeMs: 1e3 }
];
var LEVEL_BY_ID = Object.fromEntries(LEVELS.map((level) => [level.id, level]));

// site/play.ts
var getElement = (id) => {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing element #${id}.`);
  }
  return element;
};
var board = getElement("board");
var statusElement = getElement("status");
var movesElement = getElement("moves");
var levelSelect = getElement("level");
var colorSelect = getElement("color");
var passButton = getElement("pass");
var resignButton = getElement("resign");
var undoButton = getElement("undo");
var newGameButton = getElement("new-game");
var getColorLabel = (color) => color === GO_BLACK_COLOR ? "black" : "white";
var history = [createGameState()];
var moves = [];
var deadMoves = [];
var humanColor = GO_BLACK_COLOR;
var pageState = "loading";
var hoverMove = null;
var pendingRequestId = null;
var nextRequestId = 1;
var currentState = () => history[history.length - 1];
var readLevel = () => {
  const level = LEVEL_BY_ID[levelSelect.value];
  return level ?? LEVEL_BY_ID.beginner;
};
var findUndoIndex = () => {
  for (let index = history.length - 2; index >= 0; index -= 1) {
    if (history[index].nextColor === humanColor) {
      return index;
    }
  }
  return null;
};
var setPageState = (state, status) => {
  pageState = state;
  document.body.dataset.state = state;
  statusElement.textContent = status;
  renderBoard();
  const isHumanTurn = state === "human";
  passButton.disabled = !isHumanTurn;
  resignButton.disabled = !isHumanTurn;
  undoButton.disabled = !isHumanTurn || findUndoIndex() === null;
  newGameButton.disabled = state === "loading" || state === "error";
};
var renderBoard = () => renderGoban(board, {
  board: currentState().board,
  lastMove: moves.length > 0 && moves[moves.length - 1].move !== GO_PASS_MOVE ? moves[moves.length - 1].move : null,
  deadMoves,
  showCoordinates: true,
  isInteractive: pageState === "human",
  hoverMove,
  hoverColor: currentState().nextColor
});
var renderMoves = () => {
  movesElement.replaceChildren(
    ...moves.map((played) => {
      const item = document.createElement("li");
      item.textContent = `${getColorLabel(played.color)} ${played.move === GO_PASS_MOVE ? "pass" : formatMove(played.move)}`;
      return item;
    })
  );
  movesElement.scrollTop = movesElement.scrollHeight;
};
var render = () => {
  renderBoard();
  renderMoves();
};
var finishGame = () => {
  const current = currentState();
  deadMoves = getAutomaticallyDeadMoves(current);
  const score = getAreaScore(removeDeadStones(current, deadMoves));
  setPageState("over", `Game over: ${formatScore(score)} (komi 7)`);
};
var requestEngineMove = () => {
  const current = currentState();
  const requestId = nextRequestId;
  nextRequestId += 1;
  pendingRequestId = requestId;
  setPageState("thinking", `Thinking (${getColorLabel(current.nextColor)})\u2026`);
  worker.postMessage({
    type: "move",
    requestId,
    // postMessage structured-clones, so the worker can never mutate our snapshot.
    state: current,
    level: levelSelect.value
  });
};
var beginTurn = (suffix = "") => {
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
var recordMove = (move, color, nextState) => {
  moves.push({ move, color });
  history.push(nextState);
};
var applyEngineMove = (move, visits) => {
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
var playHumanMove = (move) => {
  if (pageState !== "human") {
    return;
  }
  const current = currentState();
  const nextState = playMove(current, move);
  if (!nextState) {
    return;
  }
  recordMove(move, current.nextColor, nextState);
  render();
  beginTurn();
};
var resign = () => {
  if (pageState !== "human") {
    return;
  }
  const winnerLetter = humanColor === GO_BLACK_COLOR ? "W" : "B";
  pendingRequestId = null;
  setPageState("over", `Game over: ${winnerLetter}+R (resignation)`);
};
var undo = () => {
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
var startGame = () => {
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
var worker = new Worker(new URL("./engine-worker.js?v=00d2ed542c81", import.meta.url), { type: "module" });
worker.addEventListener("message", (event) => {
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
  if (target instanceof SVGElement && target.dataset.move !== void 0) {
    playHumanMove(Number(target.dataset.move));
  }
});
board.addEventListener("pointermove", (event) => {
  if (event.pointerType !== "mouse") {
    return;
  }
  const target = event.target;
  const move = target instanceof SVGElement && target.dataset.move !== void 0 ? Number(target.dataset.move) : null;
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
setPageState("loading", "Loading the engine\u2026");
worker.postMessage({ type: "init" });
