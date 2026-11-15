// vendor/moka/src/utils/get-buffer-sha256.ts
var getBufferSha256 = async (buffer) => {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
};

// js/engine.ts
var STATUS_OK = 0;
var STATUS_WEIGHTS_NOT_LOADED = 1;
var STATUS_BAD_BATCH_COUNT = 2;
var STATUS_NON_FINITE_OUTPUT = 3;
var checkStatus = (status) => {
  if (status === STATUS_NON_FINITE_OUTPUT) {
    throw new Error("Model produced a non-finite output.");
  }
  if (status !== STATUS_OK) {
    throw new Error("Go model inference failed.");
  }
};
var createInputWriter = (exports) => {
  const featureCount = exports.feature_count();
  const input = new Float32Array(exports.memory.buffer, exports.input_ptr(), featureCount);
  return (features) => {
    if (features.length !== featureCount) {
      throw new Error(`Input features must have ${featureCount} values.`);
    }
    input.set(features);
  };
};
var toEngine = (exports) => {
  const writeInput = createInputWriter(exports);
  const policy = new Float32Array(exports.memory.buffer, exports.policy_ptr(), exports.policy_moves());
  const value = new Float64Array(exports.memory.buffer, exports.value_ptr(), 1);
  return {
    infer: (features) => {
      writeInput(features);
      checkStatus(exports.infer(1));
      return { policyLogits: policy.slice(), value: value[0] };
    }
  };
};
var toProfileEngine = (exports) => {
  const writeInput = createInputWriter(exports);
  return {
    ...toEngine(exports),
    // The bench's trunk alone: its errors keep the status, which no worker client sees.
    forwardTrunk: (features) => {
      writeInput(features);
      const status = exports.forward_trunk(1);
      if (status !== STATUS_OK) {
        throw new Error(`forward_trunk(1) returned status ${status}.`);
      }
    }
  };
};

// js/fetch-ok.ts
var fetchOk = async (url) => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Unable to load ${url} (${response.status}).`);
  }
  return response;
};

// js/search-engine.ts
var STATUS_GAME_OVER = 4;
var STATUS_POOL_FULL = 5;
var STATUS_NOT_STARTED = 6;
var STATUS_MESSAGES = {
  [STATUS_WEIGHTS_NOT_LOADED]: "The model weights are not loaded.",
  [STATUS_BAD_BATCH_COUNT]: "The batch size must be between 1 and the engine's maximum batch.",
  [STATUS_NON_FINITE_OUTPUT]: "Model produced a non-finite output.",
  [STATUS_GAME_OVER]: "The search started from a finished game.",
  [STATUS_NOT_STARTED]: "The search was not started."
};
var MAX_TARGET = 4294967295;
var NO_MOVE = -1;
var POSITION_KO_OFFSET = 0;
var POSITION_MOVE_COUNT_OFFSET = 2;
var POSITION_LAST_MOVE_OFFSET = 4;
var POSITION_PREVIOUS_MOVE_OFFSET = 6;
var POSITION_PASS_COUNT_OFFSET = 8;
var POSITION_NEXT_COLOR_OFFSET = 9;
var POSITION_BOARD_OFFSET = 10;
var MAX_PASS_COUNT = 255;
var writePosition = (exports, state) => {
  const pointer = exports.position_ptr();
  const view = new DataView(exports.memory.buffer, pointer);
  const area = state.board.length;
  view.setInt16(POSITION_KO_OFFSET, state.koMove, true);
  view.setUint16(POSITION_MOVE_COUNT_OFFSET, state.moveCount, true);
  view.setInt16(POSITION_LAST_MOVE_OFFSET, state.moveHistory.at(-1) ?? NO_MOVE, true);
  view.setInt16(POSITION_PREVIOUS_MOVE_OFFSET, state.moveHistory.at(-2) ?? NO_MOVE, true);
  view.setUint8(POSITION_PASS_COUNT_OFFSET, Math.min(state.consecutivePassCount, MAX_PASS_COUNT));
  view.setInt8(POSITION_NEXT_COLOR_OFFSET, state.nextColor);
  new Int8Array(exports.memory.buffer, pointer + POSITION_BOARD_OFFSET, area).set(state.board);
};
var checkStatus2 = (status) => {
  if (status !== STATUS_OK) {
    throw new Error(STATUS_MESSAGES[status] ?? `The search failed with status ${status}.`);
  }
};
var toSearchEngine = (exports) => {
  let rootMs = 0;
  const search = (state, { timeMs, maxVisits = Infinity, batch = 1, reuse = false, now = () => performance.now() }) => {
    const startedAt = now();
    const deadline = startedAt + timeMs;
    writePosition(exports, state);
    checkStatus2(exports.search_start(reuse ? 1 : 0));
    const reused = exports.search_visits();
    if (exports.search_reused() === 0) {
      rootMs = now() - startedAt;
    }
    let visits = reused;
    let chunk = 1;
    let poolFull = false;
    const runStartedAt = now();
    while (visits < maxVisits) {
      const chunkStartedAt = now();
      if (chunkStartedAt >= deadline) {
        break;
      }
      const status = exports.search_run(Math.min(maxVisits, visits + chunk, MAX_TARGET), batch);
      const finishedAt = now();
      const done = exports.search_visits() - visits;
      visits += done;
      if (status === STATUS_POOL_FULL) {
        poolFull = true;
        break;
      }
      checkStatus2(status);
      const searched = Math.max(visits - reused, 1);
      const msPerVisit = Math.max((finishedAt - runStartedAt) / searched, rootMs);
      const maxChunk = 2 * searched;
      chunk = Math.max(1, Math.min(maxChunk, Math.floor((deadline - finishedAt) / msPerVisit / 2)));
    }
    return { move: exports.search_best_move(), visits, reused, poolFull };
  };
  const rootVisits = () => Array.from({ length: exports.search_child_count() }, (_, index) => ({
    move: exports.search_child_move(index),
    visits: exports.search_child_visits(index)
  }));
  return { search, rootVisits, exports };
};

// js/engines.ts
var SHA256_BYTE_COUNT = 32;
var SHA256_PREFIX_LENGTH = 8;
var SIMD_ERROR = "WebAssembly SIMD is not supported in this browser.";
var ENGINE_KINDS = {
  "infer.wasm": {
    create: "createEngine",
    toEngine: (exports) => toEngine(exports)
  },
  "search.wasm": {
    create: "createSearchEngine",
    toEngine: (exports) => toSearchEngine(exports)
  },
  "infer-profile.wasm": {
    create: "createProfileEngine",
    toEngine: (exports) => toProfileEngine(exports)
  }
};
var toHex = (bytes) => Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
var describeBlockedCompilation = (location) => `WebAssembly compilation was blocked; the Content Security Policy of ${location.pathname.split("/").pop() || "this page"} must allow 'wasm-unsafe-eval'.`;
var compileWasm = async (bytes) => {
  try {
    return await WebAssembly.compile(bytes);
  } catch (error) {
    if (!WebAssembly.validate(bytes)) {
      throw new Error(SIMD_ERROR);
    }
    const location = globalThis.location;
    throw location ? new Error(describeBlockedCompilation(location)) : error;
  }
};
var instantiate = async (file, module, weights, sha256, boardSize) => {
  const instance = await WebAssembly.instantiate(module, {});
  const exports = instance.exports;
  const engineBoardSize = exports.board_size();
  if (engineBoardSize !== boardSize) {
    throw new Error(
      `${file} plays ${engineBoardSize}\xD7${engineBoardSize} games; this host plays ${boardSize}\xD7${boardSize}.`
    );
  }
  const memory = exports.memory.buffer;
  const weightsCapacity = exports.weights_capacity();
  if (weights.byteLength !== weightsCapacity) {
    throw new Error(`Model weights have the wrong length (expected ${weightsCapacity} bytes).`);
  }
  const expectedSha256 = toHex(new Uint8Array(memory, exports.expected_sha256_ptr(), SHA256_BYTE_COUNT));
  if (sha256 !== expectedSha256) {
    throw new Error(
      `Model weights do not match this engine build (expected sha256 ${expectedSha256.slice(0, SHA256_PREFIX_LENGTH)}\u2026).`
    );
  }
  new Uint8Array(memory, exports.weights_ptr(), weightsCapacity).set(new Uint8Array(weights));
  exports.load_weights();
  return exports;
};
var compileEngines = async (read, files) => {
  const modules = await Promise.all(
    files.map(async (file) => [file, await compileWasm(await read(file))])
  );
  return {
    load: async (weights, { boardSize }) => {
      const sha256 = await getBufferSha256(weights);
      const creators = await Promise.all(
        modules.map(async ([file, module]) => {
          const create = () => instantiate(file, module, weights, sha256, boardSize);
          let checked = await create();
          const createEngine = async () => {
            const exports = checked ?? await create();
            checked = null;
            return ENGINE_KINDS[file].toEngine(exports);
          };
          return [ENGINE_KINDS[file].create, createEngine];
        })
      );
      return { weights, ...Object.fromEntries(creators) };
    }
  };
};
var loadEngines = async (read, files, options) => {
  const [compiled, weights] = await Promise.all([compileEngines(read, files), read("go-model.bin")]);
  return compiled.load(weights, options);
};
var fetchFiles = (urls) => async (file) => {
  const url = urls[file];
  if (url === void 0) {
    throw new Error(`No URL given for ${file}.`);
  }
  try {
    return await (await fetchOk(url)).arrayBuffer();
  } catch {
    throw new Error(`Failed to load ${file}.`);
  }
};

// vendor/moka/src/constants.ts
var GO_BOARD_AREA = 81;
var GO_BOARD_SIZE = 9;
var GO_CURRENT_PLAYER_FEATURE_INDEX = 0;
var GO_CURRENT_PLAYER_ONE_LIBERTY_FEATURE_INDEX = 2;
var GO_CURRENT_PLAYER_TWO_LIBERTY_FEATURE_INDEX = 4;
var GO_EMPTY_COLOR = 0;
var GO_KO_FEATURE_INDEX = 6;
var GO_KOMI_NORMALIZATION_POINTS = 15;
var GO_KOMI_FEATURE_INDEX = 11;
var GO_KOMI_POINTS = 7;
var GO_MINIMUM_PASS_MOVE_COUNT = 20;
var GO_NO_KO_MOVE = -1;
var GO_OPPONENT_FEATURE_INDEX = 1;
var GO_OPPONENT_ONE_LIBERTY_FEATURE_INDEX = 3;
var GO_OPPONENT_TWO_LIBERTY_FEATURE_INDEX = 5;
var GO_PASS_FEATURE_INDEX_OFFSET = 8;
var GO_PASS_MOVE = 81;
var GO_RECENT_MOVE_FEATURE_INDEXES = [7, 8];
var GO_STUDENT_INPUT_PLANE_COUNT = 12;

// vendor/moka/src/game.ts
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
var getGroup = (board, startingMove) => {
  const color = board[startingMove];
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
      const adjacentColor = board[adjacentMove];
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
var getLegalMoves = (gameState) => {
  const legalMoves = [];
  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    if (playMove(gameState, move)) {
      legalMoves.push(move);
    }
  }
  legalMoves.push(GO_PASS_MOVE);
  return legalMoves;
};
var selectHighestLegalMove = (gameState, logits) => {
  const legalMoves = getLegalMoves(gameState);
  let selectedMove = gameState.moveCount >= GO_MINIMUM_PASS_MOVE_COUNT ? GO_PASS_MOVE : legalMoves[0];
  let selectedValue = logits[selectedMove];
  for (const legalMove of legalMoves) {
    if (legalMove === GO_PASS_MOVE && gameState.moveCount < GO_MINIMUM_PASS_MOVE_COUNT) {
      continue;
    }
    if (logits[legalMove] > selectedValue) {
      selectedMove = legalMove;
      selectedValue = logits[legalMove];
    }
  }
  return selectedMove;
};
var getLibertyFeatureIndex = (libertyCount, isCurrentPlayer) => {
  if (libertyCount === 1) {
    return isCurrentPlayer ? GO_CURRENT_PLAYER_ONE_LIBERTY_FEATURE_INDEX : GO_OPPONENT_ONE_LIBERTY_FEATURE_INDEX;
  }
  if (libertyCount === 2) {
    return isCurrentPlayer ? GO_CURRENT_PLAYER_TWO_LIBERTY_FEATURE_INDEX : GO_OPPONENT_TWO_LIBERTY_FEATURE_INDEX;
  }
  return null;
};
var encodeStudentFeatures = (gameState) => {
  const features = new Float32Array(GO_BOARD_AREA * GO_STUDENT_INPUT_PLANE_COUNT);
  const visitedMoves = /* @__PURE__ */ new Set();
  const setFeature = (move, featureIndex, value = 1) => {
    features[move * GO_STUDENT_INPUT_PLANE_COUNT + featureIndex] = value;
  };
  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    const color = gameState.board[move];
    if (color === GO_EMPTY_COLOR) {
      continue;
    }
    const isCurrentPlayer = color === gameState.nextColor;
    setFeature(move, isCurrentPlayer ? GO_CURRENT_PLAYER_FEATURE_INDEX : GO_OPPONENT_FEATURE_INDEX);
    if (visitedMoves.has(move)) {
      continue;
    }
    const group = getGroup(gameState.board, move);
    for (const stoneMove of group.stones) {
      visitedMoves.add(stoneMove);
    }
    const libertyFeatureIndex = getLibertyFeatureIndex(group.liberties.size, isCurrentPlayer);
    if (libertyFeatureIndex !== null) {
      for (const stoneMove of group.stones) {
        setFeature(stoneMove, libertyFeatureIndex);
      }
    }
  }
  if (gameState.koMove >= 0) {
    setFeature(gameState.koMove, GO_KO_FEATURE_INDEX);
  }
  for (let historyOffset = 1; historyOffset <= GO_RECENT_MOVE_FEATURE_INDEXES.length; historyOffset += 1) {
    const historyMove = gameState.moveHistory.at(-historyOffset);
    if (historyMove !== void 0 && historyMove < GO_BOARD_AREA) {
      setFeature(historyMove, GO_RECENT_MOVE_FEATURE_INDEXES[historyOffset - 1]);
    } else if (historyMove === GO_PASS_MOVE) {
      const passFeatureIndex = GO_PASS_FEATURE_INDEX_OFFSET + historyOffset;
      for (let move = 0; move < GO_BOARD_AREA; move += 1) {
        setFeature(move, passFeatureIndex);
      }
    }
  }
  const perspectiveKomi = -GO_KOMI_POINTS * gameState.nextColor / GO_KOMI_NORMALIZATION_POINTS;
  for (let move = 0; move < GO_BOARD_AREA; move += 1) {
    setFeature(move, GO_KOMI_FEATURE_INDEX, perspectiveKomi);
  }
  return features;
};

// site/engine-files.ts
var WORKER_FILES = ["infer.wasm", "search.wasm", "go-model.bin"];

// site/protocol.ts
var LEVELS = [
  { id: "beginner", label: "Beginner", usesSearch: false, timeMs: 0 },
  { id: "easy", label: "Easy", usesSearch: true, maxVisits: 16, timeMs: 500 },
  { id: "medium", label: "Medium (0.3 s)", usesSearch: true, timeMs: 300 },
  { id: "strong", label: "Strong (1 s)", usesSearch: true, timeMs: 1e3 }
];
var LEVEL_BY_ID = Object.fromEntries(LEVELS.map((level) => [level.id, level]));

// site/engine-worker.ts
var post = (response) => self.postMessage(response);
var assetUrl = (fileName) => {
  const url = new URL(`./${fileName}`, import.meta.url);
  url.search = new URL(import.meta.url).search;
  return url.href;
};
var loadSiteEngines = async () => {
  const read = fetchFiles(Object.fromEntries(WORKER_FILES.map((file) => [file, assetUrl(file)])));
  const loaded = await loadEngines(read, ["infer.wasm", "search.wasm"], { boardSize: GO_BOARD_SIZE });
  return { loaded, inferEngine: await loaded.createEngine() };
};
var siteEngines = null;
var searchEngine = null;
var pendingMessages = Promise.resolve();
var handleInit = async () => {
  const startedAt = performance.now();
  siteEngines = await loadSiteEngines();
  post({ type: "ready", durationMs: performance.now() - startedAt });
};
var handleNewGame = async () => {
  if (!siteEngines) {
    throw new Error("The site engine worker is not initialized.");
  }
  searchEngine = await siteEngines.loaded.createSearchEngine();
};
var handleMove = async (request) => {
  if (!siteEngines) {
    throw new Error("The site engine worker is not initialized.");
  }
  const level = LEVEL_BY_ID[request.level] ?? LEVEL_BY_ID.beginner;
  const startedAt = performance.now();
  let move;
  let visits;
  if (level.usesSearch) {
    if (!searchEngine) {
      throw new Error("The site engine worker has no game yet.");
    }
    const outcome = searchEngine.search(request.state, {
      timeMs: level.timeMs,
      maxVisits: level.maxVisits,
      batch: 1,
      reuse: true
    });
    move = outcome.move;
    visits = outcome.visits;
  } else {
    move = selectHighestLegalMove(
      request.state,
      siteEngines.inferEngine.infer(encodeStudentFeatures(request.state)).policyLogits
    );
    visits = 1;
  }
  post({
    type: "moved",
    requestId: request.requestId,
    move,
    visits,
    thinkingMs: performance.now() - startedAt
  });
};
var handleMessage = async (message) => {
  try {
    if (message.type === "init") {
      await handleInit();
      return;
    }
    if (message.type === "new-game") {
      await handleNewGame();
      return;
    }
    await handleMove(message);
  } catch (error) {
    post({ type: "error", error: error instanceof Error ? error.message : String(error) });
  }
};
self.onmessage = (event) => {
  pendingMessages = pendingMessages.then(() => handleMessage(event.data));
};
