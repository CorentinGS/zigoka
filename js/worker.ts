import { compileEngines, fetchFiles, type Engine } from "./engines.ts";

// Compiled as soon as the worker starts; each initialize then brings its own weights.
const compiled = compileEngines(
  fetchFiles({ "infer.wasm": new URL("./infer.wasm", import.meta.url) }),
  ["infer.wasm"],
);
compiled.catch(() => undefined);

let engine: Engine | null = null;
let pendingMessages: Promise<void> = Promise.resolve();

const handleMessage = async (
  message: GoModelWorkerInferenceMessage | GoModelWorkerInitializeMessage,
) => {
  try {
    if (message.type === "initialize") {
      engine = null;
      // worker.js serves any model (the bench copies it for each one): the client's manifest
      // gives the board it plays.
      const engines = await (await compiled).load(message.weightsBuffer, {
        boardSize: message.manifest.architecture.boardSize,
      });
      engine = await engines.createEngine();
      const response: GoModelWorkerReadyMessage = {
        requestId: message.requestId,
        type: "ready",
      };
      self.postMessage(response);
      return;
    }

    if (!engine) {
      throw new Error("Go model worker is not initialized.");
    }

    let result: GoModelInference;

    try {
      result = engine.infer(message.features);
    } catch (error) {
      if (error instanceof WebAssembly.RuntimeError) {
        engine = null;
      }

      throw error;
    }

    const response: GoModelWorkerResultMessage = {
      ...result,
      requestId: message.requestId,
      type: "result",
    };
    self.postMessage(response, { transfer: [response.policyLogits.buffer] });
  } catch (error) {
    const response: GoModelWorkerErrorMessage = {
      error: error instanceof Error ? error.message : "Go model inference failed.",
      requestId: message.requestId,
      type: "error",
    };
    self.postMessage(response);
  }
};

self.onmessage = (
  event: MessageEvent<GoModelWorkerInferenceMessage | GoModelWorkerInitializeMessage>,
) => {
  pendingMessages = pendingMessages.then(() => handleMessage(event.data));
};
