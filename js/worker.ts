import { createEngine, type Engine } from "./engine.ts";

const WASM_LOAD_ERROR = "Failed to load infer.wasm.";

const compileInferModule = async () => {
  const response = await fetch(new URL("./infer.wasm", import.meta.url));

  if (!response.ok) {
    throw new Error(WASM_LOAD_ERROR);
  }

  return WebAssembly.compile(await response.arrayBuffer());
};

const inferModule = compileInferModule();
inferModule.catch(() => undefined);

let engine: Engine | null = null;
let pendingMessages: Promise<void> = Promise.resolve();

const loadInferModule = async () => {
  try {
    return await inferModule;
  } catch {
    throw new Error(WASM_LOAD_ERROR);
  }
};

const handleMessage = async (
  message: GoModelWorkerInferenceMessage | GoModelWorkerInitializeMessage,
) => {
  try {
    if (message.type === "initialize") {
      engine = null;
      engine = await createEngine(await loadInferModule(), message.weightsBuffer);
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
