import { createEngine, type Engine } from "./engine.ts";
import { describeLoadError } from "./load-error.ts";

const fetchInferBytes = async () => {
  const response = await fetch(new URL("./infer.wasm", import.meta.url));

  if (!response.ok) {
    throw new Error(`infer.wasm answered ${response.status}.`);
  }

  return response.arrayBuffer();
};

const compileInferModule = async () => {
  let bytes: ArrayBuffer;

  try {
    bytes = await fetchInferBytes();
  } catch {
    throw new Error(describeLoadError({ fetchFailed: true, simdValid: false }));
  }

  try {
    return await WebAssembly.compile(bytes);
  } catch {
    throw new Error(
      describeLoadError({ fetchFailed: false, simdValid: WebAssembly.validate(bytes) }),
    );
  }
};

const inferModule = compileInferModule();
inferModule.catch(() => undefined);

let engine: Engine | null = null;
let pendingMessages: Promise<void> = Promise.resolve();

const handleMessage = async (
  message: GoModelWorkerInferenceMessage | GoModelWorkerInitializeMessage,
) => {
  try {
    if (message.type === "initialize") {
      engine = null;
      engine = await createEngine(await inferModule, message.weightsBuffer);
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
