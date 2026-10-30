import { GoModelWorkerClient } from "../../vendor/moka/src/client.ts";

const WORKER_URL = "/dist/worker.js";
const STRICT_WORKER_URL = "/dist/worker.js?csp=strict";
const MANIFEST_URL = "/dist/go-model.json";
const WEIGHTS_URL = "/dist/go-model.bin";

type WorkerReply =
  | GoModelWorkerErrorMessage
  | GoModelWorkerReadyMessage
  | GoModelWorkerResultMessage;

type DirectMessage =
  | Omit<GoModelWorkerInferenceMessage, "requestId">
  | Omit<GoModelWorkerInitializeMessage, "requestId">;

interface E2eApi {
  client: GoModelWorkerClient;
  manifestUrl: string;
  weightsUrl: string;
  post: (message: DirectMessage, transfer?: Transferable[]) => Promise<WorkerReply>;
  fetchModel: () => Promise<{ manifest: GoModelManifest; weights: ArrayBuffer }>;
  fetchFloat32: (url: string) => Promise<Float32Array>;
}

declare global {
  interface Window {
    e2e?: E2eApi;
  }
}

const fetchOk = async (url: string) => {
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Unable to load ${url} (${response.status}).`);
  }

  return response;
};

const isWorkerCspStrict = new URLSearchParams(window.location.search).get("worker-csp") === "strict";
const workerUrl = isWorkerCspStrict ? STRICT_WORKER_URL : WORKER_URL;

const directWorker = new Worker(workerUrl, { type: "module" });
const pendingReplies = new Map<number, (reply: WorkerReply) => void>();
let nextRequestId = 0;

directWorker.onmessage = (event: MessageEvent<WorkerReply>) => {
  const resolve = pendingReplies.get(event.data.requestId);
  pendingReplies.delete(event.data.requestId);
  resolve?.(event.data);
};

window.e2e = {
  client: new GoModelWorkerClient(new Worker(workerUrl, { type: "module" })),
  manifestUrl: MANIFEST_URL,
  weightsUrl: WEIGHTS_URL,
  post: (message, transfer = []) =>
    new Promise((resolve) => {
      const requestId = nextRequestId;
      nextRequestId += 1;
      pendingReplies.set(requestId, resolve);
      directWorker.postMessage({ ...message, requestId }, transfer);
    }),
  fetchModel: async () => ({
    manifest: (await (await fetchOk(MANIFEST_URL)).json()) as GoModelManifest,
    weights: await (await fetchOk(WEIGHTS_URL)).arrayBuffer(),
  }),
  fetchFloat32: async (url) => new Float32Array(await (await fetchOk(url)).arrayBuffer()),
};

export type { E2eApi, WorkerReply };
