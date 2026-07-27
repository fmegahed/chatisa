import os from "node:os";
import path from "node:path";
import { spawn as spawnProcess, type ChildProcess } from "node:child_process";
import type { PdfPage } from "./pdf-core";

/**
 * A small fixed pool of PDF workers with a bounded queue.
 *
 * Concurrency is the point of this application: many students use hosted
 * models at the same time, and model calls are network-bound so they scale
 * well on one thread. CPU-heavy PDF work does not, so it runs here instead,
 * capped so a burst of uploads degrades politely rather than stalling
 * everyone's chat responses.
 */

export interface ProcessedPdf {
  pageCount: number;
  pages: PdfPage[];
  classification: "text" | "mixed" | "scanned";
  warnings: string[];
  /** Rendered images for pages that need visual transcription. */
  images: { pageNumber: number; png: Uint8Array }[];
  /** Pages needing vision that were beyond the per-document cap. */
  skippedVisionPages: number[];
}

export class PdfBusyError extends Error {
  constructor() {
    super(
      "The server is handling too many uploads right now. Try again in a moment.",
    );
    this.name = "PdfBusyError";
  }
}

export class PdfWorkerError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "PdfWorkerError";
  }
}

function poolSize(): number {
  const configured = Number(process.env.CHATISA_PDF_WORKERS);
  if (Number.isFinite(configured) && configured > 0) return Math.floor(configured);
  // Leave a core for the request thread and everything else on the box.
  return Math.max(1, Math.min(4, (os.cpus()?.length ?? 2) - 1));
}

/** Uploads waiting for a free worker before we start refusing politely. */
const MAX_QUEUE = Number(process.env.CHATISA_PDF_QUEUE ?? 20);
const TASK_TIMEOUT_MS = 60_000;

interface Job {
  payload: { bytes: Uint8Array; maxVisionPages: number; deadlineMs: number };
  resolve: (value: ProcessedPdf) => void;
  reject: (reason: Error) => void;
}

interface PoolWorker {
  worker: ChildProcess;
  busy: boolean;
}

const globalForPool = globalThis as unknown as {
  __chatisaPdfPool?: { workers: PoolWorker[]; queue: Job[] };
};

function workerPath(): string {
  return path.join(process.cwd(), "workers", "pdf-worker.mjs");
}

function getPool() {
  if (!globalForPool.__chatisaPdfPool) {
    globalForPool.__chatisaPdfPool = { workers: [], queue: [] };
  }
  return globalForPool.__chatisaPdfPool;
}

/**
 * A child process, not a worker thread. Node's module loader hooks are
 * inherited by worker threads, and the framework registers a bundler resolver
 * that cannot resolve the optional native canvas binding, which broke
 * rasterization inside the server while working fine under the test runner.
 * A separate process gets clean module resolution and real CPU isolation.
 */
function workerEnv(): NodeJS.ProcessEnv {
  const env = { ...process.env };
  delete env.NODE_OPTIONS;
  delete env.NODE_PATH;
  return env;
}

function spawn(): PoolWorker {
  // Launched as an argument to the Node binary rather than through fork().
  // The bundler statically traces fork()'s first argument as a module import
  // and fails the build; an argv string is opaque to it and behaves the same,
  // with an ipc channel giving the identical send/message API.
  const child = spawnProcess(process.execPath, [workerPath()], {
    env: workerEnv(),
    // Structured clone rather than JSON, so image buffers stay binary.
    serialization: "advanced",
    stdio: ["ignore", "inherit", "inherit", "ipc"],
    windowsHide: true,
  });
  child.unref();
  child.channel?.unref();
  return { worker: child, busy: false };
}

function run(entry: PoolWorker, job: Job) {
  entry.busy = true;
  const id = Math.random().toString(36).slice(2);

  const cleanup = () => {
    entry.worker.off("message", onMessage);
    entry.worker.off("error", onError);
    clearTimeout(timer);
    entry.busy = false;
    pump();
  };

  const onMessage = (msg: {
    id?: string;
    ok?: boolean;
    result?: ProcessedPdf;
    code?: string;
    message?: string;
  }) => {
    if (msg.id !== id) return;
    cleanup();
    if (msg.ok && msg.result) {
      // Structured clone hands back Buffers; normalise to Uint8Array.
      job.resolve({
        ...msg.result,
        images: msg.result.images.map((i) => ({
          pageNumber: i.pageNumber,
          png: new Uint8Array(i.png),
        })),
      });
    }
    else
      job.reject(
        new PdfWorkerError(
          msg.code ?? "UNREADABLE_PDF",
          msg.message ?? "That PDF could not be read.",
        ),
      );
  };

  const onError = (err: Error) => {
    cleanup();
    // A crashed worker is replaced so the pool does not shrink over time.
    replace(entry);
    job.reject(new PdfWorkerError("UNREADABLE_PDF", err.message));
  };

  const timer = setTimeout(() => {
    cleanup();
    replace(entry);
    job.reject(
      new PdfWorkerError("TIMEOUT", "Reading that PDF took too long."),
    );
  }, TASK_TIMEOUT_MS);

  entry.worker.on("message", onMessage);
  entry.worker.on("error", onError);
  entry.worker.send({ id, payload: job.payload });
}

function replace(entry: PoolWorker) {
  const pool = getPool();
  const index = pool.workers.indexOf(entry);
  entry.worker.kill();
  if (index >= 0) pool.workers[index] = spawn();
}

function pump() {
  const pool = getPool();
  while (pool.queue.length > 0) {
    let free = pool.workers.find((w) => !w.busy);
    if (!free && pool.workers.length < poolSize()) {
      free = spawn();
      pool.workers.push(free);
    }
    if (!free) return;
    const job = pool.queue.shift();
    if (!job) return;
    run(free, job);
  }
}

/**
 * Reads a PDF and renders any pages needing transcription, entirely off the
 * request thread. Rejects with PdfBusyError when the queue is full, so callers
 * can tell the student to retry instead of queueing without limit.
 */
export function processPdfInWorker(params: {
  bytes: Uint8Array;
  maxVisionPages?: number;
  deadlineMs?: number;
}): Promise<ProcessedPdf> {
  const pool = getPool();
  if (pool.queue.length >= MAX_QUEUE) {
    return Promise.reject(new PdfBusyError());
  }
  return new Promise<ProcessedPdf>((resolve, reject) => {
    pool.queue.push({
      payload: {
        bytes: params.bytes,
        maxVisionPages: params.maxVisionPages ?? 40,
        deadlineMs: params.deadlineMs ?? 30_000,
      },
      resolve,
      reject,
    });
    pump();
  });
}

/** Test and shutdown helper: stops all workers. */
export async function shutdownPdfPool(): Promise<void> {
  const pool = getPool();
  for (const w of pool.workers) w.worker.kill();
  pool.workers.length = 0;
  pool.queue.length = 0;
}

export const pdfPoolStats = () => ({
  size: getPool().workers.length,
  busy: getPool().workers.filter((w) => w.busy).length,
  queued: getPool().queue.length,
  maxSize: poolSize(),
  maxQueue: MAX_QUEUE,
});
