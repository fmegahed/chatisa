import type { RunnableLanguage } from "@/lib/run/languages";
import { WS_PROXY } from "@/lib/sandbox/network";
import type { DocRequest, DocText } from "@/lib/sandbox/help-docs/doc-text";

/**
 * Runs a code snippet in a per-language Web Worker.
 *
 * One worker per language, created on first run and reused, because
 * re-initialising a WASM runtime per run would re-download and re-init it, far
 * too slowly. Requests are correlated by id.
 *
 * The timeout is the only reliable stop for a runaway snippet: without
 * cross-origin isolation a busy worker cannot be interrupted cooperatively, so
 * on timeout the worker is terminated and the next run lazily recreates it.
 */

export interface RunResult {
  text?: string;
  table?: { columns: string[]; rows: Record<string, unknown>[] };
  imageDataUrl?: string;
  /** Session introspection: the variables (or tables, for SQL) now defined. */
  variables?: SessionVariable[];
}

export interface SessionVariable {
  name: string;
  /** A class or type, e.g. "data.frame", "int", "table". */
  type: string;
  /** A short shape or preview, e.g. "40 x 6", "len 100", "3 rows". */
  info: string;
  /** For a data frame or table: its columns and their types. A variable with
   * columns can be opened in the data viewer. */
  columns?: { name: string; type: string }[];
}

/** One page of a data frame or table, for the data viewer. */
export interface DataPage {
  columns: string[];
  rows: Record<string, unknown>[];
  /** Total rows in the whole frame, so the viewer can page through. */
  totalRows: number;
}

/** One runtime autocomplete candidate. */
export interface CompletionOption {
  label: string;
  /** "method" | "property" | "variable" | "function" | "keyword" | "table". */
  type?: string;
  /** A short one-line detail (e.g. a docstring first line or column type). */
  detail?: string;
}

/** Autocomplete candidates for the word before the cursor. */
export interface CompletionResult {
  /** The already-typed text these options complete, so the caller knows the
   * replace range. */
  partial: string;
  options: CompletionOption[];
}

/** A session table dumped as CSV, for handing to the ggsql plot engine. */
export interface SqlTable {
  name: string;
  csv: string;
}

/** The result of rendering a ggsql VISUALISE query. */
export interface PlotOutcome {
  ok: boolean;
  /** A Vega-Lite specification object (present when ok). */
  spec?: unknown;
  /** Present when ok is false. Already student-readable. */
  error?: string;
}

/** How to read an uploaded file. All optional: sensible defaults are detected. */
export interface ImportOptions {
  /** Rows to skip before the data starts. */
  skipRows?: number;
  /** Whether the first (post-skip) row holds the column names. */
  header?: boolean;
  /** CSV field separator. */
  delimiter?: string;
  /** Excel: which sheet to read. */
  sheet?: string;
  /** RData: which object in the file to import. */
  object?: string;
  /** Whole-workspace restore (5d): load every object/table, not one dataset. */
  restore?: boolean;
  /** How to resolve a restored name that already exists in the session. */
  conflict?: "overwrite" | "skip" | "rename";
  /** The student confirmed they trust the file. Required to restore a pickle,
   * which can run code when it is loaded. */
  trusted?: boolean;
}

/** One object or table inside a workspace file being restored (5d). */
export interface WorkspaceMember {
  name: string;
  /** True when a variable/table of this name already exists in the session. */
  collides: boolean;
}

/** A sample of an uploaded file, to show before importing. */
export interface FilePreview {
  /** The literal file text, for text formats (csv, json); absent for binary. */
  rawText?: string;
  /** The parsed sample: columns and the first rows. */
  columns: string[];
  rows: Record<string, unknown>[];
  /** Total rows in the file, when known. */
  totalRows?: number;
  /** Excel sheet names, so the student can choose one. */
  sheets?: string[];
  /** RData object names, so the student can choose one. */
  objects?: string[];
  /** Set for a whole-workspace file (5d): the objects/tables it will restore and
   * which collide with the session, so the dialog can offer conflict handling. */
  restore?: boolean;
  members?: WorkspaceMember[];
  /** Set when the sample could not be parsed with the current options (for
   * example junk rows before the header); the raw text still shows, so the
   * student can see the file and adjust. */
  parseError?: string;
}

/** A request to preview or import an uploaded file. */
export interface FileRequest {
  /** The target variable/table name (already a valid identifier). */
  name: string;
  format: "csv" | "json" | "xlsx" | "parquet" | "rdata" | "pkl" | "sqlite";
  bytes: Uint8Array;
  options: ImportOptions;
}

/** The reply shape a worker sends for any request (run, data, completion). */
interface WorkerReply {
  ok: boolean;
  result?: RunResult;
  error?: string;
  data?: DataPage;
  completions?: CompletionResult;
  tables?: SqlTable[];
  preview?: FilePreview;
  exported?: { text?: string; bytes?: Uint8Array; skipped?: string[]; empty?: boolean };
  doc?: DocText;
}

export interface RunOutcome {
  ok: boolean;
  result?: RunResult;
  /** Present when ok is false. Already student-readable. */
  error?: string;
}

/** Default cap on a single run before the worker is terminated. A language may
 * override it (see RunnableLanguage.runTimeoutMs), for example R, whose runtime
 * and packages are far heavier to load. */
export const RUN_TIMEOUT_MS = 15_000;

/** A documentation lookup is read-only introspection, but a first-ever one may wait
 * on the runtime (or pandas) finishing loading. Give it generous headroom, but bound
 * it so a slow or missing doc falls back to the blurb + link rather than hanging the
 * pane. The fetch is off the editor's critical path, so this never blocks typing. */
const DOC_TIMEOUT_MS = 20_000;

export function isRunSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof Worker !== "undefined" &&
    typeof WebAssembly !== "undefined"
  );
}

interface Pending {
  resolve: (reply: WorkerReply) => void;
  timer: ReturnType<typeof setTimeout>;
}

class LanguageRunner {
  private worker: Worker | null = null;
  private pending = new Map<number, Pending>();
  private nextId = 1;

  constructor(private readonly language: RunnableLanguage) {}

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const worker = new Worker(this.language.workerUrl, { type: "module" });
    worker.onmessage = (event: MessageEvent) => {
      const { id, ok, result, error, data, completions, tables, preview, exported, doc } =
        event.data ?? {};
      const entry = this.pending.get(id);
      if (!entry) return;
      clearTimeout(entry.timer);
      this.pending.delete(id);
      entry.resolve({ ok, result, error, data, completions, tables, preview, exported, doc });
    };
    worker.onerror = (ev) => {
      // TEMP-DIAG
      console.error(
        "WORKER_ONERROR_DIAG",
        (ev as ErrorEvent)?.message,
        (ev as ErrorEvent)?.filename,
        (ev as ErrorEvent)?.lineno,
      );
      // A worker-level failure (for example the runtime failed to load) fails
      // every outstanding run rather than hanging them.
      for (const [, entry] of this.pending) {
        clearTimeout(entry.timer);
        entry.resolve({
          ok: false,
          error:
            "The runtime could not be loaded. Check your connection and try again.",
        });
      }
      this.pending.clear();
      this.discard();
    };
    this.worker = worker;
    return worker;
  }

  private discard(): void {
    this.worker?.terminate();
    this.worker = null;
  }

  /** Fails any in-flight runs, then discards the worker. The next run rebuilds
   * a fresh one, which for a session is how state is wiped. */
  restart(message = "The session was restarted."): void {
    for (const [, entry] of this.pending) {
      clearTimeout(entry.timer);
      entry.resolve({ ok: false, error: message });
    }
    this.pending.clear();
    this.discard();
  }

  /** Posts one request to the worker, correlating the reply by id and applying
   * the runaway timeout. Shared by code runs and data-viewer fetches. A caller
   * may override the timeout for a legitimately long operation, such as the
   * one-time package prewarm, which must not be mistaken for a runaway. */
  private dispatch(
    payload: Record<string, unknown>,
    timeoutMs = this.language.runTimeoutMs ?? RUN_TIMEOUT_MS,
  ): Promise<WorkerReply> {
    const worker = this.ensureWorker();
    const id = this.nextId++;
    return new Promise<WorkerReply>((resolve) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        // A runaway request: the worker's thread is stuck, so it is terminated
        // and the next request rebuilds it.
        this.discard();
        resolve({
          ok: false,
          error:
            "This stopped because it ran too long. Check for an endless loop, then run it again.",
        });
      }, timeoutMs);

      this.pending.set(id, { resolve, timer });
      worker.postMessage({
        id,
        ...(this.language.id === "r" ? { wsProxy: WS_PROXY } : {}),
        ...payload,
      });
    });
  }

  run(
    code: string,
    opts: { keepState?: boolean; withVariables?: boolean } = {},
  ): Promise<RunOutcome> {
    // keepState carries variables (and, for SQL, the database) between runs: off
    // for the inline Run button, on for a Sandbox session. withVariables also
    // reports the environment for the Variables pane.
    return this.dispatch({
      code,
      keepState: opts.keepState ?? false,
      withVariables: opts.withVariables ?? false,
    });
  }

  /** Fetches one page of a data frame or table for the data viewer. */
  getData(
    name: string,
    offset: number,
    limit: number,
  ): Promise<WorkerReply> {
    return this.dispatch({
      dataRequest: { name, offset, limit },
      keepState: true,
    });
  }

  /** Serializes one named tabular object to CSV/TSV text in the worker's runtime
   * and returns it. Read-only: keepState is on so it sees the session's objects,
   * and for SQL the worker re-selects the table rather than re-running the script. */
  exportObject(
    name: string,
    format: "csv" | "tsv",
  ): Promise<{ ok: boolean; text?: string; error?: string }> {
    return this.dispatch({
      exportRequest: { name, format },
      keepState: true,
    }).then((reply) =>
      reply.ok && reply.exported
        ? { ok: true, text: reply.exported.text }
        : { ok: false, error: reply.error ?? "Could not export that object." },
    );
  }

  /** Serializes the session to one binary artifact in the worker's runtime and returns the
   * bytes. Read-only: keepState is on so it sees the session, and no user code is re-run
   * (SQL serializes the DB, R saves globalenv(), Python pickles data globals). With `names`,
   * only those objects/tables are exported (5e "export selected"); without it, everything.
   * An empty selection resolves ok with empty:true and no bytes, so the UI can say so. */
  exportWorkspace(names?: string[]): Promise<{
    ok: boolean;
    bytes?: Uint8Array;
    skipped?: string[];
    empty?: boolean;
    error?: string;
  }> {
    return this.dispatch({
      exportWorkspace: true,
      names,
      keepState: true,
    }).then((reply) =>
      reply.ok && reply.exported
        ? {
            ok: true,
            bytes: reply.exported.bytes,
            skipped: reply.exported.skipped,
            empty: reply.exported.empty,
          }
        : { ok: false, error: reply.error ?? "Could not export the workspace." },
    );
  }

  /** Fetches the runtime documentation text for one symbol. Read-only: keepState is
   * on so the runtime can introspect the session's live objects, and it runs only the
   * introspection snippet, never the student's code and never a state change. Any
   * failure or missing doc resolves to { found: false } so the pane can fall back to
   * the curated blurb and the open-in-new-tab link. */
  fetchDoc(req: DocRequest): Promise<DocText> {
    return this.dispatch({ docRequest: req, keepState: true }, DOC_TIMEOUT_MS).then(
      (reply) =>
        reply.ok && reply.doc ? reply.doc : { found: false },
    );
  }

  /** Runtime autocomplete candidates for the text before the cursor. */
  complete(prefix: string): Promise<CompletionResult | null> {
    return this.dispatch({
      completeAt: { prefix },
      keepState: true,
    }).then((reply) =>
      reply.ok && reply.completions ? reply.completions : null,
    );
  }

  /** Warms the runtime and preloads bundled packages in the background, so the
   * first run is fast. Given a generous timeout because a first-ever install
   * (before it is cached) is legitimately long and must not be killed. */
  prewarm(): Promise<void> {
    return this.dispatch({ prewarm: true, keepState: true }, PREWARM_TIMEOUT_MS).then(
      () => undefined,
    );
  }

  /**
   * Warms the runtime AND fetches the packages one specific snippet needs, so
   * pressing Run does not stall on a download (2026-07-26).
   *
   * Everything about it is best-effort: it resolves rather than rejects, and the
   * run path still loads whatever is missing. That matters, because this is
   * triggered by hover and focus, and a student who never clicks Run must never
   * see an error from work they did not ask for.
   */
  prepare(code: string, packages: string[]): Promise<void> {
    return this.dispatch(
      {
        prewarm: true,
        keepState: true,
        // Each worker reads only its own field: Python resolves imports from the
        // code itself, R needs the names because its packages are not inferable
        // from `pkg::fn` by any runtime call.
        prepareCode: this.language.id === "python" ? code : undefined,
        preparePackages: this.language.id === "r" ? packages : undefined,
      },
      PREWARM_TIMEOUT_MS,
    ).then(
      () => undefined,
      () => undefined,
    );
  }

  /** Every session table as CSV, to hand to the ggsql plot engine. */
  dumpTables(): Promise<SqlTable[]> {
    return this.dispatch({ dumpTablesRequest: true, keepState: true }).then(
      (reply) => (reply.ok && reply.tables ? reply.tables : []),
    );
  }

  /** Reads a sample of an uploaded file (with the given options) to show before
   * importing. Returns null on failure so the dialog can show an error. */
  previewFile(req: FileRequest): Promise<FilePreview | null> {
    return this.dispatch(
      { fileOp: { mode: "preview", ...req }, keepState: true },
      PREWARM_TIMEOUT_MS,
    ).then((reply) => (reply.ok && reply.preview ? reply.preview : null));
  }

  /** Imports an uploaded file into the session as a named variable or table. */
  importFile(req: FileRequest): Promise<RunOutcome> {
    return this.dispatch(
      { fileOp: { mode: "import", ...req }, keepState: true, withVariables: true },
      PREWARM_TIMEOUT_MS,
    );
  }
}

/**
 * Renders a ggsql VISUALISE query to a Vega-Lite spec in a dedicated worker.
 * Experimental (alpha): only the SQL session uses it, as a plot add-on, with the
 * SQLite worker staying the SQL workhorse. A failure is returned, never thrown,
 * so the caller can fall back to the plain result table.
 */
class GgsqlPlotter {
  private worker: Worker | null = null;
  private pending = new Map<number, (reply: PlotOutcome) => void>();
  private nextId = 1;

  private ensureWorker(): Worker {
    if (this.worker) return this.worker;
    const worker = new Worker("/workers/ggsql-worker.mjs", { type: "module" });
    worker.onmessage = (event: MessageEvent) => {
      const { id, ok, spec, error } = event.data ?? {};
      const resolve = this.pending.get(id);
      if (!resolve) return;
      this.pending.delete(id);
      resolve({ ok, spec, error });
    };
    worker.onerror = () => {
      for (const [, resolve] of this.pending) {
        resolve({ ok: false, error: "The plotting engine could not be loaded." });
      }
      this.pending.clear();
      this.worker?.terminate();
      this.worker = null;
    };
    this.worker = worker;
    return worker;
  }

  private send(payload: Record<string, unknown>, timeoutMs: number): Promise<PlotOutcome> {
    const worker = this.ensureWorker();
    const id = this.nextId++;
    return new Promise<PlotOutcome>((resolve) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        resolve({ ok: false, error: "The plot took too long to render." });
      }, timeoutMs);
      this.pending.set(id, (reply) => {
        clearTimeout(timer);
        resolve(reply);
      });
      worker.postMessage({ id, ...payload });
    });
  }

  render(query: string, tables: SqlTable[]): Promise<PlotOutcome> {
    return this.send({ query, tables }, 20_000);
  }

  prewarm(): Promise<void> {
    return this.send({ prewarm: true }, 60_000).then(() => undefined);
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
    this.pending.clear();
  }
}

/** A first-ever package install can take a couple of minutes; the prewarm is
 * allowed far longer than a normal run so it is never mistaken for a runaway. */
const PREWARM_TIMEOUT_MS = 300_000;

const runners = new Map<string, LanguageRunner>();

/** Runs `code` in `language`'s worker, creating and reusing it as needed.
 * Each run is independent: the worker resets state between runs. */
export function runCode(
  language: RunnableLanguage,
  code: string,
): Promise<RunOutcome> {
  return sharedRunner(language).run(code);
}

function sharedRunner(language: RunnableLanguage): LanguageRunner {
  let runner = runners.get(language.id);
  if (!runner) {
    runner = new LanguageRunner(language);
    runners.set(language.id, runner);
  }
  return runner;
}

/**
 * Fetches a snippet's packages ahead of the student pressing Run, on the same
 * shared runner runCode uses (2026-07-26).
 *
 * Deliberately NOT called on mount. The inline Run button's whole design is that
 * a runtime loads on first Run and never at mount, so a chat page with six code
 * blocks stays light. Prewarming all six would undo that. Callers trigger this on
 * hover or keyboard focus instead: a student about to click has already done one
 * of those, and a student who scrolls past pays nothing.
 */
export function prepareCode(
  language: RunnableLanguage,
  code: string,
  packages: string[],
): Promise<void> {
  return sharedRunner(language).prepare(code, packages);
}

/** A long-lived REPL-style session for the Sandbox: variables (and, for SQL,
 * the database) persist across runs. It owns a dedicated worker, separate from
 * the inline Run button's, so the two never disturb each other's state. */
export interface RunSession {
  run(code: string): Promise<RunOutcome>;
  /** Fetch one page (rows offset..offset+limit) of a frame or table to view. */
  getData(
    name: string,
    offset: number,
    limit: number,
  ): Promise<{ ok: boolean; data?: DataPage; error?: string }>;
  /** Runtime autocomplete for the text before the cursor. */
  complete(prefix: string): Promise<CompletionResult | null>;
  /** Fetch the runtime documentation text for one symbol (docstring for Python, help
   * page for R; SQL has none). Read-only; never runs code or resets the session. */
  fetchDoc(req: DocRequest): Promise<DocText>;
  /** Serialize one tabular object to CSV/TSV text for download (5a). */
  exportObject(
    name: string,
    format: "csv" | "tsv",
  ): Promise<{ ok: boolean; text?: string; error?: string }>;
  /** Serialize the session to one binary artifact for download: everything (5c), or
   * only the named objects/tables when `names` is given (5e "export selected"). */
  exportWorkspace(names?: string[]): Promise<{
    ok: boolean;
    bytes?: Uint8Array;
    skipped?: string[];
    empty?: boolean;
    error?: string;
  }>;
  /** Warm the runtime and preload bundled packages in the background. */
  prewarm(): Promise<void>;
  /** Render a ggsql VISUALISE query against the session's tables (SQL only).
   * Returns a failure rather than throwing, so the caller can fall back. */
  plot(query: string): Promise<PlotOutcome>;
  /** Read a sample of an uploaded file (with options) to preview before import. */
  previewFile(req: FileRequest): Promise<FilePreview | null>;
  /** Import an uploaded file into the session as a named variable or table. */
  importFile(req: FileRequest): Promise<RunOutcome>;
  /** Wipe all state by tearing down the worker; the next run rebuilds it. */
  restart(): void;
  /** Release the worker entirely (on unmount). */
  dispose(): void;
}

export function createSession(language: RunnableLanguage): RunSession {
  const runner = new LanguageRunner(language);
  // Only SQL has a plot engine (ggsql). It is created lazily on the first plot.
  const plotter = language.id === "sql" ? new GgsqlPlotter() : null;
  return {
    run: (code) => runner.run(code, { keepState: true, withVariables: true }),
    getData: (name, offset, limit) => runner.getData(name, offset, limit),
    complete: (prefix) => runner.complete(prefix),
    fetchDoc: (req) => runner.fetchDoc(req),
    exportObject: (name, format) => runner.exportObject(name, format),
    exportWorkspace: (names) => runner.exportWorkspace(names),
    prewarm: () => runner.prewarm(),
    plot: async (query) => {
      if (!plotter) return { ok: false, error: "Plots are only available for SQL." };
      const tables = await runner.dumpTables();
      return plotter.render(query, tables);
    },
    previewFile: (req) => runner.previewFile(req),
    importFile: (req) => runner.importFile(req),
    restart: () => runner.restart(),
    dispose: () => {
      runner.restart("The session was closed.");
      plotter?.dispose();
    },
  };
}
