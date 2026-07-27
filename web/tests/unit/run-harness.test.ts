import { afterEach, describe, expect, it, vi } from "vitest";
import {
  languageFromClassName,
  runnerFor,
  RUNNABLE_LANGUAGES,
} from "@/lib/run/languages";

describe("runnable language gate", () => {
  it("reads the language from a react-markdown code className", () => {
    expect(languageFromClassName("language-sql")).toBe("sql");
    expect(languageFromClassName("hljs language-python foo")).toBe("python");
    expect(languageFromClassName(undefined)).toBeNull();
    expect(languageFromClassName("no-language-here")).toBe("here");
  });

  it("offers Run only for languages we can actually run", () => {
    // SQL, Python and R ship. A block with no runner must render as plain code,
    // so the gate must return null for anything not wired up.
    expect(runnerFor("sql")?.id).toBe("sql");
    expect(runnerFor("sqlite")?.id).toBe("sql");
    expect(runnerFor("python")?.id).toBe("python");
    expect(runnerFor("py")?.id).toBe("python");
    expect(runnerFor("r")?.id).toBe("r");
    expect(runnerFor("javascript")).toBeNull();
    expect(runnerFor(null)).toBeNull();
  });

  it("gives every runnable language a worker url and labels", () => {
    for (const lang of RUNNABLE_LANGUAGES) {
      expect(lang.workerUrl).toMatch(/^\/workers\/.+\.mjs$/);
      expect(lang.label.length).toBeGreaterThan(0);
      expect(lang.loadingLabel.length).toBeGreaterThan(0);
      expect(lang.aliases.length).toBeGreaterThan(0);
      // Every runnable block nudges students on bringing their own data.
      expect(lang.dataHint.length).toBeGreaterThan(0);
    }
  });
});

describe("run manager timeout and worker reuse", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    vi.resetModules();
  });

  /** A fake Worker whose responses the test drives. */
  class FakeWorker {
    static instances: FakeWorker[] = [];
    onmessage: ((e: MessageEvent) => void) | null = null;
    onerror: ((e: unknown) => void) | null = null;
    posted: { id: number; code?: string; keepState?: boolean; withVariables?: boolean; dataRequest?: { name: string; offset: number; limit: number }; exportRequest?: { name: string; format: string }; exportWorkspace?: boolean; docRequest?: { name: string; qualifier?: string; source?: string } }[] = [];
    terminated = false;

    constructor(public url: string) {
      FakeWorker.instances.push(this);
    }
    postMessage(msg: { id: number; code?: string; keepState?: boolean; withVariables?: boolean; dataRequest?: { name: string; offset: number; limit: number }; exportRequest?: { name: string; format: string }; exportWorkspace?: boolean; docRequest?: { name: string; qualifier?: string; source?: string } }) {
      this.posted.push(msg);
    }
    terminate() {
      this.terminated = true;
    }
    /** Simulate the worker replying to a posted message. */
    reply(id: number, data: object) {
      this.onmessage?.({ data: { id, ...data } } as MessageEvent);
    }
  }

  async function load() {
    vi.stubGlobal("window", {} as unknown);
    vi.stubGlobal("Worker", FakeWorker as unknown);
    vi.stubGlobal("WebAssembly", {} as unknown);
    FakeWorker.instances = [];
    return import("@/lib/run/manager");
  }

  const SQL = {
    id: "sql" as const,
    aliases: ["sql"],
    label: "SQL",
    workerUrl: "/workers/sqlite-worker.mjs",
    loadingLabel: "Loading SQLite",
    banner: "SQLite (test fixture)",
    dataHint: "Starts with an empty database.",
  };

  it("runs each inline snippet with state reset (keepState off)", async () => {
    const { runCode } = await load();
    const run = runCode(SQL, "SELECT 1");
    expect(FakeWorker.instances[0].posted[0].keepState).toBe(false);
    FakeWorker.instances[0].reply(1, { ok: true, result: { text: "1 row" } });
    expect((await run).ok).toBe(true);
  });

  it("gives a session its own worker that keeps state across runs", async () => {
    const { runCode, createSession } = await load();

    // An inline run and a session must not share a worker, or one would wipe
    // the other's state.
    const inline = runCode(SQL, "SELECT 1");
    FakeWorker.instances[0].reply(1, { ok: true, result: { text: "1 row" } });
    expect((await inline).ok).toBe(true);

    const session = createSession(SQL);
    const first = session.run("CREATE TABLE t(x)");
    // A second, distinct worker for the session.
    expect(FakeWorker.instances).toHaveLength(2);
    const worker = FakeWorker.instances[1];
    // Session runs ask the worker to keep state.
    expect(worker.posted[0].keepState).toBe(true);
    worker.reply(worker.posted[0].id, { ok: true, result: { text: "ok" } });
    expect((await first).ok).toBe(true);

    // The next session run reuses the same worker (state persists).
    const second = session.run("INSERT INTO t VALUES (1)");
    expect(FakeWorker.instances).toHaveLength(2);
    expect(worker.posted[1].keepState).toBe(true);
    worker.reply(worker.posted[1].id, { ok: true, result: { text: "ok" } });
    expect((await second).ok).toBe(true);
  });

  it("fetches a data page through the session worker", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);

    const pending = session.getData("t", 100, 50);
    const worker = FakeWorker.instances[0];
    // The request carries the frame name and page window, and keeps state.
    expect(worker.posted[0].dataRequest).toEqual({
      name: "t",
      offset: 100,
      limit: 50,
    });
    expect(worker.posted[0].keepState).toBe(true);

    worker.reply(worker.posted[0].id, {
      ok: true,
      data: { columns: ["a"], rows: [{ a: 1 }], totalRows: 300 },
    });
    const res = await pending;
    expect(res.ok).toBe(true);
    expect(res.data?.totalRows).toBe(300);
    expect(res.data?.columns).toEqual(["a"]);
  });

  it("exports a named object through the session worker", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);

    const pending = session.exportObject("t", "csv");
    const worker = FakeWorker.instances[0];
    // The request carries the object name and format, and keeps state (read-only).
    expect(worker.posted[0].exportRequest).toEqual({ name: "t", format: "csv" });
    expect(worker.posted[0].keepState).toBe(true);

    worker.reply(worker.posted[0].id, {
      ok: true,
      exported: { text: "n\n1\n2\n3\n" },
    });
    const res = await pending;
    expect(res.ok).toBe(true);
    expect(res.text).toBe("n\n1\n2\n3\n");
  });

  it("surfaces an export error without a text payload", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);
    const pending = session.exportObject("nope", "tsv");
    const worker = FakeWorker.instances[0];
    worker.reply(worker.posted[0].id, {
      ok: false,
      error: "That object is not a table you can export.",
    });
    const res = await pending;
    expect(res.ok).toBe(false);
    expect(res.error).toMatch(/not a table/i);
  });

  it("exports the whole workspace as bytes through the session worker", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);

    const pending = session.exportWorkspace();
    const worker = FakeWorker.instances[0];
    // A read-only, state-keeping, whole-session request (no object name).
    expect(worker.posted[0].exportWorkspace).toBe(true);
    expect(worker.posted[0].keepState).toBe(true);

    const bytes = new Uint8Array([0x53, 0x51, 0x4c, 0x69, 0x74, 0x65]); // "SQLite"
    worker.reply(worker.posted[0].id, {
      ok: true,
      exported: { bytes, skipped: [], empty: false },
    });
    const res = await pending;
    expect(res.ok).toBe(true);
    expect(res.bytes).toEqual(bytes);
    expect(res.empty).toBe(false);
  });

  it("reports an empty workspace without bytes", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);
    const pending = session.exportWorkspace();
    const worker = FakeWorker.instances[0];
    worker.reply(worker.posted[0].id, { ok: true, exported: { empty: true } });
    const res = await pending;
    expect(res.ok).toBe(true);
    expect(res.empty).toBe(true);
    expect(res.bytes).toBeUndefined();
  });

  it("fetches runtime documentation through the session worker", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);

    const pending = session.fetchDoc({ name: "groupby", qualifier: "df", source: "pandas" });
    const worker = FakeWorker.instances[0];
    // The request carries the symbol, receiver, and source hint, and keeps state
    // (read-only introspection: it must see the session's live objects).
    expect(worker.posted[0].docRequest).toEqual({
      name: "groupby",
      qualifier: "df",
      source: "pandas",
    });
    expect(worker.posted[0].keepState).toBe(true);

    worker.reply(worker.posted[0].id, {
      ok: true,
      doc: { found: true, text: "Group DataFrame using a mapper.", signature: "(by=None)" },
    });
    const res = await pending;
    expect(res.found).toBe(true);
    expect(res.text).toMatch(/Group DataFrame/);
    expect(res.signature).toBe("(by=None)");
  });

  it("reports no-local-docs as a clean not-found, never an error", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);

    const pending = session.fetchDoc({ name: "COUNT", source: "SQLite" });
    const worker = FakeWorker.instances[0];
    // SQLite has no runtime help; the worker answers found:false.
    worker.reply(worker.posted[0].id, { ok: true, doc: { found: false } });
    const res = await pending;
    expect(res.found).toBe(false);
    expect(res.text).toBeUndefined();
  });

  it("treats a worker failure as not-found so the pane falls back", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);

    const pending = session.fetchDoc({ name: "nope" });
    const worker = FakeWorker.instances[0];
    worker.reply(worker.posted[0].id, { ok: false, error: "boom" });
    const res = await pending;
    // No throw, no error surfaced: the pane shows the blurb + link fallback.
    expect(res.found).toBe(false);
  });

  it("restart tears down the session worker and fails in-flight runs", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);

    const pending = session.run("SELECT 1");
    const worker = FakeWorker.instances[0];
    session.restart();
    // The interrupted run resolves as an error, not a hang.
    const outcome = await pending;
    expect(outcome.ok).toBe(false);
    expect(worker.terminated).toBe(true);

    // The next run builds a fresh worker (state wiped).
    session.run("SELECT 2");
    expect(FakeWorker.instances).toHaveLength(2);
    expect(FakeWorker.instances[1].posted[0].keepState).toBe(true);
  });

  it("reuses one worker across runs of the same language", async () => {
    const { runCode } = await load();

    const first = runCode(SQL, "SELECT 1");
    FakeWorker.instances[0].reply(1, { ok: true, result: { text: "1 row" } });
    expect((await first).ok).toBe(true);

    const second = runCode(SQL, "SELECT 2");
    // Still the same single worker, not a new one.
    expect(FakeWorker.instances).toHaveLength(1);
    FakeWorker.instances[0].reply(2, { ok: true, result: { text: "1 row" } });
    expect((await second).ok).toBe(true);
  });

  it("terminates and respawns the worker when a run times out", async () => {
    const { runCode } = await load();
    vi.useFakeTimers();
    try {
      const runaway = runCode(
        SQL,
        "WITH RECURSIVE x AS (SELECT 1 UNION SELECT 1 FROM x) SELECT * FROM x",
      );
      // No reply arrives; run all pending timers so the timeout fires.
      await vi.runAllTimersAsync();
      const outcome = await runaway;

      expect(outcome.ok).toBe(false);
      expect(outcome.error).toMatch(/ran too long|endless loop/i);
      expect(FakeWorker.instances[0].terminated).toBe(true);

      // The next run builds a fresh worker rather than reusing the dead one.
      const next = runCode(SQL, "SELECT 1");
      const fresh = FakeWorker.instances[1];
      expect(FakeWorker.instances).toHaveLength(2);
      // Ids increase across runs, so reply to whatever this run actually posted.
      fresh.reply(fresh.posted[0].id, { ok: true, result: { text: "1 row" } });
      expect((await next).ok).toBe(true);
    } finally {
      vi.useRealTimers();
    }
  });

  it("fails outstanding runs cleanly if the worker errors", async () => {
    const { runCode } = await load();
    const pending = runCode(SQL, "SELECT 1");
    FakeWorker.instances[0].onerror?.(new Event("error"));
    const outcome = await pending;
    expect(outcome.ok).toBe(false);
    expect(outcome.error).toMatch(/could not be loaded/i);
  });

  it("reports run support only where Worker and WebAssembly exist", async () => {
    const { isRunSupported } = await load();
    expect(isRunSupported()).toBe(true);
    vi.unstubAllGlobals();
    // Without a window/Worker, running is not offered.
    const fresh = await import("@/lib/run/manager");
    expect(fresh.isRunSupported()).toBe(false);
  });
});
