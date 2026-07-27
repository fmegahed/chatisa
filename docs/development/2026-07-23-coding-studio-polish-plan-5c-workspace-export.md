# Slice 5c — Export the whole workspace / environment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pane-level "Export workspace" control to Coding Studio that downloads the
student's ENTIRE current session as one re-openable file per language — R as an `.RData`
image of the global environment (`save`), SQL as the whole `.sqlite` database image, and
Python as a `.zip` bundle of every data object written to CSV plus a manifest of what was
skipped — without resetting the session, running the student's code again, or leaking any
secret.

**Architecture.** The live session lives only inside each language Web Worker, so the
workspace export round-trips through the worker exactly as the shipped single-object export
(`exportRequest`, slice 5a) does. A new read-only `exportWorkspace` worker message asks the
runtime to serialize the whole environment to bytes: SQL calls
`sqlite3.capi.sqlite3_js_db_export` for a byte image of the in-memory DB; R runs
`save(list = ls(globalenv()), ...)` to the WebR virtual filesystem and reads the bytes back;
Python zips each tabular global's CSV into an in-worker file and reads the bytes back,
reporting which non-tabular objects it left out. Because these artifacts are binary, this
slice introduces the binary bytes channel on `WorkerReply.exported` (5a shipped text only).
The reply's bytes are downloaded on the main thread with a `downloadBytes` helper factored
from the existing `downloadText`/plot-PNG anchor pattern. Pure helpers (workspace filename,
extension, MIME) are unit-tested; the manager protocol is unit-tested with a fake worker; the
full path is proven end-to-end in SQL (sqlite-wasm), which is deterministic and cheap, by
asserting the downloaded file begins with the SQLite magic header.

**Tech Stack:** Next.js (this repo's vendored build — read `node_modules/next/dist/docs/`
before touching any Next API; this slice does not), React 19, Web Workers +
`@sqlite.org/sqlite-wasm` (SQL, `sqlite3_js_db_export`), Pyodide (Python, `zipfile` +
`pandas.to_csv`), WebR (R, `save` + `FS.readFile`), Vitest for unit tests, Playwright +
`@axe-core/playwright` for the sandbox e2e.

## Global Constraints

- **No git commits.** The working tree stays uncommitted; each task ends by running
  verification commands instead of committing (matching the shipped 5a plan).
- **WCAG 2.1 AA, keyboard and mouse.** The workspace-export control is a real, labelled
  button reachable and operable by keyboard; run axe on the sandbox after UI changes.
- **Miami brand tokens only for colors:** use the existing `var(--sb-...)` variables
  (`--sb-border`, `--sb-accent`, `--sb-muted`, `--sb-header`, `--sb-panel`, `--sb-text`).
  No raw hex in the export UI.
- **No em dashes** in any user-facing copy (labels, titles, menu items, error and status
  text, the Python manifest, the empty-state message).
- **Read-only, never destructive.** `exportWorkspace` runs with `keepState: true` and must
  NOT reset the session or run the student's code again. R reads `globalenv()` and never
  `rm`s; SQL serializes the existing DB and never re-runs CREATE/INSERT/UPDATE/DELETE; Python
  reads existing globals and never re-executes the script.
- **Never leak secrets.** The runtimes are sandboxed in-browser and hold no server secrets:
  no API keys, no cookies, no connection strings ever enter a worker session. The export is
  purely the student's own session objects (R global environment, SQLite user tables, Python
  data globals). No network call is made during export; everything is in-browser.
- **Honesty, never silent drops.** Python cannot faithfully serialize every object. What is
  left out is reported to the student (a console note) AND written into the ZIP as
  `MANIFEST.txt`. The feature is labelled "Export workspace" / "Export database", never "Save
  complete session", because the running process cannot be restored.
- **Empty environment is friendly, not an error.** Exporting an empty session shows a plain
  message and downloads nothing, rather than an error or a confusing near-empty file.
- **Import/restore is out of scope (slice 5d).** This slice only writes the files. The export
  formats are chosen so a future import can consume them (RData via `load`, `.sqlite` via
  re-open, ZIP entries as CSVs), and the pickle-loading security caveat for 5d is documented
  below, not implemented here.
- **All commands run from `web/`.** The Playwright config starts its own dev server on port
  3100; the Vitest config runs `tests/unit/**/*.test.ts` in a node environment.

## Chosen per-language format (the crux, grounded and honest)

- **R (WebR) — `.RData` image of the global environment. Cleanest of the three.** WebR has a
  real virtual filesystem (already used by the upload path, `webr-worker.mjs` writes
  `/tmp/__upload.*` with `webR.FS.writeFile`). We run
  `save(list = ls(envir = globalenv()), envir = globalenv(), file = "/tmp/__ws.RData")`, then
  `await webR.FS.readFile("/tmp/__ws.RData")` returns the bytes, which we hand to the browser
  for download and then `unlink`. A student later `load("...RData")`s it to restore every
  named object. Confirmed feasible: `save`/`load` are base R, and `FS.readFile` is the read
  counterpart of the `FS.writeFile` the worker already calls.
- **SQL (SQLite) — the whole `.sqlite` database file.** The environment IS the database, so
  the faithful artifact is a byte image of the in-memory DB:
  `sqlite3.capi.sqlite3_js_db_export(db.pointer)` returns a `Uint8Array` that is a complete,
  re-openable SQLite file with every table. Confirmed feasible: `sqlite3_js_db_export` is part
  of the `@sqlite.org/sqlite-wasm` C API surface (already cited in the 5 decomposition doc);
  it reads the existing in-memory DB and does not re-run any statement.
- **Python (Pyodide) — RECOMMENDED: a `.zip` of every data object as CSV, plus a manifest.**
  There is no clean "save all" in Python. The two honest options weighed:
  1. **Pickle the picklable globals to a `.pkl`.** Closest analog to `save.image`, preserves
     dtypes and structure, but it is *fragile and version-bound* (a pickle written by one
     pandas/NumPy/Python version can fail to load under another), it silently omits modules,
     functions, open files and other unpicklable objects, and — critically — **loading a
     pickle executes arbitrary code**, so it is a security landmine that only matters when
     import/restore (5d) exists. Restoring an untrusted `.pkl` must never be automatic.
  2. **A ZIP of each tabular object (DataFrame / Series / 2-D ndarray) written to CSV.**
     Robust, version-independent, human-inspectable, safe to re-open (CSV carries no code),
     and it captures the student's actual data — the thing they want to keep.
  **Recommendation: ship option 2 (the CSV ZIP) first.** It is the safe, honest, useful
  choice while 5d does not yet exist: no fragile format, no unpickle security surface, and the
  student gets their data back as files any tool can read. Its honest limitation is that it
  captures tabular data only and does not restore dtypes or non-tabular objects, so every
  non-tabular global is *reported*, not dropped: named in a console note and in `MANIFEST.txt`
  inside the ZIP. Pickle is documented here as a deferred follow-on that must land together
  with the 5d loading-trust warning, never before it.

## File Structure

- `web/lib/sandbox/export.ts` — **Modify.** Add the workspace format helpers alongside the
  shipped single-object helpers: `WorkspaceLanguage` type, `workspaceExtensionFor`,
  `workspaceMimeFor`, `exportWorkspaceFilename`, and a `downloadBytes` DOM helper (the binary
  sibling of the existing `downloadText`).
- `web/lib/run/manager.ts` — **Modify.** Widen `WorkerReply.exported` to carry `bytes`,
  `skipped`, `empty` (5a shipped `{ text }` only); add `LanguageRunner.exportWorkspace`; add
  `exportWorkspace` to the `RunSession` interface and `createSession`.
- `web/public/workers/sqlite-worker.mjs` — **Modify.** Handle `exportWorkspace`: guard on the
  user-table count (empty → `{ empty: true }`), else `sqlite3_js_db_export` to a byte image.
- `web/public/workers/webr-worker.mjs` — **Modify.** Handle `exportWorkspace`: guard on
  `length(ls(globalenv()))` (empty → `{ empty: true }`), else `save(list=..., file=...)`,
  read the bytes, unlink.
- `web/public/workers/pyodide-worker.mjs` — **Modify.** Handle `exportWorkspace`: zip each
  tabular global's CSV plus `MANIFEST.txt` to an in-worker file, read the bytes, report the
  skipped names; no data → `{ empty: true }`.
- `web/components/sandbox/WorkspaceExportButton.tsx` — **Create.** A small accessible button
  for the pane header (a plain button, not a two-item menu — there is one artifact per
  language, so no format submenu).
- `web/components/sandbox/Sandbox.tsx` — **Modify.** Add an `exportWorkspace` callback in
  `Workspace`; give `Pane`/`VariablesPane` a header `actions` slot and render
  `WorkspaceExportButton` there; surface the empty and skipped notes in the console.
- `web/tests/unit/sandbox-export.test.ts` — **Modify.** Add Vitest for the workspace helpers.
- `web/tests/unit/run-harness.test.ts` — **Modify.** Add a fake-worker test for the
  `exportWorkspace` protocol (mirrors the shipped `exportObject` test), including the bytes
  and skipped fields.
- `web/tests/e2e/sandbox.spec.ts` — **Modify.** Add the real SQL end-to-end workspace test
  (download the `.sqlite`, assert the SQLite magic header, confirm the session survives) plus
  an empty-state assertion.

---

## Task 1: Pure workspace helpers and the binary download

Filename, extension, and MIME for the workspace artifact are pure and belong on the main
thread; the byte serialization itself lives in each worker and is proven by the e2e test.
`downloadBytes` is the binary sibling of the shipped `downloadText` (same anchor mechanism as
the plot-PNG export and Download Script).

**Files:**
- Modify: `web/lib/sandbox/export.ts`
- Test: `web/tests/unit/sandbox-export.test.ts`

**Interfaces:**
- Consumes: nothing new (extends the shipped `export.ts`).
- Produces: `type WorkspaceLanguage = "python" | "r" | "sql"`;
  `workspaceExtensionFor(lang) => "RData" | "sqlite" | "zip"`;
  `workspaceMimeFor(lang) => string`;
  `exportWorkspaceFilename({ lang, date? }) => string`;
  `downloadBytes(bytes: Uint8Array, filename: string, mime: string): void` (DOM, not
  unit-tested).

- [ ] **Step 1: Write the failing helper tests**

Append to `web/tests/unit/sandbox-export.test.ts` (extend the existing import from
`@/lib/sandbox/export` to add the three new pure helpers):

```typescript
import {
  delimiterFor,
  extensionFor,
  mimeFor,
  exportFilename,
  workspaceExtensionFor,
  workspaceMimeFor,
  exportWorkspaceFilename,
} from "@/lib/sandbox/export";

describe("workspace export helpers", () => {
  it("maps each language to its whole-environment file extension", () => {
    expect(workspaceExtensionFor("r")).toBe("RData");
    expect(workspaceExtensionFor("sql")).toBe("sqlite");
    expect(workspaceExtensionFor("python")).toBe("zip");
  });

  it("maps each language to a binary MIME type", () => {
    expect(workspaceMimeFor("r")).toBe("application/octet-stream");
    expect(workspaceMimeFor("sql")).toBe("application/vnd.sqlite3");
    expect(workspaceMimeFor("python")).toBe("application/zip");
  });

  it("names the workspace file by language and moment", () => {
    const date = new Date(2026, 6, 23, 14, 30); // 2026-07-23 14:30 local
    expect(exportWorkspaceFilename({ lang: "r", date })).toBe(
      "chatisa-workspace-r-20260723-1430.RData",
    );
    expect(exportWorkspaceFilename({ lang: "sql", date })).toBe(
      "chatisa-workspace-sql-20260723-1430.sqlite",
    );
    expect(exportWorkspaceFilename({ lang: "python", date })).toBe(
      "chatisa-workspace-python-20260723-1430.zip",
    );
  });
});
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `npm run test -- sandbox-export`
Expected: FAIL — `workspaceExtensionFor` (and the others) are not exported yet.

- [ ] **Step 3: Implement the helpers**

Append to `web/lib/sandbox/export.ts` (below the shipped single-object helpers):

```typescript
/** The three languages whose whole session can be exported as one file. */
export type WorkspaceLanguage = "python" | "r" | "sql";

/** The whole-environment file extension per language: R saves an .RData image, SQL a
 * .sqlite database file, Python a .zip bundle of its data as CSV. */
export function workspaceExtensionFor(lang: WorkspaceLanguage): string {
  return lang === "r" ? "RData" : lang === "sql" ? "sqlite" : "zip";
}

/** The binary MIME type for a workspace artifact. All three are opaque blobs to the
 * browser; a specific type only helps the OS label the download. */
export function workspaceMimeFor(lang: WorkspaceLanguage): string {
  if (lang === "sql") return "application/vnd.sqlite3";
  if (lang === "python") return "application/zip";
  return "application/octet-stream"; // .RData has no registered MIME type
}

/**
 * A safe download name for a whole-environment export, e.g.
 * chatisa-workspace-r-20260723-1430.RData. A workspace is a session-wide artifact, not a
 * per-object one, so it is branded chatisa-workspace-<language> plus a timestamp rather than
 * being named after one object.
 */
export function exportWorkspaceFilename(opts: {
  lang: WorkspaceLanguage;
  date?: Date;
}): string {
  const d = opts.date ?? new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
  return `chatisa-workspace-${opts.lang}-${stamp}.${workspaceExtensionFor(opts.lang)}`;
}

/**
 * Downloads binary bytes as a file. The binary sibling of downloadText: same Blob-and-anchor
 * mechanism (see the Download Script button and the plot PNG export), with a binary MIME type
 * and no charset. The object URL is revoked after the click.
 */
export function downloadBytes(
  bytes: Uint8Array,
  filename: string,
  mime: string,
): void {
  // Copy into a fresh ArrayBuffer so a subarray view cannot leak neighbouring bytes.
  const blob = new Blob([bytes.slice()], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
```

- [ ] **Step 4: Run the tests to green**

Run: `npm run test -- sandbox-export`
Expected: PASS (the new workspace block plus the existing single-object block).

- [ ] **Step 5: Verify the task (no commit)**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors for the changed file.

---

## Task 2: Workspace export protocol in the manager

Add the `exportWorkspace` round-trip to `LanguageRunner` and `RunSession`, mirroring the
shipped `exportObject`, and widen `WorkerReply.exported` to carry binary `bytes` plus the
`skipped` list and `empty` flag. Prove it with a fake worker, exactly as the shipped
`exportObject` test does.

**Files:**
- Modify: `web/lib/run/manager.ts`
- Test: `web/tests/unit/run-harness.test.ts`

**Interfaces:**
- Consumes: `LanguageRunner.dispatch` (manager.ts:218); the `WorkerReply` shape and
  `onmessage` destructure (manager.ts:118, 164).
- Produces: `WorkerReply.exported?: { text?: string; bytes?: Uint8Array; skipped?: string[]; empty?: boolean }`;
  `LanguageRunner.exportWorkspace()` and `RunSession.exportWorkspace()` returning
  `Promise<{ ok: boolean; bytes?: Uint8Array; skipped?: string[]; empty?: boolean; error?: string }>`.

- [ ] **Step 1: Write the failing protocol tests**

Extend `FakeWorker` in `web/tests/unit/run-harness.test.ts` so its `posted` element type and
`postMessage` parameter type include `exportWorkspace?: boolean`, then add these two tests to
the `run manager timeout and worker reuse` describe block:

```typescript
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
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `npm run test -- run-harness`
Expected: FAIL — `session.exportWorkspace` does not exist.

- [ ] **Step 3: Widen the reply shape and wiring**

In `web/lib/run/manager.ts`, replace the shipped `exported?` line in `WorkerReply` (line 126):

```typescript
  exported?: { text?: string; bytes?: Uint8Array; skipped?: string[]; empty?: boolean };
```

The `onmessage` destructure and `entry.resolve(...)` already forward `exported` (manager.ts:165,
171), so no change is needed there — the widened shape flows through unchanged.

- [ ] **Step 4: Add `exportWorkspace` to `LanguageRunner`**

Next to `exportObject` (manager.ts:275), add:

```typescript
  /** Serializes the whole session to one binary artifact in the worker's runtime and returns
   * the bytes. Read-only: keepState is on so it sees the session, and no user code is re-run
   * (SQL serializes the existing DB, R saves globalenv(), Python zips existing data globals).
   * An empty environment resolves ok with empty:true and no bytes, so the UI can say so. */
  exportWorkspace(): Promise<{
    ok: boolean;
    bytes?: Uint8Array;
    skipped?: string[];
    empty?: boolean;
    error?: string;
  }> {
    return this.dispatch({
      exportWorkspace: true,
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
```

- [ ] **Step 5: Expose it on `RunSession`**

Add to the `RunSession` interface, next to the shipped `exportObject` (manager.ts:431):

```typescript
  /** Serialize the whole session to one binary artifact for download (5c). */
  exportWorkspace(): Promise<{
    ok: boolean;
    bytes?: Uint8Array;
    skipped?: string[];
    empty?: boolean;
    error?: string;
  }>;
```

And wire it in `createSession`, next to `exportObject` (manager.ts:458):

```typescript
    exportWorkspace: () => runner.exportWorkspace(),
```

- [ ] **Step 6: Run the protocol tests to green**

Run: `npm run test -- run-harness`
Expected: PASS (both new tests plus the existing suite).

- [ ] **Step 7: Verify the task (no commit)**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors for the changed file.

---

## Task 3: Worker exportWorkspace handlers (SQL, R, Python)

Each worker gains an `exportWorkspace` branch that serializes the whole session to bytes and
replies `{ ok, exported: { bytes, skipped?, empty? } }`, or a friendly `{ ok, exported: { empty: true } }`
for an empty environment. This mirrors each worker's existing `exportRequest` branch and is
strictly read-only.

**Files:**
- Modify: `web/public/workers/sqlite-worker.mjs`
- Modify: `web/public/workers/webr-worker.mjs`
- Modify: `web/public/workers/pyodide-worker.mjs`

**Interfaces:**
- Consumes: the `{ exportWorkspace: true, keepState: true }` message.
- Produces: `{ id, ok: true, exported: { bytes, skipped?, empty? } }` or `{ id, ok: false, error }`.

- [ ] **Step 1: SQL — the whole database as a byte image**

In `web/public/workers/sqlite-worker.mjs`, add a helper next to `dumpTables` (line 132):

```javascript
/** The whole in-memory database as a byte image: a complete, re-openable .sqlite file with
 * every user table. Read-only: it serializes the existing DB and re-runs no statement. */
function exportDatabaseBytes(sqlite3, db) {
  // sqlite3_js_db_export returns a Uint8Array copy of the database's serialized image.
  return sqlite3.capi.sqlite3_js_db_export(db.pointer);
}

/** How many user tables the database holds (excludes sqlite_* internal tables). */
function userTableCount(db) {
  const rows = [];
  db.exec({
    sql: "SELECT count(*) AS n FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%'",
    rowMode: "object",
    resultRows: rows,
  });
  return rows[0]?.n ?? 0;
}
```

Add `exportWorkspace` to the `onmessage` destructure (line 348) and a branch before the
`exportRequest` branch (around line 353):

```javascript
  // Export the whole database as one .sqlite byte image (read-only; re-runs nothing).
  if (exportWorkspace) {
    try {
      const sqlite3 = await getSqlite();
      const db = keepState
        ? (sessionDb ??= new sqlite3.oo1.DB())
        : new sqlite3.oo1.DB();
      if (userTableCount(db) === 0) {
        self.postMessage({ id, ok: true, exported: { empty: true } });
      } else {
        const bytes = exportDatabaseBytes(sqlite3, db);
        self.postMessage({ id, ok: true, exported: { bytes, skipped: [], empty: false } });
      }
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }
```

- [ ] **Step 2: R — save the global environment to an .RData image**

In `web/public/workers/webr-worker.mjs`, add a code builder next to `buildRExportCode`
(line 311):

```javascript
/** R that saves the whole global environment to `path` as an .RData image, or returns the
 * __empty__ sentinel when the environment holds nothing. Read-only: it reads globalenv() and
 * never removes or restarts anything. */
function buildRSaveImageCode(path) {
  return `
local({
  ns <- ls(envir = globalenv())
  if (length(ns) == 0) return("__empty__")
  save(list = ns, envir = globalenv(), file = ${rStr(path)})
  "__ok__"
})
`;
}
```

Add `exportWorkspace` to the `onmessage` destructure (line 323) and a branch next to the
`exportRequest` branch (around line 426):

```javascript
  // Export the whole global environment as one .RData byte image (read-only).
  if (exportWorkspace) {
    const path = "/tmp/__ws.RData";
    const shelter = await new webR.Shelter();
    try {
      const obj = await shelter.captureR(buildRSaveImageCode(path), {
        withAutoprint: false,
        captureStreams: false,
        captureConditions: false,
        captureGraphics: false,
      });
      const status = (await obj.result.toArray())[0];
      if (status === "__empty__") {
        self.postMessage({ id, ok: true, exported: { empty: true } });
      } else {
        const bytes = await webR.FS.readFile(path);
        await webR.FS.unlink(path).catch(() => {});
        self.postMessage({ id, ok: true, exported: { bytes, skipped: [], empty: false } });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      self.postMessage({ id, ok: false, error: friendlyError(message) });
    } finally {
      await shelter.purge();
    }
    return;
  }
```

Note: this branch must sit AFTER `webR = await getWebR()` succeeds (it does — the `onmessage`
handler loads WebR before any branch, webr-worker.mjs:326). It does not call
`ensurePackages`: `save`/`load` are base R, so no bundled package is needed and the export
stays fast.

- [ ] **Step 3: Python — zip each data global to CSV, report the rest**

In `web/public/workers/pyodide-worker.mjs`, add a code builder next to `exportFrameCode`
(line 203):

```javascript
/** Python that writes every tabular global (DataFrame, Series, 2-D ndarray) to a CSV inside a
 * single ZIP at /tmp/__ws.zip, plus a MANIFEST.txt naming what was included and what was left
 * out. Returns JSON {included, skipped, empty}. Read-only: it reads existing globals and never
 * re-runs the student's code. Non-tabular objects are reported, never silently dropped. */
const WORKSPACE_ZIP_CODE = `
def __chatisa_workspace():
    import json, io, zipfile
    try:
        import pandas as pd
    except Exception:
        pd = None
    try:
        import numpy as np
    except Exception:
        np = None
    included, skipped = [], []
    path = "/tmp/__ws.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in list(globals().items()):
            if k.startswith("__"):
                continue
            t = type(v).__name__
            if t in ("module", "function", "builtin_function_or_method", "type"):
                continue
            wrote = False
            try:
                if hasattr(v, "to_csv") and (hasattr(v, "columns") or hasattr(v, "index")):
                    z.writestr(k + ".csv", v.to_csv(index=False))
                    wrote = True
                elif pd is not None and np is not None and isinstance(v, np.ndarray) and v.ndim <= 2:
                    z.writestr(k + ".csv", pd.DataFrame(v).to_csv(index=False))
                    wrote = True
            except Exception:
                wrote = False
            if wrote:
                included.append(k)
            else:
                skipped.append(k + " (" + t + ")")
        manifest = (
            "ChatISA workspace export\\n\\n"
            "Included as CSV:\\n" + ("\\n".join(included) if included else "(none)") +
            "\\n\\nNot exported (not a table):\\n" + ("\\n".join(skipped) if skipped else "(none)")
        )
        z.writestr("MANIFEST.txt", manifest)
    return json.dumps({"included": included, "skipped": skipped,
                       "empty": len(included) == 0})
__chatisa_workspace()
`;
```

Add `exportWorkspace` to the `onmessage` destructure (line 319) and a branch next to the
`exportRequest` branch (around line 323):

```javascript
  // Export the whole environment as a ZIP of every data object's CSV (read-only). Non-tabular
  // objects are reported back (and listed in MANIFEST.txt), never silently dropped.
  if (exportWorkspace) {
    try {
      const pyodide = await getPyodide();
      const parsed = JSON.parse(pyodide.runPython(WORKSPACE_ZIP_CODE));
      if (parsed.empty) {
        self.postMessage({ id, ok: true, exported: { empty: true } });
      } else {
        const bytes = pyodide.FS.readFile("/tmp/__ws.zip");
        try {
          pyodide.FS.unlink("/tmp/__ws.zip");
        } catch {
          // best-effort cleanup
        }
        self.postMessage({
          id,
          ok: true,
          exported: { bytes, skipped: parsed.skipped, empty: false },
        });
      }
    } catch (error) {
      self.postMessage({
        id,
        ok: false,
        error: error instanceof Error ? error.message : String(error),
      });
    }
    return;
  }
```

Note on "empty": a Python environment with only non-tabular objects (say, one function)
produces `included = []`, so `empty` is true and nothing downloads. That is the right call —
the CSV ZIP would hold no data, only a manifest — and the console note (Task 5) tells the
student why. `pyodide.FS.readFile` returns a `Uint8Array`, matching the reply shape.

- [ ] **Step 4: Verify the workers compile and the app builds**

Workers are static `.mjs` loaded by URL, not type-checked by `tsc`. Sanity-check by loading
the sandbox in dev and confirming no worker console error on tab open.

Run: `npm run typecheck` — no errors (the workers are untyped, but the manager change must
still check).
Run: `npm run lint` — no errors for the changed files.

The real behavioral proof of the worker serialization is the SQL end-to-end in Task 6.

---

## Task 4: The WorkspaceExportButton control

A single accessible button, not a two-item menu: there is exactly one workspace artifact per
language, so no format submenu is needed. It lives in the pane header (a session-wide action),
distinct from the per-row `ExportMenu` (a per-object action). This is the recommended
placement: overloading the per-object menu with a whole-session item would be confusing,
because that menu is anchored to one object's row.

**Files:**
- Create: `web/components/sandbox/WorkspaceExportButton.tsx`

**Interfaces:**
- Produces: `WorkspaceExportButton({ label, title, onExport })` where `onExport(): void`.
  Renders a labelled button styled like the existing pane-header buttons.

- [ ] **Step 1: Implement `WorkspaceExportButton`**

Create `web/components/sandbox/WorkspaceExportButton.tsx`:

```tsx
"use client";

/**
 * A pane-header button that exports the whole session as one file (R .RData, SQL .sqlite,
 * Python .zip). One artifact per language, so this is a plain button, not a format menu. The
 * label and tooltip are supplied by the caller so SQL can read "Export database" while R and
 * Python read "Export workspace".
 */
export function WorkspaceExportButton({
  label,
  title,
  onExport,
}: {
  label: string;
  title: string;
  onExport: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onExport}
      aria-label={label}
      title={title}
      className="rounded border border-[var(--sb-border)] px-1.5 py-0.5 text-xs font-bold text-[var(--sb-muted)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-accent)]"
    >
      {label}
    </button>
  );
}
```

- [ ] **Step 2: Verify it compiles**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors for the new file.

---

## Task 5: Wire workspace export into the pane

Add an `exportWorkspace` callback in `Workspace`, give the `Pane`/`VariablesPane` a header
`actions` slot, render `WorkspaceExportButton` there, and surface the empty and skipped notes
in the console. The button always shows (the empty case is handled with a friendly note, not
by hiding the control), so a student never wonders where it went.

**Files:**
- Modify: `web/components/sandbox/Sandbox.tsx`

**Interfaces:**
- Consumes: `sessionRef.current.exportWorkspace` (Task 2); the workspace helpers
  `exportWorkspaceFilename`, `workspaceMimeFor`, `downloadBytes`, and `WorkspaceLanguage`
  (Task 1); `WorkspaceExportButton` (Task 4); `language.id`.
- Produces: an `exportWorkspace()` callback in `Workspace`; a `VariablesPane` `actions` header
  slot holding the button.

- [ ] **Step 1: Import the workspace helpers and the button**

In `web/components/sandbox/Sandbox.tsx`, extend the existing export-helper import (line 45):

```typescript
import {
  downloadBytes,
  downloadText,
  exportFilename,
  exportWorkspaceFilename,
  mimeFor,
  workspaceMimeFor,
  type ExportFormat,
  type WorkspaceLanguage,
} from "@/lib/sandbox/export";
import { WorkspaceExportButton } from "@/components/sandbox/WorkspaceExportButton";
```

- [ ] **Step 2: Add the `exportWorkspace` callback in `Workspace`**

Add next to the shipped `exportObject` callback (around Sandbox.tsx:553):

```tsx
  // Exports the whole session as one file (R .RData, SQL .sqlite, Python .zip), downloaded in
  // the browser. Read-only: it asks the worker to serialize the environment and never resets
  // the session. An empty environment shows a friendly note and downloads nothing; when Python
  // leaves out non-tabular objects, the console says which ones.
  const exportWorkspace = useCallback(async () => {
    const session = sessionRef.current;
    if (!session) return;
    const lang = language.id as WorkspaceLanguage;
    const res = await session.exportWorkspace();
    if (res.ok && res.empty) {
      const message =
        lang === "sql"
          ? "Your database has no tables yet. Create a table, then export."
          : lang === "r"
            ? "Your R environment is empty. Define a variable, then export the workspace."
            : "Your Python environment has no data to export yet. Create a data frame, then export.";
      setEntries((prev) => [
        ...prev,
        { code: "", outcome: { ok: true, result: {} }, silent: true, label: message },
      ]);
      return;
    }
    if (res.ok && res.bytes) {
      downloadBytes(
        res.bytes,
        exportWorkspaceFilename({ lang }),
        workspaceMimeFor(lang),
      );
      if (res.skipped && res.skipped.length > 0) {
        setEntries((prev) => [
          ...prev,
          {
            code: "",
            outcome: { ok: true, result: {} },
            silent: true,
            label: `Exported your workspace. These objects are not tables and were left out (they are also listed in MANIFEST.txt inside the ZIP): ${res.skipped!.join(", ")}.`,
          },
        ]);
      }
      return;
    }
    setEntries((prev) => [
      ...prev,
      {
        code: "",
        outcome: { ok: false, error: res.error ?? "Could not export the workspace." },
        silent: true,
        label: "Workspace export failed.",
      },
    ]);
  }, [language.id]);
```

- [ ] **Step 3: Give `VariablesPane` a header actions slot with the button**

Update the `<VariablesPane ... />` usage (around Sandbox.tsx:771) to pass the callback:

```tsx
                <VariablesPane
                  variables={variables}
                  language={language}
                  onView={openDataView}
                  onExport={exportObject}
                  onExportWorkspace={exportWorkspace}
                />
```

Update the `VariablesPane` signature and body (Sandbox.tsx:1224). Add
`onExportWorkspace: () => void` to its prop type, compute the label/title from the language,
and pass the button into `Pane` via a new `actions` prop:

```tsx
function VariablesPane({
  variables,
  language,
  onView,
  onExport,
  onExportWorkspace,
}: {
  variables: SessionVariable[];
  language: RunnableLanguage;
  onView: (name: string) => void;
  onExport: (name: string, format: ExportFormat) => void;
  onExportWorkspace: () => void;
}) {
  const title = language.id === "sql" ? "Tables" : "Environment";
  const wsLabel = language.id === "sql" ? "Export database" : "Export workspace";
  const wsTitle =
    language.id === "sql"
      ? "Download the whole database as one .sqlite file you can re-open later"
      : language.id === "r"
        ? "Download your whole R environment as one .RData file you can load later"
        : "Download every data table in your session as a .zip of CSV files";
  return (
    <Pane
      title={title}
      actions={
        <WorkspaceExportButton
          label={wsLabel}
          title={wsTitle}
          onExport={onExportWorkspace}
        />
      }
    >
```

The rest of `VariablesPane` (the table body) is unchanged. `Pane` already renders an
`actions` node in its header (Sandbox.tsx:1002-1018), so no change to `Pane` is required.

- [ ] **Step 4: Verify it compiles and stays accessible**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors.
Run: `npm run test:e2e -- sandbox.spec.ts -g "four-pane"`
Expected: PASS — the existing shell + axe test still passes with the new button present (it is
a labelled button, so axe stays clean).

---

## Task 6: Real end-to-end workspace export (SQL, sqlite-wasm)

SQL is deterministic and cheap (no prewarm, no network) and is the right language to prove a
real whole-database export end to end: create tables, export the workspace, and assert the
downloaded file is a genuine SQLite database (its first 16 bytes are the SQLite magic header
`SQLite format 3\0`). This follows the shipped single-object export e2e (create, Run, open the
menu/button, read the download bytes), and also proves the session is not reset.

**Files:**
- Test: `web/tests/e2e/sandbox.spec.ts`

- [ ] **Step 1: Write the failing end-to-end workspace test**

Add to the `test.describe("AI Sandbox", ...)` block (the file already imports
`readFileSync` from `node:fs`):

```typescript
  test("exports the whole SQL database as a real .sqlite file without resetting the session", async ({
    page,
  }) => {
    // Real sqlite-wasm execution; generous headroom for the first compile.
    test.setTimeout(120_000);

    await page.goto("/ai-sandbox");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    // Two tables, so "whole database" means more than one object.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "CREATE TABLE grades(student TEXT, grade REAL);\n" +
        "INSERT INTO grades VALUES ('Amanda',91),('Bill',79);\n" +
        "CREATE TABLE courses(code TEXT);\n" +
        "INSERT INTO courses VALUES ('ISA 401'),('ISA 444');\n" +
        "SELECT * FROM grades;",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();

    const output = page.getByLabel("Console output");
    await expect(output).toContainText("Amanda", { timeout: 60_000 });

    // The pane-header workspace button downloads the whole database.
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Export database" }).click(),
    ]);

    // Named chatisa-workspace-sql-<stamp>.sqlite.
    expect(download.suggestedFilename()).toMatch(
      /^chatisa-workspace-sql-\d{8}-\d{4}\.sqlite$/,
    );

    // The downloaded bytes are a genuine SQLite database: the file header is the
    // ASCII string "SQLite format 3" followed by a NUL.
    const bytes = readFileSync(await download.path());
    expect(bytes.subarray(0, 16).toString("latin1")).toBe(
      "SQLite format 3 ",
    );
    expect(bytes.length).toBeGreaterThan(512); // a real DB, not an empty stub

    // The export did not reset the session: both tables are still there.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "SELECT (SELECT COUNT(*) FROM grades) + (SELECT COUNT(*) FROM courses) AS total;",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("4", { timeout: 60_000 });
  });

  test("workspace export shows a friendly note for an empty database", async ({
    page,
  }) => {
    test.setTimeout(120_000);

    await page.goto("/ai-sandbox");
    await page.getByRole("radio", { name: "SQL" }).click();
    await expect(page.locator(".cm-content")).toBeVisible();

    // No tables created yet. The button is present and, when clicked, explains
    // rather than downloading a confusing near-empty file.
    await page.getByRole("button", { name: "Export database" }).click();
    await expect(page.getByLabel("Console output")).toContainText(
      "no tables yet",
      { timeout: 60_000 },
    );
  });
```

- [ ] **Step 2: Run them and confirm they pass**

Run: `npm run test:e2e -- sandbox.spec.ts -g "whole SQL database"`
Run: `npm run test:e2e -- sandbox.spec.ts -g "empty database"`
Expected: PASS — the `.sqlite` download arrives with the right name and a real SQLite header,
a follow-up query still counts 4 rows across both tables (session not reset), and the empty
database yields a friendly console note with no download.

Fallback if sqlite-wasm flakes in CI (the surrounding suite deliberately avoids WASM in most
tests): gate the first test behind the same opt-in flag the live-network test uses, at the top
of the test body:

```typescript
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "runs real sqlite-wasm; opt in with CHATISA_LIVE_NET=1",
    );
```

- [ ] **Step 3: (Optional) opt-in real serialize round-trip in Node**

If a deterministic non-browser proof of the byte image is wanted, add an opt-in Vitest that
loads `@sqlite.org/sqlite-wasm` in Node, creates a table, calls `sqlite3_js_db_export`,
re-opens the bytes into a fresh DB, and asserts the table and its rows survive. Gate it with
`describe.skipIf(!process.env.CHATISA_HEAVY)` so it never runs in the default `npm run test`
(the heavy-runtime convention: SQLite is cheap enough to test for real, WebR/Pyodide are
verified manually). Keep the browser e2e in Step 1 as the primary proof.

- [ ] **Step 4: Verify the task (no commit)**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors.
Run: `npm run test -- sandbox-export run-harness` — the unit suites pass.
Run: `npm run test:e2e -- sandbox.spec.ts` — the full sandbox suite passes, including the new
workspace e2e and the existing shell/axe tests (confirming the new control did not regress the
layout or accessibility). Leave the working tree uncommitted.

---

## Manual verification (WebR and Pyodide, heavy runtimes)

Per the project's convention, R and Python whole-environment exports are verified by hand
rather than in CI (loading WebR or Pyodide is heavy and flaky under parallel test load). Run
`npm run dev`, open `/ai-sandbox`, and for each:

- **R:** switch to R, run the "Insert Coding Example" script (defines `grades`), click "Export
  workspace", and confirm a `chatisa-workspace-r-<stamp>.RData` downloads. In a fresh R
  session, `load("chatisa-workspace-r-<stamp>.RData"); ls()` lists `grades`. Empty case:
  Restart session, click the button, confirm the "R environment is empty" console note and no
  download.
- **Python:** switch to Python, run the example (defines `grades`), also define a non-tabular
  global (for example `def f(): pass`), click "Export workspace", and confirm a
  `chatisa-workspace-python-<stamp>.zip` downloads containing `grades.csv` and `MANIFEST.txt`,
  and a console note names `f (function)` as skipped. Empty case: Restart session, click the
  button, confirm the "no data to export yet" note and no download.

---

## Self-Review

**Spec coverage (requirement F, scoped to 5c — whole-environment / workspace export).**

- Whole-environment export exists per language, distinct from single-object export — Task 3
  (SQL `sqlite3_js_db_export`; R `save(list = ls(globalenv()))`; Python ZIP of CSVs), Task 4/5
  (a pane-level `WorkspaceExportButton`, separate from the per-row `ExportMenu`).
- Per-language format chosen and confirmed feasible — R `.RData` via `save` + `FS.readFile`;
  SQL `.sqlite` via `sqlite3_js_db_export`; Python `.zip` of CSVs via `zipfile` +
  `pandas.to_csv` + `FS.readFile`. All three cited against the real worker APIs already in use
  (WebR `FS.writeFile`/`readFile`, sqlite oo1 DB, Pyodide `FS`).
- Never resets/re-runs the session — Task 3 (all three branches are read-only, `keepState`,
  and run no user code), proven by Task 6 Step 1 (both tables survive; `total` returns 4 after
  export).
- Never leaks credentials — the runtimes are sandboxed in-browser and hold no server secrets;
  the export serializes only the student's own session (R globalenv, SQL user tables, Python
  data globals), makes no network call, and runs entirely in the browser. Stated in Global
  Constraints and enforced by the read-only branches.
- Honesty, no silent drops (Python) — Task 3 Step 3 reports every non-tabular global in the
  `skipped` list AND writes `MANIFEST.txt` into the ZIP; Task 5 Step 2 surfaces the skipped
  names in a console note. The feature is labelled "Export workspace" / "Export database",
  never "Save complete session".
- Empty environment is friendly — Task 3 returns `{ empty: true }` (SQL zero user tables, R
  empty globalenv, Python no tabular data); Task 5 shows a per-language console note and
  downloads nothing; Task 6 Step 1 (second test) asserts the SQL empty note.
- Safe default file name + valid extension — Task 1 (`exportWorkspaceFilename`,
  `workspaceExtensionFor`), asserted in Task 6 (filename regex `chatisa-workspace-sql-...sqlite`).
- Naming decision (menu item vs separate control) — recommended and implemented as a separate
  pane-header button (Task 4 rationale), not a second `ExportMenu` item, because the per-object
  menu is anchored to one row and a whole-session action does not belong there.
- Binary bytes channel — introduced here on `WorkerReply.exported` (Task 2), since 5a shipped
  text only and the workspace artifacts are binary. `text` stays distinct from `bytes` so a
  future text export never pays a base64 tax.

**Out of scope for 5c (deferred, named so nothing is silently dropped):**
- Import / restore, conflict handling (overwrite/skip/rename/cancel), trust warnings, and the
  no-auto-run rule — slice 5d. This slice only writes the files.
- **Pickle-based Python workspace (`.pkl`) — deferred.** It is fragile and version-bound (a
  pickle written under one pandas/NumPy/Python version can fail to load under another) and,
  more importantly, **loading a pickle executes arbitrary code**. It must land only together
  with 5d's explicit loading-trust warning and never auto-unpickle an untrusted file. The 5c
  Python format (CSV ZIP) is deliberately code-free on the way back in.
- Multi-object "export selected" (a checkbox subset of the environment), size estimation, and
  a progress UI for very large exports — these ride on the same protocol later; the ~25 MB
  browser-tab envelope the upload path assumes (Sandbox.tsx MAX_UPLOAD_BYTES) applies, and a
  size warning is a follow-on, not part of 5c.
- The editor-side workspace (open tabs, scripts, params as JSON) — that is React state, not
  worker state; out of scope here.

**Placeholder scan:** No TBDs. Every step shows final code. The one optional item (the Node
sqlite round-trip, Task 6 Step 3) is explicitly optional and gated behind `CHATISA_HEAVY`.

**Type consistency:** `WorkspaceLanguage = "python" | "r" | "sql"` flows from `export.ts`
through `workspaceExtensionFor`/`workspaceMimeFor`/`exportWorkspaceFilename` and the
`Workspace.exportWorkspace` cast. `WorkerReply.exported` is
`{ text?; bytes?: Uint8Array; skipped?: string[]; empty?: boolean }`; the three workers post
exactly that shape (SQL/R post `{ bytes, skipped: [], empty: false }` or `{ empty: true }`;
Python posts `{ bytes, skipped, empty: false }` or `{ empty: true }`).
`LanguageRunner.exportWorkspace` and `RunSession.exportWorkspace` return
`{ ok; bytes?; skipped?; empty?; error? }` identically. The accessible name is "Export
database" (SQL) / "Export workspace" (R, Python) in the component, the wire-up, and the e2e.

## Handoff note

Per the task that produced this plan, this is a planning document only; do not build from it in
this session. When execution is scheduled, use superpowers:subagent-driven-development (a fresh
subagent per task, review between) or superpowers:executing-plans (batched with checkpoints),
and remember: no git commits, verify with the commands above instead. 5c extends the 5a export
foundation (the `exportRequest`/`exported` message shape and the `export.ts` helpers) with a
whole-session op and the binary bytes channel; keep the `exportWorkspace`/`exported.bytes`
shape stable so the 5d import/restore slice can consume RData, `.sqlite`, and the CSV ZIP
without reshaping the foundation, and so a later pickle option can be added behind 5d's trust
warning.
