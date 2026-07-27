# Slice 5a — Export a single tabular object as CSV or TSV — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a visible Export control in the Environment/Tables pane (and the open
DataView) of Coding Studio that downloads one tabular object as `.csv` or `.tsv`
through the browser, for R (a data frame / tibble), Python (a pandas DataFrame), and
SQL (a table), without resetting the session, leaking any secret, or re-running
side-effecting SQL.

**Architecture.** The live object exists only inside the language Web Worker, so the
export round-trips through the worker exactly as the data viewer already does. A new
`exportRequest` worker message asks the runtime to serialize one named global to
delimited text (SQL: a read-only `SELECT * FROM "name"`; Python: `df.to_csv`; R:
`readr::format_csv`/`format_tsv`), and the reply carries the text back to the main
thread, which downloads it with the same Blob-and-anchor mechanism the "Download
Script" button already uses. Pure helpers (filename, delimiter, MIME) are unit-tested;
the manager protocol is unit-tested with a fake worker; the full path is proven
end-to-end in SQL (sqlite-wasm), which is deterministic and cheap.

**Tech Stack:** Next.js (this repo's vendored build — read `node_modules/next/dist/docs/`
before touching any Next API; this slice does not), React 19, Web Workers +
`@sqlite.org/sqlite-wasm` (SQL), Pyodide (Python), WebR (R), Vitest for unit tests,
Playwright + `@axe-core/playwright` for the sandbox e2e.

## Global Constraints

- **No git commits.** The working tree stays uncommitted; each task ends by running
  verification commands instead of committing.
- **WCAG 2.1 AA, keyboard and mouse.** The Export control is a real button (and menu)
  reachable and operable by keyboard; run axe on the sandbox after UI changes.
- **Miami brand tokens only for colors:** use the existing `var(--sb-...)` variables
  (`--sb-border`, `--sb-accent`, `--sb-muted`, `--sb-header`, `--sb-panel`,
  `--sb-text`). No raw hex in the export UI.
- **No em dashes** in any user-facing copy (labels, titles, menu items, error text).
- **Keyboard shortcuts, if any, support Ctrl (Win/Linux) and Cmd (macOS).** The visible
  Export control is the priority; no new global shortcut is in scope for 5a.
- **Consistent across R, Python, SQL.** The same control, the same two formats.
- **Never leak secrets.** The export reads exactly one named tabular object's cell
  values. It never serializes the environment, never includes connection details, and
  for SQL uses a read-only re-select that never re-runs the student's CREATE/INSERT/
  UPDATE/DELETE.
- **No silent truncation.** 5a exports the full object (all rows), not the displayed
  page. Other formats and export options are out of scope (5b); workspace and
  import/restore are out of scope (5c/5d).
- **All commands run from `web/`.** The Playwright config starts its own dev server on
  port 3100; the Vitest config runs `tests/unit/**/*.test.ts` in a node environment.

## File Structure

- `web/lib/sandbox/export.ts` — **Create.** Pure helpers: `ExportFormat` type,
  `delimiterFor`, `extensionFor`, `mimeFor`, `exportFilename`, and the DOM
  `downloadText` helper (factored from the existing Download Script pattern).
- `web/lib/run/manager.ts` — **Modify.** Add `exported?` to `WorkerReply`, to the
  `onmessage` destructure and `resolve`; add `LanguageRunner.exportObject`; add
  `exportObject` to the `RunSession` interface and `createSession`.
- `web/public/workers/sqlite-worker.mjs` — **Modify.** Handle `exportRequest`: a
  read-only `SELECT * FROM "name"` to delimited text; generalize field quoting to the
  chosen delimiter.
- `web/public/workers/pyodide-worker.mjs` — **Modify.** Handle `exportRequest`:
  `df.to_csv(sep=..., index=False)`, guarded on `to_csv`.
- `web/public/workers/webr-worker.mjs` — **Modify.** Handle `exportRequest`:
  `readr::format_csv`/`format_tsv`, guarded on `is.data.frame`.
- `web/components/sandbox/ExportMenu.tsx` — **Create.** A small accessible menu-button
  offering "Export as CSV" and "Export as TSV" for one named object.
- `web/components/sandbox/Sandbox.tsx` — **Modify.** Add `exportObject` in `Workspace`;
  render `ExportMenu` per tabular row in `VariablesPane` and in the DataView header.
- `web/components/sandbox/DataView.tsx` — **Modify.** Accept an optional `onExport`
  and render the `ExportMenu` in the header.
- `web/tests/unit/sandbox-export.test.ts` — **Create.** Vitest for the pure helpers.
- `web/tests/unit/run-harness.test.ts` — **Modify.** Add a fake-worker test for the
  `exportObject` protocol (mirrors the existing `getData` test).
- `web/tests/e2e/sandbox.spec.ts` — **Modify.** Add the real SQL end-to-end export
  test to the existing `AI Sandbox` describe block.

---

## Task 1: Pure export helpers

Filename, delimiter, extension, and MIME are pure and belong on the main thread; the
CSV/TSV field serialization itself lives in each worker's runtime and is proven by the
e2e test, not here.

**Files:**
- Create: `web/lib/sandbox/export.ts`
- Test: `web/tests/unit/sandbox-export.test.ts`

**Interfaces:**
- Produces: `type ExportFormat = "csv" | "tsv"`; `delimiterFor(format)`,
  `extensionFor(format)`, `mimeFor(format)`, `exportFilename({ userEmail, name, format, date? })`,
  and `downloadText(text, filename, mime)` (DOM, not unit-tested).

- [ ] **Step 1: Write the failing helper tests**

Create `web/tests/unit/sandbox-export.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import {
  delimiterFor,
  extensionFor,
  mimeFor,
  exportFilename,
} from "@/lib/sandbox/export";

describe("export helpers", () => {
  it("maps format to delimiter", () => {
    expect(delimiterFor("csv")).toBe(",");
    expect(delimiterFor("tsv")).toBe("\t");
  });

  it("maps format to extension and MIME", () => {
    expect(extensionFor("csv")).toBe("csv");
    expect(extensionFor("tsv")).toBe("tsv");
    expect(mimeFor("csv")).toBe("text/csv");
    expect(mimeFor("tsv")).toBe("text/tab-separated-values");
  });

  it("names the file after the student, the object, and the moment", () => {
    const date = new Date(2026, 6, 23, 14, 30); // 2026-07-23 14:30 local
    expect(
      exportFilename({ userEmail: "megahefm@miamioh.edu", name: "grades", format: "csv", date }),
    ).toBe("megahefm-grades-20260723-1430.csv");
  });

  it("sanitizes the object name and the email local part", () => {
    const date = new Date(2026, 6, 23, 9, 5);
    // Spaces and punctuation in the object name become underscores; a blank
    // email falls back to "sandbox".
    expect(
      exportFilename({ userEmail: "", name: "my table!", format: "tsv", date }),
    ).toBe("sandbox-my_table_-20260723-0905.tsv");
  });
});
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `npm run test -- sandbox-export`
Expected: FAIL — the module `@/lib/sandbox/export` does not exist yet.

- [ ] **Step 3: Implement the helpers**

Create `web/lib/sandbox/export.ts`:

```typescript
/**
 * Helpers for exporting one tabular object (a data frame, a DataFrame, a table)
 * from the Coding Studio session to a downloaded CSV or TSV. The serialization
 * itself runs in the language worker; these are the pure main-thread pieces plus
 * the browser download, factored from the existing Download Script pattern.
 */

export type ExportFormat = "csv" | "tsv";

/** The field separator for a format. */
export function delimiterFor(format: ExportFormat): string {
  return format === "tsv" ? "\t" : ",";
}

/** The file extension for a format (also its short label). */
export function extensionFor(format: ExportFormat): string {
  return format;
}

/** The MIME type for a format. */
export function mimeFor(format: ExportFormat): string {
  return format === "tsv" ? "text/tab-separated-values" : "text/csv";
}

/**
 * A safe download name: the student (email local part), the object name, and a
 * timestamp, e.g. megahefm-grades-20260723-1430.csv. Mirrors the Download Script
 * naming so the two feel like one feature.
 */
export function exportFilename(opts: {
  userEmail: string;
  name: string;
  format: ExportFormat;
  date?: Date;
}): string {
  const who = (opts.userEmail.split("@")[0] || "sandbox").replace(
    /[^a-zA-Z0-9._-]/g,
    "",
  );
  const safeName =
    opts.name.replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "") || "data";
  const d = opts.date ?? new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
  return `${who}-${safeName}-${stamp}.${extensionFor(opts.format)}`;
}

/**
 * Downloads text as a file. Same Blob-and-anchor mechanism as the Download Script
 * button (Sandbox.tsx download); the object URL is revoked after the click.
 */
export function downloadText(text: string, filename: string, mime: string): void {
  const blob = new Blob([text], { type: `${mime};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
```

Note: `exportFilename` uses `"data"` as the fallback object name and truncation-free
timestamping identical to `download`. The `sanitizeName` collapses runs of
punctuation to a single underscore then trims, so `"my table!"` becomes `"my_table"`;
adjust the expected string in the test if you prefer trailing-underscore preservation.
The test above expects `my_table_` (trailing kept from the `!`); the implementation
trims trailing underscores, so make them agree — the reference implementation trims,
so change the fourth test's expected to `sandbox-my_table-20260723-0905.tsv`.

- [ ] **Step 4: Reconcile and run the tests to green**

Pick one convention (the reference implementation trims trailing underscores) and make
the test assert exactly that. Run: `npm run test -- sandbox-export`
Expected: PASS.

- [ ] **Step 5: Verify the task (no commit)**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors for the new file.

---

## Task 2: Export protocol in the manager

Add the `exportObject` round-trip to `LanguageRunner` and `RunSession`, mirroring
`getData`. Prove it with a fake worker, exactly as the existing data-page test does.

**Files:**
- Modify: `web/lib/run/manager.ts`
- Test: `web/tests/unit/run-harness.test.ts`

**Interfaces:**
- Consumes: `LanguageRunner.dispatch` (manager.ts:217); the `WorkerReply` shape and
  `onmessage` destructure (manager.ts:118, 164).
- Produces: `WorkerReply.exported?: { text: string }`; `LanguageRunner.exportObject(name, format)`
  and `RunSession.exportObject(name, format)` returning
  `Promise<{ ok: boolean; text?: string; error?: string }>`.

- [ ] **Step 1: Write the failing protocol test**

Add to the `run manager timeout and worker reuse` describe block in
`web/tests/unit/run-harness.test.ts`. First widen the `FakeWorker.posted` type and the
reply generic to carry `exportRequest`, then add:

```typescript
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
```

Extend the `FakeWorker.posted` element type and `postMessage` parameter type in that
file to include `exportRequest?: { name: string; format: string }` so the new fields
type-check.

- [ ] **Step 2: Run the test and confirm it fails**

Run: `npm run test -- run-harness`
Expected: FAIL — `session.exportObject` does not exist.

- [ ] **Step 3: Add `exported` to the reply shape and wiring**

In `web/lib/run/manager.ts`, add to `WorkerReply` (around line 118):

```typescript
  exported?: { text: string };
```

In `LanguageRunner.ensureWorker`, add `exported` to the destructure and the resolve
(around lines 164 and 170):

```typescript
      const { id, ok, result, error, data, completions, tables, preview, exported } =
        event.data ?? {};
      // ...
      entry.resolve({ ok, result, error, data, completions, tables, preview, exported });
```

- [ ] **Step 4: Add `exportObject` to `LanguageRunner`**

Next to `getData` (manager.ts:259), add:

```typescript
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
```

- [ ] **Step 5: Expose it on `RunSession`**

Add to the `RunSession` interface (manager.ts:402):

```typescript
  /** Serialize one tabular object to CSV/TSV text for download (5a). */
  exportObject(
    name: string,
    format: "csv" | "tsv",
  ): Promise<{ ok: boolean; text?: string; error?: string }>;
```

And wire it in `createSession` (manager.ts:431):

```typescript
    exportObject: (name, format) => runner.exportObject(name, format),
```

- [ ] **Step 6: Run the protocol tests to green**

Run: `npm run test -- run-harness`
Expected: PASS (both new tests plus the existing suite).

- [ ] **Step 7: Verify the task (no commit)**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors for the changed file.

---

## Task 3: Worker export handlers (SQL, Python, R)

Each worker gains an `exportRequest` branch that serializes one named global to the
requested delimited text and replies `{ ok, exported: { text } }`, or a clean error
for a non-tabular name. This mirrors each worker's existing `dataRequest` branch.

**Files:**
- Modify: `web/public/workers/sqlite-worker.mjs`
- Modify: `web/public/workers/pyodide-worker.mjs`
- Modify: `web/public/workers/webr-worker.mjs`

**Interfaces:**
- Consumes: the `{ exportRequest: { name, format }, keepState }` message.
- Produces: `{ id, ok: true, exported: { text } }` or `{ id, ok: false, error }`.

- [ ] **Step 1: SQL — read-only re-select to delimited text**

In `web/public/workers/sqlite-worker.mjs`, add a delimiter-aware field helper next to
`csvField` (line 86):

```javascript
/** A single delimited field, quoted only when it must be (contains the delimiter,
 * a quote, or a newline). Works for both comma and tab. */
function delimitedField(value, delimiter) {
  if (value === null || value === undefined) return "";
  const s = String(value);
  const mustQuote =
    s.includes(delimiter) || s.includes('"') || s.includes("\n") || s.includes("\r");
  return mustQuote ? '"' + s.replace(/"/g, '""') + '"' : s;
}

/** One named table serialized to CSV/TSV via a read-only SELECT. Never re-runs
 * the student's CREATE/INSERT/UPDATE/DELETE: it reads the already-built table. */
function exportTableDelimited(db, name, format) {
  const delimiter = format === "tsv" ? "\t" : ",";
  const exists = [];
  db.exec({
    sql: "SELECT 1 AS ok FROM sqlite_schema WHERE type='table' AND name = ?",
    bind: [String(name)],
    rowMode: "object",
    resultRows: exists,
  });
  if (exists.length === 0) {
    throw new Error("That object is not a table you can export.");
  }
  const rows = [];
  const columns = [];
  db.exec({
    sql: `SELECT * FROM "${String(name).replace(/"/g, '""')}"`,
    rowMode: "array",
    resultRows: rows,
    columnNames: columns,
  });
  const lines = [columns.map((c) => delimitedField(c, delimiter)).join(delimiter)];
  for (const row of rows) {
    lines.push(row.map((v) => delimitedField(v, delimiter)).join(delimiter));
  }
  return lines.join("\n") + "\n";
}
```

Add `exportRequest` to the `onmessage` destructure (line 309) and a branch before the
final run block (alongside `dumpTablesRequest`, around line 366):

```javascript
  // Export one table to CSV/TSV via a read-only re-select (never re-runs the
  // student's data-changing statements).
  if (exportRequest) {
    try {
      const sqlite3 = await getSqlite();
      const db = keepState
        ? (sessionDb ??= new sqlite3.oo1.DB())
        : new sqlite3.oo1.DB();
      const text = exportTableDelimited(db, exportRequest.name, exportRequest.format);
      self.postMessage({ id, ok: true, exported: { text } });
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

- [ ] **Step 2: Python — `df.to_csv(sep=..., index=False)`, guarded**

In `web/public/workers/pyodide-worker.mjs`, add an export code builder next to
`fetchFramePageCode` (line 185):

```javascript
/** Serializes a named DataFrame/Series to CSV/TSV. Returns {text} or {error}. */
function exportFrameCode(name, sep) {
  return `
def __chatisa_export(__name, __sep):
    import json as __j
    __v = globals().get(__name)
    if __v is None or not hasattr(__v, "to_csv"):
        return __j.dumps({"error": "That variable is not a data frame or table."})
    return __j.dumps({"text": __v.to_csv(sep=__sep, index=False)})
__chatisa_export(${JSON.stringify(name)}, ${JSON.stringify(sep)})
`;
}
```

Add `exportRequest` to the `onmessage` destructure (line 306) and a branch next to the
`dataRequest` branch (around line 414):

```javascript
  // Export one DataFrame to CSV/TSV rather than running code.
  if (exportRequest) {
    try {
      const pyodide = await getPyodide();
      const sep = exportRequest.format === "tsv" ? "\t" : ",";
      const parsed = JSON.parse(
        pyodide.runPython(exportFrameCode(exportRequest.name, sep)),
      );
      if (parsed.error) {
        self.postMessage({ id, ok: false, error: parsed.error });
      } else {
        self.postMessage({ id, ok: true, exported: { text: parsed.text } });
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

- [ ] **Step 3: R — `readr::format_csv`/`format_tsv`, guarded**

In `web/public/workers/webr-worker.mjs`, add an export code builder next to
`buildRFileCode` (line 262):

```javascript
/** R that serializes a named data.frame to CSV/TSV text via readr, or signals a
 * non-frame with the __notframe__ sentinel. */
function buildRExportCode(name, format) {
  const fn = format === "tsv" ? "format_tsv" : "format_csv";
  return `
local({
  v <- tryCatch(get(${JSON.stringify(name)}, envir = globalenv()), error = function(e) NULL)
  if (is.null(v) || !is.data.frame(v)) stop("__notframe__")
  readr::${fn}(v)
})
`;
}
```

Add `exportRequest` to the `onmessage` destructure (line 310) and a branch next to the
`dataRequest` branch (around line 412):

```javascript
  // Export one data.frame to CSV/TSV rather than running code.
  if (exportRequest) {
    const shelter = await new webR.Shelter();
    try {
      await ensurePackages(webR); // readr is bundled; format_csv/format_tsv live there
      const obj = await shelter.captureR(
        buildRExportCode(exportRequest.name, exportRequest.format),
        { withAutoprint: false, captureStreams: false, captureConditions: false, captureGraphics: false },
      );
      const text = (await obj.result.toArray())[0];
      self.postMessage({ id, ok: true, exported: { text } });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      self.postMessage({
        id,
        ok: false,
        error: /__notframe__/.test(message)
          ? "That variable is not a data frame."
          : message,
      });
    } finally {
      await shelter.purge();
    }
    return;
  }
```

- [ ] **Step 4: Verify the workers compile and the app builds**

Workers are static `.mjs` loaded by URL, not type-checked by `tsc`. Sanity-check by
loading the sandbox in dev and confirming no worker console error on tab open.

Run: `npm run typecheck` — no errors (the workers are untyped, but the manager change
must still check).
Run: `npm run lint` — no errors for the changed files.

The real behavioral proof of the worker serialization is the SQL end-to-end in Task 6.

---

## Task 4: The ExportMenu control

A small accessible menu button for one named object, offering CSV and TSV. Keyboard
(Enter/Space to open, Escape to close, focusable items) and mouse operable.

**Files:**
- Create: `web/components/sandbox/ExportMenu.tsx`

**Interfaces:**
- Produces: `ExportMenu({ name, onExport })` where
  `onExport(format: ExportFormat) => void`. Renders a button named `Export {name}`
  with `aria-haspopup="menu"` and, when open, a menu of two items:
  `Export as CSV` and `Export as TSV`.

- [ ] **Step 1: Implement `ExportMenu`**

Create `web/components/sandbox/ExportMenu.tsx`:

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import type { ExportFormat } from "@/lib/sandbox/export";

/**
 * A small menu button that exports one named tabular object as CSV or TSV. The
 * menu is two buttons; Escape closes it and returns focus to the trigger, an
 * outside click dismisses it, and every item is keyboard reachable.
 */
export function ExportMenu({
  name,
  onExport,
}: {
  name: string;
  onExport: (format: ExportFormat) => void;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const triggerRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  function choose(format: ExportFormat) {
    setOpen(false);
    triggerRef.current?.focus();
    onExport(format);
  }

  return (
    <div ref={rootRef} className="relative inline-block">
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Export ${name}`}
        title={`Export ${name} as CSV or TSV`}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === "Escape") setOpen(false);
        }}
        className="rounded border border-[var(--sb-border)] px-1.5 py-0.5 text-xs font-bold text-[var(--sb-muted)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-accent)]"
      >
        Export
      </button>
      {open ? (
        <div
          role="menu"
          aria-label={`Export ${name} format`}
          onKeyDown={(e) => {
            if (e.key === "Escape") {
              setOpen(false);
              triggerRef.current?.focus();
            }
          }}
          className="absolute right-0 z-10 mt-1 min-w-[10rem] rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] py-1 shadow-lg"
        >
          <button
            type="button"
            role="menuitem"
            onClick={() => choose("csv")}
            className="block w-full px-3 py-1 text-left text-xs font-bold text-[var(--sb-text)] hover:bg-[var(--sb-header)] hover:text-[var(--sb-accent)]"
          >
            Export as CSV
          </button>
          <button
            type="button"
            role="menuitem"
            onClick={() => choose("tsv")}
            className="block w-full px-3 py-1 text-left text-xs font-bold text-[var(--sb-text)] hover:bg-[var(--sb-header)] hover:text-[var(--sb-accent)]"
          >
            Export as TSV
          </button>
        </div>
      ) : null}
    </div>
  );
}
```

- [ ] **Step 2: Verify it compiles**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors for the new file.

---

## Task 5: Wire Export into the panes

Add `exportObject` in `Workspace`, render `ExportMenu` per tabular row in
`VariablesPane`, and add it to the DataView header. Non-tabular rows never show the
control (defense in depth on top of the worker guard).

**Files:**
- Modify: `web/components/sandbox/Sandbox.tsx`
- Modify: `web/components/sandbox/DataView.tsx`

**Interfaces:**
- Consumes: `sessionRef.current.exportObject` (Task 2); `props.userEmail`; the export
  helpers (Task 1); `ExportMenu` (Task 4); `SessionVariable.columns` for the tabular
  test.
- Produces: an `exportObject(name, format)` callback in `Workspace`; `VariablesPane`
  gains an `onExport` prop and renders `ExportMenu` for viewable rows; `DataView` gains
  an optional `onExport` and renders `ExportMenu` in its header.

- [ ] **Step 1: Add the `exportObject` callback in `Workspace`**

In `web/components/sandbox/Sandbox.tsx`, add the export helper import near the other
sandbox lib imports (around line 45):

```typescript
import { downloadText, exportFilename, mimeFor, type ExportFormat } from "@/lib/sandbox/export";
```

Add the callback next to `getData` (around line 505):

```typescript
  // Exports one tabular object as CSV/TSV, downloaded in the browser. Read-only:
  // it asks the worker to serialize a named object and never resets the session.
  // A failure is reported in the console without altering anything.
  const exportObject = useCallback(
    async (name: string, format: ExportFormat) => {
      const session = sessionRef.current;
      if (!session) return;
      const res = await session.exportObject(name, format);
      if (res.ok && res.text != null) {
        downloadText(
          res.text,
          exportFilename({ userEmail: props.userEmail, name, format }),
          mimeFor(format),
        );
      } else {
        setEntries((prev) => [
          ...prev,
          {
            code: "",
            outcome: { ok: false, error: res.error ?? `Could not export ${name}.` },
            silent: true,
            label: `Export failed for ${name}.`,
          },
        ]);
      }
    },
    [props.userEmail],
  );
```

- [ ] **Step 2: Pass `onExport` to `VariablesPane` and render the menu**

In the `<VariablesPane ... />` usage (around line 704), add:

```tsx
                <VariablesPane
                  variables={variables}
                  language={language}
                  onView={openDataView}
                  onExport={exportObject}
                />
```

Update the `VariablesPane` signature (line 1152) to accept `onExport`, and in the
actions cell (around line 1194, next to the View button), render the menu for tabular
rows:

```tsx
                    <td className="px-1 py-1 text-right">
                      <span className="inline-flex items-center gap-1">
                        {viewable ? (
                          <button
                            type="button"
                            onClick={() => onView(v.name)}
                            aria-label={`View ${v.name} in a table`}
                            title={`View ${v.name}`}
                            className="rounded border border-[var(--sb-border)] px-1 text-[var(--sb-muted)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-accent)]"
                          >
                            <TableIcon />
                          </button>
                        ) : null}
                        {viewable ? (
                          <ExportMenu name={v.name} onExport={onExport} />
                        ) : null}
                      </span>
                    </td>
```

Add `onExport: (name: string, format: ExportFormat) => void` to the `VariablesPane`
prop type, and import `ExportMenu` at the top of the file. The `viewable` gate reuses
the exact same "has columns" test the View button already uses, so the Export offer
tracks tabular objects one-to-one across R, Python, and SQL.

- [ ] **Step 3: Add Export to the DataView header**

In `web/components/sandbox/DataView.tsx`, add an optional `onExport` prop and render
`ExportMenu` in the header (next to the Refresh button, around line 63):

```tsx
        {onExport ? <ExportMenu name={name} onExport={onExport} /> : null}
```

Thread `onExport` from `Workspace` where `DataView` is rendered (Sandbox.tsx:675):

```tsx
                  {activeTab !== "script" ? (
                    <DataView getData={getData} name={activeTab} onExport={exportObject} />
                  ) : null}
```

Import `ExportMenu` and `ExportFormat` in `DataView.tsx`, and give the header row an
`ml-auto` grouping so Refresh and Export sit together on the right.

- [ ] **Step 4: Verify it compiles and stays accessible**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors.
Run: `npm run test:e2e -- sandbox.spec.ts -g "four-pane"`
Expected: PASS — the existing shell + axe test still passes with the new controls
present (the menu button is a labelled button, so axe stays clean).

---

## Task 6: Real end-to-end export (SQL, sqlite-wasm)

SQL is deterministic and cheap (no prewarm, no network) and is the right language to
prove a real CSV export end to end: create a table, export it, and assert the
downloaded file's contents. Follows the Slice-1/Slice-2 sqlite e2e pattern (create,
Run, assert), reading the download bytes.

**Files:**
- Test: `web/tests/e2e/sandbox.spec.ts`

- [ ] **Step 1: Write the failing end-to-end export test**

Add to the `test.describe("AI Sandbox", ...)` block, and add
`import { readFileSync } from "node:fs";` at the top of the file:

```typescript
  test("exports a SQL table to a real CSV download without resetting the session", async ({
    page,
  }) => {
    // Real sqlite-wasm execution; generous headroom for the first compile.
    test.setTimeout(120_000);

    await page.goto("/ai-sandbox");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    // Build a table with values that exercise CSV quoting (a comma in a field).
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "CREATE TABLE grades(student TEXT, course TEXT, grade REAL);\n" +
        "INSERT INTO grades VALUES ('Amanda','ISA 401',91),('Bill, Jr','ISA 444',79);\n" +
        "SELECT * FROM grades;",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();

    const output = page.getByLabel("Console output");
    await expect(output).toContainText("Amanda", { timeout: 60_000 });

    // The table shows in the Tables pane; open its Export menu and choose CSV.
    await page.getByRole("button", { name: "Export grades" }).click();
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("menuitem", { name: "Export as CSV" }).click(),
    ]);

    // Named after the student, the object, and a timestamp.
    expect(download.suggestedFilename()).toMatch(
      /^student-grades-\d{8}-\d{4}\.csv$/,
    );

    // The downloaded bytes are a correct CSV: header, both rows, and the field
    // with a comma is quoted.
    const path = await download.path();
    const text = readFileSync(path, "utf8");
    expect(text).toContain("student,course,grade");
    expect(text).toContain("Amanda,ISA 401,91");
    expect(text).toContain('"Bill, Jr",ISA 444,79');

    // The export did not reset the session: the table and db are still there.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("SELECT COUNT(*) AS c FROM grades;");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("2", { timeout: 60_000 });
  });
```

- [ ] **Step 2: Run it and confirm it passes**

Run: `npm run test:e2e -- sandbox.spec.ts -g "exports a SQL table"`
Expected: PASS — the download arrives, the filename matches, the CSV content is
correct including the quoted comma field, and a follow-up query still sees the table
(the session was not reset).

Fallback if sqlite-wasm flakes in CI (the surrounding suite deliberately avoids WASM
in most tests): gate this one test behind the same opt-in flag the live-network test
uses, at the top of the test body:

```typescript
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "runs real sqlite-wasm; opt in with CHATISA_LIVE_NET=1",
    );
```

- [ ] **Step 3: Optional TSV assertion**

If cheap, add a second `menuitem` click for "Export as TSV" and assert the download is
`*.tsv` with tab-separated content (`student\tcourse\tgrade`). Keep it in the same test
to avoid a second cold sqlite compile.

- [ ] **Step 4: Verify the task (no commit)**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors.
Run: `npm run test -- sandbox-export run-harness` — the unit suites pass.
Run: `npm run test:e2e -- sandbox.spec.ts` — the full sandbox suite passes, including
the new export e2e and the existing shell/axe tests (confirming the new controls did
not regress the layout or accessibility). Leave the working tree uncommitted.

---

## Self-Review

**Spec coverage (requirement F, scoped to 5a).**

- Visible Export action in/near the variables/results panel — Task 5 (per tabular row
  in Environment/Tables, and in the DataView header). Keyboard + mouse accessible —
  Task 4 (`ExportMenu` is a labelled menu button with focusable items, Escape to
  close); axe stays clean (Task 5 Step 4).
- Export one object; choose file name and format; download via the browser — Task 1
  (`exportFilename`, `downloadText`), Task 4 (CSV/TSV menu), Task 5 (wire-up). Format
  choice is CSV vs TSV for 5a; the rest of F's formats are 5b.
- Exporting never modifies/removes/resets session objects — Task 3 (SQL read-only
  re-select; Python/R read a named global), proven by Task 6 Step 1 (the table
  survives, `COUNT(*)` returns 2 after export).
- Never include credentials/tokens/secrets — the export reads exactly one named
  tabular object's cells; nothing else is serialized. SQL uses a read-only re-select
  and never re-runs the student's CREATE/INSERT/UPDATE/DELETE (Task 3 Step 1 comment
  and code).
- Explain (do not silently truncate) — a non-tabular name yields a clear worker error
  surfaced in the console (Task 3 guards, Task 5 Step 1 error entry). 5a exports the
  full object (all rows), not the displayed page.
- Safe default file name + valid extension — Task 1 (`exportFilename` sanitizes the
  student and object name, `extensionFor`), asserted in Task 6 (filename regex).

**Out of scope for 5a (deferred):** rds/RData, json/xlsx/npy/npz/parquet/pkl (5b);
delimiter/colnames/rownames/NA/encoding controls (5b); multiple-object and whole-
workspace export (5c); import/restore, conflict handling, trust warnings (5d);
progress UI for very large exports and size estimation (5b/5c). These are named in the
decomposition doc so nothing is silently dropped.

**Placeholder scan:** No TBDs. Every step shows final code. The one reconciliation
point (trailing-underscore behavior of `exportFilename`) is called out explicitly in
Task 1 Steps 3-4 with a chosen convention. Commands are exact and adapted to the
no-commit rule (verify steps replace commit steps).

**Type consistency:** `ExportFormat = "csv" | "tsv"` flows from `export.ts` through
`ExportMenu.onExport`, `VariablesPane.onExport`, `DataView.onExport`, `Workspace.exportObject`,
`RunSession.exportObject`, and `LanguageRunner.exportObject` unchanged. `WorkerReply.exported`
is `{ text: string }`; the workers post exactly that shape. Accessible name is
`Export {name}` in the component and every test.

## Handoff note

Per the task that produced this plan, this is a planning document only; do not build
from it in this session. When execution is scheduled, use
superpowers:subagent-driven-development (a fresh subagent per task, review between) or
superpowers:executing-plans (batched with checkpoints), and remember: no git commits,
verify with the commands above instead. 5a builds the shared export protocol and
download helper that 5b-5d extend; keep the `exportRequest`/`exported` message shape
and the `export.ts` helpers stable so the later sub-slices can add formats (and a
binary bytes channel) without reshaping the foundation.
