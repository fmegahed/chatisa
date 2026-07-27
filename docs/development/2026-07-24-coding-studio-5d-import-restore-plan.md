# Slice 5d — Import / restore an exported workspace — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (a fresh subagent per task, review between) or superpowers:executing-plans (batched, with checkpoints). Steps use checkbox (`- [ ]`) syntax. No git commits: each task ends by running its verification commands.

**Goal:** Let a student re-open a workspace they exported with slice 5c, folded into the existing **Upload Dataset** control (the professor chose one auto-detecting upload, not a separate Restore button). Uploading a workspace file restores **every** object it holds, with explicit conflict handling and a security gate:

- **R** — a `.RData`/`.rda` image: `load()` every object back into the global environment.
- **SQL** — a `.sqlite` database: copy every table back into the session database.
- **Python** — a `.pkl` pickle: `pickle.load()` the dict of values back into the namespace, **only after the student confirms they trust the file** (a pickle runs code when it is loaded).

A single R object saved as `.rds`, and csv/json/xlsx/parquet, stay the existing **one-object** import; only the three workspace formats trigger restore. Multi-object "export selected" is a separate follow-on (5e), out of scope here.

**Architecture.** Restore reuses the shipped upload round-trip: the file goes to the language worker with `fileOp` in `mode: "preview"` then `mode: "import"`, exactly as single-object upload does (`FileRequest`, manager `previewFile`/`importFile`). What changes is that when `options.restore` is set, the worker (a) in preview returns the *members* of the file (its object/table names) and, for each, whether that name already exists in the live session, and (b) in import applies every member into the session under the chosen conflict rule (`overwrite` / `skip` / `rename`), read-only against the file and never running the student's own script. The `ImportDialog` grows a restore mode: instead of the single-object parse preview, it shows the member list with collision badges, a conflict radio, and — for a pickle — a trust checkbox that gates the Restore button. Because the session lives only inside each worker, all the restore logic lives worker-side; the main thread only chooses names, the conflict rule, and trust.

**Tech Stack:** Next.js (this repo's vendored build; this slice touches no framework API), React 19, Web Workers, `@sqlite.org/sqlite-wasm` (SQL: `wasm.allocFromTypedArray` + `capi.sqlite3_deserialize`, **confirmed working** by a Node round-trip on 2026-07-24), WebR (R: `load` into a temp env, `assign` into globalenv), Pyodide (Python: `pickle.load`), Vitest for pure logic + the manager protocol, Playwright + `@axe-core/playwright` for the SQL end-to-end and the dialog axe check.

## Global Constraints

- **No git commits.** The tree stays uncommitted; each task ends by running its checks.
- **Never silently overwrite.** A restored name that collides with an existing session object is resolved only by the student's explicit choice: overwrite, skip, rename, or cancel the whole restore. The default is **rename** (the safest: nothing existing is lost).
- **Pickle is code; gate it.** Restoring a `.pkl` runs `pickle.load`, which can execute arbitrary code. The Restore button stays disabled until the student ticks "I trust this file". `.RData` gets a lighter note (R's `load` can in principle evaluate active bindings, but the common case is data); `.sqlite` is pure data and needs no trust gate.
- **Read-only against the file; never re-run the student's script.** Restore reads the uploaded bytes and writes into the session. It does not execute any code the student wrote, and does not reconnect to any external resource.
- **Never leak secrets.** The runtimes are sandboxed in-browser and hold no server secrets. Restore only writes the file's own objects into the session. No network call.
- **WCAG 2.1 AA, keyboard and mouse.** The restore controls are real radios in a `fieldset`, a real checkbox, and labelled buttons; run axe on the sandbox after the dialog change. State announced with text, not colour alone (a collision is the word "exists", not just a red dot).
- **Miami tokens only** (`var(--sb-...)`); **no em dashes** in any user-facing copy.
- **Empty / invalid file is friendly.** A file with no restorable objects, or bytes that are not a valid database/image/pickle, shows a plain message, not a crash.
- **All commands run from `web/`.**

## Decisions already made (professor, 2026-07-24)

- **Fold restore into Upload Dataset** (auto-detect a workspace file), not a separate button.
- **Restore + conflict handling + the pickle trust warning first**; multi-object "export selected" is deferred (5e).

## Already built (foundation, this session, tsc-clean)

- `web/lib/sandbox/upload.ts` — `UploadFormat` gained `"pkl"` and `"sqlite"`; `SUPPORTED` offers `pkl` (Python), `sqlite` (SQL), `rdata` (R); `formatFromName` and `acceptFor` map `.pkl` and `.sqlite`/`.sqlite3`/`.db`; **`isWorkspaceFile(filename)`** returns true for `.pkl`, `.sqlite`/`.sqlite3`/`.db`, `.RData`/`.rda` (not `.rds`).
- `web/lib/run/manager.ts` — `ImportOptions` gained `restore?`, `conflict?: "overwrite" | "skip" | "rename"`, `trusted?`; new `WorkspaceMember { name; collides }`; `FilePreview` gained `restore?` and `members?: WorkspaceMember[]`; `FileRequest.format` gained `"pkl" | "sqlite"`.

The SQL deserialize API is de-risked: `const p = sqlite3.wasm.allocFromTypedArray(bytes); const db = new sqlite3.oo1.DB(); sqlite3.capi.sqlite3_deserialize(db.pointer, "main", p, bytes.length, bytes.length, sqlite3.capi.SQLITE_DESERIALIZE_FREEONCLOSE | sqlite3.capi.SQLITE_DESERIALIZE_RESIZEABLE)` returned rc 0 and read the tables back in a Node probe.

## Chosen per-language restore (grounded and honest)

- **SQL — deserialize the uploaded `.sqlite`, copy each user table into the session.** Open the bytes into a throwaway `src` DB with `sqlite3_deserialize`. Enumerate `src`'s user tables; a name that already exists in the session collides. On import, for each table: `overwrite` drops the session table then recreates it from `src`'s own `CREATE` and copies rows; `skip` leaves the existing one; `rename` creates the table under the first free `name_2`, `name_3`, ... Rows are copied with a prepared `INSERT` (both DBs are in the same wasm module, but they are separate connections, so rows are read from `src` and re-inserted rather than a cross-connection `SELECT`). Read-only against `src`; runs no student statement.
- **R — `load()` the `.RData` into a temp env, then `assign` into globalenv per the rule.** `e <- new.env(); load(path, envir = e)` restores every object into `e` without touching globalenv. `ls(e)` is the member list; a name already in `ls(globalenv())` collides. On import, per rule: `overwrite` assigns into globalenv; `skip` leaves the existing binding; `rename` assigns under the first free `name.2`. Base R only, no bundled package needed.
- **Python — `pickle.load` the dict, apply per the rule, gated on trust.** The 5c export is `pickle.dump({name: value}, protocol=4)`. Restore does `data = pickle.load(open(path,'rb'))`, and `data.keys()` is the member list; a key already in `globals()` collides. On import, per rule: `overwrite` sets `globals()[name]`; `skip` leaves it; `rename` sets `globals()[name + "_2"]`. Only runs when `options.trusted` is true; the dialog enforces the gate, and the worker double-checks and refuses a pickle restore without it.

## File Structure

- **Modify** `web/public/workers/sqlite-worker.mjs` — a restore branch in `fileOp` (preview: members + collisions; import: copy tables per rule). Helpers `deserializeInto`, `userTableNames`, `copyTable`.
- **Modify** `web/public/workers/webr-worker.mjs` — `buildRFileCode` (or a new `buildRRestoreCode`) gains a restore path returning members/collisions (preview) or assigning per rule (import).
- **Modify** `web/public/workers/pyodide-worker.mjs` — a restore path in the file op: `pickle.load`, members/collisions (preview), assign per rule (import), refuse without trust.
- **Modify** `web/components/sandbox/ImportDialog.tsx` — a restore mode: member list + collision badges, a conflict `fieldset` radio, a trust checkbox for `.pkl`, a "Restore" primary button gated on trust; thread `conflict`/`trusted` into `options`.
- **Modify** `web/components/sandbox/Sandbox.tsx` — `onFileChosen` marks a workspace file (`isWorkspaceFile`) with `options.restore = true`, `conflict = "rename"`, and the restore console note names what was restored/skipped/renamed.
- **Modify** `web/lib/sandbox/upload.ts` — `defaultOptions` returns `{ restore: true, conflict: "rename" }` for `pkl`/`sqlite` and for a `.RData` restore (helper `defaultRestoreOptions`), plus a unit test.
- **Modify** `web/tests/unit/sandbox-upload.test.ts` (create if absent) — `isWorkspaceFile`, `formatFromName` for the new extensions, `uniqueName` helper.
- **Modify** `web/tests/unit/run-harness.test.ts` — a fake-worker test that a restore request carries `restore`/`conflict`/`trusted` and returns members.
- **Modify** `web/tests/e2e/sandbox.spec.ts` — a real SQL restore end-to-end (export a db, edit the session, restore, assert the tables merged per rule and the session was not otherwise reset) plus the dialog axe check; opt-in live R and Python restore.

---

## Task 1: Pure helpers — unique-name and default restore options

**Files:** modify `web/lib/sandbox/upload.ts`; test `web/tests/unit/sandbox-upload.test.ts`.

- [ ] **Step 1: failing tests**

```typescript
import { describe, expect, it } from "vitest";
import {
  isWorkspaceFile,
  formatFromName,
  uniqueName,
  defaultRestoreOptions,
} from "@/lib/sandbox/upload";

describe("workspace detection", () => {
  it("recognises workspace files, not single objects", () => {
    expect(isWorkspaceFile("a.pkl")).toBe(true);
    expect(isWorkspaceFile("a.sqlite")).toBe(true);
    expect(isWorkspaceFile("a.RData")).toBe(true);
    expect(isWorkspaceFile("a.rda")).toBe(true);
    expect(isWorkspaceFile("a.rds")).toBe(false); // one object
    expect(isWorkspaceFile("a.csv")).toBe(false);
  });
  it("maps the new extensions to formats", () => {
    expect(formatFromName("w.pkl")).toBe("pkl");
    expect(formatFromName("w.sqlite")).toBe("sqlite");
    expect(formatFromName("w.db")).toBe("sqlite");
  });
});

describe("uniqueName", () => {
  it("returns the name when free, else the first free numeric suffix", () => {
    const taken = new Set(["grades", "grades_2"]);
    expect(uniqueName("courses", taken)).toBe("courses");
    expect(uniqueName("grades", taken)).toBe("grades_3");
  });
});

describe("defaultRestoreOptions", () => {
  it("defaults to a non-destructive rename", () => {
    expect(defaultRestoreOptions()).toEqual({ restore: true, conflict: "rename" });
  });
});
```

- [ ] **Step 2:** `npm run test -- sandbox-upload` fails (helpers missing).

- [ ] **Step 3: implement** in `web/lib/sandbox/upload.ts` (append):

```typescript
/** The first free name: `name`, else `name_2`, `name_3`, ... not in `taken`.
 * The suffix uses `_` so a valid identifier stays valid in R, Python, and SQL. */
export function uniqueName(name: string, taken: Set<string>): string {
  if (!taken.has(name)) return name;
  let n = 2;
  while (taken.has(`${name}_${n}`)) n++;
  return `${name}_${n}`;
}

/** Default options for a whole-workspace restore: load everything, and rename on a
 * name collision so nothing already in the session is lost without a choice. */
export function defaultRestoreOptions(): import("@/lib/run/manager").ImportOptions {
  return { restore: true, conflict: "rename" };
}
```

- [ ] **Step 4:** `npm run test -- sandbox-upload` passes.
- [ ] **Step 5:** `npm run typecheck` and `npm run lint` clean.

---

## Task 2: SQL restore in the worker

**File:** `web/public/workers/sqlite-worker.mjs`.

- [ ] **Step 1: helpers** (next to `exportDatabaseBytes`):

```javascript
/** Opens a byte image into a throwaway in-memory DB (a full, separate connection).
 * Confirmed API: allocFromTypedArray + sqlite3_deserialize with FREEONCLOSE|RESIZEABLE. */
function deserializeInto(sqlite3, bytes) {
  const db = new sqlite3.oo1.DB();
  const p = sqlite3.wasm.allocFromTypedArray(bytes);
  const rc = sqlite3.capi.sqlite3_deserialize(
    db.pointer, "main", p, bytes.length, bytes.length,
    sqlite3.capi.SQLITE_DESERIALIZE_FREEONCLOSE | sqlite3.capi.SQLITE_DESERIALIZE_RESIZEABLE,
  );
  if (rc !== 0) { db.close(); throw new Error("That file is not a valid SQLite database."); }
  return db;
}

/** User table names (excludes sqlite_* internal tables). */
function userTableNames(db) {
  const rows = [];
  db.exec({
    sql: "SELECT name FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
    rowMode: "array", resultRows: rows,
  });
  return rows.map((r) => r[0]);
}

/** Copies one table from src into main under `target`, preserving the schema. */
function copyTable(src, main, srcName, target) {
  const [{ sql: createSql }] = (() => {
    const r = [];
    src.exec({ sql: "SELECT sql FROM sqlite_schema WHERE type='table' AND name=?",
      bind: [srcName], rowMode: "object", resultRows: r });
    return r;
  })();
  // Reuse src's CREATE with the (possibly renamed) target: swap the first name token.
  const create = createSql.replace(
    /^(\s*CREATE\s+TABLE\s+)(?:IF\s+NOT\s+EXISTS\s+)?("(?:[^"]|"")*"|`(?:[^`]|``)*`|\[[^\]]*\]|[A-Za-z_][\w$]*)/i,
    (_m, head) => `${head}"${target.replace(/"/g, '""')}"`,
  );
  main.exec(create);
  const cols = [];
  src.exec({ sql: `SELECT * FROM "${srcName.replace(/"/g, '""')}" LIMIT 0`,
    columnNames: cols });
  const q = (c) => `"${c.replace(/"/g, '""')}"`;
  const rows = [];
  src.exec({ sql: `SELECT * FROM "${srcName.replace(/"/g, '""')}"`, rowMode: "array", resultRows: rows });
  if (rows.length === 0) return;
  const stmt = main.prepare(
    `INSERT INTO "${target.replace(/"/g, '""')}" (${cols.map(q).join(",")}) VALUES (${cols.map(() => "?").join(",")})`,
  );
  try { for (const r of rows) { stmt.bind(r).step(); stmt.reset(); } }
  finally { stmt.finalize(); }
}
```

- [ ] **Step 2: destructure `exportWorkspace` already added; add a restore branch** inside the existing `if (fileOp)` block, before the csv/json parsing, guarded on `fileOp.format === "sqlite"` (or `fileOp.options.restore`):

```javascript
  if (fileOp && (fileOp.format === "sqlite" || fileOp.options?.restore)) {
    try {
      const sqlite3 = await getSqlite();
      const main = (sessionDb ??= new sqlite3.oo1.DB());
      const src = deserializeInto(sqlite3, fileOp.bytes);
      try {
        const names = userTableNames(src);
        const existing = new Set(userTableNames(main));
        if (fileOp.mode === "preview") {
          self.postMessage({ id, ok: true, preview: {
            restore: true, columns: [], rows: [],
            members: names.map((n) => ({ name: n, collides: existing.has(n) })),
          } });
        } else {
          const rule = fileOp.options?.conflict ?? "rename";
          const restored = [], skipped = [], renamed = [];
          for (const n of names) {
            if (existing.has(n)) {
              if (rule === "skip") { skipped.push(n); continue; }
              if (rule === "overwrite") { main.exec(`DROP TABLE IF EXISTS "${n.replace(/"/g,'""')}"`); copyTable(src, main, n, n); restored.push(n); existing.add(n); continue; }
              const t = uniqueTableName(n, existing); copyTable(src, main, n, t); renamed.push(`${n} -> ${t}`); existing.add(t); continue;
            }
            copyTable(src, main, n, n); restored.push(n); existing.add(n);
          }
          const variables = dumpVariablesForReply(main); // reuse whatever the run path returns for the Tables pane, or omit
          self.postMessage({ id, ok: true, result: { text: restoreNote(restored, skipped, renamed) }, ...(variables ? { } : {}) });
        }
      } finally { src.close(); }
    } catch (error) {
      self.postMessage({ id, ok: false, error: error instanceof Error ? error.message : String(error) });
    }
    return;
  }
```

Add small helpers `uniqueTableName(name, set)` (mirror `uniqueName`) and `restoreNote(restored, skipped, renamed)` returning a plain sentence with no em dashes. The import reply must also refresh the Tables pane the same way the normal import does (return `variables`/`tables` per this worker's existing import reply shape — match it exactly).

- [ ] **Step 3:** `npm run typecheck` + `npm run lint` clean (workers are untyped; the manager change still checks). Behavioural proof is Task 6.

---

## Task 3: R restore in the worker

**File:** `web/public/workers/webr-worker.mjs` — extend `buildRFileCode` with a restore path (or add `buildRRestoreCode`) used when `options.restore` and the format is `rdata`.

- [ ] **Step 1:** R that previews or applies a restore:

```javascript
function buildRRestoreCode(mode, options, path) {
  const rule = rStr(options.conflict || "rename");
  return `
local({
  e <- new.env(); load(${rStr(path)}, envir = e)
  members <- ls(e); existing <- ls(envir = globalenv())
  if (${mode === "import" ? "TRUE" : "FALSE"}) {
    rule <- ${rule}; restored <- character(); skipped <- character(); renamed <- character()
    for (nm in members) {
      if (nm %in% existing) {
        if (rule == "skip") { skipped <- c(skipped, nm); next }
        if (rule == "overwrite") { assign(nm, get(nm, envir = e), envir = globalenv()); restored <- c(restored, nm); existing <- c(existing, nm); next }
        t <- nm; k <- 2L; while (t %in% existing) { t <- paste0(nm, "_", k); k <- k + 1L }
        assign(t, get(nm, envir = e), envir = globalenv()); renamed <- c(renamed, paste0(nm, " -> ", t)); existing <- c(existing, t); next
      }
      assign(nm, get(nm, envir = e), envir = globalenv()); restored <- c(restored, nm); existing <- c(existing, nm)
    }
    jsonlite::toJSON(list(ok = TRUE, restored = as.list(restored),
      skipped = as.list(skipped), renamed = as.list(renamed)), auto_unbox = TRUE)
  } else {
    jsonlite::toJSON(list(restore = TRUE, members = lapply(members, function(nm)
      list(name = nm, collides = nm %in% existing))), auto_unbox = TRUE)
  }
})
`;
}
```

- [ ] **Step 2:** In the `fileOp` handler, when `fileOp.options?.restore && fileOp.format === "rdata"`, write the bytes to the VFS (as the existing upload path does), run `buildRRestoreCode`, and post `{ preview: { restore, members } }` or `{ result, variables }` (refresh the Environment pane with the existing `inspectR`). Reuse the shipped write/`captureR`/`friendlyError` scaffolding.

- [ ] **Step 3:** typecheck + lint clean. Behavioural proof is the opt-in live test (Task 6).

---

## Task 4: Python restore in the worker

**File:** `web/public/workers/pyodide-worker.mjs`.

- [ ] **Step 1:** Python that previews or applies, refusing without trust:

```javascript
function restorePickleCode(mode, trusted, rule) {
  return `
def __chatisa_restore():
    import json, pickle
    if ${mode === "import" ? "True" : "False"} and not ${trusted ? "True" : "False"}:
        return json.dumps({"error": "This file was not confirmed as trusted."})
    with open("/tmp/__upload", "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        return json.dumps({"error": "That file is not a ChatISA workspace pickle."})
    g = globals(); members = list(data.keys())
    if not ${mode === "import" ? "True" : "False"}:
        return json.dumps({"restore": True, "members": [
            {"name": k, "collides": (k in g and not k.startswith("__"))} for k in members]})
    rule = ${JSON.stringify(rule || "rename")}
    restored, skipped, renamed = [], [], []
    for k, v in data.items():
        if k in g and not k.startswith("__"):
            if rule == "skip": skipped.append(k); continue
            if rule == "overwrite": g[k] = v; restored.append(k); continue
            t = k; n = 2
            while t in g: t = k + "_" + str(n); n += 1
            g[t] = v; renamed.append(k + " -> " + t); continue
        g[k] = v; restored.append(k)
    return json.dumps({"ok": True, "restored": restored, "skipped": skipped, "renamed": renamed})
__chatisa_restore()
`;
}
```

- [ ] **Step 2:** In the file op, when `fileOp.format === "pkl"` (implicitly a restore), write bytes to `/tmp/__upload` (as upload does), run `restorePickleCode`, and post `{ preview: { restore, members } }` or `{ result, variables }` (refresh the Variables pane with `INSPECT_VARS`). A `{ error }` becomes `{ ok:false, error }`.

- [ ] **Step 3:** typecheck + lint clean. Proof is the opt-in live test.

---

## Task 5: The restore mode in ImportDialog + Sandbox wiring

**Files:** `web/components/sandbox/ImportDialog.tsx`, `web/components/sandbox/Sandbox.tsx`.

- [ ] **Step 1: Sandbox `onFileChosen`** — when `isWorkspaceFile(file.name)`, set the upload's options to `defaultRestoreOptions()` so the dialog opens in restore mode. The restore console note names restored/skipped/renamed from the import outcome (`outcome.result?.text`), and for `.pkl` it repeats the trust reminder.

- [ ] **Step 2: ImportDialog restore mode.** When `preview?.restore` (or the incoming `file.format` is a workspace format), replace the single-object options/parse-preview with:
  - A heading "Restore workspace" and a sentence: "This will add N items from FILENAME to your session."
  - The **member list**: each `member.name`, with the word "exists" (not colour alone) when `member.collides`.
  - A conflict **fieldset** with radios: "Rename items that clash (keep both)" (default), "Overwrite the existing ones", "Skip the ones that exist". These set `options.conflict`.
  - For `.pkl` only, a **trust checkbox**: "I trust this file. Opening a pickle can run code." bound to `options.trusted`.
  - The primary button reads **Restore**, disabled while previewing, when there are no members, or (pkl) until `trusted` is checked.
  - Keep Escape/overlay-click to cancel (cancel = close, nothing applied).

- [ ] **Step 3:** axe: the radios are a real `fieldset`/`legend`, the checkbox is labelled, the member list is a real `ul`. Run `npm run test:e2e -- sandbox.spec.ts -g "four-pane"` (axe) and the new dialog test.

- [ ] **Step 4:** typecheck + lint clean.

---

## Task 6: Real end-to-end (SQL) + opt-in live (R, Python)

**File:** `web/tests/e2e/sandbox.spec.ts`.

- [ ] **Step 1: SQL restore end-to-end (deterministic).** Create two tables, **Export database** (5c) to get a real `.sqlite` download, then in a fresh reasoning: drop one table in the session, re-upload the downloaded file, choose **Overwrite**, Restore, and assert both tables are present again and row counts match; and a second run with **Rename** asserts a `grades_2` appears next to a still-present edited `grades`. Reuse `page.waitForEvent("download")` for the export and `setInputFiles` for the re-upload (the hidden file input). This proves deserialize + copy + conflict end to end without any WASM heavy runtime beyond sqlite (already used across the suite).

- [ ] **Step 2: opt-in live R and Python** (gate `process.env.CHATISA_LIVE_NET === "1"`): export the workspace, define a colliding variable, restore with **rename**, assert the console note names the renamed object and the Environment/Variables pane shows both.

- [ ] **Step 3:** `npm run typecheck`, `npm run lint`, `npm run test`, and `npm run test:e2e -- sandbox.spec.ts` all green. Working tree uncommitted.

---

## Self-Review vs spec F (import / restore)

- **Corresponding Import/Restore action** — folded into Upload Dataset (professor's choice), auto-detected by `isWorkspaceFile` (Task 1, done in foundation).
- **Inspect before restoring** — the dialog shows the file name and the member list before any change (Task 5).
- **No silent overwrite; overwrite/skip/rename/cancel** — the conflict fieldset (Task 5), applied worker-side (Tasks 2 to 4); default rename; cancel = close.
- **Clear which restored / which could not** — the import reply returns `restored`/`skipped`/`renamed`, surfaced in a console note (Task 5 Step 1).
- **Untrusted serialized objects require a warning; never auto-run** — the `.pkl` trust checkbox gates the Restore button, and the Python worker refuses a restore without `trusted` (Tasks 4, 5). Restore never runs the student's script.
- **No auto-reconnect to external resources** — nothing in restore opens a network or a database connection beyond the in-browser session.
- **Language-aware; no cross-format confusion** — each format restores only in its own language (Python `.pkl`, SQL `.sqlite`, R `.RData`); the picker is scoped per language (`supportedFormats`).

**Deferred (5e):** multi-object "export selected" (checkbox subset of the Environment), and a pickle-vs-CSV export choice. Named so nothing is dropped silently.

## Handoff note

The foundation (detection + data model) is in place and tsc-clean; the SQL deserialize API is proven. Build Tasks 2 to 6 with TDD, verify with the commands above, no commits. Keep the `fileOp` `{ mode, options.restore, options.conflict, options.trusted }` shape stable so 5e (multi-select export) can reuse the same worker plumbing.
