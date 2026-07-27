# Slice 5 (F) — Export variables and workspaces — sub-slice decomposition

Companion to `2026-07-23-coding-studio-polish-requirements.md` (section F and the
general acceptance criteria) and `2026-07-23-coding-studio-polish-decomposition.md`
(which split F into 5a-5d). This document refines those four sub-slices against the
real Coding Studio code: the concrete approach in THIS codebase (which worker and
manager changes, which formats, per language R/Python/SQL), the security and
serialization risks, an honest read of what is feasible in the browser, and a rough
size per sub-slice. It does not implement anything.

The detailed, build-ready plan for 5a lives in
`2026-07-23-coding-studio-polish-plan-5a-export-tabular.md`.

## Where the code actually is (grounded)

- **The live objects live only in the worker, never in React.** `Workspace`
  (Sandbox.tsx:283) holds `variables: SessionVariable[]` (name/type/info/columns,
  manager.ts:24) which is a rendered summary, not the data. The real data.frame,
  the pandas DataFrame, the SQLite table exist only inside the language Web Worker
  (webr-worker.mjs / pyodide-worker.mjs / sqlite-worker.mjs). So every export must
  round-trip through the worker: ask it to serialize a named object, get bytes or
  text back, download on the main thread.
- **The round-trip protocol already exists for reads.** `LanguageRunner.dispatch`
  (manager.ts:217) posts one correlated message and awaits one reply. The workers
  already branch on message keys: `dataRequest` (data viewer, one page of a frame),
  `completeAt` (autocomplete), `dumpTablesRequest` (SQL, every table to CSV),
  `fileOp` (upload preview/import). `WorkerReply` (manager.ts:118) is the fixed
  reply shape, and the manager's `onmessage` destructures a fixed key set
  (manager.ts:164). Export is a new key in exactly this mould.
- **The reverse direction (import) is already built.** `fileOp` with `mode:"import"`
  writes an uploaded file into the worker as a named object; `previewFile`/`importFile`
  (manager.ts:299-312) and `ImportDialog` (components/sandbox/ImportDialog.tsx) drive
  it. 5d (import/restore) mirrors this plumbing rather than inventing new plumbing.
- **The download mechanism already exists twice.** `download` (Sandbox.tsx:477, the
  "Download Script" button) builds a `Blob`, `URL.createObjectURL`, a temporary
  `<a download>`, `a.click()`, `URL.revokeObjectURL`, and names the file
  `${who}-${stamp}.${ext}` from the student's email local part and a timestamp.
  `exportCurrent` (PlotsPane, Sandbox.tsx:1273) does the same for a plot PNG. Every
  export sub-slice reuses this pattern; only the payload and extension change.
- **"Which objects are tabular" is already known.** A `SessionVariable` with
  `columns.length > 0` is viewable as a table; `VariablesPane` uses exactly that
  (`viewable = (v.columns?.length ?? 0) > 0`, Sandbox.tsx:1183) to show the "View"
  button. Each worker's data-viewer path already gates on being a frame: R
  `is.data.frame` (webr-worker.mjs:146), Python `hasattr(iloc)` and `columns`
  (pyodide-worker.mjs:189), SQL "is a table in sqlite_schema" (fetchTablePage). The
  export path reuses the same "is it tabular" test, so a CSV/TSV export offer lines
  up one-to-one with the existing View affordance.

## The shared foundation (introduced by 5a, reused by 5b-5d)

All four sub-slices ride on one new worker message and one new manager method. 5a
builds it; the rest extend it.

- **New worker message:** `{ exportRequest: { name, format }, keepState: true }`.
  Each worker branches on `exportRequest` exactly as it branches on `dataRequest`,
  serializes the named global to the requested format inside the runtime, and
  replies `{ id, ok: true, exported: { text } }` (5a is text; 5b adds a bytes
  channel for binary formats, see below) or `{ id, ok: false, error }`.
- **New reply field:** add `exported?` to `WorkerReply` (manager.ts:118), to the
  `onmessage` destructure (manager.ts:164), and to the `entry.resolve(...)` call.
- **New manager/session method:** `exportObject(name, format)` on `LanguageRunner`
  and on `RunSession` (manager.ts:402), returning `{ ok, text?/bytes?, error? }`,
  wired in `createSession` (manager.ts:427) like `getData`.
- **Reused download:** a small `downloadText`/`downloadBytes` helper factored from
  the existing `download`/`exportCurrent` Blob-and-anchor pattern, plus a pure
  `exportFilename` that mirrors the Download Script naming.

For 5a the payload is text (CSV/TSV are text). 5b's binary formats (rds, xlsx, npy,
parquet, pkl) need a bytes channel: return a `Uint8Array` (structured-cloned, or
transferred) as `exported.bytes` and download with a binary MIME type. Keep the
`text` and `bytes` fields distinct so a text export never pays a base64 tax.

---

## 5a — Single tabular object to CSV or TSV (smallest, ship first)

**Scope.** Export one data frame / tibble (R), one pandas DataFrame (Python), or one
table (SQL) to `.csv` or `.tsv`, downloaded via the browser, without resetting the
session. This is the 80 percent case and delivers most of the student value.

**Approach in this codebase (per language).**

- **SQL:** a read-only `SELECT * FROM "name"` against the session database, built to
  delimited text in the worker with the delimiter chosen by format. The worker
  already has `csvField` and `dumpTables` (sqlite-worker.mjs:86, 93); 5a generalizes
  the field-quoting to honour a tab delimiter and re-selects a single named table.
  This never re-runs the CREATE/INSERT/UPDATE/DELETE in the student's script: only a
  fresh read-only SELECT of the already-materialized table.
- **Python:** `df.to_csv(sep=..., index=False)` inside Pyodide returns the text
  directly; guard on `hasattr(v, "to_csv")` so a non-tabular value returns a clean
  error rather than a traceback. Mirrors `fetchFramePageCode` (pyodide-worker.mjs:185).
- **R:** `readr::format_csv(v)` / `readr::format_tsv(v)` returns the text directly;
  readr is bundled and already ensured by `ensurePackages`. Guard on `is.data.frame`,
  reusing the `__notframe__` sentinel from `fetchFrameR` (webr-worker.mjs:146).

**UI.** An Export control per tabular row in the Environment/Tables pane (beside the
existing View button) and in the open DataView header, offering CSV and TSV. Keyboard
and mouse accessible; disabled/absent for non-tabular rows.

**Security/serialization.** Only the named frame's cell values leave the tab; nothing
secret lives in the worker session to begin with, and the export reads one named
object, not the environment. SQL uses a read-only re-select and says so. A stale or
non-tabular name yields a clear error and does not alter the session. Full object
(all rows), never a silent truncation of the displayed page.

**Rough size.** Small. One protocol message, three short worker branches, one small
UI control, pure-helper Vitest tests, and a real SQL end-to-end. Feasible in-browser
with no caveats for realistic teaching-size frames.

---

## 5b — Other per-language formats (medium-large)

**Scope.** Beyond CSV/TSV, the richer per-language formats from requirement F, on the
same single-object export control (a longer format menu, with incompatible formats
disabled per language).

**Per-language feasibility (honest).**

- **R.** `.rds` via `saveRDS(v, file)` to the WebR FS then read the bytes back
  (`webR.FS.readFile`); `.RData`/`.rda` via `save(list=..., file=...)`. Both are
  genuinely feasible: WebR has a real filesystem (already used for uploads,
  webr-worker.mjs:336). rds preserves structure and attributes; models and functions
  serialize. Warn and exclude external pointers/connections/non-serializable
  resources (the requirement is explicit). `.csv`/`.tsv` gain the delimiter / colnames
  / rownames / NA / encoding controls deferred from 5a.
- **Python.** All feasible with bundled packages: `.json` via `to_json` (identify
  non-JSON-serializable columns and offer CSV as the safe alternative), `.xlsx` via
  `to_excel` (openpyxl is bundled/hosted), `.parquet` via `to_parquet` (pyarrow
  bundled), `.npy`/`.npz` via `numpy.save`/`savez` to a `BytesIO`, `.pkl` via
  `pickle.dumps`. Pickle must be clearly labelled and carries a trust warning that it
  must only be re-opened from trusted sources. Do not present open files, sockets, db
  connections, generators, or modules as safely restorable, and do not imply a full
  process can be restored.
- **SQL.** The result set to `.csv`/`.tsv`/`.json`. State clearly displayed rows vs
  the full result: our engine is in-browser SQLite with the whole table already in
  memory, so "full result set" is a complete read-only re-select, and "prefer
  streaming / server-side export for large results" from the requirement is **not
  applicable** here (there is no server database and no live connection). Say so
  rather than pretending to stream. Preserve column names and, as CSV/JSON allow,
  nulls and dates. Never include connection credentials (there are none).

**Security/serialization.** The binary bytes channel from the shared foundation.
Pickle/rds/RData are code-or-structure-bearing on the way back in; the export side
warns, and 5d owns the untrusted-import warning. Warn (do not silently truncate) when
an object cannot be serialized safely or completely.

**Rough size.** Medium-large, mostly per-format worker code plus a format menu with
per-language enablement and size estimation. All formats are feasible in-browser with
the bundled runtimes; the honest caveat is memory, not capability.

---

## 5c — Workspace export (large)

**Scope.** Export multiple selected objects or the whole serializable workspace, and
distinguish "export selected" from "export whole workspace". Exporting must never call
a session reset.

**Per-language feasibility (honest).**

- **R.** `save(list = ls(envir = globalenv()), envir = globalenv(), file = ...)`
  writes an `.RData` of the whole environment, preserving names; a selected subset is
  `save(list = c(...))`. Genuinely feasible and the natural R workspace format. Must
  not touch the session: it reads the global environment, it does not `rm` or restart.
- **Python.** There is no faithful "save the session". Serialize the serializable
  namespace object-by-object (a zip of per-variable files, or a single pickle of a
  filtered dict), skipping modules, functions that close over runtime state, open
  handles, and unpicklable objects, and reporting what was skipped. Label the feature
  **"Export variables" / "Export serializable workspace data", never "Save complete
  session"**, because the process cannot be restored. This is the requirement's
  explicit labeling rule.
- **SQL.** The "workspace" splits in two. The database itself can be exported as a
  real `.sqlite` file via `sqlite3.capi.sqlite3_js_db_export` (a byte image of the
  in-memory DB) which is feasible and faithful. The editor-side workspace (saved
  scripts, tabs, editor contents, params, non-sensitive references) is React state,
  not worker state, and serializes as JSON. Never include live connections or
  credentials (there are none in this engine).

**Security/serialization.** The whole-environment path is where accidental leakage
would live if any secret were ever put in the session; today none is, so the job is
to keep it that way and to warn on non-serializable resources rather than dropping
them silently. Show selected objects, format, estimated size, and warnings before
export; cancel leaves the session unchanged.

**Rough size.** Large. Multi-object selection UI, per-language whole-namespace
serialization, size estimation, progress for large exports, and the honest labeling.

---

## 5d — Import / restore with conflict handling (large)

**Scope.** For the workspace formats, an Import/Restore flow that inspects
name/format/expected contents before restoring, never silently overwrites same-named
variables, reports restored vs failed, warns on untrusted serialized objects, and
never auto-runs scripts or auto-reconnects.

**Approach in this codebase.** Mirror the existing upload path: `fileOp`
preview/import in the workers and `previewFile`/`importFile` in the manager
(manager.ts:299-312), driven by an `ImportDialog`-style component. Extend `fileOp` to
understand the workspace formats (RData with multiple objects, the pickle/zip
namespace bundle, the `.sqlite` image) and to return the list of contained object
names for the inspect-before-restore step. Conflict resolution
(overwrite/skip/rename/cancel) is dialog state plus a per-object import call.

**Security/serialization (the sharp edge of the whole slice).**

- **Untrusted code-executing formats.** `.rds`/`.RData` (R can carry promises and
  environments) and `.pkl` (arbitrary code on unpickle) require an explicit trust
  warning before restore, worded plainly. Restoring must not auto-execute.
- **No auto-run, no auto-reconnect.** Restoring a workspace must not run saved
  scripts or reconnect to anything without confirmation.
- **Naming conflicts explicit.** Offer overwrite / skip / rename / cancel; report
  which objects restored and which failed; a cancel leaves the session unchanged.
- **Language-aware, no cross-language restore.** An R `.RData` restores only in R,
  etc. No cross-language workspace compatibility is implied.

**Rough size.** Large. The dialog, per-format content inspection, conflict handling,
the trust-warning UX, and the per-object restore reporting are each non-trivial, and
this is the most security-sensitive sub-slice.

---

## Recommended order within slice 5

1. **5a** — single tabular to CSV/TSV. Builds the shared export protocol and download
   helper; smallest; most student value. Ship first.
2. **5b** — other formats on the same control, adding the binary bytes channel.
3. **5c** — workspace/multi-object export, with the honest Python labeling.
4. **5d** — import/restore with conflict handling and trust warnings (do last; it is
   the security-critical reverse direction and depends on 5c's formats existing).

## Honest overall feasibility note

Everything in F is feasible in-browser with the bundled runtimes: WebR has a real FS
(rds/RData), Pyodide has pandas/numpy/pyarrow/openpyxl/pickle, and sqlite-wasm can
export a byte image of its database. The real constraints are (1) **memory**, not
capability: a browser tab handles roughly the same ~25 MB envelope the upload path
already assumes (Sandbox.tsx:421), so large exports need progress and a size warning,
not a promise of server-side streaming we cannot keep; and (2) **restore safety**:
the code-executing formats are the one genuinely dangerous surface, confined to 5d and
gated behind explicit trust warnings with no auto-execution.
