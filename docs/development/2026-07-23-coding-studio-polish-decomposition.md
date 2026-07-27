# Coding Studio polish — slice decomposition (grounded in the code)

Companion to `2026-07-23-coding-studio-polish-requirements.md`. This confirms and
refines the five suggested slices against the real Coding Studio code, and for
each large slice records the concrete approach in THIS codebase (which files and
CodeMirror APIs), the main risks, and a rough size.

## Where the code lives (map)

- `web/components/sandbox/Sandbox.tsx` — the whole four-pane workspace. `Workspace`
  holds all session UI state in React: `entries` (console output, line 300),
  `plots`, `plotIndex`, `variables`, `dataTabs`, `running`, `preparing`. The
  `Toolbar` (line 734) has Run/Source/Restart etc. `ConsolePane` (line 954) renders
  the console. `PlotsPane` (line 1236) and `VariablesPane` (line 1133) fill the
  right column. `restart` (line 491) is the only thing today that clears the
  console, and it wipes everything (session, plots, variables, tabs).
- `web/components/run/CodeEditor.tsx` — the CodeMirror wrapper. CodeMirror is
  lazy-imported (`loadCodeMirror`, line 194). Extensions are assembled at line 95:
  `basicSetup`, the language mode (`loadLanguageMode`, line 253), an update
  listener, `themeExtensions` (line 340), `runKeymap` (line 283), inline
  completion, and runtime autocomplete. Height/scroll is set inside
  `themeExtensions` via `.cm-scroller { maxHeight }`.
- `web/lib/run/manager.ts` — `createSession` (line 427) returns a `RunSession`
  backed by one `LanguageRunner` (one Web Worker per language). `run` posts code to
  the worker and gets back `{ text, table, imageDataUrl, variables }`. The console
  entries, plots, and variables panes are all fed from that single return value in
  `Sandbox.execute` (line 351). The worker owns the live state (Python/R globals,
  the SQLite database); React only holds the rendered output.
- Workers: `web/public/workers/*` (referenced by `language.workerUrl`) plus the
  ggsql plot worker. D/E/F detection all happens client-side in the editor; only F
  needs new worker messages (to serialize objects).

Key wiring fact: the console, the Environment/Tables pane, and the Plots pane are
three independent React state slices, all updated from one `run` result. Clearing
one (the console) is a pure `setEntries([])` and cannot touch the worker, the
database, the variables, or the plots. This is why A is genuinely small.

## Confirmed slice list with rough sizes

| Slice | Feature | Rough size | Notes |
| --- | --- | --- | --- |
| 1 | A Clear Console + C scrollable editor | Small (both) | Quick win. Two isolated changes, low risk. Detailed plan: `...-plan-1-clear-and-scroll.md`. |
| 2 | D language-aware Ctrl/Cmd+Enter | Large | Per-language logical-statement detection. R is the hard part (no syntax tree). |
| 3 | E continuation indent + linting | Large | Overlaps the CodeMirror language configs; adds `@codemirror/lint`. |
| 4 | B Ctrl/Cmd+Click HELP tab | Large | New HELP pane + per-language doc mapping. SQL is SQLite-only (see below). |
| 5 | F export variables/workspaces + import/restore | Largest (multi-part) | Per-language serialization in the workers, formats, security. Split further. |

The five-slice split is sound and is confirmed. Two refinements:

- Slice 2 (D) and Slice 3 (E) share a foundation: both need a per-language,
  string/comment-aware understanding of statement and bracket structure. Build D
  first; E can reuse D's boundary/indent-context helpers. Consider a shared
  `lib/sandbox/lang-structure/` module seeded by D and extended by E.
- Slice 5 (F) is too big for one pass. Recommended sub-slices: 5a Export a single
  tabular object as CSV/TSV (the 80% case, all three languages); 5b other
  per-language formats (rds/RData, json/xlsx/npy/parquet/pkl); 5c workspace export;
  5d import/restore with conflict handling. Each is independently shippable.

---

## Slice 2 — D: language-aware Ctrl/Cmd+Enter

**Today.** `runKeymap` (CodeEditor.tsx:283) binds `Mod-Enter`. With a selection it
runs the selection; with no selection it runs exactly `state.doc.lineAt(sel.head).text`
— one physical line — then advances the cursor to the next line. The requirement is
to run the complete logical statement instead.

**Approach in this codebase.**

- Selection-first is already correct (keep the `!sel.empty` branch). Only the
  no-selection branch changes: replace "one line" with "statement range containing
  the cursor".
- Add `web/lib/sandbox/statement-range.ts` exporting
  `statementRangeAt(state: EditorState, pos: number, languageId: string): {from, to, nextPos}`.
  `runKeymap` calls it and dispatches `handler(state.sliceDoc(from, to))`, then sets
  the cursor to `nextPos`.
- Python and SQL have real Lezer grammars (`@codemirror/lang-python`,
  `@codemirror/lang-sql`, already dependencies). Use `syntaxTree(state)` from
  `@codemirror/language` and walk up from the node at `pos` to the nearest
  top-level statement node (for Python: the `Statement`/compound node whose parent
  is the script/body; include a leading decorator and all connected `elif/else`,
  `except/else/finally` sections by extending to sibling clauses; for SQL: the
  `Statement` node, so `WITH ... SELECT` and parenthesised subqueries come as one
  unit). The Lezer tree already ignores syntax characters inside strings and
  comments, which satisfies the "ignore operators inside strings/comments"
  requirement for these two languages for free.
- R has NO Lezer grammar — it uses a legacy `StreamLanguage` (`loadLanguageMode`,
  CodeEditor.tsx:260), so `syntaxTree` yields no useful structure. R needs a bespoke
  scanner in `statement-range.ts`: a forward/backward line walker that tracks
  bracket depth across `() [] {}`, treats a line as continuing when the previous
  non-comment token is a pipe (`|>`, `%>%`) or a trailing `+` in a ggplot chain or
  a binary/assignment operator, and skips characters inside strings (`"`, `'`,
  backtick) and after `#` comments. Start from the cursor line, walk backward to the
  statement start, then forward to its end.

**Main risks.**

- R's hand-written detection is the bulk of the effort and the main correctness
  risk: distinguishing a continuation `+` from a `+` inside a string or comment, and
  knowing when a pipe chain has ended. Needs a thorough unit-test table.
- Python compound-statement assembly (attaching `elif/else`, `except/finally`,
  decorators) is fiddly; base it on node types and indentation, not text.
- `basicSetup` and other keymaps also bind Enter-family keys; `runKeymap` already
  wraps its bindings in `Prec.high` (CodeEditor.tsx:288), so precedence is handled.
- The Run button title copy in `Toolbar` (line 791, "runs the current line") must be
  updated to "runs the current statement".

**Rough size.** Large. Python + SQL tree walk is moderate; R scanner + its test
table is the larger half. Mostly unit-testable (`statement-range.ts` is pure), which
de-risks it.

---

## Slice 3 — E: continuation indentation + language-aware linting

**Today.** Indentation is whatever `basicSetup` plus the language mode provides.
Python's Lezer package ships an `indentService`, so Python already indents after a
colon reasonably. SQL's indentation is weak, and R (legacy stream mode) has
essentially none. There is no linter: `@codemirror/lint` is not a dependency.

**Approach in this codebase.**

- Indentation uses CodeMirror's `indentService` / `indentOnInput` facets from
  `@codemirror/language`. Add a per-language indent extension in a new
  `web/lib/sandbox/indent.ts`, wired into the extensions array in CodeEditor.tsx
  (near line 106 with the other language extensions). Python: lean on the package's
  service and enforce the 4-space, no-tabs policy. R and SQL: custom `indentService`
  that reuses the bracket/pipe/clause logic from slice 2's `statement-range.ts`
  (indent after an open bracket or a trailing pipe/`+`; dedent after a closing
  bracket; SQL dedents major clauses `FROM/WHERE/GROUP BY/...`).
- Linting: add `@codemirror/lint` and a `linter()` source per language. Keep it
  unobtrusive (underlines, not rewrites), debounced (the `linter` config has a
  `delay`), never touching strings/comments, never moving the cursor. Heavier checks
  run on pause/execute, not every keystroke, per the requirement.

**Main risks.**

- "Never change behavior / never touch quoted text" is the core constraint; auto
  indent on Enter must be undo-able in one step (CodeMirror transactions are, by
  default) and must not run a full-document formatter.
- R indentation is fully bespoke (no grammar) and is the main effort, but it can
  share code with slice 2, so ordering D before E pays off.
- Linting scope creep: keep the first pass to a few obvious rules per language, not a
  full static analyzer.

**Rough size.** Large. Smaller if slice 2 lands first and shares the R/SQL structure
helpers.

---

## Slice 4 — B: Ctrl/Cmd+Click opens a HELP tab

**Today.** No HELP tab exists. The right column is a vertical `Group` with a
Variables/Tables panel and a Plots panel (Sandbox.tsx:689-715). The editor has no
mousedown handler; CodeMirror handles clicks itself.

**Approach in this codebase.**

- Add a HELP tab beside PLOTS. Two options: (a) make the Plots panel a small tabbed
  container (Plots | Help), or (b) add a third panel to the right `Group`. Option (a)
  keeps the layout stable and matches "positioned next to PLOTS". Reuse ONE tab:
  store `helpTarget` state in `Workspace`; Ctrl/Cmd+click replaces its contents and
  selects the Help tab.
- Detect the modified click in the editor with
  `EditorView.domEventHandlers({ mousedown(event, view) })` added to the extensions
  array in CodeEditor.tsx. On `event.metaKey || event.ctrlKey`, compute
  `view.posAtCoords({x: event.clientX, y: event.clientY})`, read the token at that
  position (word range via `state.wordAt(pos)`, or the Lezer node for Python/SQL),
  call `preventDefault()` so the click does not move the caret (preserving cursor and
  script position, as required), and surface the token + language to a callback prop
  (mirror the existing `onRunLine` prop wiring).
- Map token -> documentation per language in `web/lib/sandbox/help-docs.ts`:
  - R: `summarise` -> dplyr reference, `mean` -> base R reference. Map known
    tidyverse/base names to hosted doc URLs.
  - Python: `pandas.DataFrame.groupby` -> pandas docs; `len` -> docs.python.org.
    Resolve the qualified name from the token and its receiver where possible.
  - SQL: `COUNT/AVG/DATE_TRUNC` -> function docs; `JOIN/GROUP BY/WITH` -> syntax
    docs. See the honest scope note below.
- Rendering: a strict CSP is in force for the app; embedding third-party doc sites in
  an iframe will likely be blocked, and is a moving target. Safer first pass: the
  HELP tab shows the resolved symbol, a short locally-authored blurb, and a "Open
  documentation" link that opens the canonical page in a new tab. A later pass could
  fetch and inline sanitized doc snippets if a same-origin proxy is added.

**Honest SQL scope note (dialect awareness).** The in-browser SQL engine is SQLite
(`@sqlite.org/sqlite-wasm`; see Sandbox starter copy line 69 and the SQL worker).
There is only one dialect available at runtime, so "dialect-aware docs" for
PostgreSQL, MySQL, BigQuery, and Snowflake is aspirational, not real: every SQL doc
link should point at the SQLite documentation, because that is the only engine a
student can actually run here. Build the mapping so a dialect parameter exists (for
future engines) but ship it wired to SQLite only, and say so in any UI copy. Do not
imply `DATE_TRUNC` behaves as in Postgres; SQLite lacks it.

**Main risks.** CSP/iframe embedding (mitigated by the link-first approach);
accuracy of token-to-URL mapping (especially qualified Python names and R's
which-package-owns-this-function problem); keeping exactly one HELP tab and not
disturbing the caret.

**Rough size.** Large. The click plumbing is modest; the per-language doc mapping and
the HELP pane UI are the bulk.

---

## Slice 5 — F: export variables / workspaces + import/restore

**Today.** Export exists only for plots (PlotsPane `exportCurrent`, Sandbox.tsx:1254,
a PNG download) and for the script (`download`, line 472). There is no way to export a
variable, a table, or the workspace. Variables are surfaced as `SessionVariable`
(manager.ts:24) with name/type/info/columns; the real objects live only in the
worker.

**Approach in this codebase.**

- UI: an Export affordance in `VariablesPane` (Sandbox.tsx:1133) per row, plus a
  panel-level "Export" for whole-workspace, and (SQL) an export on the displayed
  result. Reuse the download pattern already in `download`/`exportCurrent`
  (Blob + object URL + `a.click()`).
- Data path: the browser has only rendered previews, not the objects, so serialization
  must happen in the worker. Add worker messages and matching `RunSession` methods in
  manager.ts (mirror `getData`/`dumpTables`, lines 260-295): e.g.
  `exportObject(name, format)` returning bytes. Each language worker implements the
  serialization (R: csv/tsv/rds/RData via its runtime; Python: csv/tsv/json/xlsx/
  npy/npz/parquet/pkl; SQL: stream the result set, distinguishing displayed rows from
  the full result).
- Import/restore (workspace formats): a dialog that inspects name/format/contents
  before restoring, offers overwrite/skip/rename/cancel on name clashes, reports which
  objects restored vs failed, and never auto-runs code or auto-reconnects. Reuse the
  existing `ImportDialog` component pattern (`web/components/sandbox/ImportDialog.tsx`)
  and `previewFile`/`importFile` (manager.ts:299-312).

**Main risks (security-sensitive).**

- Never include credentials, tokens, or live connection details by default (the
  requirement is explicit). Since nothing secret currently lives in the worker
  session, the main job is to keep it that way and to warn on non-serializable
  resources rather than silently dropping them.
- Pickle / RData / any code-executing format needs a trust warning on import and must
  not auto-execute.
- Large exports must stream / show progress and not freeze the tab (the 25 MB upload
  ceiling in Sandbox.tsx:421 is a useful reference point for limits).
- Honest labeling: call it "Export variables" / "Export serializable workspace data",
  not "Save complete session", because a full process cannot be restored.

**Rough size.** Largest; split into 5a–5d as above. 5a (single tabular object to
CSV/TSV) is the smallest shippable increment and delivers most of the student value.

---

## Recommended build order

1. Slice 1 (A + C) — quick win, unblocks nothing but ships visible value now.
2. Slice 2 (D) — build the shared per-language structure helpers here.
3. Slice 3 (E) — reuse slice 2's helpers.
4. Slice 4 (B) — independent; can run in parallel with 2/3 if staffed separately.
5. Slice 5 (F), sub-sliced 5a → 5d — largest, do last, ship 5a first.
