# Coding Studio GUI polish — requirements (from professor, 2026-07-23)

Coding Studio supports R, Python, and SQL. These usability, documentation,
formatting, and code-execution requirements must be addressed consistently
across all three languages. This is a large, multi-feature epic: it should be
decomposed into slices (some small, some large), not built in one pass.

Standing constraints: WCAG 2.1 AA (keyboard + mouse), Miami brand tokens, no em
dashes, no secrets in the client. Keyboard shortcuts support Ctrl (Windows/Linux)
and Command (macOS).

## A. Clear Console button (small)

A clearly visible **Clear** button in the console/output panel.
- Clicking removes all current console/query output.
- Does NOT modify the script, and does NOT reset variables, imported libraries,
  database connections, plots, or the session.
- Optional keyboard shortcut; the visible button is the priority.
- For SQL: clears displayed query results, messages, and errors without
  disconnecting the active database session.

## B. Context-sensitive docs with Ctrl/Cmd + Click (large)

Ctrl+Click (Win/Linux) or Cmd+Click (macOS) on a function or recognized language
element opens its documentation in a **HELP** tab positioned next to **PLOTS**.
- Docs correspond to the specific clicked function/keyword; source matches the
  active language. Reuse ONE HELP tab (do not open multiple). Preserve the
  script position and cursor when HELP opens.
- R: `summarise()` -> dplyr docs; `mean()` -> R docs for mean.
- Python: `pandas.DataFrame.groupby()` -> pandas docs; `len()` -> Python docs.
- SQL: `COUNT()` / `AVG()` / `DATE_TRUNC()` -> active-dialect function docs;
  keywords like `JOIN`, `GROUP BY`, `WITH` -> syntax docs when function-specific
  docs do not apply.
- SQL docs must be dialect-aware where possible (PostgreSQL, SQLite, MySQL,
  BigQuery, Snowflake differ). NOTE: our in-browser SQL is SQLite (sqlite-wasm);
  dialect awareness is aspirational for other engines.

## C. Scrollable long scripts (small)

When a script exceeds the visible editor area, provide normal vertical scrolling.
- Scroll the whole script; cursor stays visible while typing/navigating.
- Mouse-wheel, trackpad, scrollbar drag, Page Up/Down, keyboard nav all work.
- Editor does not expand beyond its panel/page layout.
- Horizontal scrolling available when a line is wider than the editor, unless
  line wrapping is enabled. Consistent for R, Python, SQL.

## D. Ctrl/Cmd + Enter runs the complete logical statement (large, language-aware)

Currently Ctrl/Cmd+Enter runs one physical line; it must run the complete logical
statement containing the cursor.

Execution priority:
1. If text is selected, run only the selection.
2. Else identify the complete logical statement containing the cursor.
3. Execute only that statement/block; do not include the next unrelated
   statement.
4. Optionally move the cursor to the next executable statement after execution.

Detection must ignore operators/semicolons/brackets/keywords/continuation chars
inside strings or comments.

- **R:** single complete line runs alone. A multi-line pipe chain (`|>`, `%>%`,
  related operators) runs from its start through the end of the chain regardless
  of which line the cursor is on. A multi-line `ggplot` joined by `+` runs through
  the final layer (distinguish a continuation `+` from `+` in a string/comment).
  A multi-line call/assignment/expression enclosed by `()`/`[]`/`{}` runs whole
  (for example a multi-line `tibble(...)`).
- **Python:** a complete expression runs alone; multi-line expressions in
  `()`/`[]`/`{}` run whole from any line. Indentation-based blocks run complete
  (functions, classes, if/elif/else, for/while, try/except/else/finally, with,
  match/case). Compound structures include all connected sections (if includes
  elif/else; try includes except/else/finally). Support explicit/implicit
  continuation: open brackets, backslash, method chaining, comprehensions,
  decorators (a decorator runs with its function/class), triple-quoted strings,
  and syntax chars inside comments/strings.
- **SQL:** run the complete statement containing the cursor (ends at `;` or a
  detected boundary), not the physical line. Support CTEs (`WITH ... SELECT` runs
  together, not just the CTE or just the final SELECT), nested queries and
  parentheses (run the whole parent statement), multi-statement scripts (run only
  the statement at the cursor), and procedural blocks when the dialect allows
  (`BEGIN...END`, `CASE...END`, stored procs, functions, transactions, custom
  delimiters) without treating internal `;` as the block end.

## E. Automatic continuation indentation + language-aware linting (large)

On Enter while clearly continuing the previous statement, auto-indent to the
appropriate level (important for presentations; necessary for Python
correctness). Language-based.
- **R:** indent after opening `(`/`[`/`{`; continue after `|>`/`%>%`; continue
  after a `ggplot` layer ending in `+`; align args in multi-line calls; dedent
  after closing bracket; no continuation indent when the prior statement is
  complete.
- **Python:** indent after `:`; preserve indentation in blocks; dedent when a
  block ends; indent continued expressions in brackets; preserve valid hanging
  indentation; handle method chains/comprehensions; never change indentation in a
  way that alters behavior; consistent 4-space policy, no mixed tabs/spaces.
- **SQL:** indent columns under `SELECT`; conditions under `WHERE`/`HAVING`/`ON`;
  CTE contents; nested subqueries; align clauses; dedent major clauses
  (`FROM`/`WHERE`/`GROUP BY`/`HAVING`/`ORDER BY`/`LIMIT`); respect the active
  formatter/dialect; never change query meaning.
- **Linting:** flag obvious syntax/format problems with rules appropriate to the
  language/dialect; do not aggressively rewrite unrelated code while typing; do
  not move the cursor unexpectedly; do not change quoted text/comments/strings;
  prefer unobtrusive warnings/underlines when unsafe to auto-fix; allow undo of
  any auto-format; do not run a full-document formatter every keystroke.
  Continuation indent may happen on Enter; heavier lint/format on a pause, on
  demand, or on execute.

## F. Export variables and workspaces (large; security-sensitive)

A visible **Export** action in/near the variables/environment/results panel.
General behavior: export one object, multiple selected objects, or the full
workspace/session where supported; exporting never modifies/removes/resets
session objects; distinguish "export selected" vs "export whole workspace"; choose
file name and (when applicable) format; download via the browser; show progress +
clear errors for large exports; disable incompatible formats; keyboard + mouse
accessible; NEVER include credentials/tokens/secrets/live connection details by
default; explain (do not silently truncate) when an object cannot be serialized
safely/completely.

- **R:** export data frames/tibbles, vectors, matrices, lists, models, functions,
  serializable objects. Formats: `.csv`/`.tsv` (tabular, with delimiter/colnames/
  rownames/NA/encoding controls), `.rds` (single object, preserves structure/
  attributes), `.RData`/`.rda` (multiple objects/workspace, preserves names).
  Warn/exclude external pointers/connections/non-serializable resources. Workspace
  export must not call a session reset. Plots download via the existing plot
  workflow (not a workspace-export replacement).
- **Python:** export DataFrames/Series, ndarrays, lists/dicts/tuples/scalars,
  models, multiple vars, the serializable namespace. Formats: `.csv`/`.tsv`,
  `.json` (identify non-JSON-serializable and offer a safe alternative), `.xlsx`,
  `.npy`/`.npz`, `.parquet`, `.pkl` (clearly identified). Warn that pickle must
  only be opened from trusted sources; DataFrame controls (index/colnames/NA/
  encoding); do not present open files/sockets/db-connections/generators/modules
  as safely restorable; do not imply a full process can be restored. Label the
  feature "Export variables" / "Export serializable workspace data", NOT "Save
  complete session", when full restoration is not supported.
- **SQL:** export the displayed query result and (subject to permissions/size
  limits) the complete result set; clearly state displayed-rows vs full-result;
  preserve column names/types/nulls/dates/timestamps/timezones/precision/encoding
  as the format allows; exporting must NOT rerun data-changing statements
  (INSERT/UPDATE/DELETE/MERGE/CREATE/ALTER/DROP/side-effecting procs); if a
  read-only rerun is needed, say so; never include connection credentials;
  workspace export may include saved scripts/tabs/editor contents/params/
  non-sensitive connection refs, never live connections. Prefer streaming/
  server-side export for large results over loading all into browser memory.
- **Import/restore:** for workspace formats, provide Import/Restore. Inspect
  name/format/expected contents before restoring; never silently overwrite
  same-named variables (offer overwrite/skip/rename/cancel); report which objects
  restored vs failed; require an explicit warning for untrusted serialized objects
  (especially code-executing formats); restoring must not auto-run scripts or
  auto-reconnect to external services without confirmation; behavior is
  language-aware (no cross-language workspace compatibility implied).
- **Discoverability/feedback:** visible Export button/menu in the relevant panel;
  context menus may add Export variable/selected/results; show selected objects +
  format + estimated size + warnings before export; confirm the generated file on
  success; actionable error on failure without altering the session; cancel leaves
  the session unchanged; do not block editor interaction unnecessarily; safe
  default file names + valid extensions; accessibility labels distinguish export/
  import/download/restore.

## General acceptance criteria (apply throughout)

Consistent across R/Python/SQL; Ctrl(Win/Linux)/Cmd(macOS); selection takes
priority; no-selection runs the complete logical statement; never include the
next unrelated statement; ignore syntax chars in strings/comments; SQL is
dialect-aware where possible; Python respects indentation + connected clauses; R
supports pipes/multi-line calls/ggplot continuation; clearing console output does
not reset console/plots/variables/db-connections/session; export never resets the
session or leaks secrets; unsupported/partial serialization warns rather than
silently truncating; restore handles naming conflicts explicitly; unsafe imports
warn; export/import work by keyboard and mouse.

## Suggested decomposition (to confirm)

- Slice 1 (quick wins): A (Clear Console) + C (scrollable editor). Small
  CodeMirror/console changes.
- Slice 2: D (language-aware Ctrl+Enter statement execution). R/Python/SQL
  logical-statement detection. Substantial.
- Slice 3: E (continuation indentation + linting). Language-aware. Substantial;
  overlaps CodeMirror language configs.
- Slice 4: B (Ctrl/Cmd+Click HELP tab, dialect-aware docs). New HELP pane +
  doc sources per language. Substantial.
- Slice 5: F (export variables/workspaces + import/restore). Largest; per-language
  serialization, formats, and security. Likely its own multi-part slice.
