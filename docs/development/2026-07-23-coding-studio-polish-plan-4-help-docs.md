# Slice 4 — B: Ctrl/Cmd+Click opens context-sensitive docs in a HELP tab — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is a planning document only; do not build from it in the session that produced it.

**Goal:** Ctrl+Click (Windows/Linux) or Cmd+Click (macOS) on a function or
recognized language element in the editor resolves the clicked symbol to its
documentation and shows it in a single reusable **HELP** tab positioned next to
**PLOTS**. The HELP tab names the symbol, its source (dplyr, base R, ggplot2,
pandas, NumPy, Python, SQLite), a short locally authored blurb, and a prominent
"Open documentation" link that opens the official page in a new browser tab.
Clicking does not move the editor caret or scroll; a keyboard equivalent (F1 on
the symbol at the cursor) is provided for accessibility. R uses `summarise` ->
dplyr and `mean` -> base R; Python uses `DataFrame.groupby` -> pandas and `len`
-> Python; SQL uses `COUNT`/`AVG` -> SQLite functions and `JOIN`/`GROUP BY`/`WITH`
-> SQLite syntax.

**Architecture.** Two pure, string-in modules with no CodeMirror-view
dependency, both unit-tested in Vitest's `node` environment:

- `web/lib/sandbox/help-docs/symbol-at.ts` resolves the token under a document
  offset to a `HelpRequest` (`{ name, qualifier?, kind, language }`). Python and
  SQL reuse the same Lezer parsers already proven in Slice 2
  (`@lezer/python`'s `parser`, `SQLite.language.parser`); R uses a word scan over
  the string/comment mask from `lib/sandbox/lang-structure/mask.ts`.
- `web/lib/sandbox/help-docs/resolve.ts` maps a `HelpRequest` to a `DocEntry`
  (`{ symbol, source, url, blurb?, note? }`) using small curated per-language
  tables. Pure object lookups, no heavy imports, so it loads in the main Sandbox
  bundle.

The editor gains one modified-mousedown handler and one F1 keybinding that both
call `symbolAt(doc, pos, language)` and hand the `HelpRequest` up through a new
`onHelp` prop (mirroring the existing `onRunLine` ref wiring). The Sandbox's
`Workspace` calls `resolveDoc` and stores the result in one `helpTarget` state
slot, then selects the HELP tab. The bottom-right pane becomes a two-tab panel
(Plots | Help) so there is exactly one HELP tab, next to PLOTS.

**The COEP constraint and why we open in a new tab.** `/ai-sandbox` is
cross-origin isolated (`next.config.ts` sets COOP `same-origin` + COEP
`require-corp` on the page, the workers, and the runtimes, to give WebR its
SharedArrayBuffer channel). Under COEP `require-corp`:

- An `<iframe>` to a cross-origin docs site (dplyr.tidyverse.org,
  pandas.pydata.org, docs.python.org, sqlite.org) is **blocked** unless that site
  returns `Cross-Origin-Embedder-Policy` / `Cross-Origin-Resource-Policy` headers,
  which none of them do.
- A cross-origin `fetch()` of those pages is **CORS-blocked**, so we cannot pull
  and inline their HTML either.

Therefore the HELP tab does not embed or fetch external doc pages. It renders the
resolved symbol, source, and a bundled blurb locally, plus an
`<a target="_blank" rel="noopener noreferrer">` "Open documentation" link.
`target="_blank"` opens a new top-level browsing context, which is a separate
document not governed by our page's embedder policy, so it is not blocked. This
is the pragmatic MVP: a curated symbol-to-URL map plus open-in-new-tab. A full
offline doc corpus (bundling the actual reference HTML for every function) is out
of scope and is called out as such in the UI and below.

**Honest SQL dialect scope.** The only SQL engine that runs here is SQLite
(`@sqlite.org/sqlite-wasm`; the SQL worker is `sqlite-worker.mjs`,
`lib/run/languages.ts:33`). Every SQL doc link points at the SQLite
documentation, because that is the only engine a student can actually run.
`resolveDoc` takes an optional `dialect` parameter so a future Postgres/MySQL/
BigQuery/Snowflake engine could be wired in, but it ships resolved to SQLite
only, and when a non-SQLite dialect is requested it still returns SQLite docs
plus a `note` stating that only SQLite runs here and other dialects may differ.
`DATE_TRUNC` is handled honestly: SQLite has no `DATE_TRUNC`, so it resolves to
the SQLite date-and-time functions page with a note pointing at `strftime()` and
explaining `DATE_TRUNC` is a PostgreSQL/BigQuery function that does not run here.

**Tech Stack:** Next.js (this repo's vendored build; read
`node_modules/next/dist/docs/` before touching any Next API, though this slice
touches none), React 19, CodeMirror 6 (`@codemirror/view` `posAtCoords` /
`domEventHandlers` / `keymap`, `@codemirror/state`), the Lezer parsers
`@lezer/python` (transitive via `@codemirror/lang-python`) and
`@codemirror/lang-sql` (both already dependencies), the existing
`lib/sandbox/lang-structure/mask.ts`, `react-resizable-panels` (already used for
the panes), Vitest for the pure unit tests, Playwright + `@axe-core/playwright`
for the sandbox e2e.

## Global Constraints

- **No git commits.** The working tree stays uncommitted; each task ends by
  running verification commands instead of committing.
- **COEP require-corp is in force on `/ai-sandbox`.** No `<iframe>` to and no
  `fetch()` of a cross-origin docs site. The only way the HELP tab reaches an
  external doc page is an `<a target="_blank" rel="noopener noreferrer">` link.
  Do not add a docs proxy in this slice.
- **Pure logic is separate from CodeMirror glue.** `help-docs/symbol-at.ts` and
  `help-docs/resolve.ts` take plain strings and offsets and return plain objects.
  They must not import `@codemirror/view` or touch an `EditorView`, so they run in
  Vitest's `node` environment. Only `CodeEditor.tsx` bridges the view to
  `symbolAt`.
- **Exactly one HELP tab, reused.** Ctrl/Cmd+Click (or F1) replaces the contents
  of the single HELP tab and selects it. It never opens a second HELP tab.
- **Preserve the caret and scroll.** The modified mousedown calls
  `preventDefault()` so CodeMirror does not move the selection, and the handler
  never calls `view.focus()` or `scrollIntoView`. Selecting the HELP tab is a
  React state change in the right column and does not touch the editor.
- **SQLite is the only SQL engine.** Every SQL doc URL is a SQLite page. The
  `dialect` parameter exists for the future but ships wired to SQLite, and any
  non-SQLite request carries an honest note. Do not imply `DATE_TRUNC` behaves as
  in Postgres.
- **A full offline doc corpus is out of scope.** The MVP is a curated
  symbol-to-URL map plus a short bundled blurb and an open-in-new-tab link. Say so
  in the empty-state and unknown-symbol copy.
- **No em dashes in any user-facing copy** (tab labels, tooltips, blurbs, notes,
  empty states, the "Open documentation" link text). Use commas, colons, or
  sentence breaks.
- **Keyboard and mouse (WCAG 2.1 AA).** Ctrl/Cmd+Click is the mouse path; F1 on
  the symbol at the cursor is the keyboard path. Tabs use `role="tab"` /
  `role="tablist"` / `aria-selected`; the doc link is a real anchor.
- **Miami brand tokens.** Reuse the `--sb-*` CSS variables already used by the
  other panes; no new colors.
- All commands run from `web/` (the Next app root). Unit tests: `npm run test`.
  The Playwright config starts its own dev server on port 3100, so
  `npm run test:e2e` needs no separate server.

## File Structure

- `web/lib/sandbox/help-docs/types.ts` — Create. Shared types: `HelpLanguage`
  (`"r" | "python" | "sql"`), `SymbolKind` (`"function" | "keyword"`),
  `HelpRequest`, `DocSource`, `DocEntry`, `SqlDialect`.
- `web/lib/sandbox/help-docs/resolve.ts` — Create. The curated per-language
  symbol-to-URL tables and `resolveDoc(req, opts?): DocEntry | null`, plus
  `referenceHome(language): DocEntry` for the unknown-symbol fallback. Pure; no
  CodeMirror or Lezer imports.
- `web/lib/sandbox/help-docs/symbol-at.ts` — Create.
  `symbolAt(text, pos, language): HelpRequest | null`. Uses `@lezer/python`'s
  `parser` (Python), `SQLite.language.parser` (SQL), and `maskStringsAndComments`
  (R and the SQL string/comment guard). Lazy-loaded by the editor so the parsers
  stay out of the initial bundle.
- `web/lib/sandbox/help-docs/index.ts` — Create. Re-exports `symbolAt`,
  `resolveDoc`, `referenceHome`, and the types.
- `web/components/run/CodeEditor.tsx` — Modify. Add an `onHelp` prop and a
  `helpRef`; lazy-load `help-docs/symbol-at`; add a modified-mousedown
  `domEventHandlers` extension and an F1 keybinding, both gated on `hasHelp`;
  extend `LoadedEditor` / `loadCodeMirror`.
- `web/components/sandbox/Sandbox.tsx` — Modify. Add `helpTarget` and
  `rightLowerTab` state to `Workspace`; wire `onHelp` on `CodeEditor` through
  `resolveDoc`; replace the bottom-right `PlotsPane` with a `PlotsHelpPane`
  (a two-tab Plots | Help panel); add the `HelpBody` renderer; pass the platform
  modifier hint.
- `web/tests/unit/help-docs.test.ts` — Create. Vitest tables for `resolveDoc`
  (R/Python/SQL/dialect honesty/unknown) and `symbolAt` (R/Python/SQL token
  extraction, including `GROUP BY` and `pkg::fn`).
- `web/tests/e2e/sandbox.spec.ts` — Modify. Add Ctrl/Cmd+Click and F1 e2e tests
  (HELP appears beside PLOTS with the right symbol and a working new-tab link,
  a second click reuses the one HELP tab, the caret does not move, and axe stays
  clean).

---

## Task 1: Types and the curated symbol-to-URL resolver (pure)

`resolveDoc` is the value core of the slice: given a `HelpRequest` it returns a
`DocEntry` with the source label, the canonical URL, and a short bundled blurb.
It is pure object lookups over small curated tables, so it is trivially
unit-testable and safe to import into the main Sandbox bundle.

**Files:**
- Create: `web/lib/sandbox/help-docs/types.ts`,
  `web/lib/sandbox/help-docs/resolve.ts`
- Test: `web/tests/unit/help-docs.test.ts` (resolve section)

**Interfaces:**
- Produces:
  - `type HelpLanguage = "r" | "python" | "sql"`
  - `type SymbolKind = "function" | "keyword"`
  - `type SqlDialect = "sqlite" | "postgres" | "mysql" | "bigquery" | "snowflake"`
  - `interface HelpRequest { name: string; qualifier?: string; kind: SymbolKind; language: HelpLanguage }`
  - `interface DocEntry { symbol: string; source: string; url: string; blurb?: string; note?: string }`
  - `resolveDoc(req: HelpRequest, opts?: { dialect?: SqlDialect }): DocEntry | null`
  - `referenceHome(language: HelpLanguage): DocEntry`

- [ ] **Step 1: Write the failing resolver tests**

Create `web/tests/unit/help-docs.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { resolveDoc, referenceHome } from "@/lib/sandbox/help-docs/resolve";
import type { HelpRequest } from "@/lib/sandbox/help-docs/types";

function req(partial: Partial<HelpRequest> & { name: string }): HelpRequest {
  return {
    kind: "function",
    language: partial.language ?? "r",
    qualifier: partial.qualifier,
    name: partial.name,
  };
}

describe("resolveDoc: R", () => {
  it("maps summarise to the dplyr reference", () => {
    const d = resolveDoc(req({ name: "summarise", language: "r" }));
    expect(d?.source).toBe("dplyr");
    expect(d?.url).toBe("https://dplyr.tidyverse.org/reference/summarise.html");
    expect(d?.blurb).toBeTruthy();
  });

  it("accepts the American spelling summarize", () => {
    expect(resolveDoc(req({ name: "summarize", language: "r" }))?.url).toBe(
      "https://dplyr.tidyverse.org/reference/summarise.html",
    );
  });

  it("maps mean to base R", () => {
    const d = resolveDoc(req({ name: "mean", language: "r" }));
    expect(d?.source).toBe("base R");
    expect(d?.url).toBe(
      "https://stat.ethz.ch/R-manual/R-devel/library/base/html/mean.html",
    );
  });

  it("maps ggplot to ggplot2", () => {
    const d = resolveDoc(req({ name: "ggplot", language: "r" }));
    expect(d?.source).toBe("ggplot2");
    expect(d?.url).toBe("https://ggplot2.tidyverse.org/reference/ggplot.html");
  });

  it("returns null for an unknown R symbol", () => {
    expect(resolveDoc(req({ name: "no_such_fn_xyz", language: "r" }))).toBeNull();
  });
});

describe("resolveDoc: Python", () => {
  it("maps a DataFrame method to the pandas reference", () => {
    const d = resolveDoc(
      req({ name: "groupby", qualifier: "df", kind: "function", language: "python" }),
    );
    expect(d?.source).toBe("pandas");
    expect(d?.url).toBe(
      "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html",
    );
  });

  it("maps a pandas top-level function", () => {
    expect(
      resolveDoc(req({ name: "read_csv", qualifier: "pd", language: "python" }))?.url,
    ).toBe("https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html");
  });

  it("maps len to the Python builtins docs", () => {
    const d = resolveDoc(req({ name: "len", language: "python" }));
    expect(d?.source).toBe("Python");
    expect(d?.url).toBe("https://docs.python.org/3/library/functions.html#len");
  });

  it("returns null for an unknown Python symbol", () => {
    expect(
      resolveDoc(req({ name: "totally_made_up", language: "python" })),
    ).toBeNull();
  });
});

describe("resolveDoc: SQL (SQLite only)", () => {
  it("maps COUNT to the SQLite aggregate functions page", () => {
    const d = resolveDoc(req({ name: "COUNT", kind: "function", language: "sql" }));
    expect(d?.source).toBe("SQLite");
    expect(d?.url).toBe("https://www.sqlite.org/lang_aggfunc.html");
  });

  it("is case-insensitive for SQL (avg)", () => {
    expect(
      resolveDoc(req({ name: "avg", kind: "function", language: "sql" }))?.url,
    ).toBe("https://www.sqlite.org/lang_aggfunc.html");
  });

  it("maps JOIN and GROUP BY and WITH to SQLite syntax pages", () => {
    expect(
      resolveDoc(req({ name: "JOIN", kind: "keyword", language: "sql" }))?.url,
    ).toBe("https://www.sqlite.org/lang_select.html");
    expect(
      resolveDoc(req({ name: "GROUP BY", kind: "keyword", language: "sql" }))?.url,
    ).toBe("https://www.sqlite.org/lang_select.html");
    expect(
      resolveDoc(req({ name: "WITH", kind: "keyword", language: "sql" }))?.url,
    ).toBe("https://www.sqlite.org/lang_with.html");
  });

  it("handles DATE_TRUNC honestly (SQLite has none)", () => {
    const d = resolveDoc(req({ name: "DATE_TRUNC", kind: "function", language: "sql" }));
    expect(d?.source).toBe("SQLite");
    expect(d?.url).toBe("https://www.sqlite.org/lang_datefunc.html");
    expect(d?.note).toMatch(/SQLite has no DATE_TRUNC/i);
    expect(d?.note).toMatch(/strftime/i);
  });

  it("notes that only SQLite runs when another dialect is requested", () => {
    const d = resolveDoc(
      req({ name: "COUNT", kind: "function", language: "sql" }),
      { dialect: "postgres" },
    );
    // Still SQLite docs (the only engine that runs here), with an honest note.
    expect(d?.source).toBe("SQLite");
    expect(d?.url).toBe("https://www.sqlite.org/lang_aggfunc.html");
    expect(d?.note).toMatch(/only SQLite runs/i);
  });

  it("returns null for an unknown SQL token", () => {
    expect(
      resolveDoc(req({ name: "FLOORP", kind: "function", language: "sql" })),
    ).toBeNull();
  });
});

describe("referenceHome", () => {
  it("gives a per-language reference home for the unknown-symbol fallback", () => {
    expect(referenceHome("r").url).toContain("rdocumentation.org");
    expect(referenceHome("python").url).toContain("docs.python.org");
    expect(referenceHome("sql").url).toContain("sqlite.org");
  });
});
```

- [ ] **Step 2: Run the resolver tests and confirm they fail**

Run: `npm run test -- help-docs`
Expected: FAIL. The modules do not exist yet (import error).

- [ ] **Step 3: Create the types**

Create `web/lib/sandbox/help-docs/types.ts`:

```typescript
/** The three runnable Coding Studio languages, for documentation lookup. */
export type HelpLanguage = "r" | "python" | "sql";

/** Whether the clicked token reads as a callable or a language keyword. A hint,
 *  not authoritative: `resolveDoc` tries both tables regardless. */
export type SymbolKind = "function" | "keyword";

/** SQL dialects the resolver's signature admits. Only "sqlite" is wired here,
 *  because sqlite-wasm is the only engine that runs in the browser. */
export type SqlDialect =
  | "sqlite"
  | "postgres"
  | "mysql"
  | "bigquery"
  | "snowflake";

/** A resolved click: the token and, when known, the receiver it hangs off
 *  (for Python `df.groupby`, `qualifier` is `df`). */
export interface HelpRequest {
  name: string;
  qualifier?: string;
  kind: SymbolKind;
  language: HelpLanguage;
}

/** What the HELP tab shows: the symbol, its source, the canonical doc URL, an
 *  optional bundled blurb, and an optional honesty note (SQL dialects). */
export interface DocEntry {
  symbol: string;
  source: string;
  url: string;
  blurb?: string;
  note?: string;
}
```

- [ ] **Step 4: Create the resolver**

Create `web/lib/sandbox/help-docs/resolve.ts`:

```typescript
import type { DocEntry, HelpRequest, HelpLanguage, SqlDialect } from "./types";

interface Ref {
  source: string;
  url: string;
  blurb: string;
}

// --- R: dplyr, ggplot2, and base R ---------------------------------------
const DPLYR = "https://dplyr.tidyverse.org/reference/";
const GGPLOT = "https://ggplot2.tidyverse.org/reference/";
const BASE_R = "https://stat.ethz.ch/R-manual/R-devel/library/base/html/";

const R_REFS: Record<string, Ref> = {
  // dplyr
  summarise: { source: "dplyr", url: `${DPLYR}summarise.html`, blurb: "Summarise each group down to one row (dplyr)." },
  summarize: { source: "dplyr", url: `${DPLYR}summarise.html`, blurb: "Summarise each group down to one row (dplyr)." },
  mutate: { source: "dplyr", url: `${DPLYR}mutate.html`, blurb: "Create or change columns (dplyr)." },
  filter: { source: "dplyr", url: `${DPLYR}filter.html`, blurb: "Keep rows that match a condition (dplyr)." },
  select: { source: "dplyr", url: `${DPLYR}select.html`, blurb: "Keep or drop columns by name (dplyr)." },
  arrange: { source: "dplyr", url: `${DPLYR}arrange.html`, blurb: "Order rows by column values (dplyr)." },
  group_by: { source: "dplyr", url: `${DPLYR}group_by.html`, blurb: "Group rows for grouped operations (dplyr)." },
  count: { source: "dplyr", url: `${DPLYR}count.html`, blurb: "Count rows per group (dplyr)." },
  // ggplot2
  ggplot: { source: "ggplot2", url: `${GGPLOT}ggplot.html`, blurb: "Start a ggplot2 plot (ggplot2)." },
  aes: { source: "ggplot2", url: `${GGPLOT}aes.html`, blurb: "Map data to visual aesthetics (ggplot2)." },
  geom_point: { source: "ggplot2", url: `${GGPLOT}geom_point.html`, blurb: "Scatterplot layer (ggplot2)." },
  geom_line: { source: "ggplot2", url: `${GGPLOT}geom_path.html`, blurb: "Line layer, documented with geom_path (ggplot2)." },
  labs: { source: "ggplot2", url: `${GGPLOT}labs.html`, blurb: "Set titles and axis labels (ggplot2)." },
  theme_bw: { source: "ggplot2", url: `${GGPLOT}ggtheme.html`, blurb: "A complete black and white theme (ggplot2)." },
  // base R
  mean: { source: "base R", url: `${BASE_R}mean.html`, blurb: "Arithmetic mean of a vector (base R)." },
  sum: { source: "base R", url: `${BASE_R}sum.html`, blurb: "Sum of the values (base R)." },
  paste: { source: "base R", url: `${BASE_R}paste.html`, blurb: "Concatenate strings (base R)." },
  c: { source: "base R", url: `${BASE_R}c.html`, blurb: "Combine values into a vector (base R)." },
  seq: { source: "base R", url: `${BASE_R}seq.html`, blurb: "Generate regular sequences (base R)." },
};

// --- Python: pandas, NumPy, builtins -------------------------------------
const PANDAS_API = "https://pandas.pydata.org/docs/reference/api/";
const NUMPY_API = "https://numpy.org/doc/stable/reference/generated/";
const PY_BUILTINS = "https://docs.python.org/3/library/functions.html#";

const PANDAS_DATAFRAME_METHODS = new Set([
  "groupby", "merge", "join", "pivot_table", "head", "tail", "describe",
  "apply", "assign", "sort_values", "reset_index", "set_index", "drop",
  "fillna", "dropna", "agg", "rename",
]);
const PANDAS_TOPLEVEL: Record<string, string> = {
  read_csv: "pandas.read_csv.html",
  read_excel: "pandas.read_excel.html",
  concat: "pandas.concat.html",
  merge: "pandas.merge.html",
  DataFrame: "pandas.DataFrame.html",
  Series: "pandas.Series.html",
  to_datetime: "pandas.to_datetime.html",
};
const NUMPY_FUNCS: Record<string, string> = {
  array: "numpy.array.html",
  arange: "numpy.arange.html",
  linspace: "numpy.linspace.html",
  zeros: "numpy.zeros.html",
  ones: "numpy.ones.html",
  mean: "numpy.mean.html",
  where: "numpy.where.html",
};
const PY_BUILTIN_NAMES = new Set([
  "len", "print", "range", "sum", "min", "max", "enumerate", "zip", "map",
  "filter", "sorted", "open", "list", "dict", "set", "tuple", "int", "float",
  "str", "bool", "abs", "round", "type", "isinstance",
]);

// --- SQL (SQLite only) ----------------------------------------------------
const SQLITE_AGG = "https://www.sqlite.org/lang_aggfunc.html";
const SQLITE_DATE = "https://www.sqlite.org/lang_datefunc.html";
const SQLITE_SCALAR = "https://www.sqlite.org/lang_corefunc.html";
const SQLITE_SELECT = "https://www.sqlite.org/lang_select.html";
const SQLITE_WITH = "https://www.sqlite.org/lang_with.html";

const SQLITE_FUNCS: Record<string, Ref> = {
  COUNT: { source: "SQLite", url: SQLITE_AGG, blurb: "Count rows or non-null values (SQLite aggregate)." },
  AVG: { source: "SQLite", url: SQLITE_AGG, blurb: "Average of the values (SQLite aggregate)." },
  SUM: { source: "SQLite", url: SQLITE_AGG, blurb: "Sum of the values (SQLite aggregate)." },
  MIN: { source: "SQLite", url: SQLITE_AGG, blurb: "Minimum value (SQLite aggregate)." },
  MAX: { source: "SQLite", url: SQLITE_AGG, blurb: "Maximum value (SQLite aggregate)." },
  TOTAL: { source: "SQLite", url: SQLITE_AGG, blurb: "Sum that returns 0.0 over no rows (SQLite aggregate)." },
  GROUP_CONCAT: { source: "SQLite", url: SQLITE_AGG, blurb: "Concatenate values across a group (SQLite aggregate)." },
  DATE: { source: "SQLite", url: SQLITE_DATE, blurb: "Date value from a time string (SQLite date function)." },
  DATETIME: { source: "SQLite", url: SQLITE_DATE, blurb: "Date and time value (SQLite date function)." },
  STRFTIME: { source: "SQLite", url: SQLITE_DATE, blurb: "Format a date or time (SQLite date function)." },
  COALESCE: { source: "SQLite", url: SQLITE_SCALAR, blurb: "First non-null argument (SQLite core function)." },
  ROUND: { source: "SQLite", url: SQLITE_SCALAR, blurb: "Round a number (SQLite core function)." },
  LENGTH: { source: "SQLite", url: SQLITE_SCALAR, blurb: "Length of a string or blob (SQLite core function)." },
};
const SQLITE_KEYWORDS: Record<string, Ref> = {
  SELECT: { source: "SQLite", url: SQLITE_SELECT, blurb: "Query rows from tables (SQLite SELECT)." },
  FROM: { source: "SQLite", url: SQLITE_SELECT, blurb: "The tables a query reads from (SQLite SELECT)." },
  WHERE: { source: "SQLite", url: SQLITE_SELECT, blurb: "Filter rows in a query (SQLite SELECT)." },
  JOIN: { source: "SQLite", url: SQLITE_SELECT, blurb: "Combine rows from two tables (SQLite SELECT)." },
  "GROUP BY": { source: "SQLite", url: SQLITE_SELECT, blurb: "Group rows for aggregates (SQLite SELECT)." },
  HAVING: { source: "SQLite", url: SQLITE_SELECT, blurb: "Filter groups after aggregation (SQLite SELECT)." },
  "ORDER BY": { source: "SQLite", url: SQLITE_SELECT, blurb: "Sort the result rows (SQLite SELECT)." },
  LIMIT: { source: "SQLite", url: SQLITE_SELECT, blurb: "Cap the number of result rows (SQLite SELECT)." },
  WITH: { source: "SQLite", url: SQLITE_WITH, blurb: "Common table expressions (SQLite WITH)." },
};

function rEntry(name: string): DocEntry | null {
  const ref = R_REFS[name];
  return ref ? { symbol: name, source: ref.source, url: ref.url, blurb: ref.blurb } : null;
}

function pyEntry(req: HelpRequest): DocEntry | null {
  const { name } = req;
  if (PANDAS_DATAFRAME_METHODS.has(name)) {
    return {
      symbol: req.qualifier ? `${req.qualifier}.${name}` : name,
      source: "pandas",
      url: `${PANDAS_API}pandas.DataFrame.${name}.html`,
      blurb: `pandas.DataFrame.${name}: a DataFrame method (pandas).`,
    };
  }
  if (PANDAS_TOPLEVEL[name]) {
    return { symbol: name, source: "pandas", url: `${PANDAS_API}${PANDAS_TOPLEVEL[name]}`, blurb: `pandas.${name} (pandas).` };
  }
  if (NUMPY_FUNCS[name]) {
    return { symbol: name, source: "NumPy", url: `${NUMPY_API}${NUMPY_FUNCS[name]}`, blurb: `numpy.${name} (NumPy).` };
  }
  if (PY_BUILTIN_NAMES.has(name)) {
    return { symbol: name, source: "Python", url: `${PY_BUILTINS}${name}`, blurb: `${name}: a Python built-in function.` };
  }
  return null;
}

function sqlEntry(req: HelpRequest, dialect: SqlDialect): DocEntry | null {
  const key = req.name.toUpperCase();
  // DATE_TRUNC is a Postgres/BigQuery function with no SQLite equivalent.
  if (key === "DATE_TRUNC") {
    return {
      symbol: "DATE_TRUNC",
      source: "SQLite",
      url: SQLITE_DATE,
      blurb: "Truncate a timestamp to a unit.",
      note: "SQLite has no DATE_TRUNC. Use strftime() to truncate dates. DATE_TRUNC is a PostgreSQL and BigQuery function, which do not run here.",
    };
  }
  const ref = SQLITE_FUNCS[key] ?? SQLITE_KEYWORDS[key];
  if (!ref) return null;
  const entry: DocEntry = { symbol: key, source: ref.source, url: ref.url, blurb: ref.blurb };
  if (dialect !== "sqlite") {
    entry.note = `Coding Studio runs SQLite, so only SQLite runs here. The ${dialect} form of ${key} may differ; the link is the SQLite reference.`;
  }
  return entry;
}

/**
 * Resolves a clicked symbol to a documentation entry, or null when the symbol is
 * not in the curated tables (the HELP tab then shows a graceful fallback). The
 * curated map is intentionally small: it covers the requirement's examples and
 * the common tidyverse/pandas/SQLite names a student meets first. A full offline
 * doc corpus is out of scope.
 */
export function resolveDoc(
  req: HelpRequest,
  opts: { dialect?: SqlDialect } = {},
): DocEntry | null {
  if (req.language === "r") return rEntry(req.name);
  if (req.language === "python") return pyEntry(req);
  return sqlEntry(req, opts.dialect ?? "sqlite");
}

const HOME: Record<HelpLanguage, DocEntry> = {
  r: {
    symbol: "R",
    source: "R",
    url: "https://www.rdocumentation.org/",
    blurb: "No bundled link for this symbol. Search the R documentation.",
  },
  python: {
    symbol: "Python",
    source: "Python",
    url: "https://docs.python.org/3/",
    blurb: "No bundled link for this symbol. Search the Python documentation.",
  },
  sql: {
    symbol: "SQLite",
    source: "SQLite",
    url: "https://www.sqlite.org/docs.html",
    blurb: "No bundled link for this symbol. Browse the SQLite documentation.",
  },
};

/** A per-language reference home, shown when a clicked symbol is not in the
 *  curated map so the HELP tab still offers a useful link. */
export function referenceHome(language: HelpLanguage): DocEntry {
  return HOME[language];
}
```

- [ ] **Step 5: Run the resolver tests and confirm they pass**

Run: `npm run test -- help-docs`
Expected: PASS for every `resolveDoc` and `referenceHome` case (R summarise/
summarize/mean/ggplot/unknown, Python groupby/read_csv/len/unknown, SQL COUNT/avg/
JOIN/GROUP BY/WITH/DATE_TRUNC/dialect-note/unknown, and the three reference homes).

---

## Task 2: Token extraction under a click position (pure)

`symbolAt(text, pos, language)` turns a document offset into a `HelpRequest`.
Python and SQL reuse the Slice 2 Lezer parsers (proven to run in Vitest's `node`
environment). R uses a word scan over `maskStringsAndComments(text, "r")`, so a
click inside a string or comment yields nothing. SQL also uses the mask as a
guard so a click inside a string or comment yields nothing.

**Files:**
- Create: `web/lib/sandbox/help-docs/symbol-at.ts`,
  `web/lib/sandbox/help-docs/index.ts`
- Test: `web/tests/unit/help-docs.test.ts` (symbol-at section)

**Interfaces:**
- Consumes: `import { parser } from "@lezer/python"`;
  `import { SQLite } from "@codemirror/lang-sql"`;
  `import { maskStringsAndComments } from "@/lib/sandbox/lang-structure/mask"`.
- Produces: `symbolAt(text: string, pos: number, language: HelpLanguage): HelpRequest | null`;
  `index.ts` re-exports `symbolAt`, `resolveDoc`, `referenceHome`, and the types.

- [ ] **Step 1: Write the failing symbol-at tests**

Add to `web/tests/unit/help-docs.test.ts`:

```typescript
import { symbolAt } from "@/lib/sandbox/help-docs/symbol-at";

/** Resolve the symbol the finder returns for a cursor at `marker`. */
function at(src: string, marker: string, language: "r" | "python" | "sql") {
  const pos = src.indexOf(marker);
  return symbolAt(src, pos, language);
}

describe("symbolAt: R", () => {
  it("reads a bare function name", () => {
    expect(at("grades |> summarise(x = mean(g))", "summarise", "r")?.name).toBe(
      "summarise",
    );
  });

  it("reads the qualified name and package (dplyr::summarise)", () => {
    const s = at("dplyr::summarise(x)", "summarise", "r");
    expect(s?.name).toBe("summarise");
    expect(s?.qualifier).toBe("dplyr");
  });

  it("reads a dotted R name (theme_bw)", () => {
    expect(at("ggplot(d) + theme_bw()", "theme_bw", "r")?.name).toBe("theme_bw");
  });

  it("returns null inside a string", () => {
    expect(at('x <- "summarise here"', "summarise", "r")).toBeNull();
  });

  it("returns null inside a comment", () => {
    expect(at("x <- 1 # call summarise", "summarise", "r")).toBeNull();
  });
});

describe("symbolAt: Python", () => {
  it("reads a builtin call name", () => {
    expect(at("n = len(items)", "len", "python")?.name).toBe("len");
  });

  it("reads a method name and its receiver", () => {
    const s = at('out = df.groupby("k")', "groupby", "python");
    expect(s?.name).toBe("groupby");
    expect(s?.qualifier).toBe("df");
  });

  it("returns null inside a string", () => {
    expect(at('s = "please groupby"', "groupby", "python")).toBeNull();
  });
});

describe("symbolAt: SQL", () => {
  it("reads a function name", () => {
    expect(at("SELECT COUNT(*) FROM t;", "COUNT", "sql")?.name).toBe("COUNT");
  });

  it("combines GROUP BY into one symbol", () => {
    const s = at("SELECT k FROM t GROUP BY k;", "GROUP", "sql");
    expect(s?.name).toBe("GROUP BY");
    expect(s?.kind).toBe("keyword");
  });

  it("combines ORDER BY into one symbol", () => {
    expect(at("SELECT k FROM t ORDER BY k;", "ORDER", "sql")?.name).toBe(
      "ORDER BY",
    );
  });

  it("reads WITH", () => {
    expect(at("WITH a AS (SELECT 1) SELECT * FROM a;", "WITH", "sql")?.name).toBe(
      "WITH",
    );
  });

  it("returns null inside a string literal", () => {
    expect(at("SELECT 'COUNT me' AS s;", "COUNT", "sql")).toBeNull();
  });
});
```

- [ ] **Step 2: Run the symbol-at tests and confirm they fail**

Run: `npm run test -- help-docs`
Expected: FAIL (`symbol-at.ts` does not exist).

- [ ] **Step 3: Implement `symbol-at.ts`**

Create `web/lib/sandbox/help-docs/symbol-at.ts`:

```typescript
import { parser as pyParser } from "@lezer/python";
import { SQLite } from "@codemirror/lang-sql";
import type { SyntaxNode } from "@lezer/common";
import { maskStringsAndComments } from "@/lib/sandbox/lang-structure/mask";
import type { HelpLanguage, HelpRequest } from "./types";

const sqlParser = SQLite.language.parser;

/**
 * The documentation request for the token at `pos`, or null when there is no
 * resolvable identifier there (whitespace, punctuation, or a click inside a
 * string or comment). Python and SQL use their Lezer grammars; R uses a word
 * scan over the string/comment mask.
 */
export function symbolAt(
  text: string,
  pos: number,
  language: HelpLanguage,
): HelpRequest | null {
  if (language === "python") return pythonSymbol(text, pos);
  if (language === "sql") return sqlSymbol(text, pos);
  return rSymbol(text, pos);
}

// --- R -------------------------------------------------------------------
// R identifiers: a letter or dot, then letters, digits, dots, underscores.
const R_WORD = /[A-Za-z0-9._]/;

function rSymbol(text: string, pos: number): HelpRequest | null {
  const mask = maskStringsAndComments(text, "r");
  // A click inside a string ('x' placeholder) or comment (space) is not a symbol.
  if (pos < 0 || pos >= mask.length) return null;
  if (mask[pos] !== text[pos]) return null; // masked away (string/comment)
  if (!R_WORD.test(text[pos])) return null;
  let from = pos;
  while (from > 0 && R_WORD.test(text[from - 1]) && mask[from - 1] === text[from - 1]) from--;
  let to = pos;
  while (to < text.length && R_WORD.test(text[to]) && mask[to] === text[to]) to++;
  const name = text.slice(from, to);
  if (!name) return null;
  // Qualified call: pkg::name or pkg:::name just before the word.
  let qualifier: string | undefined;
  let q = from;
  if (text.slice(Math.max(0, from - 3), from).endsWith("::")) {
    let colon = from;
    while (colon > 0 && text[colon - 1] === ":") colon--;
    let pk = colon;
    while (pk > 0 && R_WORD.test(text[pk - 1])) pk--;
    if (pk < colon) qualifier = text.slice(pk, colon);
    q = pk;
  }
  void q;
  return { name, qualifier, kind: "function", language: "r" };
}

// --- Python --------------------------------------------------------------
function pythonSymbol(text: string, pos: number): HelpRequest | null {
  const tree = pyParser.parse(text);
  let node: SyntaxNode | null = tree.resolveInner(pos, -1);
  if (node && node.name !== "VariableName" && node.name !== "PropertyName") {
    node = tree.resolveInner(pos, 1);
  }
  if (!node || (node.name !== "VariableName" && node.name !== "PropertyName")) {
    return null;
  }
  const name = text.slice(node.from, node.to);
  // A PropertyName (`df.groupby`) hangs off a MemberExpression; the receiver is
  // its previous sibling. Read the receiver's trailing identifier as qualifier.
  let qualifier: string | undefined;
  if (node.name === "PropertyName") {
    const object = node.parent?.firstChild ?? null;
    if (object && object.to <= node.from) {
      const objText = text.slice(object.from, object.to);
      const m = objText.match(/([A-Za-z_][A-Za-z0-9_]*)\s*$/);
      qualifier = m ? m[1] : objText;
    }
  }
  return { name, qualifier, kind: "function", language: "python" };
}

// --- SQL (SQLite) --------------------------------------------------------
const SQL_WORD = /[A-Za-z0-9_]/;

function sqlSymbol(text: string, pos: number): HelpRequest | null {
  // Guard: a click inside a string or comment is not a symbol.
  const mask = maskStringsAndComments(text, "sql");
  if (pos < 0 || pos >= text.length) return null;
  if (mask[pos] !== text[pos]) return null;
  if (!SQL_WORD.test(text[pos])) return null;

  // Token boundaries. The Lezer node gives the token; fall back to a word scan.
  const tree = sqlParser.parse(text);
  const node = tree.resolveInner(pos, -1);
  let from = node.from;
  let to = node.to;
  if (!/^[A-Za-z0-9_]+$/.test(text.slice(from, to))) {
    from = pos;
    while (from > 0 && SQL_WORD.test(text[from - 1])) from--;
    to = pos;
    while (to < text.length && SQL_WORD.test(text[to])) to++;
  }
  let name = text.slice(from, to);
  if (!name) return null;
  const kind: HelpRequest["kind"] = node.name === "Keyword" ? "keyword" : "function";

  // Combine two-word clauses: GROUP BY, ORDER BY.
  const upper = name.toUpperCase();
  if (upper === "GROUP" || upper === "ORDER") {
    const after = text.slice(to).match(/^\s+([A-Za-z]+)/);
    if (after && after[1].toUpperCase() === "BY") {
      name = `${upper} BY`;
      return { name, kind: "keyword", language: "sql" };
    }
  }
  return { name, kind, language: "sql" };
}
```

Note: node names come from the installed `@lezer/python` and
`@codemirror/lang-sql` grammars. The Python identifier node types
(`VariableName`, `PropertyName`) match the Slice 2 usage in `py.ts`. For SQL,
reading `node.from`/`node.to` and the raw text is robust regardless of the exact
token-name spelling; the word-scan fallback covers any grammar that classifies a
function name as something other than a single word node. Verify the `Keyword`
node name against the installed grammar during implementation; if it differs, the
`kind` hint is only a hint (Task 1's `resolveDoc` tries both the function and
keyword tables), so a wrong hint does not change the resolved URL.

- [ ] **Step 4: Create the barrel `index.ts`**

Create `web/lib/sandbox/help-docs/index.ts`:

```typescript
export type {
  DocEntry,
  HelpLanguage,
  HelpRequest,
  SqlDialect,
  SymbolKind,
} from "./types";
export { resolveDoc, referenceHome } from "./resolve";
export { symbolAt } from "./symbol-at";
```

- [ ] **Step 5: Run the symbol-at tests and confirm they pass**

Run: `npm run test -- help-docs`
Expected: PASS for every `symbolAt` case (R bare/qualified/dotted/in-string/
in-comment, Python builtin/method-with-receiver/in-string, SQL function/GROUP BY/
ORDER BY/WITH/in-string).

Run: `npm run typecheck`
Expected: no errors. If the `@lezer/common` type import does not resolve, mirror
`py.ts`'s approach (it imports `SyntaxNode` from `@lezer/common` and compiles in
this repo).

---

## Task 3: Editor plumbing: Ctrl/Cmd+Click and F1 emit a HelpRequest

The editor adds one modified-mousedown `domEventHandlers` extension and one F1
keybinding. Both resolve the symbol and hand a `HelpRequest` up through a new
`onHelp` prop (read from a ref, like the run handlers). The mousedown calls
`preventDefault()` and never focuses or scrolls, so the caret and scroll position
are preserved. `symbolAt` rides the same lazy CodeMirror chunk so its parsers stay
out of the initial bundle.

**Files:**
- Modify: `web/components/run/CodeEditor.tsx`

**Interfaces:**
- Consumes: `symbolAt` from `@/lib/sandbox/help-docs`.
- Produces: `CodeEditor` gains an optional prop
  `onHelp?: (req: HelpRequest) => void`. New internal helpers
  `helpKeymap(...)` and `helpMouse(...)`.

- [ ] **Step 1: Add the `onHelp` prop and its ref**

In `web/components/run/CodeEditor.tsx`, import the type near the other type
imports (top of file):

```typescript
import type { HelpRequest } from "@/lib/sandbox/help-docs";
```

Add the prop to the `CodeEditor` props object (next to `onSource`, around
line 44), with a no-em-dash comment:

```typescript
  /** Resolve the symbol under a Ctrl/Cmd+Click (or F1 at the cursor) and open it
   *  in the HELP tab. Does not move the caret. */
  onHelp?: (req: HelpRequest) => void;
```

Add a ref and a `hasHelp` flag alongside the existing run wiring (near
`hasRun`, around line 65):

```typescript
  const helpRef = useRef(props.onHelp);
  const hasHelp = props.onHelp != null;
```

Keep it fresh with an effect (near the `runRef` effect, around line 79):

```typescript
  useEffect(() => {
    helpRef.current = props.onHelp;
  }, [props.onHelp]);
```

- [ ] **Step 2: Lazy-load `symbolAt` with the editor**

Extend the `LoadedEditor` interface (around line 184) with:

```typescript
  helpDocs: typeof import("@/lib/sandbox/help-docs");
```

In `loadCodeMirror` (around line 197) add the import to the `Promise.all` and the
returned object:

```typescript
  const [cm, view, lang, state, autocomplete, highlight, inline, langStructure, helpDocs, langExt] =
    await Promise.all([
      import("codemirror"),
      import("@codemirror/view"),
      import("@codemirror/language"),
      import("@codemirror/state"),
      import("@codemirror/autocomplete"),
      import("@lezer/highlight"),
      import("@/lib/sandbox/inline-completion"),
      import("@/lib/sandbox/lang-structure"),
      import("@/lib/sandbox/help-docs"),
      loadLanguageMode(languageId),
    ]);
  return {
    view,
    cm,
    lang,
    state,
    autocomplete,
    tags: highlight.tags,
    inline,
    langStructure,
    helpDocs,
    langExt,
  };
```

- [ ] **Step 3: Add the mouse and keyboard extensions to the editor build**

Update the destructure in the mount effect (around line 90) to pull `helpDocs`:

```typescript
      .then(({ view, cm, lang, state, autocomplete, tags, langExt, inline, langStructure, helpDocs }) => {
```

In the extensions array (after the `hasRun` block, around line 109) add:

```typescript
            ...(hasHelp
              ? [
                  helpMouse(view, helpRef, helpDocs.symbolAt, props.languageId),
                  helpKeymap(view, state, helpRef, helpDocs.symbolAt, props.languageId),
                ]
              : []),
```

Add `hasHelp` to the effect's dependency array (around line 139), so toggling the
handler rebuilds the editor exactly like `hasRun`:

```typescript
  }, [props.languageId, dark, fillHeight, hasCompletion, hasComplete, hasRun, hasHelp]);
```

- [ ] **Step 4: Implement `helpMouse` and `helpKeymap`**

Add these functions near `runKeymap` (after it, around line 346):

```typescript
type HelpLang = "r" | "python" | "sql";
type SymbolAt = LoadedEditor["helpDocs"]["symbolAt"];

function toHelpLang(languageId: string): HelpLang {
  return languageId === "python" ? "python" : languageId === "sql" ? "sql" : "r";
}

/**
 * Ctrl+Click (Windows/Linux) or Cmd+Click (macOS) on a symbol opens its docs in
 * the HELP tab. `preventDefault()` stops CodeMirror moving the caret, and we
 * never focus or scroll, so the script position and cursor are preserved.
 */
function helpMouse(
  view: LoadedEditor["view"],
  helpRef: { current: ((req: HelpRequest) => void) | undefined },
  symbolAt: SymbolAt,
  languageId: string,
): Extension {
  const lang = toHelpLang(languageId);
  return view.EditorView.domEventHandlers({
    mousedown(event, editor) {
      if (!(event.metaKey || event.ctrlKey)) return false;
      if (event.button !== 0) return false;
      const pos = editor.posAtCoords({ x: event.clientX, y: event.clientY });
      if (pos == null) return false;
      const req = symbolAt(editor.state.doc.toString(), pos, lang);
      // Prevent the caret move and text selection whether or not we found a
      // symbol, so a modified click never disturbs the cursor.
      event.preventDefault();
      if (req) helpRef.current?.(req);
      return true;
    },
  });
}

/**
 * F1 opens docs for the symbol at the cursor, the keyboard equivalent of a
 * modified click. It reads the caret position and does not move it.
 */
function helpKeymap(
  view: LoadedEditor["view"],
  cmState: LoadedEditor["state"],
  helpRef: { current: ((req: HelpRequest) => void) | undefined },
  symbolAt: SymbolAt,
  languageId: string,
): Extension {
  const lang = toHelpLang(languageId);
  return cmState.Prec.high(
    view.keymap.of([
      {
        key: "F1",
        preventDefault: true,
        run: (editor) => {
          const handler = helpRef.current;
          if (!handler) return false;
          const pos = editor.state.selection.main.head;
          const req = symbolAt(editor.state.doc.toString(), pos, lang);
          if (req) handler(req);
          return true;
        },
      },
    ]),
  );
}
```

- [ ] **Step 5: Verify types and lint**

Run: `npm run typecheck`
Expected: no errors.

Run: `npm run lint`
Expected: no errors for `CodeEditor.tsx`.

(The editor cannot be exercised end to end until Task 4 wires `onHelp` in the
Sandbox; the behavior is proven by the e2e in Task 5.)

---

## Task 4: The HELP tab beside PLOTS, and the Sandbox wiring

The bottom-right pane becomes a two-tab panel, Plots and Help, so there is exactly
one HELP tab next to PLOTS. `Workspace` holds `helpTarget` (the resolved
`DocEntry` plus the raw request, or null) and `rightLowerTab` (`"plots" | "help"`).
`CodeEditor`'s `onHelp` calls `resolveDoc`, stores the result, and selects the
Help tab, without touching the editor.

**Files:**
- Modify: `web/components/sandbox/Sandbox.tsx`

**Interfaces:**
- Consumes: `resolveDoc`, `referenceHome`, and the `DocEntry` / `HelpRequest`
  types from `@/lib/sandbox/help-docs`; the existing `PlotsPane` props.
- Produces: `PlotsHelpPane` (replaces the direct `PlotsPane` mount in the
  bottom-right `Panel`); `HelpBody`; `helpTarget` / `rightLowerTab` state.

- [ ] **Step 1: Import the resolver and types**

At the top of `web/components/sandbox/Sandbox.tsx`, near the other `lib/sandbox`
imports (around line 29), add:

```typescript
import {
  resolveDoc,
  referenceHome,
  type DocEntry,
  type HelpRequest,
} from "@/lib/sandbox/help-docs";
```

- [ ] **Step 2: Add HELP state and the onHelp handler in `Workspace`**

In `Workspace`, next to the plots state (around line 302), add:

```typescript
  // The single HELP tab's current target, and which of Plots/Help is showing in
  // the bottom-right pane. `helpTarget` holds the resolved doc entry (or the
  // reference-home fallback) plus the symbol; one slot, so there is one HELP tab.
  const [helpTarget, setHelpTarget] = useState<{
    symbol: string;
    entry: DocEntry;
  } | null>(null);
  const [rightLowerTab, setRightLowerTab] = useState<"plots" | "help">("plots");
```

Add the handler near the other `useCallback`s (for example after `clearConsole`,
around line 475). It resolves the request to a doc entry, falling back to the
per-language reference home when the symbol is not in the curated map, then
selects the Help tab. It never touches the editor, so the caret is preserved:

```typescript
  // Ctrl/Cmd+Click or F1 in the editor lands here. Resolve the clicked symbol to
  // a documentation entry (or the language's reference home if it is not in the
  // curated map), show it in the one HELP tab, and select that tab. SQL passes no
  // dialect override: the only engine that runs here is SQLite.
  const onHelp = useCallback((req: HelpRequest) => {
    const entry = resolveDoc(req) ?? referenceHome(req.language);
    setHelpTarget({ symbol: req.qualifier ? `${req.qualifier}.${req.name}` : req.name, entry });
    setRightLowerTab("help");
  }, []);
```

- [ ] **Step 3: Pass `onHelp` to the editor**

On the `CodeEditor` element (around line 656) add the prop after `onSource`:

```tsx
                      onSource={source}
                      onHelp={onHelp}
```

- [ ] **Step 4: Replace the bottom-right `PlotsPane` with `PlotsHelpPane`**

In the right column, change the `plots` panel body (around line 711) from:

```tsx
              <Panel id="plots" defaultSize="50" minSize="12" className="min-h-0">
                <PlotsPane
                  plots={plots}
                  index={plotIndex}
                  onIndex={setPlotIndex}
                  onDelete={deletePlot}
                  onClear={clearPlots}
                />
              </Panel>
```

to:

```tsx
              <Panel id="plots" defaultSize="50" minSize="12" className="min-h-0">
                <PlotsHelpPane
                  tab={rightLowerTab}
                  onTab={setRightLowerTab}
                  help={helpTarget}
                  plots={plots}
                  index={plotIndex}
                  onIndex={setPlotIndex}
                  onDelete={deletePlot}
                  onClear={clearPlots}
                />
              </Panel>
```

- [ ] **Step 5: Implement `PlotsHelpPane` and `HelpBody`**

Replace the existing `PlotsPane` function (Sandbox.tsx:1255-1340) with the
tabbed panel below. It keeps the plot navigation and image exactly as before, but
puts them under a Plots tab, and adds a Help tab beside it. The two tabs share one
header row (the tab strip on the left, the Plots actions on the right when Plots
is active), so there is one HELP tab, positioned next to PLOTS.

```tsx
function PlotsHelpPane({
  tab,
  onTab,
  help,
  plots,
  index,
  onIndex,
  onDelete,
  onClear,
}: {
  tab: "plots" | "help";
  onTab: (tab: "plots" | "help") => void;
  help: { symbol: string; entry: DocEntry } | null;
  plots: string[];
  index: number;
  onIndex: (i: number) => void;
  onDelete: (index: number) => void;
  onClear: () => void;
}) {
  const has = plots.length > 0;
  const safeIndex = Math.min(index, plots.length - 1);
  const btn =
    "rounded border border-[var(--sb-border)] px-1.5 py-0.5 font-bold hover:border-[var(--sb-accent)] disabled:opacity-40";
  const tabClass = (active: boolean) =>
    `px-3 py-1.5 text-xs font-bold uppercase tracking-wide ${active ? "text-[var(--sb-text)]" : "text-[var(--sb-muted)] hover:text-[var(--sb-text)]"}`;

  function exportCurrent() {
    const a = document.createElement("a");
    a.href = plots[safeIndex];
    a.download = `plot-${safeIndex + 1}.png`;
    a.click();
  }

  return (
    <section className="flex h-full flex-col overflow-hidden rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)]">
      <header className="flex items-center justify-between border-b border-[var(--sb-border)] bg-[var(--sb-header)]">
        <div role="tablist" aria-label="Plots and Help" className="flex items-stretch">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "plots"}
            onClick={() => onTab("plots")}
            className={tabClass(tab === "plots")}
          >
            Plots
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "help"}
            onClick={() => onTab("help")}
            className={tabClass(tab === "help")}
          >
            Help
          </button>
        </div>
        {tab === "plots" && has ? (
          <span className="flex items-center gap-1.5 px-2 text-xs text-[var(--sb-muted)]">
            <button
              type="button"
              onClick={() => onIndex(Math.max(0, safeIndex - 1))}
              disabled={safeIndex <= 0}
              className={btn}
              aria-label="Previous plot"
            >
              &#8249;
            </button>
            <span className="tabular-nums">
              {safeIndex + 1} / {plots.length}
            </span>
            <button
              type="button"
              onClick={() => onIndex(Math.min(plots.length - 1, safeIndex + 1))}
              disabled={safeIndex >= plots.length - 1}
              className={btn}
              aria-label="Next plot"
            >
              &#8250;
            </button>
            <button type="button" onClick={exportCurrent} className={btn}>
              Export
            </button>
            <button type="button" onClick={() => onDelete(safeIndex)} className={btn}>
              Delete
            </button>
            <button type="button" onClick={onClear} className={btn}>
              Clear all
            </button>
          </span>
        ) : null}
      </header>

      <div
        tabIndex={0}
        aria-label={tab === "plots" ? "Plots" : "Help"}
        role="tabpanel"
        className="min-h-0 flex-1 overflow-auto"
      >
        {tab === "plots" ? (
          <div className="flex h-full items-center justify-center p-2">
            {has ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={plots[safeIndex]}
                alt={`Plot ${safeIndex + 1} of ${plots.length}`}
                className="max-h-full max-w-full rounded bg-white"
              />
            ) : (
              <p className="text-sm text-[var(--sb-muted)]">
                Charts you draw appear here.
              </p>
            )}
          </div>
        ) : (
          <HelpBody help={help} />
        )}
      </div>
    </section>
  );
}

/**
 * The HELP tab body. It cannot embed or fetch the external doc site: the Coding
 * Studio page is cross-origin isolated (COEP require-corp), which blocks an
 * iframe to, and a fetch of, those cross-origin pages. So it shows the resolved
 * symbol, its source, and a bundled blurb, plus a link that opens the official
 * page in a new browser tab, which is not subject to the embedder policy.
 */
function HelpBody({ help }: { help: { symbol: string; entry: DocEntry } | null }) {
  if (!help) {
    return (
      <div className="p-3 text-sm text-[var(--sb-muted)]">
        <p>
          Ctrl or Cmd click a function or keyword in your script, or put the
          cursor on it and press F1, to see its documentation here.
        </p>
      </div>
    );
  }
  const { symbol, entry } = help;
  return (
    <div className="flex flex-col gap-2 p-3 text-sm">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="font-mono text-base font-bold text-[var(--sb-text)]">
          {symbol}
        </span>
        <span className="rounded border border-[var(--sb-border)] px-1.5 py-0.5 text-xs font-bold uppercase tracking-wide text-[var(--sb-muted)]">
          {entry.source}
        </span>
      </div>
      {entry.blurb ? (
        <p className="text-[var(--sb-text)]">{entry.blurb}</p>
      ) : null}
      {entry.note ? (
        <p className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-header)] px-2 py-1.5 text-xs text-[var(--sb-muted)]">
          {entry.note}
        </p>
      ) : null}
      <a
        href={entry.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex w-fit items-center gap-1 rounded-card bg-[var(--sb-accent)] px-3 py-1 text-sm font-bold text-white hover:opacity-90"
      >
        Open documentation
      </a>
      <p className="text-xs text-[var(--sb-muted)]">
        Opens the official {entry.source} page in a new tab. Coding Studio bundles
        a short description and this link, not the full documentation.
      </p>
    </div>
  );
}
```

- [ ] **Step 6: Verify types and lint**

Run: `npm run typecheck`
Expected: no errors.

Run: `npm run lint`
Expected: no errors for `Sandbox.tsx`. (The old `PlotsPane` is fully replaced by
`PlotsHelpPane`; confirm no remaining reference to `PlotsPane` exists.)

Run: `npm run test -- help-docs`
Expected: the full unit suite still passes.

---

## Task 5: End-to-end proof (no runtime execution needed)

Documentation resolution is pure client logic: no WASM runtime runs, so these
tests are fast and deterministic. They use the Python editor (the default
language), replace the starter with code that has a known symbol, and drive a
modified click and F1.

**Files:**
- Modify: `web/tests/e2e/sandbox.spec.ts`

- [ ] **Step 1: Add the Ctrl/Cmd+Click HELP test**

Add to the `test.describe("AI Sandbox", ...)` block:

```typescript
  test("Ctrl/Cmd+Click opens a HELP tab beside PLOTS with the symbol and a new-tab link", async ({
    page,
  }) => {
    await page.goto("/ai-sandbox");
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(editor).toBeVisible();
    // Wait for CodeMirror itself, not the load-time textarea fallback: the
    // mousedown/F1 handlers only live in the CodeMirror editor.
    await expect(page.locator(".cm-content")).toBeVisible();

    // The Help tab sits next to Plots from the start (one instance, reused).
    await expect(page.getByRole("tab", { name: "Plots" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Help" })).toHaveCount(1);

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      'import pandas as pd\ndf = pd.DataFrame()\nout = df.groupby("k")\nn = len(out)\n',
    );
    await expect(editor).toContainText("groupby");

    // Record the caret box so we can prove a modified click does not move it.
    const caretBefore = await page.evaluate(() => {
      const c = document.querySelector(".cm-cursor-primary");
      const r = c?.getBoundingClientRect();
      return r ? { x: Math.round(r.left), y: Math.round(r.top) } : null;
    });

    // Modified click on the method name. "ControlOrMeta" maps to Cmd on macOS,
    // Ctrl elsewhere, matching the requirement.
    await page.getByText("groupby").click({ modifiers: ["ControlOrMeta"] });

    // The Help tab is now selected and shows the symbol, the source, and the link.
    await expect(page.getByRole("tab", { name: "Help" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    const help = page.getByRole("tabpanel", { name: "Help" });
    await expect(help.getByText("groupby")).toBeVisible();
    await expect(help.getByText("pandas", { exact: false })).toBeVisible();
    const link = help.getByRole("link", { name: "Open documentation" });
    await expect(link).toHaveAttribute("target", "_blank");
    await expect(link).toHaveAttribute(
      "href",
      "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html",
    );

    // The caret did not move (script position and cursor preserved).
    const caretAfter = await page.evaluate(() => {
      const c = document.querySelector(".cm-cursor-primary");
      const r = c?.getBoundingClientRect();
      return r ? { x: Math.round(r.left), y: Math.round(r.top) } : null;
    });
    expect(caretAfter).toEqual(caretBefore);

    // No WCAG A/AA violations with the Help tab open.
    const axe = await new AxeBuilder({ page })
      .exclude(".cm-scroller")
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);
  });
```

- [ ] **Step 2: Run the click test and confirm it passes**

Run: `npm run test:e2e -- sandbox.spec.ts -g "opens a HELP tab beside PLOTS"`
Expected: PASS. The Help tab is selected, shows `groupby` + `pandas` + the
pandas URL on a `target="_blank"` link, the caret box is unchanged, and axe is
clean.

- [ ] **Step 3: Add the reuse and keyboard tests**

Add to the same block:

```typescript
  test("a second Ctrl/Cmd+Click reuses the one HELP tab", async ({ page }) => {
    await page.goto("/ai-sandbox");
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      'import pandas as pd\ndf = pd.DataFrame()\nout = df.groupby("k")\nn = len(out)\n',
    );

    await page.getByText("groupby").click({ modifiers: ["ControlOrMeta"] });
    const help = page.getByRole("tabpanel", { name: "Help" });
    await expect(help.getByText("groupby")).toBeVisible();

    // Click a different symbol: still one Help tab, contents replaced.
    await page.getByText("len").click({ modifiers: ["ControlOrMeta"] });
    await expect(page.getByRole("tab", { name: "Help" })).toHaveCount(1);
    await expect(help.getByText("len")).toBeVisible();
    await expect(
      help.getByRole("link", { name: "Open documentation" }),
    ).toHaveAttribute(
      "href",
      "https://docs.python.org/3/library/functions.html#len",
    );
    // The previous symbol is gone (the tab was reused, not duplicated).
    await expect(help.getByText("groupby")).toHaveCount(0);
  });

  test("F1 opens docs for the symbol at the cursor (keyboard path)", async ({
    page,
  }) => {
    await page.goto("/ai-sandbox");
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("n = len(items)\n");

    // Put the cursor inside "len" with the keyboard, then press F1.
    await page.getByText("len").click();
    await page.keyboard.press("F1");

    const help = page.getByRole("tabpanel", { name: "Help" });
    await expect(page.getByRole("tab", { name: "Help" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await expect(help.getByText("len")).toBeVisible();
    await expect(help.getByText("Python", { exact: false })).toBeVisible();
  });
```

- [ ] **Step 4: Run the reuse and keyboard tests and confirm they pass**

Run: `npm run test:e2e -- sandbox.spec.ts -g "reuses the one HELP tab"`
Expected: PASS. One Help tab throughout; the second click replaces `groupby`
with `len` and the pandas URL with the Python builtins URL.

Run: `npm run test:e2e -- sandbox.spec.ts -g "F1 opens docs"`
Expected: PASS. F1 selects the Help tab and shows `len` from Python.

- [ ] **Step 5: Verify the whole slice (no commit)**

Run: `npm run test -- help-docs`
Expected: the full `help-docs` unit suite passes (resolve + symbol-at).

Run: `npm run typecheck`
Expected: no errors.

Run: `npm run lint`
Expected: no errors for the changed files.

Run: `npm run test:e2e -- sandbox.spec.ts`
Expected: the full sandbox suite passes, including the three new HELP tests and
the existing shell/scroll/clear/statement tests. Leave the working tree
uncommitted.

---

## Self-Review

**Spec coverage (requirement B and the general acceptance criteria):**

- Ctrl+Click (Win/Linux) or Cmd+Click (macOS) on a function or recognized element
  opens its docs: Task 3's `helpMouse` fires on `event.metaKey || event.ctrlKey`;
  the e2e uses Playwright's `ControlOrMeta` modifier, which is Cmd on macOS and
  Ctrl elsewhere.
- Docs correspond to the specific clicked function/keyword, source matches the
  active language: Task 2 `symbolAt` reads the exact token (Lezer node for
  Python/SQL, word scan for R); Task 1 `resolveDoc` maps it to the correct source
  and URL, routed by `req.language`.
- HELP tab positioned next to PLOTS, reuse ONE tab: Task 4 makes the bottom-right
  pane a two-tab Plots | Help panel; `helpTarget` is a single state slot, so there
  is exactly one HELP tab (the e2e asserts `toHaveCount(1)` before and after a
  second click).
- Preserve the script position and cursor when HELP opens: `helpMouse` calls
  `preventDefault()` and never focuses or scrolls; F1's binding reads the caret
  and does not move it; selecting the tab is a right-column state change. The e2e
  asserts the caret box is byte-for-byte unchanged across a modified click.
- R: `summarise` -> dplyr, `mean` -> R docs: Task 1 R table and its tests.
- Python: `DataFrame.groupby` -> pandas, `len` -> Python: Task 1 Python resolver
  (DataFrame method table, builtins set) and tests; `symbolAt` captures the
  receiver as `qualifier`.
- SQL: `COUNT`/`AVG`/`DATE_TRUNC` -> function docs, `JOIN`/`GROUP BY`/`WITH` ->
  syntax docs: Task 1 SQL function and keyword tables; `symbolAt` combines
  `GROUP BY`/`ORDER BY`; `DATE_TRUNC` gets an honest note.
- SQL dialect-aware where possible, honestly SQLite-only: `resolveDoc` takes a
  `dialect` param but ships wired to SQLite; a non-SQLite request returns SQLite
  docs plus a note that only SQLite runs here. UI copy in `HelpBody` says the link
  opens the official SQLite page. Tested by the `dialect: "postgres"` case.
- Keyboard and mouse (WCAG 2.1 AA): mouse is the modified click; keyboard is F1 on
  the symbol at the cursor. Tabs use `role="tab"`/`tablist`/`aria-selected`; the
  doc link is a real anchor. The e2e runs axe with the Help tab open.
- No em dashes: all `HelpBody` copy, blurbs, notes, and the empty state use commas
  and sentence breaks.

**The COEP constraint, resolved.** The plan never embeds or fetches an external
doc page (both are blocked under COEP `require-corp`). The HELP tab renders the
resolved symbol, source, and a bundled blurb locally, and links out with
`target="_blank" rel="noopener noreferrer"`, which opens a separate top-level
document not governed by the page's embedder policy. `HelpBody`'s trailing note
states plainly that Coding Studio bundles a short description and a link, not the
full documentation, so the offline-corpus limitation is honest and visible.

**Symbol resolution per language, justified.**

- R has no syntax tree (legacy `StreamLanguage`), so a word scan over
  `maskStringsAndComments(text, "r")` is the right tool: it reads identifier
  characters directly and, crucially, returns null for a click inside a string or
  comment (the mask neutralizes those), and it detects a `pkg::name` qualifier by
  scanning left past `::`.
- Python and SQL have real Lezer grammars already used by Slice 2, so `symbolAt`
  reuses the same parsers in the same pure `node`-testable way. Python resolves a
  `PropertyName` and reads its receiver (`df.groupby` -> name `groupby`, qualifier
  `df`); a `VariableName` callee gives a bare builtin (`len`). SQL reads the token
  under the cursor and combines the two-word clauses, with a mask guard so a click
  inside a string is not a symbol.
- `resolveDoc`'s `kind` is only a hint; it tries both the function and keyword
  tables, so an imperfect Lezer token classification cannot produce a wrong URL.

**Scope honesty (stated in the plan and the UI).**

- SQL is SQLite-only at runtime. Every SQL URL is a SQLite page; the `dialect`
  parameter is future-facing and, when set to a non-SQLite value, still returns
  SQLite docs plus an explicit note. `DATE_TRUNC` resolves to the SQLite date
  functions page with a note that SQLite has no `DATE_TRUNC` and points at
  `strftime()`.
- Offline docs are out of scope. The MVP is a curated symbol-to-URL map plus a
  one-line bundled blurb and an open-in-new-tab link. Unknown symbols fall back to
  the language's reference home via `referenceHome`, so the tab is always useful.

**Placeholder scan:** No TBDs. Every code step shows final code; commands are
exact and adapted to the no-commit rule (verification steps replace commit steps).
The one grammar-name contingency (SQL `Keyword` node name) is hedged: reading
`node.from`/`node.to` plus the word-scan fallback works regardless, and the `kind`
hint is non-authoritative.

**Type consistency:** `HelpRequest { name; qualifier?; kind; language }` and
`DocEntry { symbol; source; url; blurb?; note? }` are defined in Task 1 and used
unchanged in Tasks 2, 3, and 4. `symbolAt(text, pos, language): HelpRequest | null`
(Task 2) is what `helpMouse`/`helpKeymap` call (Task 3) and what feeds `onHelp`
-> `resolveDoc` (Task 4). The editor accessible names in the e2e (`/Python code/i`,
`.cm-content`, the `Plots`/`Help` tabs, the `Help` tabpanel) match the code in
Task 4 and the existing `label={`${language.label} code`}` (Sandbox.tsx:660).

## Handoff note

Per the task that produced this plan, this is a planning document only; do not
build from it in this session. When execution is scheduled, use
superpowers:subagent-driven-development (a fresh subagent per task, review between)
or superpowers:executing-plans (batched with checkpoints), and remember: no git
commits, verify with the commands above instead. Build order within the slice:
Task 1 (resolver) and Task 2 (symbol-at) are independent and can be done in either
order; then Task 3 (editor plumbing), then Task 4 (HELP tab UI and wiring), then
Task 5 (e2e). Slice 4 is independent of Slices 1 to 3 and of Slice 5, so it can be
scheduled in parallel with them if staffed separately.
