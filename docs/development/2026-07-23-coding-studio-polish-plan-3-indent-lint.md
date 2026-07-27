# Slice 3 — E: continuation indentation + language-aware linting — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is a planning document only; do not build from it in the session that produced it.

**Goal:** On Enter while a statement is clearly continuing, auto-indent the new
line to the right level, per language: R (indent after an open bracket, continue
after `|>`/`%>%` and after a ggplot `+`, align args in multi-line calls, dedent
before a closing bracket, none when the statement is complete); Python (indent
after `:`, preserve block indent, dedent on block end, indent inside brackets,
consistent 4 spaces, never tabs); SQL (indent items under SELECT, conditions
under WHERE/HAVING/ON, CTE and subquery contents, dedent major clauses). Plus an
unobtrusive per-language linter that underlines obvious problems on a debounce,
never rewriting code, never moving the cursor, never touching strings or
comments.

**Architecture.** This slice reuses Slice 2's pure `web/lib/sandbox/lang-structure/`
module wholesale and never re-derives structure:

- **R indentation and R linting read Slice 2's `scanR` and `maskStringsAndComments`
  directly.** The new `rIndentColumns(text, pos)` calls `scanR(text)` to get the
  `RLine` table (per-line `depthAtStart`, `endsWithContinuation`,
  `startsWithContinuation`, `blank`) and computes: `base` (leading width of the
  statement's first line, found by walking the same continuation rule the scanner
  uses) `+` bracket depth `* unit` (indent after an open bracket, dedent after a
  close) `+` one unit when the current line ends with a pipe/`+`/binary operator at
  bracket depth 0 (the pipe and ggplot continuation). It reads the mask, so a `+`
  or bracket inside a string or comment can never trigger a false indent.
- **Python indentation reuses the same `@lezer/python` grammar Slice 2's `py.ts`
  already parses.** The `python()` language support ships tree-driven indentation
  (via `indentNodeProp`) keyed off exactly that grammar; we do not reimplement
  Python indent (reimplementing it is the "never change behavior" risk). We only
  add `indentUnit.of("    ")` to force a consistent 4-space, no-tab policy, and we
  unit-test the real behavior with `getIndentation` in Node.
- **SQL indentation reuses Slice 2's `maskStringsAndComments`** to find the enclosing
  statement (previous top-level `;`), the bracket depth (subqueries/CTE parens), and
  the governing clause keyword, all while ignoring `;`/keywords inside strings and
  comments. `statementNodeAt` from Slice 2 remains available for precise statement
  bounds but the mask-based scan is cheaper and sufficient.
- **Linting** adds `@codemirror/lint`. Python and SQL both have Lezer grammars, so
  their linter surfaces the parser's error nodes (`NodeType.isError`) as
  diagnostics, cheaply and in the browser. R has no grammar, so its linter is a
  bracket-balance and unterminated-string check over the Slice 2 mask. All three run
  on the `linter()` debounce, produce decorations only (never a document change), so
  they cannot move the cursor, cannot alter strings/comments, and are trivially
  undoable (there is nothing to undo).

The only editor change is additive CodeMirror extensions in `CodeEditor.tsx`: an
indent extension per language and one linter, wired next to the existing language
extensions. The pure decision logic (`rIndentColumns`, `sqlIndentColumns`, the lint
problem finders) lives in `lang-structure/` with no CodeMirror-view dependency and
is unit-tested in Vitest's Node environment.

**Grounding (verified against the installed packages).**

- `@codemirror/language` exports `indentService` (a `Facet<(context, pos) => number
  | null | undefined>`), `indentUnit` (a `Facet<string>`), `getIndentation(context,
  pos)`, and `IndentContext` (verified in `node_modules/@codemirror/language/dist/index.d.ts`,
  lines 428, 434, 458, 470). CodeMirror's default `insertNewlineAndIndent` (bound by
  `basicSetup`) consults `getIndentation`, which "will first consult any indent
  services" before language-based indentation, so an `indentService` we add wins for
  R and SQL and returns an absolute column count.
- `indentUnit.of("    ")` (four spaces) makes both the tree-driven Python indent and
  the whitespace `insertNewlineAndIndent` emits space-based, so no tab is ever
  produced. Our R/SQL services return explicit even column counts, so their output is
  space runs regardless.
- `@codemirror/lint@6.9.7` is already resolvable in `node_modules` (it ships inside
  the `codemirror` meta-package that `basicSetup` comes from) but is **not** a
  declared dependency in `web/package.json`. Since this slice imports `@codemirror/lint`
  directly, Task 1 adds it as an explicit dependency. It exports `linter(source,
  { delay })` and the `Diagnostic` type. A `Diagnostic` is a pure decoration
  `{ from, to, severity, message }`; applying it never changes the document.
- `NodeType.isError` exists on Lezer nodes (`node_modules/@lezer/common/dist/index.d.ts`
  line 460), so iterating a parsed tree and collecting error nodes gives real syntax
  diagnostics for Python (`@lezer/python`) and SQL (`SQLite.language.parser`), the same
  two parsers Slice 2 already uses in Node.
- `python()` (`@codemirror/lang-python`) and `SQLite` (`@codemirror/lang-sql`) both
  construct in a plain Node process (no DOM), proven by Slice 2's unit tests, so the
  Python `getIndentation` test and the SQL error-node test run under Vitest
  `environment: "node"`.
- R uses `StreamLanguage.define(r)` (CodeEditor.tsx:268-272) with no useful tree, so
  R indentation and R linting are bespoke, exactly as Slice 2's R statement finder is.

**Tech Stack:** Next.js (this repo's vendored build — read `node_modules/next/dist/docs/`
before touching any Next API; this slice does not), React 19, CodeMirror 6
(`@codemirror/view`, `@codemirror/state`, `@codemirror/language`, and now
`@codemirror/lint`), the Lezer parsers `@lezer/python` and `@codemirror/lang-sql`
(both already dependencies, reused from Slice 2), Vitest for the pure unit tests,
Playwright + `@axe-core/playwright` for the sandbox e2e.

## Global Constraints

- **No git commits.** The working tree stays uncommitted; each task ends by running
  verification commands instead of committing.
- **Reuse Slice 2, do not fork it.** `rIndentColumns`, `sqlIndentColumns`, and the
  R linter import `scanR` / `maskStringsAndComments` from the existing
  `lang-structure/` files. Python indentation reuses `python()`'s tree indentation
  (the same `@lezer/python` grammar as `py.ts`). Do not reimplement Python
  indentation, and do not duplicate the mask or the R line scanner.
- **Pure logic is separate from CodeMirror glue.** Everything in `lang-structure/`
  takes `(text: string, pos: number)` (indent) or `(text, languageId)` (lint) and
  returns numbers or plain problem records. It must not import `@codemirror/view`
  or touch an `EditorView`, so it stays testable in Vitest's `node` environment.
  The `indentService`/`linter` glue lives in `CodeEditor.tsx`.
- **Never change program behavior via auto-indent, especially Python.** Python uses
  the package's own tree indentation; we add only `indentUnit.of("    ")`. Indent is
  computed on Enter only. No full-document formatter runs on any keystroke.
- **Linting is unobtrusive.** Diagnostics are decorations only: they never edit the
  document, never move the cursor, never touch quoted text or comments, and run on
  the `linter()` debounce (`delay: 400`), not per keystroke. Heavier truth (a real
  SQLite parse error, a real Python exception) already surfaces in the console on
  execute; the linter only flags cheap, obvious, in-browser problems.
- **Consistent 4-space Python, no mixed tabs/spaces.** `indentUnit.of("    ")` plus a
  warning-level lint rule flagging any line whose leading whitespace mixes tabs and
  spaces.
- **No em dashes in any user-facing copy** (diagnostic messages, tooltips). Use
  commas, colons, or sentence breaks.
- **Keyboard shortcuts support Ctrl (Windows/Linux) and Cmd (macOS).** This slice
  adds no shortcut; Enter and the run keys are unchanged.
- All commands run from `web/` (the Next app root). Unit tests: `npm run test`. The
  Playwright config starts its own dev server on port 3100, so no separate server is
  needed for `npm run test:e2e`.

## File Structure

- `web/package.json` — Modify. Add `"@codemirror/lint": "^6.9.7"` to `dependencies`
  (already resolvable transitively; declare it because we import it directly).
- `web/lib/sandbox/lang-structure/indent.ts` — Create. Pure indent-decision helpers:
  `rIndentColumns(text, pos)`, `sqlIndentColumns(text, pos)`, and the shared
  `R_INDENT_UNIT` / `SQL_INDENT_UNIT` constants. Imports `scanR` and
  `maskStringsAndComments` from Slice 2.
- `web/lib/sandbox/lang-structure/lint.ts` — Create. Pure lint finders:
  `rBalanceProblems(text)`, `treeErrorProblems(text, lang)`, `pyTabSpaceProblems(text)`,
  and the `lintProblems(text, languageId)` dispatch returning `LintProblem[]`. Imports
  `maskStringsAndComments` from Slice 2 and the `@lezer/python` / `@codemirror/lang-sql`
  parsers.
- `web/lib/sandbox/lang-structure/index.ts` — Modify. Re-export `rIndentColumns`,
  `sqlIndentColumns`, `R_INDENT_UNIT`, `SQL_INDENT_UNIT`, `lintProblems`, and the
  `LintProblem` type so `CodeEditor.tsx` reaches them through the one lazy chunk.
- `web/components/run/CodeEditor.tsx` — Modify. Add `@codemirror/lint` to
  `loadCodeMirror`'s `Promise.all` and `LoadedEditor`; add two extension builders
  (`indentExtensions`, `lintExtension`); insert them into the editor's extensions
  array.
- `web/tests/unit/indent.test.ts` — Create. Pure tables for R and SQL indent columns
  plus a Node `getIndentation` check for Python, using the requirement's snippets.
- `web/tests/unit/lint.test.ts` — Create. Pure tables for the R balance linter, the
  Python/SQL tree-error linter, and the Python tab/space warning.
- `web/tests/e2e/sandbox.spec.ts` — Modify. Add Enter-indentation e2e for R, Python,
  and SQL, and one lint-marker e2e for an obvious R error.

---

## Task 1: Declare the lint dependency and confirm the indent APIs

`@codemirror/lint` already resolves (it ships in the `codemirror` meta-package that
`basicSetup` imports), but we import it directly in this slice, so it must be a
declared dependency. `@codemirror/language` already provides `indentService`,
`indentUnit`, `getIndentation`, and `IndentContext`; no new install is needed for
indentation.

**Files:**
- Modify: `web/package.json`

**Interfaces:**
- Produces: a declared `@codemirror/lint` dependency so `import("@codemirror/lint")`
  is a supported direct import.

- [ ] **Step 1: Add the dependency line**

In `web/package.json`, add to `dependencies` (keep the block alphabetically ordered,
so it sits right after `@codemirror/language`):

```json
    "@codemirror/lint": "^6.9.7",
```

- [ ] **Step 2: Install and confirm the version resolves**

Run: `npm install`
Expected: completes without adding a new download (6.9.7 is already in the tree);
`package-lock.json` now records `@codemirror/lint` as a direct dependency.

- [ ] **Step 3: Confirm the indent and lint APIs are importable**

Run:
```bash
node -e "const l=require('@codemirror/language'); console.log(typeof l.indentService, typeof l.indentUnit, typeof l.getIndentation, typeof l.IndentContext); const lint=require('@codemirror/lint'); console.log(typeof lint.linter);"
```
Expected: `object object function function` then `function` (facets print as objects;
`getIndentation`, `IndentContext`, and `linter` are functions/classes).

- [ ] **Step 4: Typecheck the untouched tree**

Run: `npm run typecheck`
Expected: no errors (nothing else changed yet).

---

## Task 2: R indent-decision helper (reuses `scanR` + the mask)

The R helper is the core reuse of Slice 2. It reads the same `RLine` table `scanR`
produces (per-line bracket depth and continuation flags over the string/comment
mask) and turns it into a column count for the new line.

### The algorithm (concrete)

Given `text` (the document before Enter) and `pos` (the cursor):

1. `const { lines } = scanR(text)` and `const mask = maskStringsAndComments(text, "r")`.
   Everything below reads the mask, so brackets/operators inside strings or comments
   are invisible (this is exactly what makes a ggplot continuation `+` distinct from
   a `+` in a string, per the requirement).
2. `cur` = the `RLine` containing `pos`.
3. **Open-bracket depth at the cursor** = `lines[cur].depthAtStart` plus the net
   `(`/`[`/`{` minus `)`/`]`/`}` in the mask from the line start up to `pos`. This is
   "indent after an open bracket, dedent after a close" made numeric.
4. **Base indent** = leading whitespace width of the statement's first line. Walk
   back from `cur` over the same continuation rule the scanner uses: a line continues
   its predecessor when it opens inside brackets (`depthAtStart > 0`), starts with a
   continuation token (`startsWithContinuation`), or the previous non-blank line ends
   with one (`endsWithContinuation`). Stop at the first line that does not continue;
   that is the statement start.
5. **Trailing continuation** = the current line up to `pos`, right-trimmed on the
   mask, ends with a pipe / `+` / binary operator. Add one unit only at bracket depth
   0 (inside brackets, the bracket depth already supplies the indent, so args align
   rather than stair-step).
6. `indent = base + depth * R_INDENT_UNIT + (depth === 0 && trailingCont ? R_INDENT_UNIT : 0)`.
7. **Dedent before a closing bracket**: if the first non-space character at/after
   `pos` is `)`/`]`/`}`, subtract one unit (so pressing Enter with the caret just
   before an existing closer lands the closer at the opener's level).

`R_INDENT_UNIT = 2` (the common R convention).

**Files:**
- Create: `web/lib/sandbox/lang-structure/indent.ts`
- Test: `web/tests/unit/indent.test.ts` (R section)

**Interfaces:**
- Consumes: `scanR`, `maskStringsAndComments` from Slice 2.
- Produces: `rIndentColumns(text: string, pos: number): number`; exported constants
  `R_INDENT_UNIT = 2`, `SQL_INDENT_UNIT = 2`.

- [ ] **Step 1: Write the failing R indent tests**

Create `web/tests/unit/indent.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import { rIndentColumns } from "@/lib/sandbox/lang-structure/indent";

/** Columns the new line would get if Enter were pressed at the end of `line`. */
function rEnterAfter(src: string, line: string): number {
  const pos = src.indexOf(line) + line.length;
  return rIndentColumns(src, pos);
}

describe("rIndentColumns", () => {
  it("gives no continuation indent after a complete statement", () => {
    const src = "x <- 1\n";
    expect(rEnterAfter(src, "x <- 1")).toBe(0);
  });

  it("indents after a trailing native pipe", () => {
    const src = "df |>\n";
    expect(rEnterAfter(src, "df |>")).toBe(2);
  });

  it("keeps the pipe indent on subsequent chain lines", () => {
    const src = "df |>\n  filter(x) |>\n";
    expect(rEnterAfter(src, "  filter(x) |>")).toBe(2);
  });

  it("indents after a magrittr pipe", () => {
    const src = "df %>%\n";
    expect(rEnterAfter(src, "df %>%")).toBe(2);
  });

  it("indents after a ggplot layer ending in +", () => {
    const src = "ggplot(d, aes(x, y)) +\n";
    expect(rEnterAfter(src, "ggplot(d, aes(x, y)) +")).toBe(2);
  });

  it("indents inside an open call and aligns following args", () => {
    const src = "t <- tibble(\n  a = 1,\n";
    expect(rEnterAfter(src, "t <- tibble(")).toBe(2);
    expect(rEnterAfter(src, "  a = 1,")).toBe(2);
  });

  it("does not treat a + inside a string as a continuation", () => {
    const src = 'lab <- "a + b"\n';
    expect(rEnterAfter(src, 'lab <- "a + b"')).toBe(0);
  });

  it("does not treat a + inside a comment as a continuation", () => {
    const src = "x <- 1 # add + more\n";
    expect(rEnterAfter(src, "x <- 1 # add + more")).toBe(0);
  });

  it("dedents when the caret is just before a closing bracket", () => {
    const src = "tibble(\n  a = 1\n)";
    const pos = src.indexOf("\n)") + 1; // caret right before ")"
    expect(rIndentColumns(src, pos)).toBe(0);
  });
});
```

- [ ] **Step 2: Run and confirm the R indent tests fail**

Run: `npm run test -- indent`
Expected: FAIL (`indent.ts` does not exist).

- [ ] **Step 3: Implement `indent.ts` (R half)**

Create `web/lib/sandbox/lang-structure/indent.ts`:

```typescript
import { scanR } from "./r-scan";
import { maskStringsAndComments } from "./mask";

export const R_INDENT_UNIT = 2;
export const SQL_INDENT_UNIT = 2;

const OPENERS = new Set(["(", "[", "{"]);
const CLOSERS = new Set([")", "]", "}"]);
// Pipe, magrittr-style %...%, ggplot/arithmetic +, and other binary operators
// that cannot end an R expression. Read from the mask, so string/comment content
// never matches.
const R_TRAILING_CONT = /(\|>|%[^%\s]*%|[-+*/^~:?<>=&|!])$/;

function leadingWidth(text: string, from: number, to: number): number {
  let n = 0;
  for (let i = from; i < to; i++) {
    if (text[i] === " ") n++;
    else if (text[i] === "\t") n++;
    else break;
  }
  return n;
}

/**
 * Columns of indentation for a new line created by pressing Enter at `pos` in R.
 * Reuses Slice 2's `scanR` line table (bracket depth + continuation flags) and the
 * string/comment mask, so a bracket or operator inside a string or comment can
 * never influence the result.
 */
export function rIndentColumns(text: string, pos: number): number {
  const clamped = Math.max(0, Math.min(pos, text.length));
  const { lines } = scanR(text);
  if (lines.length === 0) return 0;
  const mask = maskStringsAndComments(text, "r");

  let cur = lines.findIndex((l) => clamped >= l.from && clamped <= l.to);
  if (cur < 0) cur = lines.length - 1;

  // Open-bracket depth at the cursor.
  let depth = lines[cur].depthAtStart;
  for (let i = lines[cur].from; i < clamped && i < lines[cur].to; i++) {
    const c = mask[i];
    if (OPENERS.has(c)) depth++;
    else if (CLOSERS.has(c)) depth = Math.max(0, depth - 1);
  }

  // Base indent: leading whitespace of the statement's first line. Walk back over
  // the same continuation rule the scanner uses.
  let start = cur;
  while (start > 0) {
    const L = lines[start];
    let p = start - 1;
    while (p >= 0 && lines[p].blank) p--;
    const continues =
      L.depthAtStart > 0 ||
      L.startsWithContinuation ||
      (p >= 0 && lines[p].endsWithContinuation);
    if (!continues || p < 0) break;
    start = p;
  }
  const base = leadingWidth(text, lines[start].from, lines[start].to);

  // Trailing continuation operator on the current line up to the cursor.
  const curSlice = mask.slice(lines[cur].from, clamped).replace(/\s+$/, "");
  const trailingCont = R_TRAILING_CONT.test(curSlice);

  let indent = base + depth * R_INDENT_UNIT;
  if (depth === 0 && trailingCont) indent += R_INDENT_UNIT;

  // Dedent when the new line will begin with an existing closing bracket.
  let q = clamped;
  while (q < text.length && (text[q] === " " || text[q] === "\t")) q++;
  if (q < text.length && CLOSERS.has(text[q])) {
    indent = Math.max(0, indent - R_INDENT_UNIT);
  }

  return indent;
}
```

- [ ] **Step 4: Run and confirm the R indent tests pass**

Run: `npm run test -- indent`
Expected: PASS for every `rIndentColumns` case (complete statement, native pipe,
chain continuation, magrittr pipe, ggplot `+`, open call and arg alignment,
plus-in-string, plus-in-comment, closing-bracket dedent).

---

## Task 3: SQL indent-decision helper (reuses the mask)

SQL indentation reuses Slice 2's `maskStringsAndComments` to ignore `;`, keywords,
and brackets inside strings and comments, and computes columns from bracket depth
(subqueries/CTE parens) plus the governing clause keyword.

### The algorithm (concrete)

Given `text` and `pos`:

1. `const mask = maskStringsAndComments(text, "sql")`.
2. **Bracket depth at `pos`** over the mask (subquery / CTE paren nesting).
3. **Statement start** = one past the previous top-level `;` (depth 0) in the mask, or
   0. This is the same "ignore a `;` inside a string" guarantee Slice 2 relies on.
4. **Base indent** = leading whitespace width of the statement's first non-blank line.
5. **Governing clause** = scan the statement's lines up to and including the current
   line; remember the last line whose first token is a clause keyword. A clause is a
   "body clause" (its items indent one level under it) when it is `SELECT`, `WHERE`,
   `HAVING`, `ON`, `GROUP BY`, or `ORDER BY`. `FROM`, `LIMIT`, and `WITH` are not body
   clauses (their line, and the line after them, sits at the statement level).
6. `indent = base + depth * SQL_INDENT_UNIT + (governingIsBody ? SQL_INDENT_UNIT : 0)`.

This yields, for a flat query, columns under SELECT, conditions under WHERE, CTE and
subquery contents indented by their paren depth, and major clauses (FROM/WHERE/GROUP
BY/...) back at the statement level. `SQL_INDENT_UNIT = 2`.

**Honest scope.** This is a pragmatic keyword-and-bracket approximation, not a full
SQL formatter. It handles the requirement's cases (items under SELECT, conditions
under WHERE/HAVING/ON, CTE and subquery contents, major-clause dedent) on the common
statement shapes. It does not attempt column-precise alignment to the character after
`SELECT`, and a deeply interleaved subquery can carry a stale governing clause; those
are acceptable, documented limits. The engine is SQLite (`sqlite-wasm`), so
dialect-specific procedural indentation is out of scope, consistent with Slice 2's SQL
note.

**Files:**
- Modify: `web/lib/sandbox/lang-structure/indent.ts` (add the SQL half)
- Test: `web/tests/unit/indent.test.ts` (SQL section)

**Interfaces:**
- Consumes: `maskStringsAndComments` from Slice 2.
- Produces: `sqlIndentColumns(text: string, pos: number): number`.

- [ ] **Step 1: Write the failing SQL indent tests**

Add to `web/tests/unit/indent.test.ts`:

```typescript
import { sqlIndentColumns } from "@/lib/sandbox/lang-structure/indent";

function sqlEnterAfter(src: string, line: string): number {
  const pos = src.indexOf(line) + line.length;
  return sqlIndentColumns(src, pos);
}

describe("sqlIndentColumns", () => {
  it("indents columns under SELECT", () => {
    const src = "SELECT\n";
    expect(sqlEnterAfter(src, "SELECT")).toBe(2);
  });

  it("keeps column items aligned under SELECT", () => {
    const src = "SELECT\n  a,\n";
    expect(sqlEnterAfter(src, "  a,")).toBe(2);
  });

  it("returns to the statement level for a major clause line", () => {
    const src = "SELECT a\nFROM t\n";
    expect(sqlEnterAfter(src, "FROM t")).toBe(0);
  });

  it("indents conditions under WHERE", () => {
    const src = "SELECT a\nFROM t\nWHERE\n";
    expect(sqlEnterAfter(src, "WHERE")).toBe(2);
  });

  it("indents CTE contents inside the parens", () => {
    const src = "WITH a AS (\n";
    expect(sqlEnterAfter(src, "WITH a AS (")).toBe(2);
  });

  it("indents subquery contents by paren depth plus the inner clause", () => {
    const src = "SELECT * FROM t WHERE n > (\n";
    // depth 1 paren (2) + governing SELECT body (2) = 4.
    expect(sqlEnterAfter(src, "SELECT * FROM t WHERE n > (")).toBe(4);
  });

  it("does not split on a semicolon inside a string", () => {
    const src = "SELECT ';' AS s,\n";
    expect(sqlEnterAfter(src, "SELECT ';' AS s,")).toBe(2);
  });
});
```

- [ ] **Step 2: Run and confirm the SQL indent tests fail**

Run: `npm run test -- indent`
Expected: FAIL (`sqlIndentColumns` not exported yet).

- [ ] **Step 3: Add the SQL half to `indent.ts`**

Append to `web/lib/sandbox/lang-structure/indent.ts`:

```typescript
const SQL_CLAUSE =
  /^(select|from|where|having|group\s+by|order\s+by|limit|on|union(\s+all)?|values|inner\s+join|left\s+join|right\s+join|join|with)\b/i;
const SQL_BODY_CLAUSE = /^(select|where|having|on|group\s+by|order\s+by)\b/i;

function lineStart(text: string, p: number): number {
  let s = Math.max(0, Math.min(p, text.length));
  while (s > 0 && text[s - 1] !== "\n") s--;
  return s;
}

function lineEnd(text: string, p: number): number {
  let e = Math.max(0, Math.min(p, text.length));
  while (e < text.length && text[e] !== "\n") e++;
  return e;
}

/**
 * Columns of indentation for a new line created by pressing Enter at `pos` in SQL.
 * Reuses Slice 2's mask so a `;`, keyword, or bracket inside a string or comment is
 * ignored. A pragmatic keyword-and-bracket approximation, not a full formatter.
 */
export function sqlIndentColumns(text: string, pos: number): number {
  const clamped = Math.max(0, Math.min(pos, text.length));
  const mask = maskStringsAndComments(text, "sql");

  // Bracket depth at the cursor.
  let depth = 0;
  for (let i = 0; i < clamped; i++) {
    const c = mask[i];
    if (OPENERS.has(c)) depth++;
    else if (CLOSERS.has(c)) depth = Math.max(0, depth - 1);
  }

  // Statement start: one past the previous top-level ";".
  let stmtStart = 0;
  let d = 0;
  for (let i = clamped - 1; i >= 0; i--) {
    const c = mask[i];
    if (CLOSERS.has(c)) d++;
    else if (OPENERS.has(c)) d = Math.max(0, d - 1);
    else if (c === ";" && d === 0) {
      stmtStart = i + 1;
      break;
    }
  }

  // Base indent = leading whitespace of the statement's first non-blank line.
  let s = stmtStart;
  while (s < text.length && /\s/.test(text[s])) s++;
  const base = leadingWidth(text, lineStart(text, s), s);

  // Governing clause: last clause keyword seen from the statement start through the
  // current line.
  let governingIsBody = false;
  const curStart = lineStart(text, clamped);
  let i = lineStart(text, stmtStart);
  while (i <= curStart && i < text.length) {
    const le = lineEnd(text, i);
    const first = mask.slice(i, le).replace(/^\s+/, "");
    if (SQL_CLAUSE.test(first)) governingIsBody = SQL_BODY_CLAUSE.test(first);
    if (le >= curStart) break;
    i = le + 1;
  }

  return base + depth * SQL_INDENT_UNIT + (governingIsBody ? SQL_INDENT_UNIT : 0);
}
```

- [ ] **Step 4: Run and confirm the SQL indent tests pass**

Run: `npm run test -- indent`
Expected: PASS for every `sqlIndentColumns` case (columns under SELECT, aligned
items, major-clause dedent, conditions under WHERE, CTE contents, subquery depth,
semicolon-in-string).

---

## Task 4: Python indentation policy and its Node-level proof

Python reuses the `@lezer/python` grammar's own tree-driven indentation, which
already indents after `:`, preserves block indent, dedents on block end, and indents
inside brackets. We do not reimplement it (that is the "never change behavior" risk).
We only add `indentUnit.of("    ")` so every level is exactly four spaces and never a
tab. The proof is a Node test that drives the real `getIndentation` used by
`insertNewlineAndIndent`, not a reimplementation.

**Files:**
- Test: `web/tests/unit/indent.test.ts` (Python section)

**Interfaces:**
- Consumes: `python()` from `@codemirror/lang-python`; `EditorState` from
  `@codemirror/state`; `indentUnit`, `getIndentation`, `IndentContext` from
  `@codemirror/language`. No new production code (the policy is an editor extension
  added in Task 5).

- [ ] **Step 1: Write the Python indentation proof test**

Add to `web/tests/unit/indent.test.ts`:

```typescript
import { EditorState } from "@codemirror/state";
import { getIndentation, IndentContext, indentUnit } from "@codemirror/language";
import { python } from "@codemirror/lang-python";

/** The indentation columns the editor would give a new line broken at `pos`. */
function pyIndentAt(src: string, pos: number): number | null {
  const state = EditorState.create({
    doc: src,
    extensions: [python(), indentUnit.of("    ")],
  });
  const cx = new IndentContext(state, { simulateBreak: pos });
  return getIndentation(cx, pos);
}

describe("python indentation policy (built-in tree indent, 4-space unit)", () => {
  it("indents one 4-space level after a colon header", () => {
    const src = "def f(x):";
    expect(pyIndentAt(src, src.length)).toBe(4);
  });

  it("preserves the block indent for the next simple statement", () => {
    const src = "def f(x):\n    y = 1";
    expect(pyIndentAt(src, src.length)).toBe(4);
  });

  it("indents nested blocks by another level", () => {
    const src = "def f(x):\n    if x:";
    expect(pyIndentAt(src, src.length)).toBe(8);
  });

  it("indents inside an open bracket", () => {
    const src = "total = (";
    expect(pyIndentAt(src, src.length)).toBeGreaterThanOrEqual(4);
  });
});
```

- [ ] **Step 2: Run and confirm the Python indentation proof passes**

Run: `npm run test -- indent`
Expected: PASS. This proves `python() + indentUnit.of("    ")` already delivers the
Python indentation the requirement asks for, so Task 5 only has to add that one
extension and no bespoke Python logic.

Contingency: if `python()` cannot construct under Vitest `node` (it constructs fine
in Slice 2, so this is not expected), replace this test with the two Python
Enter-indentation e2e assertions in Task 8 and note in the file that Python
indentation is covered end-to-end rather than at the unit level. Do not reimplement
Python indentation to satisfy a unit test.

---

## Task 5: Wire the indent extensions into the editor

Add one indent extension per language: `indentUnit.of("    ")` everywhere (space-based,
no tabs), plus a custom `indentService` for R and SQL that returns the pure helper's
column count. Python gets no custom service; `python()`'s own tree indentation stands.

**Files:**
- Modify: `web/lib/sandbox/lang-structure/index.ts` (re-exports)
- Modify: `web/components/run/CodeEditor.tsx` (extensions array, one new builder)

**Interfaces:**
- Consumes: `rIndentColumns`, `sqlIndentColumns` (via the `langStructure` lazy
  import); `lang.indentService`, `lang.indentUnit` (already the loaded
  `@codemirror/language`).
- Produces: `indentExtensions(lang, langStructure, languageId): Extension[]` in
  `CodeEditor.tsx`.

- [ ] **Step 1: Re-export the indent helpers from the module index**

In `web/lib/sandbox/lang-structure/index.ts`, add after the existing re-exports
(after the `maskStringsAndComments` line):

```typescript
export {
  rIndentColumns,
  sqlIndentColumns,
  R_INDENT_UNIT,
  SQL_INDENT_UNIT,
} from "./indent";
```

- [ ] **Step 2: Add the indent builder to `CodeEditor.tsx`**

Add this function near `themeExtensions` in `web/components/run/CodeEditor.tsx`:

```typescript
/** Per-language indentation on Enter. `indentUnit` is four spaces for all three
 *  languages so indentation is never a tab and Python is a consistent 4-space
 *  policy. R and SQL add a custom indentService backed by the pure column helpers;
 *  Python relies on the @lezer/python tree indentation already in `python()`. */
function indentExtensions(
  lang: LoadedEditor["lang"],
  langStructure: LoadedEditor["langStructure"],
  languageId: string,
): Extension[] {
  const exts: Extension[] = [lang.indentUnit.of("    ")];
  if (languageId === "r") {
    exts.push(
      lang.indentService.of((cx, pos) =>
        langStructure.rIndentColumns(cx.state.doc.toString(), pos),
      ),
    );
  } else if (languageId === "sql") {
    exts.push(
      lang.indentService.of((cx, pos) =>
        langStructure.sqlIndentColumns(cx.state.doc.toString(), pos),
      ),
    );
  }
  return exts;
}
```

- [ ] **Step 3: Insert the indent extensions into the editor build**

In the mount effect's `extensions` array (currently ending after the theme and run
keymap, around lines 106-121), add the indent extensions right after `langExt`:

```typescript
            cm.basicSetup,
            langExt,
            ...indentExtensions(lang, langStructure, props.languageId),
```

(`lang` and `langStructure` are already destructured from `loadCodeMirror` in the
`.then(...)` on line 90.)

- [ ] **Step 4: Typecheck**

Run: `npm run typecheck`
Expected: no errors. `indentService`/`indentUnit` are on the loaded
`@codemirror/language` (`lang`); `rIndentColumns`/`sqlIndentColumns` are on
`langStructure`.

- [ ] **Step 5: Confirm the unit suite still passes**

Run: `npm run test -- indent`
Expected: PASS (unchanged; the pure helpers are untouched, only wired).

---

## Task 6: Pure lint finders (tree errors, R balance, tab/space)

Three cheap, in-browser finders, each returning plain `LintProblem` records. Python
and SQL reuse the Lezer parsers Slice 2 already runs in Node; R reuses the Slice 2
mask.

**Files:**
- Create: `web/lib/sandbox/lang-structure/lint.ts`
- Modify: `web/lib/sandbox/lang-structure/index.ts` (re-export)
- Test: `web/tests/unit/lint.test.ts`

**Interfaces:**
- Consumes: `maskStringsAndComments` (Slice 2); `parser` from `@lezer/python`;
  `SQLite` from `@codemirror/lang-sql`.
- Produces: `LintProblem = { from: number; to: number; severity: "error" |
  "warning"; message: string }`; `rBalanceProblems(text)`,
  `treeErrorProblems(text, lang: "python" | "sql")`, `pyTabSpaceProblems(text)`,
  and `lintProblems(text, languageId): LintProblem[]`.

- [ ] **Step 1: Write the failing lint tests**

Create `web/tests/unit/lint.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import {
  rBalanceProblems,
  treeErrorProblems,
  pyTabSpaceProblems,
  lintProblems,
} from "@/lib/sandbox/lang-structure/lint";

describe("rBalanceProblems", () => {
  it("flags an unclosed bracket", () => {
    const p = rBalanceProblems("x <- (1 + 2\n");
    expect(p.some((d) => /Unclosed/.test(d.message))).toBe(true);
  });

  it("flags an unmatched closing bracket", () => {
    const p = rBalanceProblems("x <- 1)\n");
    expect(p.some((d) => /Unmatched/.test(d.message))).toBe(true);
  });

  it("does not flag a balanced statement", () => {
    expect(rBalanceProblems("f(g(1), h(2))\n")).toEqual([]);
  });

  it("ignores brackets inside strings and comments", () => {
    expect(rBalanceProblems('x <- "((" # ))\n')).toEqual([]);
  });

  it("flags an unterminated string", () => {
    const p = rBalanceProblems('x <- "oops\n');
    expect(p.some((d) => /Unterminated/.test(d.message))).toBe(true);
  });
});

describe("treeErrorProblems", () => {
  it("flags a Python syntax error", () => {
    const p = treeErrorProblems("def f(:\n", "python");
    expect(p.length).toBeGreaterThan(0);
    expect(p[0].severity).toBe("error");
  });

  it("accepts valid Python", () => {
    expect(treeErrorProblems("x = 1\n", "python")).toEqual([]);
  });

  it("flags a SQL syntax error", () => {
    const p = treeErrorProblems("SELECT FROM;\n", "sql");
    expect(p.length).toBeGreaterThan(0);
  });

  it("accepts valid SQL", () => {
    expect(treeErrorProblems("SELECT 1;\n", "sql")).toEqual([]);
  });
});

describe("pyTabSpaceProblems", () => {
  it("warns on a line whose indentation mixes tabs and spaces", () => {
    const p = pyTabSpaceProblems("def f():\n \ty = 1\n");
    expect(p.length).toBe(1);
    expect(p[0].severity).toBe("warning");
  });

  it("does not warn on pure-space indentation", () => {
    expect(pyTabSpaceProblems("def f():\n    y = 1\n")).toEqual([]);
  });
});

describe("lintProblems dispatch", () => {
  it("routes R to the balance finder", () => {
    expect(lintProblems("x <- (1\n", "r").length).toBeGreaterThan(0);
  });
  it("routes Python to the tree-error and tab/space finders", () => {
    expect(lintProblems("def f(:\n", "python").length).toBeGreaterThan(0);
  });
  it("routes SQL to the tree-error finder", () => {
    expect(lintProblems("SELECT FROM;\n", "sql").length).toBeGreaterThan(0);
  });
});
```

- [ ] **Step 2: Run and confirm the lint tests fail**

Run: `npm run test -- lint`
Expected: FAIL (`lint.ts` does not exist).

- [ ] **Step 3: Implement `lint.ts`**

Create `web/lib/sandbox/lang-structure/lint.ts`:

```typescript
import { parser as pyParser } from "@lezer/python";
import { SQLite } from "@codemirror/lang-sql";
import { maskStringsAndComments } from "./mask";
import type { LanguageId } from "./types";

export interface LintProblem {
  from: number;
  to: number;
  severity: "error" | "warning";
  message: string;
}

const OPEN: Record<string, string> = { ")": "(", "]": "[", "}": "{" };

/** Unbalanced brackets and unterminated strings in R (no grammar available). Reads
 *  the Slice 2 mask so brackets/quotes inside strings and comments are ignored. */
export function rBalanceProblems(text: string): LintProblem[] {
  const mask = maskStringsAndComments(text, "r");
  const problems: LintProblem[] = [];
  const stack: { ch: string; pos: number }[] = [];
  for (let i = 0; i < mask.length; i++) {
    const c = mask[i];
    if (c === "(" || c === "[" || c === "{") stack.push({ ch: c, pos: i });
    else if (c === ")" || c === "]" || c === "}") {
      const top = stack.pop();
      if (!top || top.ch !== OPEN[c]) {
        problems.push({ from: i, to: i + 1, severity: "error", message: `Unmatched ${c}` });
      }
    }
  }
  for (const open of stack) {
    problems.push({
      from: open.pos,
      to: open.pos + 1,
      severity: "error",
      message: `Unclosed ${open.ch}`,
    });
  }
  problems.push(...unterminatedStrings(text));
  return problems;
}

/** Quote runs that reach end of file without closing (R/Python quoting). */
function unterminatedStrings(text: string): LintProblem[] {
  const out: LintProblem[] = [];
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === "#") {
      while (i < text.length && text[i] !== "\n") i++;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") {
      const quote = ch;
      const start = i;
      i++;
      let closed = false;
      while (i < text.length) {
        if (text[i] === "\\" && quote !== "`") {
          i += 2;
          continue;
        }
        if (text[i] === quote) {
          i++;
          closed = true;
          break;
        }
        i++;
      }
      if (!closed) {
        out.push({ from: start, to: text.length, severity: "error", message: "Unterminated string" });
      }
      continue;
    }
    i++;
  }
  return out;
}

/** Parser error nodes for a language with a Lezer grammar (Python or SQL). */
export function treeErrorProblems(text: string, lang: "python" | "sql"): LintProblem[] {
  const tree = lang === "python" ? pyParser.parse(text) : SQLite.language.parser.parse(text);
  const out: LintProblem[] = [];
  const cursor = tree.cursor();
  do {
    if (cursor.type.isError) {
      const from = cursor.from;
      const to = Math.max(cursor.to, from + 1);
      // Collapse a run of adjacent error nodes into one marker.
      const last = out[out.length - 1];
      if (last && from <= last.to) last.to = Math.max(last.to, to);
      else out.push({ from, to, severity: "error", message: "Syntax error" });
    }
  } while (cursor.next());
  return out;
}

/** Lines whose leading whitespace mixes tabs and spaces (Python correctness). */
export function pyTabSpaceProblems(text: string): LintProblem[] {
  const out: LintProblem[] = [];
  let from = 0;
  for (const line of text.split("\n")) {
    const lead = /^[ \t]*/.exec(line)![0];
    if (lead.includes("\t") && lead.includes(" ")) {
      out.push({
        from,
        to: from + lead.length,
        severity: "warning",
        message: "Mixed tabs and spaces in indentation",
      });
    }
    from += line.length + 1; // + the newline
  }
  return out;
}

/** The unobtrusive lint problems for a document, by language. */
export function lintProblems(text: string, languageId: LanguageId): LintProblem[] {
  if (languageId === "r") return rBalanceProblems(text);
  if (languageId === "python") {
    return [...treeErrorProblems(text, "python"), ...pyTabSpaceProblems(text)];
  }
  return treeErrorProblems(text, "sql");
}
```

- [ ] **Step 4: Re-export from the module index**

In `web/lib/sandbox/lang-structure/index.ts`, add:

```typescript
export { lintProblems } from "./lint";
export type { LintProblem } from "./lint";
```

- [ ] **Step 5: Run and confirm the lint tests pass**

Run: `npm run test -- lint`
Expected: PASS for every case (R balance: unclosed, unmatched, balanced, string/comment
ignored, unterminated; tree errors: Python bad/good, SQL bad/good; tab/space warn/clean;
dispatch routing).

---

## Task 7: Wire the linter into the editor (debounced, unobtrusive)

Add one `linter()` per editor, reading `lintProblems` on the built-in debounce. The
linter maps our `LintProblem[]` to CodeMirror `Diagnostic[]`. Diagnostics are pure
decorations, so this cannot move the cursor, cannot change strings/comments, and needs
no undo. `delay: 400` keeps it off the keystroke path; heavier truth still surfaces in
the console on execute.

**Files:**
- Modify: `web/components/run/CodeEditor.tsx` (`LoadedEditor`, `loadCodeMirror`, the
  extensions array, one new builder)

**Interfaces:**
- Consumes: `lintProblems` (via `langStructure`); `@codemirror/lint`'s `linter`.
- Produces: `lintExtension(lintMod, langStructure, languageId): Extension` in
  `CodeEditor.tsx`; `LoadedEditor.lint: typeof import("@codemirror/lint")`.

- [ ] **Step 1: Add `@codemirror/lint` to the lazy load**

In `web/components/run/CodeEditor.tsx`, extend the `LoadedEditor` interface (around
line 184):

```typescript
  lint: typeof import("@codemirror/lint");
```

In `loadCodeMirror` (around line 198), add the import to the `Promise.all` and the
return object:

```typescript
  const [cm, view, lang, state, autocomplete, highlight, inline, langStructure, lint, langExt] =
    await Promise.all([
      import("codemirror"),
      import("@codemirror/view"),
      import("@codemirror/language"),
      import("@codemirror/state"),
      import("@codemirror/autocomplete"),
      import("@lezer/highlight"),
      import("@/lib/sandbox/inline-completion"),
      import("@/lib/sandbox/lang-structure"),
      import("@codemirror/lint"),
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
    lint,
    langExt,
  };
```

- [ ] **Step 2: Add the linter builder**

Add near `indentExtensions` in `CodeEditor.tsx`:

```typescript
/** An unobtrusive, debounced linter: underlines obvious problems (R bracket/quote
 *  balance; Python and SQL parser error nodes; Python tab/space mixing). It only
 *  produces diagnostics, so it never edits the document, never moves the cursor, and
 *  needs no undo. Heavier truth surfaces in the console on execute. */
function lintExtension(
  lintMod: LoadedEditor["lint"],
  langStructure: LoadedEditor["langStructure"],
  languageId: string,
): Extension {
  const lang: "r" | "python" | "sql" =
    languageId === "python" ? "python" : languageId === "sql" ? "sql" : "r";
  return lintMod.linter(
    (view) =>
      langStructure.lintProblems(view.state.doc.toString(), lang).map((p) => ({
        from: p.from,
        to: p.to,
        severity: p.severity,
        message: p.message,
      })),
    { delay: 400 },
  );
}
```

- [ ] **Step 3: Insert the linter into the editor build**

In the mount effect's `.then(...)` destructure (line 90), add `lint`; then add the
extension to the array (after the run keymap block, before the completion blocks):

```typescript
      .then(({ view, cm, lang, state, autocomplete, tags, langExt, inline, langStructure, lint }) => {
```

and in the extensions array:

```typescript
            lintExtension(lint, langStructure, props.languageId),
```

- [ ] **Step 4: Typecheck and lint the source**

Run: `npm run typecheck`
Expected: no errors.

Run: `npm run lint`
Expected: no errors for the changed files.

- [ ] **Step 5: Confirm the full unit suite still passes**

Run: `npm run test -- indent lint`
Expected: PASS for both files.

---

## Task 8: End-to-end proof (Enter indentation and a lint marker)

Prove real editor behavior: pressing Enter inside a continuing statement indents the
new line for R, Python, and SQL, and an obvious error draws a lint underline. All
deterministic: no runtime execution, no streaming, no network. Indentation and lint
are pure client-side CodeMirror behavior, so these tests never touch a WASM runtime.

**Files:**
- Modify: `web/tests/e2e/sandbox.spec.ts`

- [ ] **Step 1: Add a shared helper to read a rendered editor line**

At the top of the sandbox spec's describe block (or near the existing helpers), add
a helper that reads the exact text (including leading spaces) of a rendered CodeMirror
line:

```typescript
  /** The exact textContent of the nth rendered editor line (0-based), spaces kept. */
  async function editorLineText(page: import("@playwright/test").Page, n: number) {
    return page.locator(".cm-line").nth(n).evaluate((el) => el.textContent ?? "");
  }
```

- [ ] **Step 2: Add the R pipe indentation test**

```typescript
  test("Enter after an R pipe indents the continuation line", async ({ page }) => {
    await page.goto("/ai-sandbox");
    await page.getByRole("radio", { name: "R" }).click();
    const editor = page.getByRole("textbox", { name: /R code/i });
    await expect(editor).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("df |>");
    await page.keyboard.press("Enter");
    await page.keyboard.type("filter(x)");

    // The second line was auto-indented two spaces before "filter(x)".
    expect(await editorLineText(page, 1)).toBe("  filter(x)");
  });
```

- [ ] **Step 3: Add the Python def indentation test**

```typescript
  test("Enter after a Python colon header indents four spaces", async ({ page }) => {
    await page.goto("/ai-sandbox");
    await page.getByRole("radio", { name: "Python" }).click();
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(editor).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("def f(x):");
    await page.keyboard.press("Enter");
    await page.keyboard.type("return x");

    expect(await editorLineText(page, 1)).toBe("    return x");
  });
```

- [ ] **Step 4: Add the SQL SELECT indentation test**

```typescript
  test("Enter after SQL SELECT indents the column list", async ({ page }) => {
    await page.goto("/ai-sandbox");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("SELECT");
    await page.keyboard.press("Enter");
    await page.keyboard.type("a,");

    expect(await editorLineText(page, 1)).toBe("  a,");
  });
```

- [ ] **Step 5: Add the lint-marker test (obvious R error)**

```typescript
  test("an unclosed R bracket draws a lint underline", async ({ page }) => {
    await page.goto("/ai-sandbox");
    await page.getByRole("radio", { name: "R" }).click();
    const editor = page.getByRole("textbox", { name: /R code/i });
    await expect(editor).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("x <- (1 + 2");

    // The linter runs on a 400ms debounce and underlines the unclosed "(".
    await expect(page.locator(".cm-lintRange-error").first()).toBeVisible({
      timeout: 5000,
    });
  });
```

- [ ] **Step 6: Run the new e2e tests**

Run: `npm run test:e2e -- sandbox.spec.ts -g "indent|indents|lint underline"`
Expected: PASS. R shows `  filter(x)`, Python shows `    return x`, SQL shows `  a,`,
and the lint underline appears for the unclosed bracket.

Note on the typed keystrokes: `page.keyboard.type` sends each character, so CodeMirror
runs `insertNewlineAndIndent` on `Enter` (consulting the new indentService) exactly as
a real user would. If a rendered `.cm-line` collapses its whitespace in a given browser
build, switch the assertion to read the document via the editor's value: the sandbox is
controlled, so the typed text round-trips through React; assert on the last-known value
instead. Do not weaken the assertion to `toContainText`, which would not verify the
space count.

- [ ] **Step 7: Full slice verification (no commit)**

Run: `npm run test -- indent lint`
Expected: both unit files pass.

Run: `npm run typecheck`
Expected: no errors.

Run: `npm run lint`
Expected: no errors for the changed files.

Run: `npm run test:e2e -- sandbox.spec.ts`
Expected: the full sandbox suite passes, including the four new tests and the existing
shell/axe tests. Leave the working tree uncommitted.

---

## Self-Review

**Spec coverage (requirement E and the general acceptance criteria):**

- **R indent:** indent after opening `(`/`[`/`{` (bracket depth term); continue after
  `|>`/`%>%` (trailing-op term); continue after a ggplot `+` (trailing-op term, read
  from the mask so a `+` in a string/comment does not count); align args in multi-line
  calls (depth term keeps sibling args level); dedent before a closing bracket (the
  closer look-ahead); no continuation indent when the statement is complete (base only)
  — Task 2 tests, all passing on the requirement's snippet shapes.
- **Python indent:** indent after `:`; preserve block indent; dedent when a block ends;
  indent inside brackets; consistent 4-space, no tab/space — delivered by `python()`'s
  tree indentation plus `indentUnit.of("    ")`, proven in Task 4 with the real
  `getIndentation`, never reimplemented, so behavior cannot drift.
- **SQL indent:** items under SELECT; conditions under WHERE/HAVING/ON; CTE contents;
  nested subqueries (paren depth term); dedent major clauses FROM/WHERE/GROUP BY/HAVING/
  ORDER BY/LIMIT (governing-clause term) — Task 3 tests. Honest limits (no
  character-precise alignment, SQLite-only) are documented.
- **Linting:** flags obvious problems (R bracket/quote balance; Python/SQL parser error
  nodes; Python tab/space mixing) with unobtrusive underlines; does not aggressively
  rewrite (diagnostics only); does not move the cursor (no document change); does not
  change quoted text/comments/strings (R reads the mask; tree parsers keep strings in
  String nodes); undoable (nothing to undo); does not run a full-document formatter per
  keystroke (`delay: 400`) — Tasks 6, 7, and the Task 8 lint-marker test.
- **Continuation indent on Enter; heavier lint on a pause:** indent is an
  `indentService` consulted by `insertNewlineAndIndent`; lint is a debounced `linter`;
  real syntax/runtime errors still surface in the console on execute — the split the
  requirement asks for.
- **Never change behavior via auto-indent (especially Python):** Python has no custom
  service; indent only fires on Enter; no formatter runs.
- **No em dashes** in the two diagnostic messages and the JSDoc copy.

**Reuse of Slice 2 (explicit):**

- `scanR` + its `RLine` table (bracket `depthAtStart`, `endsWithContinuation`,
  `startsWithContinuation`, `blank`) drive `rIndentColumns`: bracket depth gives the
  open-bracket indent and close dedent; the continuation flags find the statement's
  base line; the trailing-op check gives the pipe/ggplot continuation. `rBalanceProblems`
  reuses the same understanding through the mask.
- `maskStringsAndComments` is reused by `rIndentColumns`, `sqlIndentColumns`, and
  `rBalanceProblems`, so no indent or lint decision is ever influenced by a bracket,
  operator, `;`, or quote inside a string or comment.
- The `@lezer/python` grammar (the one Slice 2's `py.ts` parses) supplies Python
  indentation via `python()` and Python syntax diagnostics via `treeErrorProblems`.
  The `SQLite` parser (Slice 2's `sql.ts`) supplies SQL diagnostics. `statementNodeAt`
  remains available for precise SQL statement bounds if the mask scan ever needs
  tightening.

**Whether `@codemirror/lint` needs adding:** it is already resolvable in `node_modules`
(6.9.7, via the `codemirror` meta-package), but it is **not** declared in
`package.json`; Task 1 adds it as an explicit dependency because this slice imports it
directly. No install download is expected.

**Constraints from `CodeEditor.tsx` that shaped the design:**

- The editor rebuilds when `props.languageId` changes (dependency array, line 139), so
  passing `languageId` into `indentExtensions`/`lintExtension` at build time is safe: a
  language switch tears down and rebuilds with the correct services.
- CodeMirror is lazy-loaded via `loadCodeMirror`; adding `@codemirror/lint` and reaching
  the indent/lint helpers through the already-lazy `langStructure` keeps them out of the
  initial bundle. The pure helpers never import `@codemirror/view`, so they also load in
  the Vitest `node` environment.
- `basicSetup` already binds Enter to `insertNewlineAndIndent` and includes
  `indentOnInput()`; our `indentService` is consulted first by `getIndentation`, so it
  wins for R and SQL without touching the keymap.

**Placeholder scan:** no TBDs; every code step shows final code; commands are exact and
adapted to the no-commit rule (verify steps replace commit steps). The two contingencies
(Python `getIndentation` under Node; `.cm-line` whitespace rendering) each have an inline
fallback that does not weaken coverage.

**Type consistency:** `rIndentColumns(text, pos): number`, `sqlIndentColumns(text, pos):
number`, `R_INDENT_UNIT`/`SQL_INDENT_UNIT` numbers; `LintProblem = { from; to; severity:
"error" | "warning"; message }`; `lintProblems(text, languageId): LintProblem[]`. The
`CodeEditor` builders narrow `languageId: string` to the `"r" | "python" | "sql"` union
before calling. Editor accessible names in the e2e (`/R code/i`, `/Python code/i`,
`/SQL code/i`) match `label={`${language.label} code`}` (Sandbox.tsx:660); radios match
`role="radiogroup"` (Sandbox.tsx:766). The lint underline class `.cm-lintRange-error` is
the standard `@codemirror/lint` decoration.

## Handoff note

Per the task that produced this plan, this is a planning document only; do not build
from it in this session. When execution is scheduled, use
superpowers:subagent-driven-development (a fresh subagent per task, review between) or
superpowers:executing-plans (batched with checkpoints), and remember: no git commits,
verify with the commands above instead. Build order within the slice: Task 1 (declare
lint) first, then Tasks 2-4 (R indent, SQL indent, Python indent proof, independent of
each other), Task 5 (wire indent), Task 6 (lint finders), Task 7 (wire lint), Task 8
(e2e and full verification). This slice depends on Slice 2 (D) having landed, since it
imports `scanR`, `maskStringsAndComments`, and the Lezer helpers from
`web/lib/sandbox/lang-structure/`.
