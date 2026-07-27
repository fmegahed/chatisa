# Slice 2 — D: Ctrl/Cmd+Enter runs the complete logical statement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is a planning document only; do not build from it in the session that produced it.

**Goal:** When there is no selection, Ctrl/Cmd+Enter runs the complete logical
statement containing the cursor (not one physical line), consistently for R,
Python, and SQL. A multi-line pipe chain, a ggplot `+` chain, a multi-line
bracketed call, a Python block (with its connected `elif/else`,
`except/finally`, and decorators), and a SQL statement (CTE plus its final
SELECT, nested subqueries, one statement out of a multi-statement script) each
run whole. Detection ignores operators, semicolons, brackets, keywords, and
continuation characters that appear inside strings or comments. Selection-first
behavior is unchanged.

**Architecture.** The detection logic is a pure, string-in / range-out module,
`web/lib/sandbox/lang-structure/`, that has no CodeMirror-view dependency and is
unit-tested with Vitest. Python and SQL reuse their existing Lezer grammars: the
module imports the Lezer parser directly (`@lezer/python`'s `parser`, and
`SQLite.language.parser` from `@codemirror/lang-sql`), parses the document text,
and walks up from the cursor node to the enclosing top-level statement. R has no
grammar (it runs on a legacy CodeMirror stream mode), so R uses a bespoke,
string/comment-aware line scanner that tracks bracket depth and continuation
operators. The only editor change is the `Mod-Enter` no-selection branch in
`CodeEditor.tsx`, which calls `statementRangeAt(...)` instead of reading one
line. The module is deliberately shaped so Slice 3 (E, continuation indentation)
reuses the same per-language structure helpers.

**Grounding (verified against the installed packages).**

- Python `@lezer/python` `parser.parse(src)` produces a `Script` whose direct
  children are the top-level statements. A decorated function is a single
  `DecoratedStatement` node wrapping `Decorator` + `FunctionDefinition`. An
  `IfStatement` node already contains its `elif`/`else`; a `TryStatement`
  contains its `except`/`else`/`finally`. So "climb to the node whose parent is
  `Script`" yields the whole connected block in one step. Verified:
  `@dec\ndef f(x):\n    return x` parses to
  `Script > DecoratedStatement(Decorator, FunctionDefinition(...))`.
- SQL `SQLite.language.parser.parse(src)` produces a `Script` whose children are
  `Statement` nodes (each including its trailing `;`), interleaved with
  `LineComment`/`BlockComment` nodes. A `WITH ... SELECT` CTE is one `Statement`;
  a nested subquery is a `Parens` node inside it; a `;` inside a string is inside
  a `String` node, never a top-level statement boundary. Verified: from a cursor
  inside a nested subquery inside a CTE, walking up to `Statement` returns the
  entire `WITH a AS (...) SELECT * FROM a;` and excludes the following
  `SELECT 99;`.
- Both Lezer parsers import and run in a plain Node (Vitest `environment: "node"`)
  process with no DOM, so the pure module is unit-testable without a browser.
- R uses `StreamLanguage.define(r)` (CodeEditor.tsx:260-267); `syntaxTree` yields
  no useful structure, confirming the bespoke scanner is required.

**Tech Stack:** Next.js (this repo's vendored build — read
`node_modules/next/dist/docs/` before touching any Next API; this slice does
not), React 19, CodeMirror 6 (`@codemirror/view`, `@codemirror/state`,
`@codemirror/language`), the Lezer parsers `@lezer/python` (transitive via
`@codemirror/lang-python`) and `@codemirror/lang-sql` (both already
dependencies), Vitest for the pure unit tests, Playwright +
`@axe-core/playwright` for the sandbox e2e.

## Global Constraints

- **No git commits.** The working tree stays uncommitted; each task ends by
  running verification commands instead of committing.
- **Pure logic is separate from CodeMirror glue.** Everything in
  `web/lib/sandbox/lang-structure/` takes `(text: string, pos: number)` and
  returns offsets. It must not import `@codemirror/view` or touch an
  `EditorView`. This keeps it testable in Vitest's `node` environment and reusable
  by Slice 3.
- **Ignore strings and comments in all three languages.** Python and SQL get this
  for free from the Lezer tree (a `;`/operator/bracket inside a `String` or
  comment node is not a boundary). R gets it from the scanner's lexical mask.
- **Selection-first is unchanged.** Only the no-selection branch of `Mod-Enter`
  changes. When text is selected, the selection still runs verbatim.
- **Never run less than one physical line.** Every per-language function falls
  back to the physical line at the cursor when it cannot resolve a statement, so
  behavior is never worse than today.
- **No em dashes in any user-facing copy** (button titles, tooltips, the toolbar
  hint text). Use commas, colons, or sentence breaks.
- **Keyboard shortcuts support Ctrl (Windows/Linux) and Cmd (macOS).** The binding
  is already `Mod-Enter`, which CodeMirror maps to both; no change needed there.
- **Consistent across R, Python, and SQL.** The dispatch in `index.ts` routes by
  `languageId`; the e2e proves real execution.
- All commands run from `web/` (the Next app root). Unit tests:
  `npm run test`. The Playwright config starts its own dev server on port 3100,
  so no separate server needs starting for `npm run test:e2e`.

## File Structure

- `web/lib/sandbox/lang-structure/types.ts` — Create. Shared types: `LanguageId`,
  `StatementRange` (`{ from: number; to: number; nextPos: number }`).
- `web/lib/sandbox/lang-structure/mask.ts` — Create. `maskStringsAndComments(text,
  languageId)`: a same-length "code view" string where string and comment
  characters are replaced by placeholders, so any bracket/operator/`;` scan
  ignores strings and comments. Used by the R scanner now and by Slice 3's R and
  SQL indentation later.
- `web/lib/sandbox/lang-structure/r-scan.ts` — Create. The bespoke R scanner:
  `scanR(text)` (line table + mask, the reusable primitive) and
  `rStatementRange(text, pos)`.
- `web/lib/sandbox/lang-structure/py.ts` — Create. `pyStatementRange(text, pos)`
  and the exported helper `topLevelStatementAt(tree, pos)` (reused by Slice 3 for
  block/indent context).
- `web/lib/sandbox/lang-structure/sql.ts` — Create. `sqlStatementRange(text, pos)`
  and the exported helper `statementNodeAt(tree, pos)`.
- `web/lib/sandbox/lang-structure/index.ts` — Create. `statementRangeAt(text, pos,
  languageId): StatementRange` dispatch, plus `nextStatementPos(text, to)`, and
  re-exports of the per-language helpers for Slice 3.
- `web/components/run/CodeEditor.tsx` — Modify. Pass `languageId` and the loaded
  `statementRangeAt` into `runKeymap`; rewrite the `Mod-Enter` no-selection branch
  to use it; update the `onRunLine` JSDoc.
- `web/components/sandbox/Sandbox.tsx` — Modify. Update the Run button title copy
  (line 797, "runs the current line") and the toolbar hint (line 812,
  "{mod}+Enter: line") to say "statement".
- `web/tests/unit/lang-structure.test.ts` — Create. Pure Vitest tables for R,
  Python, and SQL using the requirement's example snippets.
- `web/tests/e2e/sandbox.spec.ts` — Modify. Add SQLite real-execution e2e tests
  that run a multi-line statement via Ctrl+Enter from a middle line.

---

## Task 1: Shared types and the string/comment mask

The mask is the foundation the R scanner stands on and the primitive Slice 3
reuses. `maskStringsAndComments(text, languageId)` returns a string of the same
length as `text` in which:

- comment characters become a single space `' '` (so a masked comment is blank),
- string delimiter and content characters become the placeholder `'x'` (so a
  string still counts as "code present" on its line but exposes no operators,
  brackets, or `;`),
- every other character (including newlines and real code) is copied verbatim.

Because offsets are preserved 1:1, callers scan the mask and index back into the
original text with the same positions.

**Files:**
- Create: `web/lib/sandbox/lang-structure/types.ts`,
  `web/lib/sandbox/lang-structure/mask.ts`
- Test: `web/tests/unit/lang-structure.test.ts` (mask section)

**Interfaces:**
- Produces: `LanguageId = "r" | "python" | "sql"`; `StatementRange = { from:
  number; to: number; nextPos: number }`; `maskStringsAndComments(text: string,
  languageId: LanguageId): string`.

- [ ] **Step 1: Write the failing mask tests**

Create `web/tests/unit/lang-structure.test.ts` with:

```typescript
import { describe, expect, it } from "vitest";
import { maskStringsAndComments } from "@/lib/sandbox/lang-structure/mask";

describe("maskStringsAndComments", () => {
  it("blanks comments and neutralizes strings, preserving length and offsets", () => {
    const src = 'x <- "a + b" # tail + comment';
    const m = maskStringsAndComments(src, "r");
    expect(m.length).toBe(src.length);
    // Real code is preserved.
    expect(m.startsWith("x <- ")).toBe(true);
    // The "+" inside the string is hidden (placeholder, not an operator).
    expect(m.includes("+")).toBe(false);
    // The comment is blank (all spaces from '#').
    expect(m.slice(src.indexOf("#")).trim()).toBe("");
  });

  it("keeps a hash inside a string from starting a comment (R)", () => {
    const src = 'y <- "a#b" + 1';
    const m = maskStringsAndComments(src, "r");
    // The trailing "+ 1" is real code, still visible after the string.
    expect(m.trimEnd().endsWith("+ 1")).toBe(true);
  });

  it("handles escaped quotes inside R strings", () => {
    const src = 'z <- "he said \\"hi\\"" + 2';
    const m = maskStringsAndComments(src, "r");
    expect(m.trimEnd().endsWith("+ 2")).toBe(true);
  });

  it("blanks SQL line and block comments and neutralizes quoted semicolons", () => {
    const src = "SELECT ';' -- c;\n/* b;lock */ , 2";
    const m = maskStringsAndComments(src, "sql");
    expect(m.includes(";")).toBe(false); // no ; survives from string or comments
    expect(m.trimEnd().endsWith(", 2")).toBe(true);
  });
});
```

- [ ] **Step 2: Run the mask tests and confirm they fail**

Run: `npm run test -- lang-structure`
Expected: FAIL. The module does not exist yet (import error).

- [ ] **Step 3: Create the types**

Create `web/lib/sandbox/lang-structure/types.ts`:

```typescript
/** The three runnable Coding Studio languages. */
export type LanguageId = "r" | "python" | "sql";

/**
 * A resolved logical-statement range in a document.
 * `from`/`to` bound the statement (to excludes the trailing newline); `nextPos`
 * is where the caret advances after running (start of the next executable
 * statement, or the document end).
 */
export interface StatementRange {
  from: number;
  to: number;
  nextPos: number;
}
```

- [ ] **Step 4: Create the mask**

Create `web/lib/sandbox/lang-structure/mask.ts`:

```typescript
import type { LanguageId } from "./types";

const SPACE = " ";
const STRING = "x";

/**
 * Returns a same-length "code view" of `text` where comment characters are
 * spaces and string characters are the placeholder `x`. Offsets are preserved,
 * so a scan of the result indexes back into `text` at the same positions. This
 * lets bracket, operator, and semicolon detection ignore anything inside a
 * string or comment for R (and, for Slice 3, SQL indentation).
 *
 * String rules: R and Python use `"`, `'`, and backtick with backslash escapes
 * (backtick in R has no escape but closes on the next backtick, which is fine
 * here). SQL uses `'` (doubled `''` escape) and `"`/backtick as quoted
 * identifiers. Comment rules: `#` to end of line for R and Python; `--` to end
 * of line and `/* ... *​/` blocks for SQL. Python triple-quoted strings are
 * covered because the same delimiter opens and the scanner consumes until it
 * recurs; the Python and SQL statement finders use the Lezer tree and do not
 * rely on this mask, so triple-quote subtleties never affect them. The mask is
 * primarily for R.
 */
export function maskStringsAndComments(
  text: string,
  languageId: LanguageId,
): string {
  const out = new Array<string>(text.length);
  const lineComment = languageId === "sql" ? "--" : "#";
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    // Line comment.
    if (text.startsWith(lineComment, i)) {
      while (i < text.length && text[i] !== "\n") out[i++] = SPACE;
      continue;
    }
    // SQL block comment.
    if (languageId === "sql" && text.startsWith("/*", i)) {
      while (i < text.length && !text.startsWith("*/", i)) {
        out[i] = text[i] === "\n" ? "\n" : SPACE;
        i++;
      }
      if (i < text.length) {
        out[i++] = SPACE;
        if (i < text.length) out[i++] = SPACE; // the closing "/"
      }
      continue;
    }
    // String / quoted identifier.
    if (ch === '"' || ch === "'" || ch === "`") {
      const quote = ch;
      const sqlDoubled = languageId === "sql" && quote === "'";
      out[i++] = STRING; // opening quote
      while (i < text.length) {
        const c = text[i];
        if (c === "\\" && languageId !== "sql") {
          out[i] = STRING;
          if (i + 1 < text.length) out[i + 1] = STRING;
          i += 2;
          continue;
        }
        if (c === quote) {
          if (sqlDoubled && text[i + 1] === "'") {
            out[i] = STRING;
            out[i + 1] = STRING;
            i += 2;
            continue;
          }
          out[i++] = STRING; // closing quote
          break;
        }
        out[i] = c === "\n" ? "\n" : STRING;
        i++;
      }
      continue;
    }
    out[i] = ch;
    i++;
  }
  return out.join("");
}
```

- [ ] **Step 5: Run the mask tests and confirm they pass**

Run: `npm run test -- lang-structure`
Expected: PASS for the four mask tests.

---

## Task 2: Python statement range (Lezer tree walk)

Parse with `@lezer/python`'s `parser`, resolve the node at the cursor, and climb
to the node whose parent is `Script`. That single node is the complete top-level
logical statement: for a compound block it already contains all connected
clauses (`IfStatement` holds its `elif`/`else`, `TryStatement` holds its
`except`/`else`/`finally`), and a decorated definition is a single
`DecoratedStatement`. Multi-line expressions in `()`/`[]`/`{}`, method chains,
comprehensions, and triple-quoted strings are inside their statement node, so
they come along whole. A `+`/`;`/bracket inside a `String` or `Comment` node is
never a boundary, satisfying the ignore-strings-and-comments rule for free.

**Files:**
- Create: `web/lib/sandbox/lang-structure/py.ts`
- Test: `web/tests/unit/lang-structure.test.ts` (Python section)

**Interfaces:**
- Consumes: `import { parser } from "@lezer/python"`.
- Produces: `pyStatementRange(text: string, pos: number): { from: number; to:
  number }`; `topLevelStatementAt(tree, pos)` (exported for Slice 3).

- [ ] **Step 1: Write the failing Python tests**

Add to `web/tests/unit/lang-structure.test.ts`:

```typescript
import { pyStatementRange } from "@/lib/sandbox/lang-structure/py";

/** The statement text the finder would run for a cursor at `marker`. */
function pyAt(src: string, marker: string) {
  const pos = src.indexOf(marker);
  const { from, to } = pyStatementRange(src, pos);
  return src.slice(from, to);
}

describe("pyStatementRange", () => {
  it("runs a simple expression alone", () => {
    const src = "x = 1\nprint(x)\ny = 2\n";
    expect(pyAt(src, "print")).toBe("print(x)");
  });

  it("runs a def block whole from a line inside its body", () => {
    const src = "def f(x):\n    y = x + 1\n    return y\n\nf(3)\n";
    expect(pyAt(src, "return y")).toBe("def f(x):\n    y = x + 1\n    return y");
  });

  it("keeps a decorator with its function", () => {
    const src = "@cache\ndef g():\n    return 1\n";
    expect(pyAt(src, "return 1")).toBe("@cache\ndef g():\n    return 1");
  });

  it("keeps if / elif / else together", () => {
    const src = "if a:\n    p()\nelif b:\n    q()\nelse:\n    r()\nz = 1\n";
    expect(pyAt(src, "q()")).toBe(
      "if a:\n    p()\nelif b:\n    q()\nelse:\n    r()",
    );
  });

  it("keeps try / except / finally together", () => {
    const src = "try:\n    a()\nexcept E:\n    b()\nfinally:\n    c()\n";
    expect(pyAt(src, "b()")).toContain("finally:");
  });

  it("runs a multi-line bracketed expression whole from any line", () => {
    const src = "total = (\n    a\n    + b\n    + c\n)\n";
    expect(pyAt(src, "+ b")).toBe("total = (\n    a\n    + b\n    + c\n)");
  });

  it("ignores a colon inside a string", () => {
    const src = 's = "a: b"\nn = 2\n';
    expect(pyAt(src, "a: b")).toBe('s = "a: b"');
  });
});
```

- [ ] **Step 2: Run and confirm the Python tests fail**

Run: `npm run test -- lang-structure`
Expected: FAIL (`py.ts` does not exist).

- [ ] **Step 3: Implement `py.ts`**

Create `web/lib/sandbox/lang-structure/py.ts`:

```typescript
import { parser } from "@lezer/python";
import type { SyntaxNode, Tree } from "@lezer/common";

/**
 * The top-level statement node containing `pos`: the ancestor whose parent is
 * the `Script` root. Returns null only for an empty document or a cursor in
 * leading whitespace with no statement to attach to.
 */
export function topLevelStatementAt(tree: Tree, pos: number): SyntaxNode | null {
  // Bias left so a cursor at a line end stays in the statement it terminates;
  // if that lands on the Script root (blank line / gap), try biasing right, then
  // the nearest child before/after the position.
  let node: SyntaxNode | null = tree.resolveInner(pos, -1);
  if (node.name === "Script") node = tree.resolveInner(pos, 1);
  if (node.name === "Script") {
    node = tree.topNode.childBefore(pos) ?? tree.topNode.childAfter(pos);
  }
  if (!node) return null;
  while (node.parent && node.parent.name !== "Script") node = node.parent;
  return node.parent ? node : null; // must be a direct child of Script
}

/**
 * The range of the complete Python logical statement/block containing `pos`.
 * Falls back to the physical line when no statement resolves, so the caller
 * never runs less than a line.
 */
export function pyStatementRange(
  text: string,
  pos: number,
): { from: number; to: number } {
  const tree = parser.parse(text);
  const node = topLevelStatementAt(tree, pos);
  if (node) return { from: node.from, to: node.to };
  return physicalLine(text, pos);
}

function physicalLine(text: string, pos: number): { from: number; to: number } {
  let from = pos;
  while (from > 0 && text[from - 1] !== "\n") from--;
  let to = pos;
  while (to < text.length && text[to] !== "\n") to++;
  return { from, to };
}
```

Note: `@lezer/common` is already present transitively (every Lezer grammar
depends on it); if `tsc` cannot resolve the type-only import, change it to
`import type { SyntaxNode, Tree } from "@lezer/python"`'s re-exported
`parser.parse` return type, or drop the annotations and let inference cover it.

- [ ] **Step 4: Run and confirm the Python tests pass**

Run: `npm run test -- lang-structure`
Expected: PASS for all `pyStatementRange` cases (simple expression, def block
from inside the body, decorator, if/elif/else, try/except/finally, multi-line
bracketed expression, colon-in-string).

---

## Task 3: SQL statement range (Lezer tree walk)

Parse with `SQLite.language.parser`, resolve the node at the cursor, and climb to
the enclosing `Statement` node. A `WITH ... SELECT` CTE is one `Statement`; a
nested subquery is a `Parens` inside it; `;` inside a `String` is not a boundary.
A multi-statement script has one `Statement` per top-level `;`, so only the
statement at the cursor runs.

**Files:**
- Create: `web/lib/sandbox/lang-structure/sql.ts`
- Test: `web/tests/unit/lang-structure.test.ts` (SQL section)

**Interfaces:**
- Consumes: `import { SQLite } from "@codemirror/lang-sql"`; parse via
  `SQLite.language.parser.parse(text)` (verified accessor in the installed
  `@codemirror/lang-sql` build).
- Produces: `sqlStatementRange(text: string, pos: number): { from: number; to:
  number }`; `statementNodeAt(tree, pos)` (exported for Slice 3).

- [ ] **Step 1: Write the failing SQL tests**

Add to `web/tests/unit/lang-structure.test.ts`:

```typescript
import { sqlStatementRange } from "@/lib/sandbox/lang-structure/sql";

function sqlAt(src: string, marker: string) {
  const pos = src.indexOf(marker);
  const { from, to } = sqlStatementRange(src, pos);
  return src.slice(from, to).trim();
}

describe("sqlStatementRange", () => {
  it("runs a single statement to its semicolon", () => {
    const src = "SELECT 1;\nSELECT 2;\n";
    expect(sqlAt(src, "SELECT 1")).toBe("SELECT 1;");
  });

  it("runs a CTE and its final SELECT together, from a line inside the CTE", () => {
    const src =
      "WITH a AS (\n  SELECT n FROM t\n)\nSELECT * FROM a;\nSELECT 9;\n";
    expect(sqlAt(src, "SELECT n FROM t")).toBe(
      "WITH a AS (\n  SELECT n FROM t\n)\nSELECT * FROM a;",
    );
  });

  it("runs the whole parent statement from inside a nested subquery", () => {
    const src = "SELECT * FROM t WHERE n > (SELECT AVG(n) FROM t);\nSELECT 9;\n";
    expect(sqlAt(src, "AVG(n)")).toBe(
      "SELECT * FROM t WHERE n > (SELECT AVG(n) FROM t);",
    );
  });

  it("runs only the statement at the cursor in a multi-statement script", () => {
    const src = "CREATE TABLE t(n);\nINSERT INTO t VALUES (1);\nSELECT * FROM t;\n";
    expect(sqlAt(src, "INSERT")).toBe("INSERT INTO t VALUES (1);");
  });

  it("does not split on a semicolon inside a string literal", () => {
    const src = "SELECT ';' AS s, 2 AS n;\nSELECT 3;\n";
    expect(sqlAt(src, "AS s")).toBe("SELECT ';' AS s, 2 AS n;");
  });
});
```

- [ ] **Step 2: Run and confirm the SQL tests fail**

Run: `npm run test -- lang-structure`
Expected: FAIL (`sql.ts` does not exist).

- [ ] **Step 3: Implement `sql.ts`**

Create `web/lib/sandbox/lang-structure/sql.ts`:

```typescript
import { SQLite } from "@codemirror/lang-sql";
import type { SyntaxNode, Tree } from "@lezer/common";

const sqlParser = SQLite.language.parser;

/**
 * The `Statement` node containing `pos`, or (for a cursor in a gap between
 * statements or in a comment) the nearest `Statement`. Returns null only for an
 * empty document.
 */
export function statementNodeAt(tree: Tree, pos: number): SyntaxNode | null {
  let node: SyntaxNode | null = tree.resolveInner(pos, -1);
  while (node && node.name !== "Statement" && node.parent) node = node.parent;
  if (node && node.name === "Statement") return node;
  // Cursor sits between statements (blank line/comment): pick the nearest one.
  const before = tree.topNode.childBefore(pos);
  if (before && before.name === "Statement") return before;
  const after = tree.topNode.childAfter(pos);
  if (after && after.name === "Statement") return after;
  return null;
}

/**
 * The range of the complete SQL statement containing `pos` (through its `;` or
 * the end of the script). Falls back to the physical line when nothing resolves.
 */
export function sqlStatementRange(
  text: string,
  pos: number,
): { from: number; to: number } {
  const tree = sqlParser.parse(text);
  const node = statementNodeAt(tree, pos);
  if (node) return { from: node.from, to: node.to };
  let from = pos;
  while (from > 0 && text[from - 1] !== "\n") from--;
  let to = pos;
  while (to < text.length && text[to] !== "\n") to++;
  return { from, to };
}
```

Note on procedural blocks (`BEGIN ... END`, `CASE ... END`, custom delimiters):
the requirement flags these as dialect-dependent, and the in-browser engine is
SQLite (`sqlite-wasm`), where a student cannot run a stored procedure or custom
delimiter script. The grammar treats a top-level `;` as the boundary, but a `;`
inside a string or comment is inside a `String`/comment node and is not a
boundary (proven by the string test), which is the case that actually matters
here. `CASE ... END` and a parenthesised subquery never contain a top-level `;`,
so they already stay inside one `Statement`. Full `BEGIN ... END` block handling
is out of scope for SQLite and noted as such.

- [ ] **Step 4: Run and confirm the SQL tests pass**

Run: `npm run test -- lang-structure`
Expected: PASS for all `sqlStatementRange` cases (single statement, CTE from
inside, nested subquery, multi-statement selection, semicolon-in-string).

---

## Task 4: R bespoke scanner (the core of this slice)

R has no syntax tree, so R statement detection is a hand-written, string/comment
aware line scanner. This is the largest and highest-risk part; it gets the
fullest test table.

### The scanning algorithm (concrete)

**Phase 1 — mask.** Run `maskStringsAndComments(text, "r")` (Task 1). Everything
below reads the mask, never the raw text, so operators, brackets, `#`, and `+`
inside strings or comments are invisible. Offsets map 1:1 back to `text`.

**Phase 2 — line table.** Split the mask on `\n` into lines, tracking each line's
absolute `from`/`to` offsets in the document. Walk the mask left to right
maintaining a running bracket depth: `+1` for each `(`, `[`, `{` and `-1` for each
`)`, `]`, `}` seen in the mask (string/comment brackets are already masked away,
so they do not count). For each line record:

- `from`, `to` (document offsets; `to` excludes the newline).
- `blank`: the mask slice for the line is all whitespace.
- `depthAtStart`: bracket depth at the line's first character.
- `endsWithContinuation`: on the right-trimmed mask slice, the final token is a
  binary or pipe operator that cannot end an expression. Match:
  `/(\|>|%[^%\s]*%|<<-|<-|->>|->|[-+*/^~:?<>=&|!,])$/`. Closing brackets
  `)` `]` `}` are deliberately excluded: a line ending in `)` is complete.
- `startsWithContinuation`: on the left-trimmed mask slice, the first token is a
  continuation that binds to the previous line. Match:
  `/^(\|>|%[^%\s]*%|\+)/`. Kept conservative: pipes, `%...%` operators, and a
  leading `+` (the ggplot leading-plus style). Leading `-`/`*` are ambiguous
  (unary, dereference) and are not treated as continuations; the trailing-operator
  rule and bracket depth cover the normal cases.

**Phase 3 — statement starts.** A non-blank line `L` starts a new statement iff
all of:

- `depthAtStart(L) === 0` (not inside an open bracket), and
- the previous non-blank line `P` (if any) does not have `endsWithContinuation`,
  and
- `L` does not have `startsWithContinuation`.

If any condition fails, `L` continues the current statement. A line with
`depthAtStart > 0` is always a continuation (it is inside brackets opened above),
which is what carries multi-line `tibble(...)` / `c(...)` / `{ ... }` bodies.

**Phase 4 — resolve the cursor.**

1. `cursorLine` = the line containing `pos`. If it is blank, snap to the nearest
   non-blank line: prefer the previous non-blank line (RStudio-style: run the
   statement just above), else the next non-blank line. If the document has no
   non-blank line, fall back to the physical line at `pos`.
2. `startLine` = walk backward over non-blank lines from `cursorLine` until a line
   that starts a new statement (Phase 3). Because a continuation line never
   "starts new", this lands on the first line of the chain (the `df` before a
   `|>`, the `ggplot(...)` before the `+`, the line that opened the bracket).
3. `endLine` = walk forward: while the next non-blank line `N` after the current
   end does not start a new statement, extend the end to `N`. A line inside open
   brackets or after a trailing continuation operator never starts new, so the
   chain extends to its natural close; the walk stops at the next real statement
   or EOF.
4. `from` = the offset of the first non-whitespace character on `startLine`
   (trim leading indentation). `to` = `endLine.to`.

**Phase 5 — optional same-line `;` split.** R rarely uses `;`, but if the resolved
single-line statement contains one or more top-level `;` in the mask (depth 0),
split it into segments at those semicolons and return the segment containing
`pos`. This keeps `a; b` on one line running only the half at the cursor. Skip
this when the statement spans multiple lines.

`nextPos` (the caret advance) is computed by the caller in Task 6 via
`nextStatementPos(text, to)`: skip whitespace and blank lines after `to` to the
first non-whitespace character, snap to its line start; if none, the document
end.

**Why trailing-operator plus bracket depth is enough.** R's own rule for
continuing a line is "is the expression syntactically incomplete at the
newline?", which reduces to an open bracket or a dangling binary operator. The
scanner detects exactly those two, plus the leading-token style some students
use, and reads them from the mask so string/comment content can never trigger a
false continuation. This is the distinction the requirement calls out: a
trailing `+` that is a ggplot continuation versus a `+` inside a string or
comment.

**Files:**
- Create: `web/lib/sandbox/lang-structure/r-scan.ts`
- Test: `web/tests/unit/lang-structure.test.ts` (R section)

**Interfaces:**
- Consumes: `maskStringsAndComments` from `./mask`.
- Produces: `scanR(text): { lines: RLine[]; mask: string }` (the reusable
  primitive for Slice 3); `rStatementRange(text: string, pos: number): { from:
  number; to: number }`. `RLine = { from; to; blank; depthAtStart;
  endsWithContinuation; startsWithContinuation }`.

- [ ] **Step 1: Write the failing R tests**

Add to `web/tests/unit/lang-structure.test.ts`:

```typescript
import { rStatementRange } from "@/lib/sandbox/lang-structure/r-scan";

function rAt(src: string, marker: string) {
  const pos = src.indexOf(marker);
  const { from, to } = rStatementRange(src, pos);
  return src.slice(from, to);
}

describe("rStatementRange", () => {
  it("runs a single complete line alone", () => {
    const src = "x <- 1\ny <- 2\nz <- 3\n";
    expect(rAt(src, "y <- 2")).toBe("y <- 2");
  });

  it("runs a trailing-pipe chain whole from any line (|>)", () => {
    const src = "df |>\n  filter(x > 1) |>\n  summarise(n = n())\nz <- 1\n";
    expect(rAt(src, "filter")).toBe(
      "df |>\n  filter(x > 1) |>\n  summarise(n = n())",
    );
  });

  it("runs a magrittr pipe chain whole (%>%)", () => {
    const src = "df %>%\n  mutate(a = b) %>%\n  arrange(a)\n";
    expect(rAt(src, "arrange")).toBe("df %>%\n  mutate(a = b) %>%\n  arrange(a)");
  });

  it("runs a ggplot + chain through the final layer", () => {
    const src =
      "ggplot(d, aes(x, y)) +\n  geom_point() +\n  theme_bw()\nmsg <- 1\n";
    expect(rAt(src, "geom_point")).toBe(
      "ggplot(d, aes(x, y)) +\n  geom_point() +\n  theme_bw()",
    );
  });

  it("runs a multi-line bracketed call whole (tibble)", () => {
    const src = "t <- tibble(\n  a = 1,\n  b = 2\n)\nq <- 9\n";
    expect(rAt(src, "b = 2")).toBe("t <- tibble(\n  a = 1,\n  b = 2\n)");
  });

  it("does not treat a + inside a string as a continuation", () => {
    const src = 'lab <- "a + b"\ny <- 2\n';
    expect(rAt(src, "lab")).toBe('lab <- "a + b"');
  });

  it("does not treat a + inside a comment as a continuation", () => {
    const src = "x <- 1 # add + more\ny <- 2\n";
    expect(rAt(src, "x <- 1")).toBe("x <- 1 # add + more");
  });

  it("supports the leading-pipe style", () => {
    const src = "df\n  |> filter(x)\n  |> summarise(n())\n";
    expect(rAt(src, "filter")).toBe("df\n  |> filter(x)\n  |> summarise(n())");
  });

  it("runs the statement above when the cursor is on a blank line", () => {
    const src = "a <- 1\n\nb <- 2\n";
    const pos = src.indexOf("\n\n") + 1; // the blank line
    const { from, to } = rStatementRange(src, pos);
    expect(src.slice(from, to)).toBe("a <- 1");
  });
});
```

- [ ] **Step 2: Run and confirm the R tests fail**

Run: `npm run test -- lang-structure`
Expected: FAIL (`r-scan.ts` does not exist).

- [ ] **Step 3: Implement `r-scan.ts`**

Create `web/lib/sandbox/lang-structure/r-scan.ts`:

```typescript
import { maskStringsAndComments } from "./mask";

export interface RLine {
  from: number;
  to: number;
  blank: boolean;
  depthAtStart: number;
  endsWithContinuation: boolean;
  startsWithContinuation: boolean;
}

const TRAILING_OP = /(\|>|%[^%\s]*%|<<-|<-|->>|->|[-+*/^~:?<>=&|!,])$/;
const LEADING_OP = /^(\|>|%[^%\s]*%|\+)/;

/** Lexes R into a line table over the string/comment mask. Reused by Slice 3. */
export function scanR(text: string): { lines: RLine[]; mask: string } {
  const mask = maskStringsAndComments(text, "r");
  const lines: RLine[] = [];
  let depth = 0;
  let from = 0;
  for (let i = 0; i <= mask.length; i++) {
    if (i === mask.length || mask[i] === "\n") {
      const slice = mask.slice(from, i);
      const depthAtStart = depth;
      for (const ch of slice) {
        if (ch === "(" || ch === "[" || ch === "{") depth++;
        else if (ch === ")" || ch === "]" || ch === "}") depth = Math.max(0, depth - 1);
      }
      const trimmed = slice.trim();
      lines.push({
        from,
        to: i,
        blank: trimmed === "",
        depthAtStart,
        endsWithContinuation: TRAILING_OP.test(slice.replace(/\s+$/, "")),
        startsWithContinuation: LEADING_OP.test(slice.replace(/^\s+/, "")),
      });
      from = i + 1;
    }
  }
  return { lines, mask };
}

function startsNew(lines: RLine[], idx: number): boolean {
  const L = lines[idx];
  if (L.blank || L.depthAtStart > 0 || L.startsWithContinuation) return false;
  for (let p = idx - 1; p >= 0; p--) {
    if (lines[p].blank) continue;
    return !lines[p].endsWithContinuation;
  }
  return true;
}

/** The range of the complete R logical statement containing `pos`. */
export function rStatementRange(
  text: string,
  pos: number,
): { from: number; to: number } {
  const { lines, mask } = scanR(text);
  if (lines.length === 0) return { from: 0, to: 0 };

  let cur = lines.findIndex((l) => pos >= l.from && pos <= l.to);
  if (cur < 0) cur = lines.length - 1;

  // Blank line: snap to the previous non-blank, else the next non-blank.
  if (lines[cur].blank) {
    let up = cur - 1;
    while (up >= 0 && lines[up].blank) up--;
    if (up >= 0) cur = up;
    else {
      let down = cur + 1;
      while (down < lines.length && lines[down].blank) down++;
      if (down < lines.length) cur = down;
      else return physicalLine(text, pos);
    }
  }

  // Walk back to the statement start.
  let start = cur;
  while (start > 0 && !startsNew(lines, start)) {
    let p = start - 1;
    while (p >= 0 && lines[p].blank) p--;
    if (p < 0) break;
    start = p;
  }

  // Walk forward while the next non-blank line does not start a new statement.
  let end = start;
  let j = end + 1;
  while (j < lines.length) {
    if (lines[j].blank) {
      j++;
      continue;
    }
    if (startsNew(lines, j)) break;
    end = j;
    j++;
  }

  // Trim leading indentation on the start line.
  let f = lines[start].from;
  while (f < lines[start].to && /\s/.test(text[f])) f++;
  const range = { from: f, to: lines[end].to };

  // Single-line statement: honor top-level semicolons at the cursor.
  if (start === end) {
    const seg = splitTopLevelSemicolons(mask, range.from, range.to, pos);
    if (seg) return seg;
  }
  return range;
}

function splitTopLevelSemicolons(
  mask: string,
  from: number,
  to: number,
  pos: number,
): { from: number; to: number } | null {
  const stops: number[] = [];
  let depth = 0;
  for (let i = from; i < to; i++) {
    const ch = mask[i];
    if (ch === "(" || ch === "[" || ch === "{") depth++;
    else if (ch === ")" || ch === "]" || ch === "}") depth = Math.max(0, depth - 1);
    else if (ch === ";" && depth === 0) stops.push(i);
  }
  if (stops.length === 0) return null;
  const bounds = [from, ...stops, to];
  for (let k = 0; k < bounds.length - 1; k++) {
    const segFrom = k === 0 ? bounds[0] : bounds[k] + 1;
    const segTo = bounds[k + 1];
    if (pos >= segFrom && pos <= segTo) {
      let f = segFrom;
      while (f < segTo && /\s/.test(mask[f])) f++;
      return { from: f, to: segTo };
    }
  }
  return null;
}

function physicalLine(text: string, pos: number): { from: number; to: number } {
  let from = pos;
  while (from > 0 && text[from - 1] !== "\n") from--;
  let to = pos;
  while (to < text.length && text[to] !== "\n") to++;
  return { from, to };
}
```

- [ ] **Step 4: Run and confirm the R tests pass**

Run: `npm run test -- lang-structure`
Expected: PASS for all `rStatementRange` cases (single line, `|>` chain, `%>%`
chain, ggplot `+` chain, multi-line tibble, plus-in-string, plus-in-comment,
leading-pipe style, blank-line snap).

---

## Task 5: The dispatch and next-statement helpers

`statementRangeAt` routes by language and adds `nextPos` so the editor can
advance the caret. This is the one function `CodeEditor.tsx` calls.

**Files:**
- Create: `web/lib/sandbox/lang-structure/index.ts`
- Test: `web/tests/unit/lang-structure.test.ts` (dispatch section)

**Interfaces:**
- Produces: `statementRangeAt(text, pos, languageId): StatementRange`;
  `nextStatementPos(text, to): number`. Re-exports `pyStatementRange`,
  `sqlStatementRange`, `rStatementRange`, `scanR`, `topLevelStatementAt`,
  `statementNodeAt`, `maskStringsAndComments` for Slice 3.

- [ ] **Step 1: Write the failing dispatch tests**

Add to `web/tests/unit/lang-structure.test.ts`:

```typescript
import { statementRangeAt } from "@/lib/sandbox/lang-structure";

describe("statementRangeAt", () => {
  it("routes to the R scanner and advances past the statement", () => {
    const src = "df |>\n  filter(x)\n\ny <- 2\n";
    const pos = src.indexOf("filter");
    const r = statementRangeAt(src, pos, "r");
    expect(src.slice(r.from, r.to)).toBe("df |>\n  filter(x)");
    // nextPos lands on the next executable statement.
    expect(src.slice(r.nextPos)).toMatch(/^y <- 2/);
  });

  it("routes to Python", () => {
    const src = "def f():\n    return 1\n\ng()\n";
    const r = statementRangeAt(src, src.indexOf("return"), "python");
    expect(src.slice(r.from, r.to)).toBe("def f():\n    return 1");
    expect(src.slice(r.nextPos)).toMatch(/^g\(\)/);
  });

  it("routes to SQL", () => {
    const src = "SELECT 1;\nSELECT 2;\n";
    const r = statementRangeAt(src, src.indexOf("SELECT 1"), "sql");
    expect(src.slice(r.from, r.to).trim()).toBe("SELECT 1;");
    expect(src.slice(r.nextPos)).toMatch(/^SELECT 2/);
  });
});
```

- [ ] **Step 2: Run and confirm the dispatch tests fail**

Run: `npm run test -- lang-structure`
Expected: FAIL (`index.ts` does not exist).

- [ ] **Step 3: Implement `index.ts`**

Create `web/lib/sandbox/lang-structure/index.ts`:

```typescript
import type { LanguageId, StatementRange } from "./types";
import { rStatementRange } from "./r-scan";
import { pyStatementRange } from "./py";
import { sqlStatementRange } from "./sql";

export type { LanguageId, StatementRange } from "./types";
export { rStatementRange, scanR } from "./r-scan";
export { pyStatementRange, topLevelStatementAt } from "./py";
export { sqlStatementRange, statementNodeAt } from "./sql";
export { maskStringsAndComments } from "./mask";

/** The next executable position after `to`: the start of the next non-blank
 *  line, or the document end. */
export function nextStatementPos(text: string, to: number): number {
  let i = to;
  while (i < text.length && (text[i] === "\n" || /\s/.test(text[i]))) {
    if (text[i] === "\n") {
      // Snap to the first non-whitespace on/after the next line.
      let j = i + 1;
      while (j < text.length && text[j] !== "\n" && /\s/.test(text[j])) j++;
      if (j < text.length && text[j] !== "\n") return j;
    }
    i++;
  }
  return text.length;
}

/**
 * The complete logical statement containing `pos`, for the given language.
 * `nextPos` is where the caret advances after running. The per-language finders
 * fall back to the physical line, so this never returns less than a line.
 */
export function statementRangeAt(
  text: string,
  pos: number,
  languageId: LanguageId,
): StatementRange {
  const clamped = Math.max(0, Math.min(pos, text.length));
  const { from, to } =
    languageId === "python"
      ? pyStatementRange(text, clamped)
      : languageId === "sql"
        ? sqlStatementRange(text, clamped)
        : rStatementRange(text, clamped);
  return { from, to, nextPos: nextStatementPos(text, to) };
}
```

- [ ] **Step 4: Run and confirm the full unit suite passes**

Run: `npm run test -- lang-structure`
Expected: PASS for every describe block (mask, Python, SQL, R, dispatch).

Run: `npm run typecheck`
Expected: no errors. If the `@lezer/common` type import does not resolve, apply
the fallback noted in Task 2 Step 3.

---

## Task 6: Wire the no-selection branch and update the copy

Only the `Mod-Enter` no-selection branch changes. `runKeymap` needs the current
`languageId` and the loaded `statementRangeAt`, so it is threaded through
`loadCodeMirror`. The selection branch and the two other run keys are untouched.

**Files:**
- Modify: `web/components/run/CodeEditor.tsx` (runKeymap, `loadCodeMirror`, the
  extensions array, the `onRunLine` JSDoc)
- Modify: `web/components/sandbox/Sandbox.tsx` (Run title line 797, hint line 812)

**Interfaces:**
- Consumes: `statementRangeAt` from `@/lib/sandbox/lang-structure`; the existing
  `runRef` handlers; `props.languageId`.
- Produces: no prop signature change to `CodeEditor`; `runKeymap` gains
  `languageId` and a `statementRangeAt` argument.

- [ ] **Step 1: Add `statementRangeAt` to the lazy editor load**

In `web/components/run/CodeEditor.tsx`, extend `LoadedEditor` and `loadCodeMirror`
so the pure module rides the same lazy chunk (keeping it out of the initial
bundle):

In the `LoadedEditor` interface (around line 182) add:

```typescript
  langStructure: typeof import("@/lib/sandbox/lang-structure");
```

In `loadCodeMirror` (around line 195) add the import to the `Promise.all` and the
return object:

```typescript
  const [cm, view, lang, state, autocomplete, highlight, inline, langStructure, langExt] =
    await Promise.all([
      import("codemirror"),
      import("@codemirror/view"),
      import("@codemirror/language"),
      import("@codemirror/state"),
      import("@codemirror/autocomplete"),
      import("@lezer/highlight"),
      import("@/lib/sandbox/inline-completion"),
      import("@/lib/sandbox/lang-structure"),
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
    langExt,
  };
```

Update the destructure in the mount effect (around line 90) to pull
`langStructure`, and pass it plus the language id into `runKeymap`:

```typescript
      .then(({ view, cm, lang, state, autocomplete, tags, langExt, inline, langStructure }) => {
```

and in the extensions array (around line 107):

```typescript
            ...(hasRun
              ? [runKeymap(view, state, runRef, langStructure.statementRangeAt, props.languageId)]
              : []),
```

- [ ] **Step 2: Rewrite the no-selection branch of `runKeymap`**

Replace the `runKeymap` signature and its `Mod-Enter` `run` body (lines 283-317)
with the statement-aware version:

```typescript
function runKeymap(
  view: LoadedEditor["view"],
  cmState: LoadedEditor["state"],
  runRef: { current: RunHandlers },
  statementRangeAt: LoadedEditor["langStructure"]["statementRangeAt"],
  languageId: string,
): Extension {
  const lang: "r" | "python" | "sql" =
    languageId === "python" ? "python" : languageId === "sql" ? "sql" : "r";
  return cmState.Prec.high(
    view.keymap.of([
      {
        key: "Mod-Enter",
        preventDefault: true,
        run: (editor) => {
          const handler = runRef.current.onRunLine;
          if (!handler) return false;
          const state = editor.state;
          const sel = state.selection.main;
          let code: string;
          let nextPos: number;
          if (!sel.empty) {
            // Selection-first: run exactly what is selected (unchanged).
            code = state.sliceDoc(sel.from, sel.to);
            const endLine = state.doc.lineAt(sel.to).number;
            nextPos =
              endLine < state.doc.lines
                ? state.doc.line(endLine + 1).from
                : state.doc.line(endLine).to;
          } else {
            // Run the complete logical statement containing the cursor.
            const r = statementRangeAt(state.doc.toString(), sel.head, lang);
            code = state.sliceDoc(r.from, r.to);
            nextPos = r.nextPos;
          }
          editor.dispatch({ selection: { anchor: nextPos }, scrollIntoView: true });
          if (code.trim()) handler(code);
          return true;
        },
      },
      {
        key: "Mod-Shift-Enter",
        preventDefault: true,
        run: () => {
          runRef.current.onRunAll?.();
          return true;
        },
      },
      {
        key: "Mod-Shift-s",
        preventDefault: true,
        run: () => {
          runRef.current.onSource?.();
          return true;
        },
      },
    ]),
  );
}
```

- [ ] **Step 3: Update the `onRunLine` JSDoc (copy only)**

In the `CodeEditor` props (line 39), change the comment to reflect statement
execution (no em dashes):

```typescript
  /** Run the current statement or selection (Mod-Enter), advancing the cursor. */
  onRunLine?: (code: string) => void;
```

The prop name stays `onRunLine` to avoid churn across `Sandbox.tsx`; only the
doc comment changes.

- [ ] **Step 4: Update the toolbar copy in `Sandbox.tsx`**

In `web/components/sandbox/Sandbox.tsx`, the Run button title (line 797):

```tsx
          title={`Run the whole script, output shown in the console (${mod}+Shift+Enter). In the editor, ${mod}+Enter runs the current statement.`}
```

and the inline hint (line 812):

```tsx
          {mod}+Enter: statement
```

Both are free of em dashes.

- [ ] **Step 5: Verify types and lint**

Run: `npm run typecheck`
Expected: no errors.

Run: `npm run lint`
Expected: no errors for the changed files.

---

## Task 7: End-to-end proof with real SQLite execution

Prove the whole statement actually runs (not one physical line) by running a
multi-line CTE from a middle line via Ctrl+Enter and asserting the aggregate
result. SQLite is the deterministic choice (no prewarm, no network), following
the Slice 1 plan's sqlite pattern.

**Files:**
- Modify: `web/tests/e2e/sandbox.spec.ts`

- [ ] **Step 1: Add the multi-line-statement execution test**

Add to the `test.describe("AI Sandbox", ...)` block:

```typescript
  test("Ctrl+Enter runs the whole multi-line SQL statement from a middle line", async ({
    page,
  }) => {
    // Real sqlite-wasm execution; generous headroom for the first compile.
    test.setTimeout(120_000);

    await page.goto("/ai-sandbox");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    // A CTE that only yields a value if the WHOLE statement runs. Running any
    // single physical line here is a syntax error, so a passing result proves
    // the complete logical statement executed.
    await page.keyboard.insertText(
      "WITH nums(n) AS (\n" +
        "  VALUES (1), (2), (3)\n" +
        ")\n" +
        "SELECT SUM(n) AS total FROM nums;",
    );

    // Put the cursor on a middle line (the VALUES row), not the start.
    await page.getByText("VALUES (1), (2), (3)").click();
    await page.keyboard.press("ControlOrMeta+Enter");

    const output = page.getByLabel("Console output");
    await expect(output).toContainText("total", { timeout: 60_000 });
    await expect(output).toContainText("6");
  });
```

- [ ] **Step 2: Run the multi-line test and confirm it passes**

Run: `npm run test:e2e -- sandbox.spec.ts -g "whole multi-line SQL"`
Expected: PASS. The console shows `total` and `6`, proving the CTE plus its final
SELECT ran as one statement from a middle-line cursor. (Before this slice, the
same keystroke would send only `  VALUES (1), (2), (3)`, a SQLite syntax error,
so this test also guards against regression.)

- [ ] **Step 3: Add the multi-statement selectivity test**

Prove that in a multi-statement script only the statement at the cursor runs:

```typescript
  test("Ctrl+Enter runs only the statement at the cursor, not the next one", async ({
    page,
  }) => {
    test.setTimeout(120_000);

    await page.goto("/ai-sandbox");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "SELECT 111 AS first_only;\nSELECT 222 AS second_only;",
    );

    // Cursor on the first statement.
    await page.getByText("SELECT 111 AS first_only;").click();
    await page.keyboard.press("ControlOrMeta+Enter");

    const output = page.getByLabel("Console output");
    await expect(output).toContainText("111", { timeout: 60_000 });
    // The second statement did not run.
    await expect(output).not.toContainText("222");
  });
```

- [ ] **Step 4: Run the selectivity test and confirm it passes**

Run: `npm run test:e2e -- sandbox.spec.ts -g "only the statement at the cursor"`
Expected: PASS. `111` appears, `222` does not.

Fallback if sqlite-wasm proves flaky in CI (the surrounding suite deliberately
avoids WASM execution): gate both tests behind the same opt-in flag the Slice 1
plan uses, at the top of each test body:

```typescript
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "runs real sqlite-wasm; opt in with CHATISA_LIVE_NET=1",
    );
```

- [ ] **Step 5: Verify the slice (no commit)**

Run: `npm run test -- lang-structure`
Expected: the full unit suite passes.

Run: `npm run typecheck`
Expected: no errors.

Run: `npm run lint`
Expected: no errors for the changed files.

Run: `npm run test:e2e -- sandbox.spec.ts`
Expected: the full sandbox suite passes, including the two new execution tests and
the existing shell/axe tests. Leave the working tree uncommitted.

---

## Self-Review

**Spec coverage (requirement D and the general acceptance criteria):**

- Selection-first: run only the selection — Task 6 Step 2 keeps the `!sel.empty`
  branch verbatim.
- No selection: run the complete logical statement containing the cursor — Tasks
  2, 3, 4, 5 build `statementRangeAt`; Task 6 wires it.
- Do not include the next unrelated statement — Python climbs to a single
  child-of-`Script` node; SQL climbs to a single `Statement`; R stops the forward
  walk at the next statement-start. Proven by the multi-statement unit tests and
  the e2e selectivity test (Task 7 Step 3).
- Optionally advance the cursor to the next executable statement — `nextPos` /
  `nextStatementPos` (Task 5), applied in Task 6.
- Ignore operators/semicolons/brackets/keywords/continuation chars inside strings
  and comments — Python/SQL via the Lezer tree (String/comment nodes are never
  boundaries; the SQL semicolon-in-string test proves it); R via the mask (the
  plus-in-string and plus-in-comment tests, and the SQL-mask semicolon test).
- R: single line alone; multi-line `|>` and `%>%` chains from any line; ggplot `+`
  chain through the final layer distinguishing a continuation `+` from a string /
  comment `+`; multi-line bracketed call/assignment (tibble) whole — Task 4 tests.
- Python: expression alone; multi-line bracketed expression whole from any line;
  indentation blocks whole (def/class/if-elif-else/for/while/try-except-finally/
  with/match handled by climbing to the top-level node); decorator runs with its
  target (`DecoratedStatement`) — Task 2 tests.
- SQL: complete statement to `;`; CTE `WITH ... SELECT` together; nested
  subqueries and parentheses run the whole parent; multi-statement scripts run one
  statement; procedural `BEGIN ... END` noted as out of scope for SQLite with the
  `;`-in-string boundary safety kept — Task 3 tests and note.
- Ctrl (Win/Linux) and Cmd (macOS): the binding is `Mod-Enter`, unchanged.
- Run button copy no longer says "current line" — Task 6 Step 4 (title and hint,
  no em dashes).

**Reuse for Slice 3 (E, indentation).** The module is split so E imports the same
per-language primitives rather than re-deriving structure:

- `maskStringsAndComments` — E's R and SQL indent services scan the mask, so they
  never indent based on a bracket or keyword inside a string or comment.
- `scanR` — E's R `indentService` reads the same `RLine` table (bracket depth per
  line, trailing/leading continuation flags) to indent after an open bracket or a
  trailing `|>`/`+`, and to dedent after a close.
- `topLevelStatementAt` (Python) and `statementNodeAt` (SQL) — E reuses these tree
  walks for block context (indent after a Python `:`, dedent SQL major clauses)
  instead of writing new tree traversals.

Building D first therefore front-loads the shared foundation, exactly as the
decomposition recommended.

**Constraints from `CodeEditor.tsx` that shaped the design:**

- The editor is rebuilt (the mount effect re-runs) when `props.languageId`
  changes (dependency array, line 137), so passing `languageId` into `runKeymap`
  at build time is safe: a language switch tears down and rebuilds the keymap with
  the new id. No per-keystroke language lookup is needed.
- Run handlers are read through `runRef` (a ref updated by an effect), so changing
  handlers does not rebuild the editor; `statementRangeAt` is a stable import and
  is passed by value at build time.
- `runKeymap` is already wrapped in `Prec.high` (line 288), so its `Mod-Enter`
  wins over `basicSetup`'s Enter-family bindings; the rewrite keeps that wrapper.
- CodeMirror is lazy-loaded via `loadCodeMirror`; adding `@/lib/sandbox/lang-structure`
  to that same `Promise.all` keeps the Lezer parsers and the R scanner out of the
  initial bundle and in the editor's lazy chunk. The pure module never imports
  `@codemirror/view`, so it also loads cleanly in the Vitest `node` environment.
- The no-selection branch calls `state.doc.toString()` and re-parses on each
  `Mod-Enter`. That is one parse on one keypress (negligible). A later optimization
  could pass the live `syntaxTree(state)` for Python/SQL to skip the re-parse, but
  it would couple the module to CodeMirror and is intentionally not done here.

**Placeholder scan:** No TBDs. Every code step shows final code. Commands are exact
and adapted to the no-commit rule (verify steps replace commit steps). The one
noted contingency (the `@lezer/common` type import) has an inline fallback.

**Type consistency:** `statementRangeAt(text, pos, languageId): StatementRange`
with `StatementRange = { from; to; nextPos }`. Per-language finders return
`{ from; to }`; the dispatch adds `nextPos`. `runKeymap` narrows `languageId:
string` to the `"r" | "python" | "sql"` union before calling. Editor accessible
names in the e2e (`/SQL code/i`, `Console output`) match the existing
`label={`${language.label} code`}` (Sandbox.tsx:660) and `aria-label="Console
output"` (Sandbox.tsx:1026).

## Handoff note

Per the task that produced this plan, this is a planning document only; do not
build from it in this session. When execution is scheduled, use
superpowers:subagent-driven-development (a fresh subagent per task, review between)
or superpowers:executing-plans (batched with checkpoints), and remember: no git
commits, verify with the commands above instead. Build order within the slice:
Task 1 (mask) first, then Tasks 2-4 (the three finders, independent of each
other), then Task 5 (dispatch), then Task 6 (editor wiring), then Task 7 (e2e).
