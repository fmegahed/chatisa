import { describe, expect, it } from "vitest";
import {
  rIndentColumns,
  sqlIndentColumns,
} from "@/lib/sandbox/lang-structure/indent";
import { EditorState } from "@codemirror/state";
import { getIndentation, IndentContext, indentUnit } from "@codemirror/language";
import { python } from "@codemirror/lang-python";

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
