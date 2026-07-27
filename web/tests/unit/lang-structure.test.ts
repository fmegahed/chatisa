import { describe, expect, it } from "vitest";
import { maskStringsAndComments, cursorInMasked } from "@/lib/sandbox/lang-structure/mask";
import { pyStatementRange } from "@/lib/sandbox/lang-structure/py";
import { sqlStatementRange } from "@/lib/sandbox/lang-structure/sql";
import { rStatementRange } from "@/lib/sandbox/lang-structure/r-scan";
import { statementRangeAt } from "@/lib/sandbox/lang-structure";

describe("cursorInMasked", () => {
  it("is true inside a comment, false in code, per language", () => {
    // R / Python line comments.
    expect(cursorInMasked("# help me compute summary stat", "r")).toBe(true);
    expect(cursorInMasked("x <- 1 # note ", "r")).toBe(true);
    expect(cursorInMasked("df.gr", "python")).toBe(false);
    expect(cursorInMasked("summary(df", "r")).toBe(false);
    // SQL uses -- for line comments.
    expect(cursorInMasked("SELECT * -- pick ", "sql")).toBe(true);
    expect(cursorInMasked("SELECT cou", "sql")).toBe(false);
  });

  it("is true inside an open string, false once it closes", () => {
    expect(cursorInMasked('name <- "Ama', "r")).toBe(true);
    expect(cursorInMasked('name <- "Amanda"', "r")).toBe(false);
    expect(cursorInMasked('x = "abc" + y', "python")).toBe(false);
  });
});

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
