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
    // The SQLite grammar is lenient about keyword shape but does flag structural
    // breaks; an unclosed subquery paren yields an error node (the SQL analogue of
    // an unclosed R bracket).
    const p = treeErrorProblems("SELECT * FROM (", "sql");
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
    expect(lintProblems("SELECT * FROM (", "sql").length).toBeGreaterThan(0);
  });
});
