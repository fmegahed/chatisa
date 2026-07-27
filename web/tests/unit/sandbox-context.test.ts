import { describe, expect, it } from "vitest";
import { buildSandboxContext } from "@/lib/sandbox/context";

describe("buildSandboxContext", () => {
  it("includes the language, script, last result, and variables with columns", () => {
    const ctx = buildSandboxContext({
      languageLabel: "Python",
      script: "df.head()",
      lastRun: {
        code: "df.describe()",
        outcome: { ok: true, result: { text: "count 3" } },
      },
      variables: [
        {
          name: "df",
          type: "DataFrame",
          info: "3 x 2",
          columns: [
            { name: "name", type: "object" },
            { name: "score", type: "int64" },
          ],
        },
        { name: "x", type: "int", info: "42" },
      ],
    });

    expect(ctx).toContain("Language: Python");
    expect(ctx).toContain("df.head()");
    expect(ctx).toContain("df.describe()");
    expect(ctx).toContain("count 3");
    // Column names and types are surfaced for data frames.
    expect(ctx).toContain("name (object)");
    expect(ctx).toContain("score (int64)");
    // A plain scalar has no columns section.
    expect(ctx).toContain("- x: int [42]");
    expect(ctx).not.toContain("x: int [42]; columns");
  });

  it("shows the error when the last run failed", () => {
    const ctx = buildSandboxContext({
      languageLabel: "R",
      script: "summary(z)",
      lastRun: {
        code: "summary(z)",
        outcome: { ok: false, error: "object 'z' not found" },
      },
      variables: [],
    });
    expect(ctx).toContain("Last run error:");
    expect(ctx).toContain("object 'z' not found");
  });

  it("omits sections that have nothing in them", () => {
    const ctx = buildSandboxContext({
      languageLabel: "SQL",
      script: "",
      variables: [],
    });
    expect(ctx).toBe("Language: SQL");
  });

  it("truncates a very long script rather than sending all of it", () => {
    const ctx = buildSandboxContext({
      languageLabel: "Python",
      script: "x = 1\n".repeat(5000),
      variables: [],
    });
    expect(ctx).toContain("(truncated)");
    expect(ctx.length).toBeLessThan(6000);
  });
});
