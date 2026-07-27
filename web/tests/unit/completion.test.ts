import { describe, expect, it } from "vitest";
import {
  buildCompletionPrompt,
  parseCompletion,
} from "@/lib/sandbox/completion";

describe("buildCompletionPrompt", () => {
  it("marks the cursor between prefix and suffix and names the language", () => {
    const p = buildCompletionPrompt("python", "df.", "\nprint(df)");
    expect(p).toContain("python");
    expect(p).toContain("df.<CURSOR>");
    expect(p).toContain("<CURSOR>\nprint(df)");
  });
});

describe("parseCompletion", () => {
  it("keeps a plain single-line completion", () => {
    expect(parseCompletion("groupby('region').sum()")).toBe(
      "groupby('region').sum()",
    );
  });

  it("strips a wrapping code fence", () => {
    expect(parseCompletion("```python\nx = 1\n```")).toBe("x = 1");
  });

  it("preserves leading indentation but trims trailing whitespace", () => {
    expect(parseCompletion("    return x   \n\n")).toBe("    return x");
  });

  it("caps the number of lines", () => {
    const many = Array.from({ length: 20 }, (_, i) => `line${i}`).join("\n");
    const out = parseCompletion(many);
    expect(out.split("\n").length).toBeLessThanOrEqual(8);
  });

  it("caps the length", () => {
    expect(parseCompletion("x".repeat(1000)).length).toBeLessThanOrEqual(240);
  });

  it("handles empty input", () => {
    expect(parseCompletion("")).toBe("");
  });
});
