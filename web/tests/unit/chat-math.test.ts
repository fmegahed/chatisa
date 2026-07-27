import { describe, expect, it } from "vitest";
import { normalizeMathDelimiters } from "@/lib/chat/math";

describe("math delimiter normalization", () => {
  it("converts inline bracket math to dollars", () => {
    expect(normalizeMathDelimiters("Solve \\(x^2 = 4\\) for x.")).toBe(
      "Solve $x^2 = 4$ for x.",
    );
  });

  it("converts display bracket math to double dollars", () => {
    const out = normalizeMathDelimiters("Result:\n\\[e^{i\\pi} + 1 = 0\\]\ndone");
    expect(out).toContain("$$\ne^{i\\pi} + 1 = 0\n$$");
    expect(out).not.toContain("\\[");
  });

  it("leaves fenced code blocks byte-identical", () => {
    const md = "Use this:\n```tex\n\\(x\\) and \\[y\\]\n```\nAnd \\(z\\) outside.";
    const out = normalizeMathDelimiters(md);
    expect(out).toContain("```tex\n\\(x\\) and \\[y\\]\n```");
    expect(out).toContain("And $z$ outside.");
  });

  it("leaves inline code spans untouched", () => {
    expect(normalizeMathDelimiters("Regex `\\(a\\)` but math \\(b\\)")).toBe(
      "Regex `\\(a\\)` but math $b$",
    );
  });

  it("passes plain text and dollar amounts through unchanged", () => {
    const text = "That costs $5 and $10 total.";
    expect(normalizeMathDelimiters(text)).toBe(text);
  });

  it("handles multiline inline groups across a paragraph", () => {
    const out = normalizeMathDelimiters(
      "First \\(a + b\\), then \\(c \\neq 0\\).",
    );
    expect(out).toBe("First $a + b$, then $c \\neq 0$.");
  });
});
