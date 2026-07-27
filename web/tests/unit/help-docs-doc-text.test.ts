import { describe, expect, it } from "vitest";
import {
  buildDocRequest,
  stripOverstrike,
  truncateDocText,
  DOC_MAX_LINES,
} from "@/lib/sandbox/help-docs/doc-text";
import type { DocEntry, HelpRequest } from "@/lib/sandbox/help-docs/types";

function req(p: Partial<HelpRequest> & { name: string }): HelpRequest {
  return { kind: "function", language: p.language ?? "python", qualifier: p.qualifier, name: p.name };
}
function entry(p: Partial<DocEntry> & { source: string }): DocEntry {
  return { symbol: p.symbol ?? "x", source: p.source, url: p.url ?? "https://example.test", blurb: p.blurb };
}

describe("buildDocRequest", () => {
  it("carries name, qualifier, and the resolved source as a hint", () => {
    const r = buildDocRequest(
      req({ name: "groupby", qualifier: "df", language: "python" }),
      entry({ source: "pandas" }),
    );
    expect(r).toEqual({ name: "groupby", qualifier: "df", source: "pandas" });
  });

  it("passes the source so R can pick a package for a bare name", () => {
    const r = buildDocRequest(
      req({ name: "summarise", language: "r" }),
      entry({ source: "dplyr" }),
    );
    expect(r.name).toBe("summarise");
    expect(r.qualifier).toBeUndefined();
    expect(r.source).toBe("dplyr");
  });
});

describe("truncateDocText", () => {
  it("leaves short text unchanged and not truncated", () => {
    const { text, truncated } = truncateDocText("one\ntwo\nthree");
    expect(text).toBe("one\ntwo\nthree");
    expect(truncated).toBe(false);
  });

  it("caps by line count and flags truncation", () => {
    const raw = Array.from({ length: DOC_MAX_LINES + 40 }, (_, i) => `line ${i}`).join("\n");
    const { text, truncated } = truncateDocText(raw);
    expect(truncated).toBe(true);
    expect(text.split("\n").length).toBeLessThanOrEqual(DOC_MAX_LINES);
  });

  it("caps by character count and flags truncation", () => {
    const raw = "x".repeat(20000);
    const { text, truncated } = truncateDocText(raw, { maxChars: 100 });
    expect(truncated).toBe(true);
    expect(text.length).toBeLessThanOrEqual(100);
  });

  it("strips Rd2txt overstrike so help reads cleanly", () => {
    // Underline: "_\bB..." for "Build"; bold: "X\bX". As produced by R's Rd2txt.
    const underlined = "_\bB_\bu_\bi_\bl_\bd";
    const bold = "t\bti\bit\btl\ble\be";
    const { text } = truncateDocText(`${underlined} a ${bold}`);
    expect(text).toBe("Build a title");
    expect(text).not.toContain("\b");
    expect(text).not.toContain("_B");
  });
});

describe("stripOverstrike", () => {
  it("collapses bold, underline, and double overstrike; leaves plain text alone", () => {
    expect(stripOverstrike("a\bab\bbc\bc")).toBe("abc"); // bold
    expect(stripOverstrike("_\bx_\by")).toBe("xy"); // underline
    expect(stripOverstrike("_\bB\bB")).toBe("B"); // bold + underline
    expect(stripOverstrike("plain text")).toBe("plain text"); // no-op
    expect(stripOverstrike("line1\nline2")).toBe("line1\nline2"); // newlines kept
  });
});
