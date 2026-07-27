import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";

/**
 * Brand-compliance guardrails: the stylesheet must use the Miami brand
 * guide's exact hex values (pp. 28–29) and must not reintroduce the legacy
 * off-brand colors or alpha-tinted brand colors (p. 28: no tints/shades).
 */
const css = readFileSync(
  path.resolve(__dirname, "../../app/globals.css"),
  "utf-8",
).toLowerCase();

describe("Miami brand tokens", () => {
  it.each([
    ["Miami Red", "#c41230"],
    ["Accent Red", "#ad102a"],
    ["Light Tan", "#edece2"],
    ["Medium Tan", "#ccc9b8"],
    ["Dark Tan", "#70685c"],
    ["Warm White", "#faf9f7"],
    ["Medium Gray", "#666666"],
    ["Corn Yellow", "#efdb72"],
    ["Slate Blue", "#3e5468"],
  ])("defines %s as %s", (_name, hex) => {
    expect(css).toContain(hex);
  });

  it("does not contain the legacy off-brand reds", () => {
    expect(css).not.toContain("#c3142d");
    expect(css).not.toContain("200, 16, 45");
  });

  it("does not alpha-tint brand colors (rgb of Miami Red with alpha)", () => {
    expect(css).not.toMatch(/196[, ]+18[, ]+48[, ]*\/?\s*0?\./);
    expect(css).not.toMatch(/#c41230[0-9a-f]{2}\b/);
  });
});
