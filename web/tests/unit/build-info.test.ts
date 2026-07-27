import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { BUILD_DATE } from "@/lib/config/build-info";

/**
 * The footer date must be FROZEN at bundle time: a student opening the app in
 * October should still see the date the running bundle was made. That only
 * holds while the value is a compiled-in literal, so these tests guard the
 * shape rather than the value, plus the two ways it could regress into being
 * evaluated per request.
 */
/** Source with block and line comments removed, so prose about a pattern is
 * not mistaken for a use of it. */
function codeOnly(source: string): string {
  return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
}

describe("footer build date", () => {
  it("is a human-readable month day, year", () => {
    expect(BUILD_DATE).toMatch(
      /^(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}$/,
    );
  });

  it("is a plain literal, not computed at import time", () => {
    const source = readFileSync(
      join(process.cwd(), "lib", "config", "build-info.ts"),
      "utf8",
    );
    expect(source).toMatch(/export const BUILD_DATE = "[^"]+";/);
    // A literal survives into the build and stays put. Any of these in the CODE
    // would make the date drift per request or per render. Comments are stripped
    // first: the file's own docs explain why those approaches were rejected.
    expect(codeOnly(source)).not.toMatch(
      /new Date|Date\.now|toLocaleDateString|process\.env/,
    );
  });

  it("is the only place the footer's date comes from", () => {
    const footer = readFileSync(
      join(process.cwd(), "components", "SiteFooter.tsx"),
      "utf8",
    );
    expect(footer).toContain("BUILD_DATE");
    // No second, hand-maintained date to drift out of sync with the stamp.
    expect(codeOnly(footer)).not.toMatch(/"\w+ \d{1,2}, \d{4}"/);
    expect(codeOnly(footer)).not.toMatch(/new Date|Date\.now/);
  });

  it("is stamped by the bundle script before the build runs", () => {
    const script = readFileSync(
      join(process.cwd(), "scripts", "make-deploy-bundle.mjs"),
      "utf8",
    );
    const stampAt = script.indexOf("await writeFile(buildInfoPath");
    const buildAt = script.indexOf('spawnSync("npm", ["run", "build"]');
    expect(stampAt).toBeGreaterThan(-1);
    expect(buildAt).toBeGreaterThan(-1);
    // Stamping after the build would ship the previous bundle's date.
    expect(stampAt).toBeLessThan(buildAt);
  });
});
