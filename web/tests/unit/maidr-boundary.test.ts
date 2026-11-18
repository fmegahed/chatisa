import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";
import pin from "@/lib/run/maidr-pin.json";

/**
 * The licence boundary around MAIDR (#40).
 *
 * maidr.js is GPL-3.0-or-later; ChatISA is MIT. The professor's decision
 * (2026-11-17) was to ship maidr.js as a separate, unmodified file loaded at
 * runtime, never compiled into ChatISA's code. That boundary is easy to erode
 * by accident (one `import { Maidr } from "maidr/react"` would do it), so it is
 * checked here, together with the pieces that keep the shipped copy honest:
 * the pinned hashes, the bundle check and the third-party notice.
 */

const web = process.cwd();
const repo = join(web, "..");

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    if (name === "node_modules" || name.startsWith(".")) continue;
    const full = join(dir, name);
    if (statSync(full).isDirectory()) sourceFiles(full, out);
    else if (/\.(ts|tsx|js|mjs|jsx)$/.test(name)) out.push(full);
  }
  return out;
}

describe("MAIDR licence boundary (#40)", () => {
  it("no ChatISA source imports maidr", () => {
    const importsMaidr = /(?:from\s+|import\s*\(\s*|require\s*\(\s*)["']maidr(?:\/[^"']*)?["']/;
    const offenders = ["app", "components", "lib", "public/workers"]
      .flatMap((d) => sourceFiles(join(web, d)))
      .filter((f) => importsMaidr.test(readFileSync(f, "utf8")))
      .map((f) => relative(web, f));
    expect(offenders).toEqual([]);
  });

  it("maidr is not an npm dependency", () => {
    const pkg = JSON.parse(readFileSync(join(web, "package.json"), "utf8"));
    expect(Object.keys({ ...pkg.dependencies, ...pkg.devDependencies })).not.toContain("maidr");
  });

  it("the setup verifies the download against the pin, and the bundle checks it again", () => {
    const setup = readFileSync(join(web, "scripts", "setup-runtimes.mjs"), "utf8");
    expect(setup).toContain("maidr-pin.json");
    expect(setup).toMatch(/integrity !== pin\.integrity/);
    expect(setup).toMatch(/sha256 !== pin\.maidrJsSha256/);
    const bundle = readFileSync(join(web, "scripts", "make-deploy-bundle.mjs"), "utf8");
    expect(bundle).toContain("maidr-pin.json");
    expect(bundle).toMatch(/pin\.maidrJsSha256/);
  });

  it("the third-party notice names the pinned version, integrity and source", () => {
    const notice = readFileSync(join(repo, "THIRD_PARTY_NOTICES.md"), "utf8");
    expect(notice).toContain(`Version:** ${pin.version}`);
    expect(notice).toContain(pin.integrity);
    expect(notice).toContain(pin.source);
    expect(notice).toMatch(/GPL-3\.0-or-later/);
  });
});
