import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";

/**
 * Guards against a mistake made twice in this project, both times silently.
 *
 * A regex word boundary written as `\b` through a tool that interprets escape
 * sequences becomes a literal backspace character (0x08). The file still looks
 * correct in an editor, in `grep`, and in a diff, because the character is
 * invisible. But `/\bword\b/` becomes `/<BS>word<BS>/`, which matches nothing,
 * so the pattern is dead and whatever it was guarding silently stops being
 * guarded.
 *
 * It happened first in the Exam Ally self-contained-question gate, where every
 * pattern was dead on arrival, and again in the job posting detector, where it
 * caused a real posting to be rejected. Repairing the second occurrence found
 * two more survivors in the first file.
 *
 * A test is the right place for this: it is invisible to review by
 * construction, so only a machine will catch it.
 */

const SKIP = new Set([
  "node_modules",
  ".next",
  ".git",
  "test-results",
  "playwright-report",
  "drizzle",
  "coverage",
]);

/** Control characters that have no business in source, with their names. */
const FORBIDDEN: { code: number; name: string; why: string }[] = [
  { code: 0x08, name: "backspace", why: "usually a `\\b` regex boundary mangled by an escape-interpreting tool" },
  { code: 0x00, name: "null", why: "never intentional in source" },
  { code: 0x1b, name: "escape", why: "usually a stray terminal escape sequence" },
];

function sourceFiles(dir: string, found: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (SKIP.has(entry)) continue;
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) {
      sourceFiles(path, found);
    } else if (/\.(ts|tsx|mjs|js|jsx)$/.test(entry)) {
      found.push(path);
    }
  }
  return found;
}

describe("source hygiene", () => {
  it("contains no invisible control characters", () => {
    const offenders: string[] = [];

    for (const path of sourceFiles(process.cwd())) {
      const text = readFileSync(path, "utf8");
      for (const { code, name, why } of FORBIDDEN) {
        const character = String.fromCharCode(code);
        if (!text.includes(character)) continue;
        // Report the line so the fix is obvious despite being invisible.
        const line =
          text.slice(0, text.indexOf(character)).split("\n").length;
        offenders.push(
          `${path}:${line} contains a ${name} character (${why})`,
        );
      }
    }

    expect(offenders, offenders.join("\n")).toEqual([]);
  });
});
