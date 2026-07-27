import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  BUNDLED_PYTHON,
  BUNDLED_R,
  KNOWN_UNAVAILABLE_PYTHON,
  MIRRORED_R,
} from "@/lib/sandbox/packages";
import { runningCodeRules } from "@/lib/prompts/running-code";
import { CODING_COMPANION_SYSTEM_PROMPT } from "@/lib/prompts/coding-companion";

const ROOT = resolve(__dirname, "..", "..");

/**
 * The Coding Tutor was told on 2026-07-26 that its code blocks really run, and
 * which packages exist in the browser runtimes. A prompt that lists packages is a
 * promise, and a stale promise is worse than none: it produces code that cannot
 * run, in a module whose entire value is code that runs.
 *
 * So the lists are generated from lib/sandbox/packages rather than typed into
 * prose, and the R list is checked against the worker that actually installs it.
 */
describe("runningCodeRules", () => {
  it("tells the model the blocks are runnable, not copy-and-paste", () => {
    const rules = runningCodeRules();
    expect(rules).toMatch(/Run button/);
    expect(rules).toMatch(/in their own browser/i);
  });

  it("forbids the guessed-selector answer that prompted this block", () => {
    // The live failure: selectors the model invented, each labelled
    // "VERIFY & REPLACE THIS SELECTOR", none of which existed on the page.
    const rules = runningCodeRules();
    expect(rules).toMatch(/never hand over something you could have checked/i);
    expect(rules).toMatch(/replace this/i);
    expect(rules).toMatch(/two steps/i);
  });

  it("generates the package lists rather than restating them", () => {
    const rules = runningCodeRules();
    for (const pkg of BUNDLED_PYTHON) expect(rules).toContain(pkg);
    for (const pkg of BUNDLED_R) expect(rules).toContain(pkg);
    for (const pkg of MIRRORED_R) expect(rules).toContain(pkg);
    for (const pkg of KNOWN_UNAVAILABLE_PYTHON) expect(rules).toContain(pkg);
  });

  it("says state is not kept between separate Run presses", () => {
    // True of the inline Run button (keepState: false in lib/run/manager), and
    // the opposite of the Coding Studio. A model that assumes otherwise writes
    // snippets that depend on a previous block having been run.
    expect(runningCodeRules()).toMatch(/not kept between/i);
  });

  it("rides in the Coding Tutor prompt", () => {
    expect(CODING_COMPANION_SYSTEM_PROMPT).toContain(runningCodeRules());
  });
});

describe("the R bundle list matches the worker that installs it", () => {
  it("agrees with BUNDLED_PACKAGES in the webR worker", () => {
    // The worker is a static ES module loaded by URL, so it cannot be imported
    // here and the list exists in two places by necessity. This is the guard:
    // parse the worker's own array and compare.
    const worker = readFileSync(
      join(ROOT, "public", "workers", "webr-worker.mjs"),
      "utf8",
    );
    const match = /const BUNDLED_PACKAGES = \[([^\]]*)\]/.exec(worker);
    expect(match, "BUNDLED_PACKAGES not found in webr-worker.mjs").not.toBeNull();
    const fromWorker = (match?.[1] ?? "")
      .split(",")
      .map((s) => s.trim().replace(/^["']|["']$/g, ""))
      .filter(Boolean);
    expect(fromWorker).toEqual([...BUNDLED_R]);
  });

  it("names every mirrored R package the setup script fetches", () => {
    // setup-runtimes mirrors BUNDLED_R plus MIRRORED_R. If it ever mirrors more,
    // the prompt is understating what a student can use; if fewer, it is
    // promising a package that will 404 on install.
    const setup = readFileSync(
      join(ROOT, "scripts", "setup-runtimes.mjs"),
      "utf8",
    );
    const match = /const WEBR_PACKAGES = \[([^\]]*)\]/.exec(setup);
    expect(match, "WEBR_PACKAGES not found in setup-runtimes.mjs").not.toBeNull();
    const mirrored = (match?.[1] ?? "")
      .split(",")
      .map((s) => s.trim().replace(/^["']|["']$/g, ""))
      .filter(Boolean);
    expect([...mirrored].sort()).toEqual([...BUNDLED_R, ...MIRRORED_R].sort());
  });
});
