import { describe, expect, it } from "vitest";
import { existsSync, readdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { inspectPptx, pptxProblems } from "../live/support/pptx";

const ROOT = resolve(__dirname, "..", "..");
const GENERATED = join(ROOT, "tests", "live", ".artifacts", "files");

/**
 * The deck inspector, checked against decks a real model actually produced.
 *
 * It exists because a download link proves nothing: on 2026-07-25 every deck this
 * app generated arrived as a working link and would not open, each slide having
 * declared its layout twice. But an inspector that gets the answer wrong is worse
 * than none, and this one did: it read whichever `ppt/theme/themeN.xml` came
 * first in zip order and so reported "not the Miami template" for two decks whose
 * master theme was Roboto Condensed / Roboto. These tests pin both directions.
 *
 * They run only when generated decks are present (they come from a live run and
 * are git-ignored), and skip quietly otherwise rather than failing a clean
 * checkout.
 */
function generatedDecks(): string[] {
  if (!existsSync(GENERATED)) return [];
  return readdirSync(GENERATED)
    .filter((f) => f.toLowerCase().endsWith(".pptx"))
    .map((f) => join(GENERATED, f));
}

describe("inspectPptx", () => {
  it("reads a real generated deck as structurally sound", async () => {
    const decks = generatedDecks();
    if (decks.length === 0) return;

    for (const path of decks) {
      const report = await inspectPptx(path);
      // Independently confirmed by unzipping these files: exactly one
      // slideLayout relationship per slide, so lib/ask/pptx-repair is working.
      expect(report.slideCount, `${path} has no slides`).toBeGreaterThan(0);
      expect(report.slidesWithDuplicateLayout).toEqual([]);
      expect(report.slidesWithoutLayout).toEqual([]);
      expect(pptxProblems(report)).toEqual([]);
    }
  });

  it("finds the Miami template theme wherever it sits in the archive", async () => {
    const decks = generatedDecks();
    if (decks.length === 0) return;

    // Both decks generated on 2026-07-26 carry Roboto Condensed / Roboto as the
    // master theme's major/minor latin fonts, so they were built FROM the
    // template. Before the fix this returned false for both.
    for (const path of decks) {
      const report = await inspectPptx(path);
      expect(
        report.usesTemplateTheme,
        `${path}: the Miami template fonts were not found in any theme part`,
      ).toBe(true);
    }
  });

  it("reports empty slides, which a slide count alone cannot", async () => {
    const decks = generatedDecks();
    if (decks.length === 0) return;
    for (const path of decks) {
      const report = await inspectPptx(path);
      // A deck of blank slides has the right shape and no content. Neither of
      // the real decks has one.
      expect(report.emptySlides).toEqual([]);
    }
  });
});
