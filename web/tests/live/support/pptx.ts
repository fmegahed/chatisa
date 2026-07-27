import { readFileSync } from "node:fs";
import JSZip from "jszip";

/**
 * Opens a generated .pptx far enough to tell whether PowerPoint would.
 *
 * A download link is not evidence. On 2026-07-25 every deck this app produced
 * arrived as a working link and failed to open, because each slide's
 * relationships declared its slideLayout twice; lib/ask/pptx-repair.ts now
 * strips the duplicate on the way out. A live test that only checked the link
 * existed would have passed that deck, so this reads the package.
 *
 * What it cannot tell you is whether the deck is any GOOD: the slides could be
 * empty or the content wrong. Slide count, layout wiring, and text presence are
 * structural checks, and the specs save the file so a human can open it.
 */

export interface PptxReport {
  /** Number of ppt/slides/slideN.xml parts. */
  slideCount: number;
  /** Slides whose rels declare more than one slideLayout: the 2026-07-25 bug. */
  slidesWithDuplicateLayout: string[];
  /** Slides whose rels declare NO slideLayout, which PowerPoint also rejects. */
  slidesWithoutLayout: string[];
  /** True when the template's own layouts are present, so branding survived. */
  hasSlideLayouts: boolean;
  /** True when the Miami template's theme fonts are still referenced. */
  usesTemplateTheme: boolean;
  /** Slides carrying no visible text at all. */
  emptySlides: string[];
  /** Embedded images (charts built with matplotlib land here). */
  imageCount: number;
}

const LAYOUT_TYPE = "officeDocument/2006/relationships/slideLayout";

export async function inspectPptx(path: string): Promise<PptxReport> {
  const zip = await JSZip.loadAsync(readFileSync(path));

  const slideNames = Object.keys(zip.files)
    .filter((name) => /^ppt\/slides\/slide\d+\.xml$/.test(name))
    .sort((a, b) => {
      const n = (s: string) => Number(/(\d+)/.exec(s)?.[1] ?? 0);
      return n(a) - n(b);
    });

  const duplicates: string[] = [];
  const missing: string[] = [];
  const empty: string[] = [];

  for (const slide of slideNames) {
    const relsPath = slide.replace(
      /^ppt\/slides\/(slide\d+\.xml)$/,
      "ppt/slides/_rels/$1.rels",
    );
    const relsFile = zip.file(relsPath);
    const rels = relsFile ? await relsFile.async("string") : "";
    const layoutRefs = (rels.match(/<Relationship\b[^>]*?\/>/g) ?? []).filter((tag) =>
      tag.includes(LAYOUT_TYPE),
    );
    if (layoutRefs.length > 1) duplicates.push(slide);
    if (layoutRefs.length === 0) missing.push(slide);

    const xml = await (zip.file(slide)?.async("string") ?? Promise.resolve(""));
    // <a:t> holds every run of visible text. A slide with none is a blank.
    const text = (xml.match(/<a:t>([\s\S]*?)<\/a:t>/g) ?? [])
      .map((t) => t.replace(/<[^>]+>/g, ""))
      .join("")
      .trim();
    if (!text) empty.push(slide);
  }

  const names = Object.keys(zip.files);
  // EVERY theme part, not the first one in zip order. An earlier version read
  // whichever matched first and reported "not the Miami template" for decks
  // whose master theme was Roboto Condensed / Roboto: a package carries several
  // themes (slide master, notes master, handout master) and their order in the
  // archive is not meaningful. Reporting a correct deck as off-brand is exactly
  // the kind of wrong answer that wastes the reader's time.
  const themeFiles = names.filter((n) => /^ppt\/theme\/theme\d+\.xml$/.test(n));
  const themes = await Promise.all(
    themeFiles.map((n) => zip.file(n)?.async("string") ?? Promise.resolve("")),
  );
  const theme = themes.join("\n");

  return {
    slideCount: slideNames.length,
    slidesWithDuplicateLayout: duplicates,
    slidesWithoutLayout: missing,
    hasSlideLayouts: names.some((n) => /^ppt\/slideLayouts\/slideLayout\d+\.xml$/.test(n)),
    // The Miami template's typefaces, which python-pptx carries over only when
    // the deck was opened FROM the template rather than from a blank
    // Presentation(). The system prompt requires the template, and nothing else
    // in the package reveals whether it was used.
    usesTemplateTheme: /Roboto/i.test(theme),
    emptySlides: empty,
    imageCount: names.filter((n) => /^ppt\/media\/.+\.(png|jpe?g|svg|emf)$/i.test(n))
      .length,
  };
}

/** Structural problems that would make PowerPoint refuse or a student complain. */
export function pptxProblems(report: PptxReport): string[] {
  const problems: string[] = [];
  if (report.slideCount === 0) problems.push("the package contains no slides");
  for (const slide of report.slidesWithDuplicateLayout) {
    problems.push(`${slide} declares its slideLayout more than once`);
  }
  for (const slide of report.slidesWithoutLayout) {
    problems.push(`${slide} declares no slideLayout`);
  }
  if (!report.hasSlideLayouts) {
    problems.push("the package has no slide layouts, so it carries no branding");
  }
  if (report.emptySlides.length) {
    problems.push(`slides with no text at all: ${report.emptySlides.join(", ")}`);
  }
  return problems;
}
