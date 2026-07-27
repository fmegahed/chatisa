import "server-only";
import { readFile } from "node:fs/promises";
import path from "node:path";
import {
  HOSTED_CHART_EXAMPLE,
  PYTHON_CHART_EXAMPLE,
  R_CHART_EXAMPLE,
} from "@/lib/ask/chart-examples";
import {
  PIE_POLICY,
  annotationRules,
  paletteRules,
  portableRules,
} from "@/lib/ask/chart-style";

/**
 * The get_miami_style tool's backing store (slice C): brand assets distilled
 * from the professor's figure set and deck template, served to the model on
 * demand so Miami-themed output costs tokens only when a student asks for it.
 * Files live in web/assets/brand/ (server-readable, not public/).
 *
 * The chart kinds (2026-07-25) are GENERATED rather than read from a file: the
 * palette contract, the tier split, and the exemplar have to agree, so they are
 * assembled from lib/ask/chart-style and lib/ask/chart-examples, which the unit
 * tests pin.
 */

export const MIAMI_STYLE_KINDS = [
  "tikz",
  "gantt",
  "colors",
  "latex-doc",
  "charts-r",
  "charts-python",
] as const;
export type MiamiStyleKind = (typeof MIAMI_STYLE_KINDS)[number];

const STYLE_OUTPUT_MAX = 8_000;

/** The chart guidance for one language: rules first, then the exemplar. The
 * hosted variant rides along in the Python answer, because the sandboxes that
 * build decks are Python and must not import the browser-only packages. */
function chartGuidance(language: "r" | "python"): string {
  const sections = [
    "## Palette",
    paletteRules(),
    "",
    "## Titles, subtitles, and form (these hold in EVERY runtime)",
    portableRules(),
    "",
    "## Pie charts",
    PIE_POLICY,
    "",
    "## Annotations",
    annotationRules(language),
    "",
    language === "r"
      ? "## Exemplar (R, browser runtime)"
      : "## Exemplar (Python, browser runtime)",
    "```" + (language === "r" ? "r" : "python"),
    language === "r" ? R_CHART_EXAMPLE : PYTHON_CHART_EXAMPLE,
    "```",
  ];
  if (language === "python") {
    sections.push(
      "",
      "## Exemplar (Python, HOSTED sandbox: no extra packages, for deck and document charts)",
      "```python",
      HOSTED_CHART_EXAMPLE,
      "```",
    );
  }
  return sections.join("\n");
}

const GENERATED: Partial<Record<MiamiStyleKind, { note: string; content: () => string }>> = {
  "charts-r": {
    note: "The house chart style for R. Follow the palette and the title contract exactly; they are checked, not preferences. Adapt the exemplar to the student's data rather than starting from a blank ggplot.",
    content: () => chartGuidance("r"),
  },
  "charts-python": {
    note: "The house chart style for Python. Follow the palette and the title contract exactly. Use the browser exemplar for run_python and the hosted exemplar for charts inside a deck or document, which must not import extra packages.",
    content: () => chartGuidance("python"),
  },
};

const FILES: Record<string, { file: string; note: string }> = {
  tikz: {
    file: "miami-tikz-style.tex",
    note: "Start from this preamble and vocabulary. Keep the palette and box/arrow styles; replace the minimal example with the figure's content. Output a complete, compilable .tex file (students compile in Overleaf).",
  },
  gantt: {
    file: "exemplar-gantt.tex",
    note: "A Miami-styled pgfgantt exemplar. Keep its gantt chart option block (colors, fonts, grid) and replace the groups, bars, and milestones with the student's timeline.",
  },
  colors: {
    file: "miami-colors.md",
    note: "The palette and composition conventions, for any Miami-themed output.",
  },
  "latex-doc": {
    file: "miami-colors.md",
    note: "Use the 'LaTeX documents' section for report/memo preambles: helvet, geometry 1in, Miami-red section headings and links, with the palette above.",
  },
};

const cache = new Map<MiamiStyleKind, string>();

export async function getMiamiStyle(
  kind: string,
): Promise<{ kind: string; note: string; content: string } | { error: string }> {
  const k = MIAMI_STYLE_KINDS.find((s) => s === kind);
  if (!k) {
    return {
      error: `Unknown kind "${kind}". Use one of: ${MIAMI_STYLE_KINDS.join(", ")}.`,
    };
  }
  const generated = GENERATED[k];
  if (generated) {
    return { kind: k, note: generated.note, content: generated.content() };
  }
  let content = cache.get(k);
  if (content === undefined) {
    try {
      const file = path.join(process.cwd(), "assets", "brand", FILES[k].file);
      content = await readFile(file, "utf8");
      cache.set(k, content);
    } catch {
      return { error: "The style assets are not available on this server." };
    }
  }
  return {
    kind: k,
    note: FILES[k].note,
    content:
      content.length > STYLE_OUTPUT_MAX
        ? `${content.slice(0, STYLE_OUTPUT_MAX)}\n% [truncated]`
        : content,
  };
}
