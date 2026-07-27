import {
  PIE_POLICY,
  paletteRules,
  portableRules,
} from "@/lib/ask/chart-style";

/**
 * The house chart rules for a module whose model has NO get_miami_style tool
 * (added 2026-07-25 at the professor's instruction: "Coding Tutor should take
 * the same plotting instructions and styling instructions as that in the Ask
 * Anything").
 *
 * Ask Anything carries only the short version in its system prompt and fetches
 * the full contract, plus a runnable exemplar, through get_miami_style. The
 * Coding Tutor is a plain chat with no tools, so the contract has to be inline
 * or it is not there at all. The palette and the title rules are imported from
 * lib/ask/chart-style rather than restated, so the two modules can never drift:
 * that file is the single source of truth and its unit tests pin every hex.
 *
 * What is NOT inlined is the code exemplar. It would roughly quadruple the
 * prompt on every tutoring turn, including the many that never mention a chart,
 * and its R form opens with library(ggplot2), which directly contradicts the
 * CODING_STYLE_RULES this module also carries. The reconciliation below states
 * the package-qualified form instead.
 */
export function chartRulesForPrompt(): string {
  return [
    "## Charts and plots",
    "",
    "Every figure you write code for follows the house style below. These are checked conventions, not preferences, and they apply to R and to Python alike.",
    "",
    "### Palette",
    paletteRules(),
    "",
    "### Titles, subtitles, and form",
    portableRules(),
    "",
    "### Pie charts",
    PIE_POLICY,
    "",
    "### Labels",
    "When there are few enough points to label, label them, and place the labels so they never sit on a geom or on each other.",
    "In R use ggrepel::geom_text_repel(min.segment.length = 0, box.padding = 0.5) for the labels, and ggtext::element_markdown() to colour the series words inside the subtitle so the legend can be dropped entirely.",
    "In Python use adjustText.adjust_text(texts, arrowprops=dict(arrowstyle=\"-\", color=\"#585E60\")) for the labels, and highlight_text.ax_text to colour the series words in the subtitle.",
    "",
    "### How this fits the code style above",
    "The R style rules still apply to chart code: call functions as package::function(), guard each package with if(require(package)==FALSE) install.packages(package), and use the native pipe. So it is ggplot2::ggplot() and ggrepel::geom_text_repel(), never library(ggplot2).",
    "Explain the styling choice in one line when it teaches something, for example why two series are red and charcoal rather than two bright colours. Do not lecture about the palette on every chart.",
  ].join("\n");
}
