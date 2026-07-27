/**
 * The house chart style for anything ChatISA plots (design 2026-07-25).
 *
 * Pure and client-safe: no "server-only", no fs, no credentials. The
 * get_miami_style tool serves the guidance to the model, the system prompt
 * carries the short version, and the tests here pin the palette.
 *
 * Every hex below was checked with a palette validator (CVD separation,
 * normal-vision separation, lightness band, chroma floor, contrast against the
 * surface) rather than chosen by eye. The results drove the contract:
 *
 *   - Charcoal #585E60 is a pure gray (chroma 0.008). It works beautifully as
 *     the SECOND colour, where gray means "context" and red means "the series
 *     you care about", but any other dark hue placed beside it collapses.
 *     Slate blue #3E5468 sits at normal-vision deltaE 5.5 from it (floor 15),
 *     which is why the blue here is #1D5FAD instead.
 *   - No categorical palette stays readable on colour alone past 3 slots, so
 *     from 3 series up, direct labels or distinct shapes are REQUIRED, not
 *     optional. That is also what makes the 4-colour brand palette legitimate.
 *   - ColorBrewer Paired was rejected for 9 to 12: its orange and green are
 *     deltaE 0.6 apart under protanopia. Past 8, the honest answer is to change
 *     the question, not the palette.
 */

/** The chart surface. Plain white, per the brand guide. */
export const CHART_SURFACE = "#FFFFFF";

/** Text, axes, ticks, and annotation ink. Never a series colour: next to
 * charcoal it is indistinguishable, and the brewer escalation means a fifth
 * brand colour is never needed. */
export const CHART_INK = "#000000";

/** Miami series colours, in assignment order. Order is load-bearing: blue
 * precedes orange, and charcoal is second so the two-series case is the
 * red-on-gray highlight pattern. */
export const MIAMI_SERIES = [
  "#C3142D", // Miami red, template style guide
  "#585E60", // Charcoal, template style guide
  "#1D5FAD", // Agent blue, assets/brand/miami-colors.md
  "#FF7436", // Orange accent, template style guide
] as const;

/**
 * Corn yellow / Highlight. Deliberately NOT in MIAMI_SERIES: at 1.31:1 against
 * white it is invisible as a line or a small point. Legal only as a large fill
 * with an outline and a visible label.
 */
export const MIAMI_HIGHLIGHT = "#FFDF65";
export const MIAMI_HIGHLIGHT_OUTLINE = "#585E60";

/** Teal from the template style guide, which reserves it for hyperlinks. Kept
 * here so the guidance can say "not a data colour" with the value attached. */
export const MIAMI_LINK_TEAL = "#84D6D3";

export const DARK2_MAX = 8;
export const CATEGORICAL_MAX = 12;

export interface PaletteChoice {
  /** Which palette to use. "refuse" means the chart form is wrong for the data. */
  kind: "miami" | "dark2" | "refuse";
  /** Literal hex codes, when the palette is enumerable. */
  colors: string[];
  /** The brewer palette name for ggplot2/matplotlib, when kind is "dark2". */
  brewer: string | null;
  /**
   * Whether colour alone is enough. False means the chart MUST also carry
   * direct labels, distinct shapes, or distinct line types.
   */
  colorAloneIsEnough: boolean;
  /** One line for the model, and for the student if it explains a refusal. */
  note: string;
}

/**
 * The palette for n series, with the rule about whether colour can carry
 * identity by itself.
 */
export function paletteFor(n: number): PaletteChoice {
  const count = Math.max(1, Math.floor(n));
  if (count <= MIAMI_SERIES.length) {
    return {
      kind: "miami",
      colors: MIAMI_SERIES.slice(0, count),
      brewer: null,
      colorAloneIsEnough: count <= 2,
      note:
        count === 1
          ? "One series: Miami red on white."
          : count === 2
            ? "Two series: Miami red for the series in focus, charcoal for context."
            : `${count} series: add direct labels, distinct shapes, or distinct line types. Colour alone does not separate these reliably.`,
    };
  }
  if (count <= DARK2_MAX) {
    return {
      kind: "dark2",
      colors: [],
      brewer: "Dark2",
      colorAloneIsEnough: false,
      note: `${count} series is past the Miami palette. Use ColorBrewer Dark2, and label or shape every series: Dark2 does not separate reliably on colour alone past three either.`,
    };
  }
  return {
    kind: "refuse",
    colors: [],
    brewer: null,
    colorAloneIsEnough: false,
    note:
      count <= CATEGORICAL_MAX
        ? `${count} categories cannot be told apart by colour in any palette. Group the small ones into "Other", split into small multiples (facets), or plot the ranked top few. Say which you did.`
        : `${count} categories is far past what a categorical chart can show. Aggregate, facet, or answer with a sorted table instead.`,
  };
}

/** The palette contract as lines of text, for the tool output and the prompt. */
export function paletteRules(): string {
  return [
    `Surface: ${CHART_SURFACE}. Text, axes, and annotations: ${CHART_INK}.`,
    `Series colours in this order: ${MIAMI_SERIES.join(", ")}.`,
    "1 series: red. 2 series: red for the focus, charcoal for context.",
    "3 or 4 series: the above PLUS direct labels, shapes, or line types. Colour alone is not enough.",
    "5 to 8 series: ColorBrewer Dark2, still with labels or shapes.",
    "9 or more: do not colour them. Group the tail into Other, facet, or plot the ranked top few, and say so.",
    `${MIAMI_HIGHLIGHT} (corn yellow) is a fill only, with a ${MIAMI_HIGHLIGHT_OUTLINE} outline and a visible label. It is invisible as a line or small point on white.`,
    `${MIAMI_LINK_TEAL} (teal) is for hyperlinks only, never data.`,
    "Never use a colour ramp for categories, and never a second y axis.",
  ].join("\n");
}

/** Rules that hold in every runtime, including the network-isolated provider
 * sandboxes. No package beyond ggplot2 or matplotlib is named here. */
export function portableRules(): string {
  return [
    "Title states the finding, not the variables: \"ISA 401 grades ran higher for three of four students\", not \"Grades by course\".",
    "Subtitle carries the insight or the caveat. Neither line restates the axis labels.",
    "Axis titles are capitalized and name the units. Sentence case for category labels.",
    "No pie or donut charts (see the pie rule). No 3D, no second y axis, no colour ramp for categories.",
    "Keep gridlines minimal and light, drop the panel border, and put the legend at the bottom when there is one.",
    "Fonts: ask for a condensed sans and let it fall back. The deck template uses Roboto Condensed and Roboto, which a sandbox almost never has, and matplotlib substitutes silently rather than failing. Never hardcode a single family.",
    "Start bar and column axes at zero. Sort categorical bars by value unless the categories have a natural order.",
  ].join("\n");
}

/** The pie-chart policy, stated once and identically wherever it appears. */
export const PIE_POLICY =
  "Do not build a pie or donut chart by default. Tell the student once that pie charts are a suboptimal way to show data, because people compare angles and areas poorly, and that a bar chart ranks categories more accurately while a dot chart handles many categories or a two-value comparison better. Offer the better form. If the student still wants a pie, build it without arguing again.";

export type ChartLanguage = "r" | "python";

/**
 * The annotation guidance. Split by runtime, because the provider sandboxes
 * that build decks have no network and cannot install anything: naming ggrepel
 * or adjustText there produces code that fails on import.
 */
export function annotationRules(language: ChartLanguage): string {
  const rich =
    language === "r"
      ? [
          "Browser runtime (run_r): ggrepel and ggtext are installed.",
          "Label a small number of points with ggrepel::geom_text_repel(min.segment.length = 0, box.padding = 0.5, seg.colour = \"#585E60\"), which moves labels off the geoms and off each other.",
          "Put the series colours into the subtitle as coloured words with ggtext::element_markdown() and drop the legend entirely: legend.position = \"none\" plus a subtitle like \"<span style='color:#C3142D'>ISA 401</span> vs <span style='color:#585E60'>ISA 444</span>\".",
        ]
      : [
          "Browser runtime (run_python): adjustText and highlight_text are installed.",
          "Label a small number of points with adjustText.adjust_text(texts, arrowprops=dict(arrowstyle=\"-\", color=\"#585E60\")), which pushes labels apart and off the marks.",
          "Colour words inside the subtitle with highlight_text.ax_text and drop the legend, so identity lives in the subtitle instead of a legend box.",
        ];
  const portable = [
    "Hosted provider sandbox (code_execution or code_interpreter): NO extra packages. Do not import ggrepel, ggtext, adjustText, or highlight_text there; the container has no network and the import will fail. Annotation is not required in a deck chart; the palette, the title, and the subtitle are.",
    "Without those packages, label only when there are few points, offset each label by the sign of its value, keep labels inside expanded axis limits, and never place one on top of a geom.",
  ];
  return [...rich, ...portable].join("\n");
}
