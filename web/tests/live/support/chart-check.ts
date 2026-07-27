import { MIAMI_SERIES, MIAMI_LINK_TEAL } from "@/lib/ask/chart-style";

/**
 * Checks generated chart code against the house style.
 *
 * Static analysis of the code, not of the rendered picture. That is a real limit
 * and it is stated rather than papered over: this can tell that a palette hex was
 * used, not that it landed on the right series; it can tell a title exists, not
 * that the title is true. Those need a human, and the specs save the code and the
 * plot so a human can look.
 *
 * What it does catch is the class of failure that actually happens: the model
 * ignoring the palette entirely and reaching for viridis, rainbow, or ggplot's
 * default hues, and colouring more categories than colour can separate.
 */

export interface ChartFindings {
  /** Miami palette hexes present in the code, in the order they appear. */
  miamiHexes: string[];
  /** Palette machinery that the house style rules out. */
  bannedPalettes: string[];
  /** True when a title is set at all. */
  hasTitle: boolean;
  /** The title text, for a human to judge. */
  title: string | null;
  /** The subtitle text, if any. */
  subtitle: string | null;
  /** A secondary encoding beyond colour: shapes, line types, or direct labels. */
  hasSecondaryEncoding: boolean;
  /** Pie or donut geometry, which the style forbids by default. */
  usesPie: boolean;
  /** A second y axis, which the style forbids outright. */
  usesSecondAxis: boolean;
  /** Teal used as a data colour, which is reserved for links. */
  usesLinkTealAsData: boolean;
}

/** Palette calls that contradict the contract. Dark2 is the one allowed brewer
 * escalation, so it is excluded from the scale_*_brewer match. */
const BANNED = [
  /viridis/i,
  /rainbow\s*\(/i,
  /\bheat\.colors\b/i,
  /\bterrain\.colors\b/i,
  /cm\.colors/i,
  /\bjet\b/,
  /plt\.cm\.(?!gray|Grays)[A-Za-z]+/,
  /scale_(?:colour|color|fill)_brewer\((?![^)]*Dark2)/,
  /scale_(?:colour|color|fill)_viridis/,
  /scale_(?:colour|color|fill)_gradient/,
];

function firstMatch(code: string, patterns: RegExp[]): string[] {
  return patterns
    .filter((p) => p.test(code))
    .map((p) => (code.match(p) ?? [p.source])[0]);
}

/** Pulls a quoted string out of `title = "..."` or `set_title("...")`. */
function extractLabel(code: string, keys: string[]): string | null {
  for (const key of keys) {
    const patterns = [
      new RegExp(`${key}\\s*=\\s*"([^"]{3,})"`),
      new RegExp(`${key}\\s*=\\s*'([^']{3,})'`),
      new RegExp(`${key}\\s*\\(\\s*"([^"]{3,})"`),
      new RegExp(`${key}\\s*\\(\\s*'([^']{3,})'`),
    ];
    for (const pattern of patterns) {
      const match = pattern.exec(code);
      if (match) return match[1];
    }
  }
  return null;
}

export function inspectChartCode(code: string): ChartFindings {
  const upper = code.toUpperCase();
  const title =
    extractLabel(code, ["title", "set_title", "suptitle"]) ?? null;
  const subtitle = extractLabel(code, ["subtitle"]) ?? null;

  return {
    miamiHexes: MIAMI_SERIES.filter((hex) => upper.includes(hex.toUpperCase())),
    bannedPalettes: firstMatch(code, BANNED),
    hasTitle: title !== null,
    title,
    subtitle,
    hasSecondaryEncoding:
      /shape\s*=|linetype\s*=|scale_shape|scale_linetype|marker\s*=|geom_text|geom_label|geom_text_repel|adjust_text|ax\.text|annotate\(|ax\.annotate/.test(
        code,
      ),
    usesPie: /coord_polar|pie\s*\(|plt\.pie|ax\.pie|geom_arc_bar/.test(code),
    usesSecondAxis: /sec\.axis|sec_axis|twinx\s*\(|twiny\s*\(/.test(code),
    usesLinkTealAsData: upper.includes(MIAMI_LINK_TEAL.toUpperCase()),
  };
}

/**
 * The house-style verdict for one chart, as a list of human-readable problems.
 * Empty means nothing detectable is wrong, which is weaker than "correct".
 */
export function chartStyleProblems(f: ChartFindings): string[] {
  const problems: string[] = [];
  if (f.miamiHexes.length === 0) {
    problems.push(
      "no Miami palette colour appears in the code, so the chart is not in the house style",
    );
  }
  for (const banned of f.bannedPalettes) {
    problems.push(`uses a palette the style rules out: ${banned}`);
  }
  if (!f.hasTitle) problems.push("no title is set");
  if (f.usesPie) problems.push("builds a pie or donut chart");
  if (f.usesSecondAxis) problems.push("uses a second y axis");
  if (f.usesLinkTealAsData) {
    problems.push("uses the link teal as a data colour");
  }
  return problems;
}
