/**
 * Renders a ggsql (experimental) Vega-Lite spec to an SVG image, so a SQL plot
 * drops into the Plots pane alongside the matplotlib and R plots.
 *
 * ggsql returns a bare spec: it has no size (so it would collapse), keeps column
 * names as axis titles, and leaves colours to the default palette. This module
 * normalises all of that to a clean, on-brand look before rendering. Client-only
 * (uses the DOM and a lazily imported Vega renderer).
 */

// A small Miami-leaning categorical palette: red and corn-gold first, so a
// two-category plot (like the ISA course example) reads on-brand, then a few
// distinct supporting hues for plots with more categories.
const BRAND_CATEGORY = [
  "#C3142D",
  "#F0B323",
  "#1f5fa8",
  "#4a7c59",
  "#8a6d3b",
  "#6b3fa0",
];

type AnySpec = Record<string, unknown>;

/** The first meaningful axis title across layers (ggsql repeats column names,
 * which Vega-Lite would otherwise concatenate into "g401, Grade, Grade"). */
function sharedTitle(layers: AnySpec[], channel: string): unknown {
  for (const layer of layers) {
    const enc = layer.encoding as Record<string, AnySpec> | undefined;
    const title = enc?.[channel]?.title;
    if (typeof title === "string" && title && !title.startsWith("_")) return title;
  }
  return undefined;
}

/** Applies sizing, a single axis title per channel, larger points, and a clean
 * branded theme. Mutates a copy, so the caller's spec is untouched. */
export function normalizeSpec(spec: AnySpec, dark: boolean): AnySpec {
  const s: AnySpec = structuredClone(spec);
  // A faceted spec nests the marks under `spec`; a layered one has `layer` at top.
  const inner = (s.spec as AnySpec) ?? s;
  // ggsql emits width/height "container", which measures the parent element and
  // so collapses to 0 when rendered off-screen. Force a fixed size unless the
  // student set an explicit number.
  if (typeof inner.width !== "number") inner.width = 520;
  if (typeof inner.height !== "number") inner.height = 320;

  const layers = (inner.layer as AnySpec[]) ?? [];
  const xTitle = sharedTitle(layers, "x");
  const yTitle = sharedTitle(layers, "y");
  for (const layer of layers) {
    const enc = (layer.encoding ?? {}) as Record<string, AnySpec>;
    if (enc.x && xTitle != null) enc.x.title = xTitle;
    if (enc.y && yTitle != null) enc.y.title = yTitle;
    const markType =
      typeof layer.mark === "object"
        ? (layer.mark as AnySpec).type
        : layer.mark;
    if (markType === "point" || markType === "circle") {
      layer.mark =
        typeof layer.mark === "object"
          ? { ...(layer.mark as AnySpec), filled: true }
          : { type: markType, filled: true };
      enc.size = { value: 260 };
      layer.encoding = enc;
    }
    // Force the brand palette even when ggsql wrote an explicit colour scale
    // (its default blue/orange would otherwise win over config.range.category).
    for (const channel of ["fill", "color"]) {
      const e = enc[channel];
      if (e && typeof e === "object") {
        e.scale = { ...((e.scale as AnySpec) ?? {}), range: BRAND_CATEGORY };
      }
    }
  }

  const ink = dark ? "#e6e6e6" : "#1a1a1a";
  const line = dark ? "#8a8a8a" : "#54585A";
  // Our theme wins over ggsql's defaults for the keys we set, but keeps the rest.
  s.config = {
    ...((s.config as AnySpec) ?? {}),
    background: dark ? "#0a0a0a" : "white",
    view: { stroke: null },
    range: { category: BRAND_CATEGORY },
    axis: {
      grid: false,
      domainColor: line,
      tickColor: line,
      labelColor: ink,
      titleColor: ink,
    },
    rule: { color: line },
    line: { color: line },
  };
  return s;
}

/**
 * Renders a Vega-Lite spec to an SVG data URL. Draws into a throwaway off-screen
 * element so it never disturbs the page, and always cleans it up.
 */
export async function renderSpecToImage(
  spec: AnySpec,
  opts: { dark: boolean },
): Promise<string> {
  const vegaEmbed = (await import("vega-embed")).default;
  const normalized = normalizeSpec(spec, opts.dark);
  const host = document.createElement("div");
  host.style.position = "fixed";
  host.style.left = "-99999px";
  host.style.top = "0";
  document.body.appendChild(host);
  try {
    const result = await vegaEmbed(host, normalized as never, {
      actions: false,
      renderer: "svg",
    });
    const svg = await result.view.toSVG();
    result.finalize();
    return "data:image/svg+xml;charset=utf-8," + encodeURIComponent(svg);
  } finally {
    host.remove();
  }
}

/**
 * A copy of the SQL with comments and string literals replaced by spaces of the
 * same length, so keyword and statement-boundary searches ignore a VISUALISE or
 * a `;` that sits inside a `-- comment`, a block comment, or a 'string'. Indices
 * line up with the original, so callers can slice the original by them.
 */
function maskSqlNoise(code: string): string {
  let out = "";
  let i = 0;
  while (i < code.length) {
    const two = code.slice(i, i + 2);
    if (two === "--") {
      let j = i;
      while (j < code.length && code[j] !== "\n") j++;
      out += " ".repeat(j - i);
      i = j;
    } else if (two === "/*") {
      let j = i + 2;
      while (j < code.length && code.slice(j, j + 2) !== "*/") j++;
      j = Math.min(code.length, j + 2);
      out += " ".repeat(j - i);
      i = j;
    } else if (code[i] === "'") {
      let j = i + 1;
      while (j < code.length && code[j] !== "'") j++;
      j = Math.min(code.length, j + 1);
      out += " ".repeat(j - i);
      i = j;
    } else {
      out += code[i];
      i++;
    }
  }
  return out;
}

/** True when a SQL snippet contains a ggsql VISUALISE clause (outside comments
 * and strings), so it must be routed to the plot engine rather than run as
 * plain SQL. */
export function hasVisualise(code: string): boolean {
  return /\bVISUALISE\b/i.test(maskSqlNoise(code));
}

/**
 * The plain-SQL part of a ggsql query: everything up to the VISUALISE clause.
 * Running this in SQLite populates the tables and yields a result table, which
 * also serves as the fallback if the plot fails. A trailing semicolon is kept
 * off so the SELECT stays a single statement.
 */
export function stripVisualise(code: string): string {
  const at = maskSqlNoise(code).search(/\bVISUALISE\b/i);
  const head = at < 0 ? code : code.slice(0, at);
  return head.trim().replace(/;?\s*$/, "");
}

/**
 * Just the VISUALISE statement, without any preceding CREATE/INSERT: everything
 * from the start of the statement that contains VISUALISE (the last `;` before
 * it) to the end. ggsql receives this, with the referenced tables handed over
 * separately as CSV, so the data-building statements must not be sent (ggsql
 * already has those tables registered and would collide on CREATE).
 */
export function extractVisualiseQuery(code: string): string {
  const masked = maskSqlNoise(code);
  const at = masked.search(/\bVISUALISE\b/i);
  if (at < 0) return code.trim();
  const lastSemicolon = masked.slice(0, at).lastIndexOf(";");
  return code.slice(lastSemicolon + 1).trim();
}
