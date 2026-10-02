/**
 * Text alternatives for plots the runtimes draw (#23).
 *
 * A plot arrives as a PNG or SVG data URL, so the picture itself says nothing to
 * a screen reader. What we can say cheaply comes from two places:
 *  - the figure: the Python worker reads the matplotlib figure before it closes
 *    it (title, axis labels, mark types, legend entries, panel count);
 *  - the code: titles, axis labels and chart calls written in the script
 *    (ggplot2 labs()/geom_*, base R plot(main=), matplotlib set_title, ggsql
 *    DRAW/LABEL). This is the only source for R and SQL.
 * The figure wins where both have a value. When neither says anything we fall
 * back to "Plot 1 of 2 from your Python code (no title)", which at least places
 * the plot and is honest about what is missing.
 */

export type PlotKind =
  | "lines"
  | "points"
  | "bars"
  | "histogram"
  | "box"
  | "violin"
  | "density"
  | "heatmap"
  | "area"
  | "pie"
  | "segments"
  | "text";

export interface PlotInfo {
  /** Mark types, in the order first seen. */
  kinds?: PlotKind[];
  title?: string;
  xLabel?: string;
  yLabel?: string;
  /** Legend entries (series or groups). */
  series?: string[];
  /** Number of panels (subplots); more than one is worth saying. */
  panels?: number;
  /** Where the description came from, for the "Describe" disclosure. */
  source?: "figure" | "code";
}

const SINGLE: Record<PlotKind, string> = {
  lines: "line chart",
  points: "scatter plot",
  bars: "bar chart",
  histogram: "histogram",
  box: "box plot",
  violin: "violin plot",
  density: "density plot",
  heatmap: "heatmap",
  area: "area chart",
  pie: "pie chart",
  segments: "segment chart",
  text: "text chart",
};

const PLURAL: Record<PlotKind, string> = {
  lines: "lines",
  points: "points",
  bars: "bars",
  histogram: "histogram bars",
  box: "box plots",
  violin: "violins",
  density: "density curves",
  heatmap: "a heatmap",
  area: "shaded areas",
  pie: "a pie",
  segments: "segments",
  text: "text labels",
};

/** Kinds the figure cannot tell apart from plainer marks (a histogram is bars,
 * a box plot is lines), so the code's word for them is better. */
const SPECIFIC: PlotKind[] = ["histogram", "box", "violin", "density", "heatmap", "pie"];

const LANGUAGE_LABEL: Record<string, string> = {
  python: "Python",
  r: "R",
  sql: "SQL",
};

function clean(s: string | undefined): string | undefined {
  if (typeof s !== "string") return undefined;
  // Drop matplotlib mathtext dollars and collapse whitespace; keep it short.
  const t = s.replace(/\$/g, "").replace(/\s+/g, " ").trim();
  if (!t) return undefined;
  return t.length > 80 ? `${t.slice(0, 77)}...` : t;
}

function list(items: string[]): string {
  if (items.length <= 1) return items.join("");
  if (items.length === 2) return `${items[0]} and ${items[1]}`;
  return `${items.slice(0, -1).join(", ")} and ${items[items.length - 1]}`;
}

/** "scatter plot", or "chart with lines and points" for a layered figure. */
export function kindPhrase(kinds: PlotKind[] | undefined): string | undefined {
  if (!kinds || kinds.length === 0) return undefined;
  if (kinds.length === 1) return SINGLE[kinds[0]];
  return `chart with ${list(kinds.slice(0, 4).map((k) => PLURAL[k]))}`;
}

/** Combines figure facts with code hints: the figure wins per field, except
 * that a specific chart word from the code (histogram, box plot) replaces the
 * figure's plainer bars or lines. */
export function mergePlotInfo(
  figure: PlotInfo | undefined,
  code: PlotInfo | undefined,
): PlotInfo {
  if (!figure) return code ?? {};
  if (!code) return figure;
  const codeSpecific = (code.kinds ?? []).filter((k) => SPECIFIC.includes(k));
  const kinds =
    codeSpecific.length > 0
      ? codeSpecific
      : figure.kinds && figure.kinds.length > 0
        ? figure.kinds
        : code.kinds;
  return {
    kinds,
    title: clean(figure.title) ?? clean(code.title),
    xLabel: clean(figure.xLabel) ?? clean(code.xLabel),
    yLabel: clean(figure.yLabel) ?? clean(code.yLabel),
    series: figure.series && figure.series.length > 0 ? figure.series : code.series,
    panels: figure.panels ?? code.panels,
    source: "figure",
  };
}

/**
 * The alt text for one plot. `index`/`total` place it among the session's
 * plots when there is more than one.
 */
export function plotAltText(opts: {
  language: string;
  info?: PlotInfo;
  index?: number;
  total?: number;
}): string {
  const info = opts.info ?? {};
  const lang = LANGUAGE_LABEL[opts.language] ?? opts.language;
  const where =
    opts.total && opts.total > 1 && opts.index != null
      ? `Plot ${opts.index + 1} of ${opts.total}`
      : "Plot";
  const kind = kindPhrase(info.kinds);
  const title = clean(info.title);
  const x = clean(info.xLabel);
  const y = clean(info.yLabel);
  const series = (info.series ?? []).map(clean).filter(Boolean) as string[];

  if (!kind && !title && !x && !y && series.length === 0) {
    return `${where} from your ${lang} code (no title)`;
  }

  const parts: string[] = [];
  let head = kind ? `${where}: ${kind}` : where;
  if (title) head += ` titled "${title}"`;
  parts.push(head);
  if (x) parts.push(`x axis ${x}`);
  if (y) parts.push(`y axis ${y}`);
  if (series.length > 0) {
    const shown = series.slice(0, 5);
    parts.push(
      `legend ${list(shown)}${series.length > shown.length ? ` and ${series.length - shown.length} more` : ""}`,
    );
  }
  if (info.panels && info.panels > 1) parts.push(`${info.panels} panels`);
  return `${parts.join(", ")}. From your ${lang} code.`;
}

/** Label/value rows for a "Describe this plot" disclosure. */
export function plotDetails(info: PlotInfo | undefined, language: string) {
  const rows: { label: string; value: string }[] = [];
  const i = info ?? {};
  const kind = kindPhrase(i.kinds);
  if (kind) rows.push({ label: "Chart type", value: kind[0].toUpperCase() + kind.slice(1) });
  const title = clean(i.title);
  if (title) rows.push({ label: "Title", value: title });
  const x = clean(i.xLabel);
  if (x) rows.push({ label: "X axis", value: x });
  const y = clean(i.yLabel);
  if (y) rows.push({ label: "Y axis", value: y });
  const series = (i.series ?? []).map(clean).filter(Boolean) as string[];
  if (series.length > 0) rows.push({ label: "Legend", value: series.join(", ") });
  if (i.panels && i.panels > 1) rows.push({ label: "Panels", value: String(i.panels) });
  rows.push({ label: "Language", value: LANGUAGE_LABEL[language] ?? language });
  rows.push({
    label: "Described from",
    value:
      i.source === "figure"
        ? "The figure itself"
        : "Your code (titles, labels and chart calls it contains)",
  });
  return rows;
}

// ---------------------------------------------------------------------------
// Code hints

/** A quoted string literal: '...' or "..." (no escapes across quotes). */
const Q = `(?:"([^"\\n]*)"|'([^'\\n]*)')`;

function firstQuoted(code: string, prefix: string): string | undefined {
  const m = new RegExp(`${prefix}\\s*${Q}`).exec(code);
  return m ? (m[1] ?? m[2]) : undefined;
}

/** Drops full-line comments so a commented-out chart is not described. */
function stripComments(code: string, marker: string): string {
  return code
    .split("\n")
    .filter((line) => !line.trimStart().startsWith(marker))
    .join("\n");
}

function addKind(kinds: PlotKind[], k: PlotKind | undefined) {
  if (k && !kinds.includes(k)) kinds.push(k);
}

/** The argument text of the first call to `name(` (balanced parentheses). */
function callArgs(code: string, name: RegExp): string | undefined {
  const m = name.exec(code);
  if (!m) return undefined;
  let depth = 1;
  const start = m.index + m[0].length;
  for (let i = start; i < code.length; i++) {
    const c = code[i];
    if (c === "(") depth++;
    else if (c === ")" && --depth === 0) return code.slice(start, i);
  }
  return code.slice(start);
}

const GEOMS: Record<string, PlotKind> = {
  point: "points",
  jitter: "points",
  count: "points",
  line: "lines",
  path: "lines",
  step: "lines",
  smooth: "lines",
  col: "bars",
  bar: "bars",
  histogram: "histogram",
  freqpoly: "lines",
  boxplot: "box",
  violin: "violin",
  density: "density",
  tile: "heatmap",
  raster: "heatmap",
  area: "area",
  ribbon: "area",
  segment: "segments",
  text: "text",
  label: "text",
};

function rHints(code: string): PlotInfo {
  const src = stripComments(code, "#");
  const kinds: PlotKind[] = [];
  for (const m of src.matchAll(/\bgeom_(\w+)\s*\(/g)) addKind(kinds, GEOMS[m[1]]);
  const labs = callArgs(src, /\blabs\s*\(/);
  let title = labs ? firstQuoted(labs, `\\btitle\\s*=`) : undefined;
  let x = labs ? firstQuoted(labs, `(?:^|[\\s,(])x\\s*=`) : undefined;
  let y = labs ? firstQuoted(labs, `(?:^|[\\s,(])y\\s*=`) : undefined;
  title ??= firstQuoted(src, `\\bggtitle\\s*\\(`);
  x ??= firstQuoted(src, `\\bxlab\\s*\\(`);
  y ??= firstQuoted(src, `\\bylab\\s*\\(`);
  // Base graphics.
  if (/\bhist\s*\(/.test(src)) addKind(kinds, "histogram");
  if (/\bbarplot\s*\(/.test(src)) addKind(kinds, "bars");
  if (/\bboxplot\s*\(/.test(src)) addKind(kinds, "box");
  if (/\bpie\s*\(/.test(src)) addKind(kinds, "pie");
  const plotArgs = callArgs(src, /(?:^|[^\w.])plot\s*\(/);
  if (plotArgs != null) {
    addKind(kinds, /\btype\s*=\s*["'][lb]["']/.test(plotArgs) ? "lines" : "points");
  }
  if (/\blines\s*\(/.test(src)) addKind(kinds, "lines");
  title ??= firstQuoted(src, `\\bmain\\s*=`);
  x ??= firstQuoted(src, `\\bxlab\\s*=`);
  y ??= firstQuoted(src, `\\bylab\\s*=`);
  return { kinds, title, xLabel: x, yLabel: y, source: "code" };
}

const SEABORN: Record<string, PlotKind> = {
  scatterplot: "points",
  stripplot: "points",
  swarmplot: "points",
  regplot: "points",
  lmplot: "points",
  relplot: "points",
  lineplot: "lines",
  barplot: "bars",
  countplot: "bars",
  histplot: "histogram",
  displot: "histogram",
  boxplot: "box",
  violinplot: "violin",
  kdeplot: "density",
  heatmap: "heatmap",
};

const PANDAS_KIND: Record<string, PlotKind> = {
  line: "lines",
  bar: "bars",
  barh: "bars",
  hist: "histogram",
  box: "box",
  kde: "density",
  density: "density",
  area: "area",
  pie: "pie",
  scatter: "points",
};

function pythonHints(code: string): PlotInfo {
  const src = stripComments(code, "#");
  const kinds: PlotKind[] = [];
  for (const m of src.matchAll(/\bsns\.(\w+)\s*\(/g)) addKind(kinds, SEABORN[m[1]]);
  const pandasKind = /\.plot\s*\([^)]*\bkind\s*=\s*["'](\w+)["']/.exec(src);
  if (pandasKind) addKind(kinds, PANDAS_KIND[pandasKind[1]]);
  for (const m of src.matchAll(/\.(plot|scatter|bar|barh|hist|boxplot|violinplot|imshow|pie|fill_between|step)\s*\(/g)) {
    const map: Record<string, PlotKind> = {
      plot: "lines",
      step: "lines",
      scatter: "points",
      bar: "bars",
      barh: "bars",
      hist: "histogram",
      boxplot: "box",
      violinplot: "violin",
      imshow: "heatmap",
      pie: "pie",
      fill_between: "area",
    };
    if (m[1] === "plot" && pandasKind) continue;
    addKind(kinds, map[m[1]]);
  }
  // A legend's title= names the legend, not the chart.
  const noLegend = src.replace(/\blegend\s*\([^)]*\)/g, "");
  const title =
    firstQuoted(src, `\\.(?:set_title|title|suptitle)\\s*\\(`) ??
    firstQuoted(noLegend, `\\btitle\\s*=`);
  const x =
    firstQuoted(src, `\\.(?:set_xlabel|xlabel)\\s*\\(`) ??
    firstQuoted(src, `\\bxlabel\\s*=`);
  const y =
    firstQuoted(src, `\\.(?:set_ylabel|ylabel)\\s*\\(`) ??
    firstQuoted(src, `\\bylabel\\s*=`);
  return { kinds, title, xLabel: x, yLabel: y, source: "code" };
}

const GGSQL_MARKS: Record<string, PlotKind> = {
  point: "points",
  line: "lines",
  path: "lines",
  bar: "bars",
  col: "bars",
  histogram: "histogram",
  boxplot: "box",
  violin: "violin",
  density: "density",
  area: "area",
  ribbon: "area",
  tile: "heatmap",
  segment: "segments",
  rule: "lines",
  text: "text",
};

function sqlHints(code: string): PlotInfo {
  const src = stripComments(code, "--");
  const kinds: PlotKind[] = [];
  for (const m of src.matchAll(/\bDRAW\s+(\w+)/gi)) {
    addKind(kinds, GGSQL_MARKS[m[1].toLowerCase()]);
  }
  const label = /\bLABEL\b([\s\S]*)$/i.exec(src)?.[1] ?? "";
  const pick = (key: string) => firstQuoted(label, `(?:^|[\\s,])${key}\\s*=>`);
  return {
    kinds,
    title: pick("title"),
    xLabel: pick("x"),
    yLabel: pick("y"),
    source: "code",
  };
}

/** Best-effort hints from the code that drew the plot. */
export function plotInfoFromCode(code: string, language: string): PlotInfo {
  try {
    if (language === "r") return rHints(code);
    if (language === "python") return pythonHints(code);
    if (language === "sql") return sqlHints(code);
  } catch {
    // A hint is optional; never let it break a run.
  }
  return { source: "code" };
}

/** Validates the figure facts a worker sent (they cross a postMessage). */
export function parsePlotInfo(raw: unknown): PlotInfo | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const r = raw as Record<string, unknown>;
  const str = (v: unknown) => (typeof v === "string" ? v : undefined);
  const kinds = Array.isArray(r.kinds)
    ? (r.kinds.filter((k) => typeof k === "string" && k in SINGLE) as PlotKind[])
    : undefined;
  const series = Array.isArray(r.series)
    ? (r.series.filter((s) => typeof s === "string" && s.trim()) as string[]).slice(0, 12)
    : undefined;
  return {
    kinds,
    title: str(r.title),
    xLabel: str(r.xLabel),
    yLabel: str(r.yLabel),
    series,
    panels: typeof r.panels === "number" ? r.panels : undefined,
    source: "figure",
  };
}

/** Describes a plot from what is known: worker facts (if any) plus code hints. */
export function describePlot(
  code: string,
  language: string,
  figure?: unknown,
): PlotInfo {
  return mergePlotInfo(parsePlotInfo(figure), plotInfoFromCode(code, language));
}
