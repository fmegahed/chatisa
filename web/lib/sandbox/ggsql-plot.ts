/**
 * Renders a ggsql (experimental) Vega-Lite spec to an SVG image, so a SQL plot
 * drops into the Plots pane alongside the matplotlib and R plots.
 *
 * ggsql returns a bare spec: it has no size (so it would collapse), keeps column
 * names as axis titles, and leaves colours to the default palette. This module
 * normalises all of that to a clean, on-brand look before rendering. Client-only
 * (uses the DOM and a lazily imported Vega renderer).
 */

import { parsePlotData, type PlotData } from "@/lib/run/maidr-schema";

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

// ---------------------------------------------------------------------------
// The plotted data, for keyboard and sound exploration (#40).

type Row = Record<string, unknown>;

function field(enc: Record<string, AnySpec>, channel: string): string | undefined {
  const f = enc[channel]?.field;
  return typeof f === "string" ? f : undefined;
}

function isNominal(enc: Record<string, AnySpec>, channel: string): boolean {
  const t = enc[channel]?.type;
  return t === "nominal" || t === "ordinal";
}

function numberOf(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

/** Distinct values in first-seen order. */
function distinct(values: unknown[]): string[] {
  return [...new Set(values.filter((v) => v != null).map(String))];
}

/**
 * The data behind a ggsql chart, in the neutral shape MAIDR's builder reads
 * (lib/run/maidr-schema.ts). Verified against ggsql-wasm 0.4.1's output
 * (tests/unit/ggsql-plot-data.test.ts runs it live):
 *  - all layers share one inline `data.values` array; each layer picks its rows
 *    with a filter on `__ggsql_source__`;
 *  - field names are generated (`__ggsql_aes_pos1__`...), so every name is read
 *    from the layer's `encoding`, never assumed;
 *  - a bar's height is y - y2 (ggsql stacks repeated x values itself);
 *  - a boxplot is four layers whose sources end in `box` (q1 to q3),
 *    `lower_whisker`, `upper_whisker` and `median`.
 * A point layer with one category axis (the Coding Studio's dumbbell example)
 * is read as bars, one series per layer, so "Student is Amanda, Grade is 91"
 * can be announced. Faceted charts return undefined: picture and description
 * only, as before.
 */
export function plotDataFromSpec(spec: unknown): PlotData | undefined {
  try {
    const s = (spec ?? {}) as AnySpec;
    if (s.facet || s.spec || s.repeat || s.concat || s.hconcat || s.vconcat) return undefined;
    const values = (((s.data as AnySpec | undefined)?.values ?? []) as Row[]).filter(
      (r) => r && typeof r === "object",
    );
    const layers = ((s.layer as AnySpec[] | undefined) ?? [s]).filter(Boolean);
    const bySource = new Map<string, Row[]>();
    for (const row of values) {
      const key = String(row.__ggsql_source__ ?? "");
      bySource.set(key, [...(bySource.get(key) ?? []), row]);
    }
    const sourceOf = (layer: AnySpec): string | undefined => {
      for (const t of (layer.transform as AnySpec[] | undefined) ?? []) {
        const eq = (t.filter as AnySpec | undefined)?.equal;
        if (typeof eq === "string") return eq;
      }
      return undefined;
    };
    const rowsOf = (layer: AnySpec) => {
      const src = sourceOf(layer);
      return src ? (bySource.get(src) ?? []) : values;
    };
    const rawTitle = s.title;
    const title =
      typeof rawTitle === "string"
        ? rawTitle
        : typeof (rawTitle as AnySpec | undefined)?.text === "string"
          ? ((rawTitle as AnySpec).text as string)
          : undefined;
    const label = (channel: string) => {
      const t = sharedTitle(layers, channel);
      return typeof t === "string" ? t : undefined;
    };

    const out: unknown[] = [];
    const boxes = new Map<string, Map<string, Record<string, number>>>();
    // Point layers with one category axis become one bar layer, a series each.
    const pointBars: {
      horizontal: boolean;
      series: { name?: string; points: { x: string; y: number }[] }[];
    } = { horizontal: false, series: [] };

    for (const layer of layers) {
      const markRaw = layer.mark;
      const mark = typeof markRaw === "object" ? (markRaw as AnySpec)?.type : markRaw;
      const enc = (layer.encoding ?? {}) as Record<string, AnySpec>;
      const rows = rowsOf(layer);
      const xf = field(enc, "x");
      const yf = field(enc, "y");
      const source = sourceOf(layer) ?? "";

      const boxPart = /^(__ggsql_layer_\d+__)(box|lower_whisker|upper_whisker|median)$/.exec(source);
      if (boxPart && xf && yf) {
        const group = boxes.get(boxPart[1]) ?? new Map<string, Record<string, number>>();
        boxes.set(boxPart[1], group);
        const y2f = field(enc, "y2");
        for (const row of rows) {
          const key = String(row[xf]);
          const stats = group.get(key) ?? {};
          group.set(key, stats);
          const a = numberOf(row[yf]);
          const b = y2f ? numberOf(row[y2f]) : undefined;
          if (a === undefined) continue;
          const lo = b === undefined ? a : Math.min(a, b);
          const hi = b === undefined ? a : Math.max(a, b);
          if (boxPart[2] === "box") Object.assign(stats, { q1: lo, q3: hi });
          else if (boxPart[2] === "lower_whisker") stats.min = lo;
          else if (boxPart[2] === "upper_whisker") stats.max = hi;
          else stats.q2 = a;
        }
        continue;
      }
      if (!xf || !yf) continue;
      // The field that splits rows into series: a colour, stroke or detail.
      const groupField = ["fill", "stroke", "color", "detail"]
        .map((c) => (isNominal(enc, c) ? field(enc, c) : undefined))
        .find(Boolean);
      const inGroup = (g: string | undefined) => (r: Row) =>
        g === undefined || String(r[groupField!]) === g;

      if (mark === "bar") {
        const x2f = field(enc, "x2");
        const y2f = field(enc, "y2");
        if (x2f) {
          out.push({
            kind: "hist",
            bins: rows.map((r) => ({
              xMin: numberOf(r[xf]),
              xMax: numberOf(r[x2f]),
              y: (numberOf(r[yf]) ?? 0) - (y2f ? (numberOf(r[y2f]) ?? 0) : 0),
            })),
          });
          continue;
        }
        const horizontal = isNominal(enc, "y") && !isNominal(enc, "x");
        const [cat, val, val2] = horizontal ? [yf, xf, field(enc, "x2")] : [xf, yf, y2f];
        const point = (r: Row) => ({
          x: String(r[cat]),
          y: (numberOf(r[val]) ?? Number.NaN) - (val2 ? (numberOf(r[val2]) ?? 0) : 0),
        });
        const groups = groupField ? distinct(rows.map((r) => r[groupField])) : [undefined];
        out.push({
          kind: "bar",
          horizontal,
          stacked: !!groupField,
          series: groups.map((g) => ({ name: g, points: rows.filter(inGroup(g)).map(point) })),
        });
      } else if (mark === "line") {
        const groups = groupField ? distinct(rows.map((r) => r[groupField])) : [undefined];
        out.push({
          kind: "line",
          series: groups.map((g) => {
            const pts = rows
              .filter(inGroup(g))
              .map((r) => ({ x: numberOf(r[xf]) ?? r[xf], y: numberOf(r[yf]) }));
            // Lines are announced left to right.
            if (pts.every((p) => typeof p.x === "number")) {
              pts.sort((a, b) => (a.x as number) - (b.x as number));
            }
            return { name: g, points: pts };
          }),
        });
      } else if (mark === "point" || mark === "circle") {
        const xCat = isNominal(enc, "x");
        const yCat = isNominal(enc, "y");
        // A constant colour ('ISA 401' AS fill) names the whole layer.
        const fillField = field(enc, "fill");
        const constant =
          fillField && rows.length > 0 && distinct(rows.map((r) => r[fillField])).length === 1
            ? String(rows[0][fillField])
            : undefined;
        if (xCat !== yCat) {
          const [cat, val] = yCat ? [yf, xf] : [xf, yf];
          pointBars.horizontal = yCat;
          pointBars.series.push({
            name: constant,
            points: rows.map((r) => ({ x: String(r[cat]), y: numberOf(r[val]) ?? Number.NaN })),
          });
          continue;
        }
        if (xCat) continue; // both axes categorical: nothing to measure
        const groups = groupField ? distinct(rows.map((r) => r[groupField])) : [undefined];
        for (const g of groups) {
          out.push({
            kind: "point",
            name: g ?? constant,
            points: rows.filter(inGroup(g)).map((r) => ({ x: numberOf(r[xf]), y: numberOf(r[yf]) })),
          });
        }
      } else if (mark === "rect") {
        const ff = field(enc, "fill");
        if (!ff || isNominal(enc, "fill")) continue;
        const domain = (channel: string, f: string) => {
          const d = (enc[channel]?.scale as AnySpec | undefined)?.domain;
          return Array.isArray(d) ? d.map(String) : distinct(rows.map((r) => r[f]));
        };
        const xs = domain("x", xf);
        const ys = domain("y", yf);
        const z: (number | null)[][] = ys.map(() => xs.map(() => null));
        for (const r of rows) {
          const xi = xs.indexOf(String(r[xf]));
          const yi = ys.indexOf(String(r[yf]));
          if (xi >= 0 && yi >= 0) z[yi][xi] = numberOf(r[ff]) ?? null;
        }
        out.push({ kind: "heat", x: xs, y: ys, z });
      }
      // rule, area, text and other marks have no navigable values here.
    }

    if (pointBars.series.length > 0) {
      out.push({ kind: "bar", horizontal: pointBars.horizontal, stacked: false, series: pointBars.series });
    }
    for (const group of boxes.values()) {
      out.push({
        kind: "box",
        boxes: [...group.entries()].map(([lab, st]) => ({
          label: lab,
          ...st,
          lowerOutliers: [],
          upperOutliers: [],
        })),
      });
    }
    // The colour legend's title names the groups ("Course is ISA 401").
    const zLabel = label("fill") ?? label("stroke") ?? label("color");
    return parsePlotData({ title, xLabel: label("x"), yLabel: label("y"), zLabel, layers: out });
  } catch {
    return undefined;
  }
}
