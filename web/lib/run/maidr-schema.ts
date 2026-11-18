/**
 * Navigable, audible plots (#40): turns the data a runtime extracted from a
 * plot into the JSON that MAIDR (maidr.ai) reads.
 *
 * MAIDR lets a student explore a chart from the keyboard: arrow keys move from
 * point to point, each point is announced ("Course is ISA 401, Average grade is
 * 3.2"), S plays it as a tone, B shows it in braille, T switches text detail.
 * The September 2026 accessibility audit recommended it.
 *
 * The pipeline, and why this module sits in the middle of it:
 *  - The Python worker reads the matplotlib figure, the R worker reads the
 *    ggplot it drew (ggplot_build), and the SQL path reads the Vega-Lite spec
 *    ggsql produced (lib/sandbox/ggsql-plot.ts). Each sends a NEUTRAL PlotData:
 *    plain numbers and strings, no MAIDR vocabulary.
 *  - parsePlotData() validates it. Worker data crosses a postMessage and comes
 *    from code the student (or a model) wrote, so it is untrusted.
 *  - toMaidrSchema() builds the MAIDR JSON field by field. It never copies an
 *    input object, because maidr.js executes one schema field as code: an
 *    axis `format.function` string is passed to `Function()` (maidr.js 4.16.0,
 *    resolveFormat). Nothing here ever emits `format`, `selectors` or any other
 *    field MAIDR could act on beyond plain data.
 *  - components/run/AccessiblePlot.tsx puts the JSON on the plot's <img> and
 *    loads maidr.js.
 *
 * maidr.js is GPL-3.0-or-later and ChatISA is MIT, so maidr.js is never
 * imported here: it is a separate, unmodified file under /runtimes/maidr/,
 * loaded at runtime (see scripts/setup-runtimes.mjs). This module only knows
 * MAIDR's data format.
 *
 * PNG plots get sound, text and braille, but no visual highlight: the picture
 * is a bitmap, so there is nothing for MAIDR to outline.
 */
import pin from "./maidr-pin.json";

export const MAIDR_VERSION: string = pin.version;
export const MAIDR_SCRIPT_URL = `/runtimes/maidr/${pin.version}/maidr.js`;

// ---------------------------------------------------------------------------
// The neutral shape the runtimes send.

export interface BarSeries {
  name?: string;
  points: { x: string; y: number }[];
}

export interface LineSeries {
  name?: string;
  points: { x: number | string; y: number }[];
}

export interface BoxStats {
  label: string;
  min: number;
  q1: number;
  q2: number;
  q3: number;
  max: number;
  lowerOutliers: number[];
  upperOutliers: number[];
}

export type PlotLayerData =
  /** Bars; several series are dodged unless `stacked`. `horizontal` means the
   * categories run up the y axis (matplotlib barh, ggplot coord_flip). */
  | { kind: "bar"; series: BarSeries[]; stacked?: boolean; horizontal?: boolean }
  | { kind: "hist"; bins: { xMin: number; xMax: number; y: number }[] }
  | { kind: "line"; series: LineSeries[] }
  | { kind: "point"; name?: string; points: { x: number; y: number }[] }
  | { kind: "box"; boxes: BoxStats[]; horizontal?: boolean }
  /** `z[row][col]`, rows in `y` order (top to bottom as drawn), columns in `x`. */
  | { kind: "heat"; x: string[]; y: string[]; z: (number | null)[][] };

export interface PlotData {
  title?: string;
  xLabel?: string;
  yLabel?: string;
  /** What the colours mean: the legend title ("Course"). It names a bar or line
   * layer's groups and a heatmap's cell values; without it MAIDR says "Level"
   * ("Level is ISA 401"). */
  zLabel?: string;
  /** More than one panel is not navigable in this release (description only). */
  panels?: number;
  layers: PlotLayerData[];
  /** Data points before sampling, when a layer was too large to send whole. */
  total?: number;
  /** Data points after sampling. */
  shown?: number;
}

// ---------------------------------------------------------------------------
// Limits. A plot with 50,000 points is fine as a picture but not as something
// to arrow through, announce, or post across a worker boundary; past these
// sizes the data is sampled at even intervals and the hint says so.

export const MAX_LAYERS = 6;
export const MAX_SERIES = 10;
export const MAX_BARS = 200;
export const MAX_LINE_POINTS = 2000;
export const MAX_SCATTER_POINTS = 1000;
export const MAX_BOXES = 50;
export const MAX_OUTLIERS = 200;
export const MAX_HEAT = 60;
const MAX_LABEL = 120;

/** Every `step`-th item so that at most `max` remain, keeping the first and
 * spreading the rest evenly. */
export function sample<T>(items: T[], max: number): T[] {
  if (items.length <= max) return items;
  const out: T[] = [];
  const step = items.length / max;
  for (let i = 0; i < max; i++) out.push(items[Math.floor(i * step)]);
  return out;
}

// ---------------------------------------------------------------------------
// Validation of untrusted worker data.

/** A short, plain label: no `$` (MAIDR would try to render it as math), no
 * control characters, collapsed whitespace, clipped. */
export function cleanLabel(v: unknown): string | undefined {
  if (typeof v !== "string" && typeof v !== "number") return undefined;
  const t = String(v)
    .replace(/\$/g, "")
    .replace(/[\u0000-\u001f\u007f]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!t) return undefined;
  return t.length > MAX_LABEL ? `${t.slice(0, MAX_LABEL - 3)}...` : t;
}

function num(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function arr(v: unknown): unknown[] {
  return Array.isArray(v) ? v : [];
}

function obj(v: unknown): Record<string, unknown> {
  return v && typeof v === "object" && !Array.isArray(v) ? (v as Record<string, unknown>) : {};
}

interface Counter {
  total: number;
  shown: number;
}

function capped<T>(items: T[], max: number, count: Counter): T[] {
  const out = sample(items, max);
  count.total += items.length;
  count.shown += out.length;
  return out;
}

function parseLayer(raw: unknown, count: Counter): PlotLayerData | undefined {
  const r = obj(raw);
  switch (r.kind) {
    case "bar": {
      const series = arr(r.series)
        .slice(0, MAX_SERIES)
        .map((s): BarSeries => {
          const so = obj(s);
          const points = arr(so.points).flatMap((p) => {
            const po = obj(p);
            const x = cleanLabel(po.x);
            const y = num(po.y);
            return x !== undefined && y !== undefined ? [{ x, y }] : [];
          });
          return { name: cleanLabel(so.name), points: capped(points, MAX_BARS, count) };
        })
        .filter((s) => s.points.length > 0);
      return series.length > 0
        ? { kind: "bar", series, stacked: r.stacked === true, horizontal: r.horizontal === true }
        : undefined;
    }
    case "hist": {
      const bins = arr(r.bins).flatMap((b) => {
        const bo = obj(b);
        const xMin = num(bo.xMin);
        const xMax = num(bo.xMax);
        const y = num(bo.y);
        return xMin !== undefined && xMax !== undefined && y !== undefined ? [{ xMin, xMax, y }] : [];
      });
      return bins.length > 0 ? { kind: "hist", bins: capped(bins, MAX_BARS, count) } : undefined;
    }
    case "line": {
      const series = arr(r.series)
        .slice(0, MAX_SERIES)
        .map((s): LineSeries => {
          const so = obj(s);
          const points = arr(so.points).flatMap((p) => {
            const po = obj(p);
            const x = num(po.x) ?? cleanLabel(po.x);
            const y = num(po.y);
            return x !== undefined && y !== undefined ? [{ x, y }] : [];
          });
          return { name: cleanLabel(so.name), points: capped(points, MAX_LINE_POINTS, count) };
        })
        .filter((s) => s.points.length > 1);
      return series.length > 0 ? { kind: "line", series } : undefined;
    }
    case "point": {
      const points = arr(r.points).flatMap((p) => {
        const po = obj(p);
        const x = num(po.x);
        const y = num(po.y);
        return x !== undefined && y !== undefined ? [{ x, y }] : [];
      });
      return points.length > 0
        ? { kind: "point", name: cleanLabel(r.name), points: capped(points, MAX_SCATTER_POINTS, count) }
        : undefined;
    }
    case "box": {
      const boxes = arr(r.boxes).flatMap((b, i): BoxStats[] => {
        const bo = obj(b);
        const min = num(bo.min);
        const q1 = num(bo.q1);
        const q2 = num(bo.q2);
        const q3 = num(bo.q3);
        const max = num(bo.max);
        if ([min, q1, q2, q3, max].some((v) => v === undefined)) return [];
        const outliers = (v: unknown) =>
          sample(arr(v).flatMap((o) => (num(o) !== undefined ? [num(o)!] : [])), MAX_OUTLIERS);
        return [
          {
            label: cleanLabel(bo.label) ?? `Group ${i + 1}`,
            min: min!,
            q1: q1!,
            q2: q2!,
            q3: q3!,
            max: max!,
            lowerOutliers: outliers(bo.lowerOutliers),
            upperOutliers: outliers(bo.upperOutliers),
          },
        ];
      });
      return boxes.length > 0
        ? { kind: "box", boxes: capped(boxes, MAX_BOXES, count), horizontal: r.horizontal === true }
        : undefined;
    }
    case "heat": {
      const x = arr(r.x).map((v, i) => cleanLabel(v) ?? String(i + 1));
      const y = arr(r.y).map((v, i) => cleanLabel(v) ?? String(i + 1));
      const z = arr(r.z).map((row) => arr(row).map((v) => num(v) ?? null));
      // The grid must be rectangular and match its labels, or MAIDR would
      // announce the wrong cell.
      if (x.length === 0 || y.length === 0 || z.length !== y.length) return undefined;
      if (z.some((row) => row.length !== x.length)) return undefined;
      if (x.length > MAX_HEAT || y.length > MAX_HEAT) return undefined;
      count.total += x.length * y.length;
      count.shown += x.length * y.length;
      return { kind: "heat", x, y, z };
    }
    default:
      return undefined;
  }
}

/** Validates what a worker sent. Undefined when nothing navigable remains. */
export function parsePlotData(raw: unknown): PlotData | undefined {
  const r = obj(raw);
  const count: Counter = { total: 0, shown: 0 };
  const layers = arr(r.layers)
    .slice(0, MAX_LAYERS)
    .map((l) => parseLayer(l, count))
    .filter((l): l is PlotLayerData => l !== undefined);
  const panels = num(r.panels);
  if (layers.length === 0 && !(panels && panels > 1)) return undefined;
  // A worker that sampled before sending reports the true size in `total`.
  const total = Math.max(count.total, num(r.total) ?? 0);
  return {
    title: cleanLabel(r.title),
    xLabel: cleanLabel(r.xLabel),
    yLabel: cleanLabel(r.yLabel),
    zLabel: cleanLabel(r.zLabel),
    panels: panels && panels > 0 ? Math.floor(panels) : undefined,
    layers,
    total,
    shown: count.shown,
  };
}

/** "Explores 1,000 of 12,345 points." when the data was sampled. */
export function sampledNote(data: PlotData | undefined): string | undefined {
  if (!data || !data.total || !data.shown || data.total <= data.shown) return undefined;
  const f = (n: number) => n.toLocaleString("en-US");
  return `Explores ${f(data.shown)} of ${f(data.total)} points.`;
}

// ---------------------------------------------------------------------------
// The MAIDR JSON (maidr 4.16.0 MaidrData), built field by field.

interface MaidrAxis {
  label?: string;
}

export interface MaidrLayer {
  id: string;
  type: "bar" | "dodged_bar" | "stacked_bar" | "hist" | "line" | "point" | "box" | "heat";
  title?: string;
  /** `z` names the groups (dodged/stacked bars, several lines) or a heatmap's
   * cell values; MAIDR reads it as `axes.z.label`. */
  axes: { x: MaidrAxis; y: MaidrAxis; z?: MaidrAxis };
  data: unknown;
}

export interface MaidrSchema {
  id: string;
  title?: string;
  subplots: { layers: MaidrLayer[] }[][];
}

function axis(label: string | undefined): MaidrAxis {
  return label ? { label } : {};
}

let nextId = 0;

/** A page-unique id. MAIDR uses it as a DOM id prefix, so it stays [a-z0-9-]. */
export function makeMaidrId(): string {
  nextId += 1;
  return `chatisa-plot-${nextId}`;
}

/**
 * The MAIDR JSON for one plot, or null when there is nothing MAIDR can
 * navigate (several panels, no supported layer). Null means "show the picture
 * and its description only", which is exactly what happened before #40.
 *
 * Horizontal bars and boxes are sent as vertical ones with the axis labels
 * swapped: the announcement ("Course is ISA 401, Grade is 3.2") is the same,
 * and a PNG has no outline to place either way.
 */
export function toMaidrSchema(
  data: PlotData | undefined,
  opts: { id: string; title?: string; xLabel?: string; yLabel?: string },
): MaidrSchema | null {
  if (!data) return null;
  if (data.panels && data.panels > 1) return null;
  const title = cleanLabel(opts.title) ?? data.title;
  const xLabel = cleanLabel(opts.xLabel) ?? data.xLabel;
  const yLabel = cleanLabel(opts.yLabel) ?? data.yLabel;
  const layers: MaidrLayer[] = [];

  for (const layer of data.layers) {
    const id = String(layers.length);
    const swap = (layer.kind === "bar" || layer.kind === "box") && layer.horizontal;
    // The legend title names the third thing MAIDR announces, only on layers
    // that have one: groups of bars or lines, or a heatmap's values.
    const grouped =
      ((layer.kind === "bar" || layer.kind === "line") && layer.series.length > 1) || layer.kind === "heat";
    const axes: MaidrLayer["axes"] = swap
      ? { x: axis(yLabel), y: axis(xLabel) }
      : { x: axis(xLabel), y: axis(yLabel) };
    if (grouped && data.zLabel) axes.z = axis(data.zLabel);
    switch (layer.kind) {
      case "bar": {
        if (layer.series.length === 1) {
          layers.push({
            id,
            type: "bar",
            axes,
            data: layer.series[0].points.map((p) => ({ x: p.x, y: p.y })),
          });
        } else {
          layers.push({
            id,
            type: layer.stacked ? "stacked_bar" : "dodged_bar",
            axes,
            data: layer.series.map((s, i) => {
              const z = s.name ?? `Series ${i + 1}`;
              return s.points.map((p) => ({ x: p.x, y: p.y, z }));
            }),
          });
        }
        break;
      }
      case "hist":
        layers.push({
          id,
          type: "hist",
          axes,
          data: layer.bins.map((b) => ({
            x: (b.xMin + b.xMax) / 2,
            y: b.y,
            xMin: b.xMin,
            xMax: b.xMax,
            yMin: 0,
            yMax: b.y,
          })),
        });
        break;
      case "line":
        layers.push({
          id,
          type: "line",
          axes,
          data: layer.series.map((s, i) => {
            const z = s.name ?? (layer.series.length > 1 ? `Series ${i + 1}` : undefined);
            return s.points.map((p) => (z ? { x: p.x, y: p.y, z } : { x: p.x, y: p.y }));
          }),
        });
        break;
      case "point":
        layers.push({
          id,
          type: "point",
          ...(layer.name ? { title: layer.name } : {}),
          axes,
          data: layer.points.map((p) => ({ x: p.x, y: p.y })),
        });
        break;
      case "box":
        layers.push({
          id,
          type: "box",
          axes,
          data: layer.boxes.map((b) => ({
            z: b.label,
            fill: b.label,
            lowerOutliers: [...b.lowerOutliers],
            min: b.min,
            q1: b.q1,
            q2: b.q2,
            q3: b.q3,
            max: b.max,
            upperOutliers: [...b.upperOutliers],
          })),
        });
        break;
      case "heat":
        layers.push({
          id,
          type: "heat",
          axes,
          data: { x: [...layer.x], y: [...layer.y], points: layer.z.map((row) => [...row]) },
        });
        break;
    }
  }
  if (layers.length === 0) return null;
  return {
    id: opts.id,
    ...(title ? { title } : {}),
    subplots: [[{ layers }]],
  };
}
