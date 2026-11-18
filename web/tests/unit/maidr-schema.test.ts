import { describe, expect, it } from "vitest";
import {
  MAIDR_SCRIPT_URL,
  MAX_BARS,
  MAX_SCATTER_POINTS,
  cleanLabel,
  makeMaidrId,
  parsePlotData,
  sample,
  sampledNote,
  toMaidrSchema,
  type MaidrSchema,
} from "@/lib/run/maidr-schema";
import pin from "@/lib/run/maidr-pin.json";

/**
 * Navigable, audible plots (#40). The MAIDR JSON is built here from data the
 * runtimes extracted, and maidr.js executes one schema field as code (an axis
 * `format.function` goes to Function()). So the builder must emit only plain
 * data, whatever the worker sends.
 */

const schema = (raw: unknown, opts: Partial<{ title: string; xLabel: string; yLabel: string }> = {}) =>
  toMaidrSchema(parsePlotData(raw), { id: "chatisa-plot-1", ...opts });

const layer0 = (s: MaidrSchema | null) => s!.subplots[0][0].layers[0];

/** Every key anywhere in a value, for the "never emits" checks. */
function allKeys(v: unknown, out: string[] = []): string[] {
  if (Array.isArray(v)) v.forEach((x) => allKeys(x, out));
  else if (v && typeof v === "object") {
    for (const [k, x] of Object.entries(v)) {
      out.push(k);
      allKeys(x, out);
    }
  }
  return out;
}

describe("MAIDR schema from extracted plot data (#40)", () => {
  it("maps a single-series bar chart to a bar layer with labelled axes", () => {
    const s = schema({
      title: "Grades",
      xLabel: "Course",
      yLabel: "Average grade",
      layers: [{ kind: "bar", series: [{ points: [{ x: "ISA 401", y: 3.2 }, { x: "ISA 444", y: 3.6 }] }] }],
    });
    expect(s).toEqual({
      id: "chatisa-plot-1",
      title: "Grades",
      subplots: [
        [
          {
            layers: [
              {
                id: "0",
                type: "bar",
                axes: { x: { label: "Course" }, y: { label: "Average grade" } },
                data: [
                  { x: "ISA 401", y: 3.2 },
                  { x: "ISA 444", y: 3.6 },
                ],
              },
            ],
          },
        ],
      ],
    });
  });

  it("maps several bar series to dodged bars, or stacked bars when stacked", () => {
    const raw = (stacked: boolean) => ({
      layers: [
        {
          kind: "bar",
          stacked,
          series: [
            { name: "ISA 401", points: [{ x: "Fall", y: 10 }, { x: "Spring", y: 12 }] },
            { name: "ISA 444", points: [{ x: "Fall", y: 7 }, { x: "Spring", y: 9 }] },
          ],
        },
      ],
    });
    const dodged = layer0(schema(raw(false)));
    expect(dodged.type).toBe("dodged_bar");
    expect(dodged.data).toEqual([
      [
        { x: "Fall", y: 10, z: "ISA 401" },
        { x: "Spring", y: 12, z: "ISA 401" },
      ],
      [
        { x: "Fall", y: 7, z: "ISA 444" },
        { x: "Spring", y: 9, z: "ISA 444" },
      ],
    ]);
    expect(layer0(schema(raw(true))).type).toBe("stacked_bar");
  });

  it("names the colour groups after the legend, so MAIDR says 'Course is ISA 401', not 'Level is'", () => {
    const series = [
      { name: "ISA 401", points: [{ x: "Fall", y: 10 }] },
      { name: "ISA 444", points: [{ x: "Fall", y: 7 }] },
    ];
    const line = (name: string) => ({ name, points: [{ x: 1, y: 2 }, { x: 2, y: 3 }] });
    const heat = { kind: "heat", x: ["Amanda"], y: ["ISA 401"], z: [[91]] };
    const s = schema({
      xLabel: "Term",
      yLabel: "Students",
      zLabel: "Cou$rse\n",
      layers: [
        { kind: "bar", series },
        { kind: "line", series: [line("ISA 401"), line("ISA 444")] },
        heat,
      ],
    });
    const layers = s!.subplots[0][0].layers;
    // Cleaned like every other label.
    expect(layers.map((l) => l.axes)).toEqual([
      { x: { label: "Term" }, y: { label: "Students" }, z: { label: "Course" } },
      { x: { label: "Term" }, y: { label: "Students" }, z: { label: "Course" } },
      { x: { label: "Term" }, y: { label: "Students" }, z: { label: "Course" } },
    ]);
    // Layers without groups never get one: a single bar series, one line, a
    // scatter layer (named by its title), a box plot (its boxes are the x axis).
    const plain = schema({
      zLabel: "Course",
      layers: [
        { kind: "bar", series: [series[0]] },
        { kind: "line", series: [line("ISA 401")] },
        { kind: "point", points: [{ x: 1, y: 2 }] },
        { kind: "box", boxes: [{ label: "A", min: 1, q1: 2, q2: 3, q3: 4, max: 5 }] },
      ],
    });
    for (const l of plain!.subplots[0][0].layers) expect(l.axes).not.toHaveProperty("z");
    // No legend title, no z axis: MAIDR keeps its own default.
    expect(layer0(schema({ layers: [{ kind: "bar", series }] })).axes).toEqual({ x: {}, y: {} });
  });

  it("sends horizontal bars as vertical ones with the axis labels swapped", () => {
    const s = schema({
      xLabel: "Average grade",
      yLabel: "Course",
      layers: [{ kind: "bar", horizontal: true, series: [{ points: [{ x: "ISA 401", y: 3.2 }] }] }],
    });
    expect(layer0(s).axes).toEqual({ x: { label: "Course" }, y: { label: "Average grade" } });
    expect(layer0(s)).not.toHaveProperty("orientation");
  });

  it("maps histogram bins to MAIDR's bin shape", () => {
    const s = schema({ layers: [{ kind: "hist", bins: [{ xMin: 1, xMax: 2, y: 4 }] }] });
    expect(layer0(s)).toMatchObject({
      type: "hist",
      data: [{ x: 1.5, y: 4, xMin: 1, xMax: 2, yMin: 0, yMax: 4 }],
    });
  });

  it("maps line series to one inner array per series, named only when several", () => {
    const one = schema({ layers: [{ kind: "line", series: [{ points: [{ x: 1, y: 2 }, { x: 2, y: 3 }] }] }] });
    expect(layer0(one).data).toEqual([[{ x: 1, y: 2 }, { x: 2, y: 3 }]]);
    const two = schema({
      layers: [
        {
          kind: "line",
          series: [
            { name: "A", points: [{ x: "2026-01-01", y: 1 }, { x: "2026-02-01", y: 2 }] },
            { points: [{ x: "2026-01-01", y: 3 }, { x: "2026-02-01", y: 4 }] },
          ],
        },
      ],
    });
    expect(layer0(two).data).toEqual([
      [
        { x: "2026-01-01", y: 1, z: "A" },
        { x: "2026-02-01", y: 2, z: "A" },
      ],
      [
        { x: "2026-01-01", y: 3, z: "Series 2" },
        { x: "2026-02-01", y: 4, z: "Series 2" },
      ],
    ]);
  });

  it("maps scatter points, one layer per group with the group as its title", () => {
    const s = schema({
      layers: [
        { kind: "point", name: "ISA 401", points: [{ x: 1, y: 2 }] },
        { kind: "point", name: "ISA 444", points: [{ x: 3, y: 4 }] },
      ],
    });
    const layers = s!.subplots[0][0].layers;
    expect(layers.map((l) => [l.id, l.type, l.title])).toEqual([
      ["0", "point", "ISA 401"],
      ["1", "point", "ISA 444"],
    ]);
  });

  it("maps box statistics, labelling each box", () => {
    const s = schema({
      layers: [
        {
          kind: "box",
          boxes: [{ label: "ISA 401", min: 1, q1: 2, q2: 3, q3: 3.5, max: 4, lowerOutliers: [0.2], upperOutliers: [] }],
        },
      ],
    });
    expect(layer0(s)).toMatchObject({
      type: "box",
      data: [
        { z: "ISA 401", fill: "ISA 401", lowerOutliers: [0.2], min: 1, q1: 2, q2: 3, q3: 3.5, max: 4, upperOutliers: [] },
      ],
    });
  });

  it("maps a heatmap grid and rejects a grid that does not match its labels", () => {
    const ok = schema({ layers: [{ kind: "heat", x: ["Mon", "Tue"], y: ["9am", "10am"], z: [[1, 2], [3, null]] }] });
    expect(layer0(ok)).toMatchObject({
      type: "heat",
      data: { x: ["Mon", "Tue"], y: ["9am", "10am"], points: [[1, 2], [3, null]] },
    });
    expect(schema({ layers: [{ kind: "heat", x: ["Mon"], y: ["9am", "10am"], z: [[1, 2]] }] })).toBeNull();
  });

  it("never emits a field maidr.js could execute or act on, whatever the input carries", () => {
    const hostile = {
      title: "T",
      format: { function: "alert(1)" },
      selectors: "body",
      layers: [
        {
          kind: "bar",
          format: { function: "alert(2)" },
          axes: { x: { format: { function: "alert(3)" } } },
          selectors: "*",
          __proto__: { polluted: true },
          series: [{ name: "s", points: [{ x: "a", y: 1, format: { function: "alert(4)" } }] }],
        },
      ],
    };
    const s = schema(hostile);
    const keys = allKeys(s);
    for (const bad of ["format", "function", "selectors", "domMapping", "hoverMode", "polluted"]) {
      expect(keys).not.toContain(bad);
    }
    // Axes are always objects (MAIDR rejects bare strings).
    expect(typeof layer0(s).axes.x).toBe("object");
  });

  it("drops non-finite numbers and cleans labels", () => {
    const s = schema({
      title: "Price in $USD\n(2026)",
      layers: [{ kind: "bar", series: [{ points: [{ x: "a", y: Number.NaN }, { x: "b", y: Infinity }, { x: "c", y: 2 }] }] }],
    });
    expect(s!.title).toBe("Price in USD (2026)");
    expect(layer0(s).data).toEqual([{ x: "c", y: 2 }]);
    expect(cleanLabel("x".repeat(200))!.length).toBe(120);
  });

  it("samples oversized data at even intervals and says so", () => {
    const points = Array.from({ length: 5000 }, (_, i) => ({ x: i, y: i * 2 }));
    const data = parsePlotData({ layers: [{ kind: "point", points }] })!;
    expect((data.layers[0] as { points: unknown[] }).points).toHaveLength(MAX_SCATTER_POINTS);
    expect(sampledNote(data)).toBe("Explores 1,000 of 5,000 points.");
    const bars = parsePlotData({
      layers: [{ kind: "bar", series: [{ points: Array.from({ length: 900 }, (_, i) => ({ x: `c${i}`, y: i })) }] }],
    })!;
    expect((bars.layers[0] as { series: { points: unknown[] }[] }).series[0].points).toHaveLength(MAX_BARS);
    // A worker that sampled before sending reports the true size.
    expect(sampledNote(parsePlotData({ total: 40000, layers: [{ kind: "point", points: points.slice(0, 900) }] }))).toBe(
      "Explores 900 of 40,000 points.",
    );
    expect(sampledNote(parsePlotData({ layers: [{ kind: "point", points: points.slice(0, 10) }] }))).toBeUndefined();
    expect(sample([1, 2, 3, 4, 5, 6], 3)).toEqual([1, 3, 5]);
  });

  it("returns null when there is nothing to navigate, so the picture keeps its description only", () => {
    expect(toMaidrSchema(undefined, { id: "x" })).toBeNull();
    expect(schema({ layers: [] })).toBeNull();
    expect(schema({ layers: [{ kind: "pie", values: [1, 2] }] })).toBeNull();
    expect(schema({ layers: "not a list" })).toBeNull();
    expect(schema(null)).toBeNull();
    // Several panels are described, not navigated, in this release.
    expect(schema({ panels: 4, layers: [{ kind: "point", points: [{ x: 1, y: 2 }] }] })).toBeNull();
    // A line needs at least two points to be a line.
    expect(schema({ layers: [{ kind: "line", series: [{ points: [{ x: 1, y: 2 }] }] }] })).toBeNull();
  });

  it("lets the caller's title and labels win over the worker's", () => {
    const s = schema(
      { title: "From figure", xLabel: "fx", layers: [{ kind: "point", points: [{ x: 1, y: 2 }] }] },
      { title: "Chosen" },
    );
    expect(s!.title).toBe("Chosen");
    expect(layer0(s).axes.x).toEqual({ label: "fx" });
  });

  it("makes page-unique ids MAIDR can use as DOM id prefixes", () => {
    const a = makeMaidrId();
    const b = makeMaidrId();
    expect(a).not.toBe(b);
    expect(a).toMatch(/^[a-z0-9-]+$/);
  });

  it("serves the pinned maidr.js from our own origin, versioned", () => {
    expect(MAIDR_SCRIPT_URL).toBe(`/runtimes/maidr/${pin.version}/maidr.js`);
    expect(pin.integrity).toMatch(/^sha512-/);
    expect(pin.maidrJsSha256).toMatch(/^[0-9a-f]{64}$/);
  });
});
