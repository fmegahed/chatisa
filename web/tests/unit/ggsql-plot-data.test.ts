import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { plotDataFromSpec } from "@/lib/sandbox/ggsql-plot";
import { toMaidrSchema } from "@/lib/run/maidr-schema";

/**
 * SQL plots become explorable from ggsql's own Vega-Lite spec (#40).
 *
 * These run the real ggsql-wasm in Node rather than hand-written fixtures:
 * ggsql is alpha, its field names are generated, and a release that changes
 * the spec's conventions (one inline dataset, `__ggsql_source__` filters, the
 * four-layer boxplot) must fail here, not silently in a student's Plots pane.
 */

interface GgsqlContext {
  register_csv(name: string, bytes: Uint8Array): void;
  execute(query: string): string;
}

let ctx: GgsqlContext;

beforeAll(async () => {
  const dir = join(process.cwd(), "node_modules", "ggsql-wasm");
  const mod = await import(pathToFileURL(join(dir, "ggsql_wasm.js")).href);
  await mod.default({ module_or_path: await readFile(join(dir, "ggsql_wasm_bg.wasm")) });
  ctx = new mod.GgsqlContext();
  const csv = [
    "student,course,grade,hours",
    "Amanda,ISA 401,91,6",
    "Amanda,ISA 444,85,5",
    "Bill,ISA 401,88,4",
    "Bill,ISA 444,79,3",
    "Cara,ISA 401,82,5",
    "Cara,ISA 444,90,7",
    "Dan,ISA 401,76,2",
    "Dan,ISA 444,71,1",
  ].join("\n");
  ctx.register_csv("grades", new TextEncoder().encode(csv));
  const wide = [
    "student,g401,g444",
    "Amanda,91,85",
    "Bill,88,79",
    "Cara,82,90",
    "Dan,76,71",
  ].join("\n");
  ctx.register_csv("wide", new TextEncoder().encode(wide));
}, 60_000);

const run = (query: string) => plotDataFromSpec(JSON.parse(ctx.execute(query)));

describe("ggsql spec to plot data (#40, live ggsql-wasm)", () => {
  it("reads a bar chart, with its title and axis labels", () => {
    const data = run(
      "SELECT course, AVG(grade) AS avg FROM grades GROUP BY course VISUALISE course AS x, avg AS y DRAW bar LABEL title => 'Average grade', x => 'Course', y => 'Average'",
    );
    expect(data).toMatchObject({
      title: "Average grade",
      xLabel: "Course",
      yLabel: "Average",
      layers: [
        {
          kind: "bar",
          stacked: false,
          series: [{ points: [{ x: "ISA 401", y: 84.25 }, { x: "ISA 444", y: 81.25 }] }],
        },
      ],
    });
  });

  it("reads bars split by a fill as stacked series with each segment's own height", () => {
    const data = run("SELECT * FROM grades VISUALISE student AS x, grade AS y, course AS fill DRAW bar");
    const layer = data!.layers[0] as { kind: string; stacked: boolean; series: { name: string; points: { x: string; y: number }[] }[] };
    expect(layer.kind).toBe("bar");
    expect(layer.stacked).toBe(true);
    expect(layer.series.map((s) => s.name)).toEqual(["ISA 401", "ISA 444"]);
    // Amanda's ISA 444 segment is 85 tall, wherever ggsql stacked it.
    expect(layer.series[1].points.find((p) => p.x === "Amanda")!.y).toBe(85);
  });

  it("reads a histogram's bins", () => {
    const data = run("SELECT * FROM grades VISUALISE grade AS x DRAW histogram");
    const layer = data!.layers[0] as { kind: string; bins: { xMin: number; xMax: number; y: number }[] };
    expect(layer.kind).toBe("hist");
    expect(layer.bins.length).toBeGreaterThan(1);
    expect(layer.bins.reduce((n, b) => n + b.y, 0)).toBe(8);
    for (const b of layer.bins) expect(b.xMax).toBeGreaterThan(b.xMin);
  });

  it("reads lines as one series per stroke, ordered left to right", () => {
    const data = run("SELECT * FROM grades VISUALISE hours AS x, grade AS y, course AS stroke DRAW line");
    const layer = data!.layers[0] as { kind: string; series: { name: string; points: { x: number }[] }[] };
    expect(layer.kind).toBe("line");
    expect(layer.series.map((s) => s.name).sort()).toEqual(["ISA 401", "ISA 444"]);
    const xs = layer.series[0].points.map((p) => p.x);
    expect(xs).toEqual([...xs].sort((a, b) => a - b));
  });

  it("reads scatter points, one layer per colour group", () => {
    const plain = run("SELECT * FROM grades VISUALISE hours AS x, grade AS y DRAW point LABEL x => 'Hours', y => 'Grade'");
    expect(plain!.layers).toHaveLength(1);
    expect(plain!.layers[0]).toMatchObject({ kind: "point" });
    expect((plain!.layers[0] as { points: unknown[] }).points).toHaveLength(8);
    const grouped = run("SELECT * FROM grades VISUALISE hours AS x, grade AS y, course AS fill DRAW point");
    expect(grouped!.layers.map((l) => (l as { name?: string }).name)).toEqual(["ISA 401", "ISA 444"]);
  });

  it("reassembles a boxplot from its four layers", () => {
    const data = run("SELECT * FROM grades VISUALISE course AS x, grade AS y DRAW boxplot");
    expect(data!.layers).toHaveLength(1);
    const layer = data!.layers[0];
    if (layer.kind !== "box") throw new Error(`expected a box layer, got ${layer.kind}`);
    const isa401 = layer.boxes.find((b) => b.label === "ISA 401")!;
    expect(isa401).toMatchObject({ min: 76, max: 91, q2: 85 });
    expect(isa401.q1).toBeLessThanOrEqual(isa401.q3);
  });

  it("reads a tile chart as a heatmap grid matching its labels", () => {
    const data = run("SELECT * FROM grades VISUALISE student AS x, course AS y, grade AS fill DRAW tile");
    const layer = data!.layers[0] as { kind: string; x: string[]; y: string[]; z: number[][] };
    expect(layer.kind).toBe("heat");
    expect(layer.x).toEqual(["Amanda", "Bill", "Cara", "Dan"]);
    expect(layer.y).toEqual(["ISA 401", "ISA 444"]);
    expect(layer.z[layer.y.indexOf("ISA 444")][layer.x.indexOf("Cara")]).toBe(90);
  });

  it("makes the Coding Studio's dumbbell example navigable, one series per course", () => {
    const data = run(`SELECT * FROM wide
VISUALISE student AS y
DRAW segment MAPPING g401 AS x, g444 AS xend, student AS yend
DRAW point MAPPING g401 AS x, 'ISA 401' AS fill
DRAW point MAPPING g444 AS x, 'ISA 444' AS fill
LABEL x => 'Grade', y => 'Student', fill => 'Course'`);
    expect(data!.layers).toHaveLength(1);
    const layer = data!.layers[0] as { kind: string; horizontal: boolean; series: { name: string; points: { x: string; y: number }[] }[] };
    expect(layer).toMatchObject({ kind: "bar", horizontal: true });
    expect(layer.series.map((s) => s.name)).toEqual(["ISA 401", "ISA 444"]);
    expect(layer.series[0].points).toContainEqual({ x: "Amanda", y: 91 });
    // Categories run up the y axis, so the announcement reads
    // "Student is Amanda, Grade is 91".
    // "Course" from LABEL fill => 'Course' names the groups: "..., Course is
    // ISA 401" rather than MAIDR's default "Level is ISA 401".
    expect(data!.zLabel).toBe("Course");
    const schema = toMaidrSchema(data, { id: "chatisa-plot-9" });
    const l0 = schema!.subplots[0][0].layers[0];
    expect(l0.type).toBe("dodged_bar");
    expect(l0.axes).toEqual({ x: { label: "Student" }, y: { label: "Grade" }, z: { label: "Course" } });
  });

  it("names the colour groups after the fill or stroke legend", () => {
    expect(run("SELECT * FROM grades VISUALISE student AS x, grade AS y, course AS fill DRAW bar")!.zLabel).toBe("course");
    expect(run("SELECT * FROM grades VISUALISE hours AS x, grade AS y, course AS stroke DRAW line")!.zLabel).toBe("course");
    expect(run("SELECT * FROM grades VISUALISE student AS x, course AS y, grade AS fill DRAW tile")!.zLabel).toBe("grade");
    expect(run("SELECT * FROM grades VISUALISE grade AS x DRAW histogram")!.zLabel).toBeUndefined();
  });

  it("returns undefined for a faceted chart or a spec it cannot read", () => {
    expect(plotDataFromSpec({ facet: { field: "x" }, spec: { layer: [] } })).toBeUndefined();
    expect(plotDataFromSpec(null)).toBeUndefined();
    expect(plotDataFromSpec({ layer: [{ mark: "rule" }] })).toBeUndefined();
  });
});
