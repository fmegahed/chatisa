import { describe, expect, it } from "vitest";
import {
  CHART_INK,
  CHART_SURFACE,
  MIAMI_HIGHLIGHT,
  MIAMI_LINK_TEAL,
  MIAMI_SERIES,
  PIE_POLICY,
  annotationRules,
  paletteFor,
  paletteRules,
  portableRules,
} from "@/lib/ask/chart-style";
import { getMiamiStyle, MIAMI_STYLE_KINDS } from "@/lib/ask/miami-style";
import { anthropicTemplateMessage, DECK_TEMPLATE } from "@/lib/ask/hosted";

/**
 * The palette here is not a matter of taste: every hex was checked with a
 * palette validator, and these tests pin the outcome so a well-meaning edit
 * cannot quietly reintroduce a combination students cannot read.
 */
describe("paletteFor", () => {
  it("uses Miami colours in a fixed order up to four series", () => {
    expect(paletteFor(1).colors).toEqual(["#C3142D"]);
    expect(paletteFor(2).colors).toEqual(["#C3142D", "#585E60"]);
    expect(paletteFor(3).colors).toEqual(["#C3142D", "#585E60", "#1D5FAD"]);
    expect(paletteFor(4).colors).toEqual([
      "#C3142D",
      "#585E60",
      "#1D5FAD",
      "#FF7436",
    ]);
    for (const n of [1, 2, 3, 4]) expect(paletteFor(n).kind).toBe("miami");
  });

  it("allows colour alone only for one or two series", () => {
    expect(paletteFor(1).colorAloneIsEnough).toBe(true);
    expect(paletteFor(2).colorAloneIsEnough).toBe(true);
    // Three or more needs labels or shapes: no categorical palette separates
    // reliably on colour alone past two brand colours.
    for (const n of [3, 4, 5, 8, 9, 20]) {
      expect(paletteFor(n).colorAloneIsEnough).toBe(false);
    }
  });

  it("escalates to Dark2 from five to eight series", () => {
    for (const n of [5, 6, 7, 8]) {
      const p = paletteFor(n);
      expect(p.kind).toBe("dark2");
      expect(p.brewer).toBe("Dark2");
    }
  });

  it("refuses to colour nine or more categories", () => {
    // ColorBrewer Paired was measured at deltaE 0.6 between its orange and
    // green under protanopia, so past eight the form changes, not the palette.
    for (const n of [9, 12, 13, 40]) {
      expect(paletteFor(n).kind).toBe("refuse");
      expect(paletteFor(n).brewer).toBeNull();
    }
    expect(paletteFor(9).note).toMatch(/Other|facet/i);
    expect(paletteFor(40).note).toMatch(/aggregate|table/i);
  });

  it("never offers black or corn yellow as a series colour", () => {
    for (let n = 1; n <= 12; n += 1) {
      expect(paletteFor(n).colors).not.toContain(CHART_INK);
      expect(paletteFor(n).colors).not.toContain(MIAMI_HIGHLIGHT);
      expect(paletteFor(n).colors).not.toContain(MIAMI_LINK_TEAL);
      // Slate blue reads at normal-vision deltaE 5.5 from charcoal.
      expect(paletteFor(n).colors).not.toContain("#3E5468");
    }
  });

  it("clamps nonsense counts instead of returning an empty palette", () => {
    expect(paletteFor(0).colors).toEqual(["#C3142D"]);
    expect(paletteFor(-3).colors).toEqual(["#C3142D"]);
    expect(paletteFor(2.7).colors).toEqual(["#C3142D", "#585E60"]);
  });

  it("keeps the surface white and the ink black", () => {
    expect(CHART_SURFACE).toBe("#FFFFFF");
    expect(CHART_INK).toBe("#000000");
    expect(MIAMI_SERIES).toHaveLength(4);
  });
});

describe("guidance text", () => {
  it("states the palette, the escalation, and the yellow restriction", () => {
    const rules = paletteRules();
    for (const hex of MIAMI_SERIES) expect(rules).toContain(hex);
    expect(rules).toContain("Dark2");
    expect(rules).toMatch(/fill only/i);
    expect(rules).toMatch(/hyperlinks only/i);
  });

  it("puts the finding in the title and forbids restating the axes", () => {
    expect(portableRules()).toMatch(/states the finding/i);
    expect(portableRules()).toMatch(/restates the axis/i);
    // Fonts must be a fallback list: matplotlib substitutes silently.
    expect(portableRules()).toMatch(/fall back|fallback/i);
  });

  it("tells the student pie charts are suboptimal but builds one if pressed", () => {
    expect(PIE_POLICY).toMatch(/suboptimal/i);
    expect(PIE_POLICY).toMatch(/bar chart/i);
    expect(PIE_POLICY).toMatch(/dot chart/i);
    expect(PIE_POLICY).toMatch(/still wants a pie, build it/i);
  });

  it("never suggests an uninstallable package for the hosted sandbox", () => {
    for (const language of ["r", "python"] as const) {
      const text = annotationRules(language);
      const hosted = text.slice(text.indexOf("Hosted provider sandbox"));
      expect(hosted.length).toBeGreaterThan(0);
      for (const pkg of ["ggrepel", "ggtext", "adjustText", "highlight_text"]) {
        // Named only inside the "do not import" sentence, never as advice.
        expect(hosted).toMatch(new RegExp(`do not import[^.]*${pkg}`, "i"));
      }
    }
  });

  it("offers the rich packages for the browser runtime, per language", () => {
    expect(annotationRules("r")).toMatch(/geom_text_repel/);
    expect(annotationRules("r")).toMatch(/element_markdown/);
    expect(annotationRules("python")).toMatch(/adjust_text/);
    expect(annotationRules("python")).toMatch(/ax_text/);
  });
});

describe("get_miami_style chart kinds", () => {
  it("offers a kind per language", () => {
    expect(MIAMI_STYLE_KINDS).toContain("charts-r");
    expect(MIAMI_STYLE_KINDS).toContain("charts-python");
  });

  it("returns rules plus a runnable exemplar for R", async () => {
    const out = await getMiamiStyle("charts-r");
    expect(out).not.toHaveProperty("error");
    const { content } = out as { content: string };
    expect(content).toContain("#C3142D");
    expect(content).toContain("library(ggtext)");
    expect(content).toContain("geom_text_repel");
    expect(content).toMatch(/ISA 401 grades ran higher/);
    // R guidance must not hand the model matplotlib.
    expect(content).not.toContain("import matplotlib");
  });

  it("gives Python both a browser and a hosted exemplar", async () => {
    const out = await getMiamiStyle("charts-python");
    const { content } = out as { content: string };
    expect(content).toContain("from adjustText import adjust_text");
    expect(content).toMatch(/HOSTED sandbox/);
    // The hosted exemplar is the tail; it must be import-clean.
    const hosted = content.slice(content.indexOf("HOSTED sandbox"));
    expect(hosted).not.toContain("import adjustText");
    expect(hosted).not.toContain("from adjustText");
    expect(hosted).not.toContain("highlight_text");
    expect(hosted).toContain("#C3142D");
  });

  it("stays inside a sane tool-output budget", async () => {
    for (const kind of ["charts-r", "charts-python"]) {
      const { content } = (await getMiamiStyle(kind)) as { content: string };
      expect(content.length).toBeLessThan(9_000);
    }
  });

  it("still rejects an unknown kind", async () => {
    expect(await getMiamiStyle("charts")).toHaveProperty("error");
  });
});

describe("hosted deck template", () => {
  it("names the new template in one place", () => {
    expect(DECK_TEMPLATE).toBe("miami_template_by_fadel_megahed.pptx");
  });

  it("carries the palette and title rules into the container note", () => {
    const message = anthropicTemplateMessage("file_test1234");
    const parts = message.content as { type: string; text?: string }[];
    const text = parts
      .filter((p) => p.type === "text")
      .map((p) => p.text ?? "")
      .join("\n");
    expect(text).toContain(DECK_TEMPLATE);
    expect(text).toContain("#C3142D");
    expect(text).toMatch(/states the finding/i);
    expect(text).toMatch(/suboptimal/i);
    expect(text).toMatch(/do not import/i);
  });
});
