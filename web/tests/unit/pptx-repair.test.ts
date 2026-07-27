import JSZip from "jszip";
import { describe, expect, it } from "vitest";
import { dedupeLayoutRelationships, repairPptx } from "@/lib/ask/pptx-repair";

/**
 * A generated deck arrived with two slideLayout relationships per slide, which
 * makes PowerPoint refuse the whole package. These tests pin the repair, and
 * pin that a valid deck is passed through byte-for-byte.
 */

const LAYOUT = "officeDocument/2006/relationships/slideLayout";
const NOTES = "officeDocument/2006/relationships/notesSlide";

function rels(...tags: string[]): string {
  return (
    `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>` +
    `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">` +
    tags.join("") +
    `</Relationships>`
  );
}

const layoutTag = (id: string, target = "../slideLayouts/slideLayout1.xml") =>
  `<Relationship Id="${id}" Type="http://schemas.openxmlformats.org/${LAYOUT}" Target="${target}"/>`;

const notesTag = (id: string) =>
  `<Relationship Id="${id}" Type="http://schemas.openxmlformats.org/${NOTES}" Target="../notesSlides/notesSlide1.xml"/>`;

async function deck(relsBySlide: Record<string, string>): Promise<Uint8Array> {
  const zip = new JSZip();
  zip.file("[Content_Types].xml", "<Types/>");
  zip.file("ppt/presentation.xml", "<p:presentation/>");
  for (const [name, xml] of Object.entries(relsBySlide)) {
    zip.file(`ppt/slides/_rels/${name}.xml.rels`, xml);
  }
  return zip.generateAsync({ type: "uint8array" });
}

describe("dedupeLayoutRelationships", () => {
  it("keeps the first slideLayout and drops later duplicates", () => {
    const { xml, removed } = dedupeLayoutRelationships(
      rels(layoutTag("rId1"), layoutTag("rId2")),
    );
    expect(removed).toBe(1);
    expect(xml).toContain('Id="rId1"');
    expect(xml).not.toContain('Id="rId2"');
  });

  it("leaves a correct slide untouched, notesSlide included", () => {
    const input = rels(notesTag("rId2"), layoutTag("rId1"));
    const { xml, removed } = dedupeLayoutRelationships(input);
    expect(removed).toBe(0);
    expect(xml).toBe(input);
  });

  it("does not treat a second non-layout relationship as a duplicate", () => {
    const { removed } = dedupeLayoutRelationships(
      rels(layoutTag("rId1"), notesTag("rId2"), notesTag("rId3")),
    );
    expect(removed).toBe(0);
  });

  it("removes two duplicates when three layouts are present", () => {
    const { removed } = dedupeLayoutRelationships(
      rels(layoutTag("rId1"), layoutTag("rId2"), layoutTag("rId3")),
    );
    expect(removed).toBe(2);
  });
});

describe("repairPptx", () => {
  it("repairs every slide in a broken deck", async () => {
    const broken = await deck({
      slide1: rels(layoutTag("rId1"), layoutTag("rId2")),
      slide2: rels(layoutTag("rId1"), layoutTag("rId2")),
      slide3: rels(layoutTag("rId1"), layoutTag("rId2")),
    });
    const { bytes, removed } = await repairPptx(broken);
    expect(removed).toBe(3);

    // The output must still be a readable package with exactly one layout
    // relationship per slide.
    const zip = await JSZip.loadAsync(bytes);
    for (const n of ["slide1", "slide2", "slide3"]) {
      const xml = await zip.file(`ppt/slides/_rels/${n}.xml.rels`)!.async("string");
      const layouts = xml.match(new RegExp(LAYOUT, "g")) ?? [];
      expect(layouts).toHaveLength(1);
    }
  });

  it("passes a valid deck through unchanged, byte for byte", async () => {
    const clean = await deck({ slide1: rels(layoutTag("rId1"), notesTag("rId2")) });
    const { bytes, removed } = await repairPptx(clean);
    expect(removed).toBe(0);
    // Identity, not just equality of content: a valid deck must not be re-zipped.
    expect(bytes).toBe(clean);
  });

  it("returns non-zip bytes unchanged rather than throwing", async () => {
    const notAZip = new TextEncoder().encode("this is not a pptx at all");
    const { bytes, removed } = await repairPptx(notAZip);
    expect(removed).toBe(0);
    expect(bytes).toBe(notAZip);
  });

  it("leaves a zip with no slide rels alone", async () => {
    const zip = new JSZip();
    zip.file("ppt/presentation.xml", "<p:presentation/>");
    const input = await zip.generateAsync({ type: "uint8array" });
    const { removed } = await repairPptx(input);
    expect(removed).toBe(0);
  });
});
