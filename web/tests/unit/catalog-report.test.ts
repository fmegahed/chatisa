import { describe, expect, it } from "vitest";
import { renderReport } from "@/scripts/catalog/report";

const empty = {
  seeding: false,
  report: { added: [], retired: [], retitled: [], suspectedRenumbering: [], prereqChanged: [] },
  unparsed: [],
  missingFromBulletin: [],
  counts: { programs: 13, courses: 166, withSyllabus: 115 },
};

describe("renderReport", () => {
  it("says plainly when nothing changed", () => {
    const md = renderReport(empty, null);
    expect(md).toContain("No changes to courses or programs");
    expect(md).toContain("166 courses");
  });

  it("lists every kind of change a reviewer must see, in plain words", () => {
    const md = renderReport({
      ...empty,
      report: {
        added: ["FIN 402"],
        retired: ["ISA 245"],
        retitled: [{ code: "ISA 491", from: "Introduction to Data Mining in Business", to: "Machine Learning" }],
        suspectedRenumbering: [{ from: "ISA 245", to: "ISA 345", title: "Database Design" }],
        prereqChanged: ["ISA 401"],
      },
      unparsed: [{ program: "marketing", row: "Something new" }],
      missingFromBulletin: ["MKT 999"],
    }, {
      mapped: ["FIN 402"],
      agreed: 5,
      disputed: [{ course: "FIN 402", skillId: "financial_modeling", a: "anchor", b: "applied" }],
      skippedByCap: ["ECO 999"],
      costUsd: 0.21,
    });
    expect(md).toContain("ISA 491: Introduction to Data Mining in Business → Machine Learning");
    expect(md).toContain("ISA 245 → ISA 345 (Database Design)");
    expect(md).toContain("Needs your decision");
    expect(md).toContain("FIN 402 · financial_modeling: anchor or applied");
    expect(md).toContain("Not understood");
    expect(md).toContain("marketing: Something new");
    expect(md).toContain("ECO 999");
    expect(md).toContain("$0.21");
  });

  it("shows new anchors with evidence, failed courses, and decisions still open from earlier runs (review fix)", () => {
    const md = renderReport(empty, {
      mapped: ["FIN 402"], agreed: 2, disputed: [], skippedByCap: [], costUsd: 0.1,
      failed: [{ course: "ECO 311", error: "timeout" }],
      written: [
        { course: "FIN 402", skillId: "financial_modeling", level: "anchor", evidence: "built a three-statement model" },
        { course: "FIN 402", skillId: "excel", level: "applied" },
      ],
    }, ["GEO 442", "FIN 402"]);
    expect(md).toContain("FIN 402 · financial_modeling (anchor): \"built a three-statement model\"");
    expect(md).toContain("FIN 402 · excel (applied)");
    expect(md).toContain("ECO 311: timeout");
    expect(md).toContain("Still waiting for a decision from earlier runs");
    // FIN 402 was mapped this run, so it is not listed as left over.
    expect(md).toMatch(/earlier runs[^#]*GEO 442/);
    expect(md).not.toMatch(/earlier runs[^#]*- FIN 402/);
  });
});
