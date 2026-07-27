// tests/unit/project-docx.test.ts
import { describe, expect, it } from "vitest";
import JSZip from "jszip";
import { renderProjectDeliverablesDocx, scopingBlocks, genericBlocks } from "@/lib/documents/coach-docx";
import { renderScopingDocx } from "@/lib/documents/scoping-docx";
import { emptyScopingContent } from "@/lib/project/scoping";
import { COACH_SPECS } from "@/lib/project/coach-specs";
import { buildEmptyContent } from "@/lib/project/coach-framework";

const header = {
  projectName: "Retail dashboard",
  courseLabel: "ISA 496: Business Analytics Practicum",
  organization: "Kroger",
  members: ["Lead", "Mate"],
};

async function documentXml(buf: Buffer): Promise<string> {
  const zip = await JSZip.loadAsync(buf);
  return zip.file("word/document.xml")!.async("string");
}

describe("project export", () => {
  it("combines multiple deliverables into one document", async () => {
    const scoping = emptyScopingContent();
    scoping.projectName = "Retail dashboard";
    const premortem = COACH_SPECS.premortem;
    const pmContent = buildEmptyContent(premortem);
    pmContent.fields.projectDescription = "A forecasting tool";

    const buf = await renderProjectDeliverablesDocx(header, [
      { title: "Project Scoping", blocks: scopingBlocks(scoping) },
      { title: "Premortem", blocks: genericBlocks(premortem, pmContent) },
    ]);
    const xml = await documentXml(buf);
    expect(buf.subarray(0, 2).toString("latin1")).toBe("PK");
    expect(xml).toContain("A forecasting tool");
    // Both section titles present.
    expect(xml).toContain("Project Scoping");
    expect(xml).toContain("Premortem");
  });

  it("keeps the title and headings Miami Red (regression)", async () => {
    const xml = await documentXml(await renderScopingDocx(emptyScopingContent(), header));
    expect((xml.match(/C41230/g) ?? []).length).toBeGreaterThanOrEqual(2);
  });
});
