import { describe, expect, it } from "vitest";
import { unzipSync, strFromU8 } from "fflate";
import { renderCoverLetterDocx, renderResumeDocx } from "@/lib/documents/docx";
import { LAYOUT } from "@/lib/prompts/fsb-standards";
import type { CoverLetterContent, ResumeContent } from "@/lib/documents/schema";

/**
 * A .docx is a zip of XML, so these open the produced file and read it rather
 * than asserting it merely did not throw. The metrics checked here are the ones
 * read out of the FSB Word originals, so a drift from the school's standard
 * fails the build.
 */
function openDocx(buffer: Buffer): string {
  const files = unzipSync(new Uint8Array(buffer));
  return strFromU8(files["word/document.xml"]);
}

const RESUME: ResumeContent = {
  name: "Kaitlin Jones",
  contact: {
    email: "joneskl@MiamiOH.edu",
    phone: "(513) 555-5555",
    linkedin: "linkedin.com/in/kaitlinjones",
  },
  education: {
    school: "Miami University",
    location: "Oxford, OH",
    degree: "Bachelors of Science",
    majorMinor: "Business Analytics / Statistics",
    graduation: "Expected Graduation 2027",
    gpa: "3.6",
    honors: ["Dean's List, four semesters"],
  },
  sections: [
    {
      heading: "Relevant Experience",
      entries: [
        {
          organization: "Acme Logistics",
          title: "Data Analytics Intern",
          location: "Cincinnati, OH",
          dates: "Summer 2025",
          bullets: [
            { text: "Automated weekly operations reporting in SQL", sourceLine: null },
            { text: "Reconciled shipment data across three systems", sourceLine: null },
          ],
        },
      ],
    },
  ],
  skills: ["R", "Python", "SQL", "Tableau"],
};

const LETTER: CoverLetterContent = {
  name: "Kaitlin Jones",
  contact: {
    email: "joneskl@MiamiOH.edu",
    phone: "(513) 555-5555",
    linkedin: null,
  },
  date: "September 22, 2026",
  recipient: {
    name: "Loretta Cooper",
    company: "XYZ Marketing",
    address: "718 12th Street, Chicago, IL 61234",
  },
  salutation: "Dear Ms. Cooper:",
  paragraphs: [
    { text: "Through Handshake I learned of the analytics internship.", addresses: null, sourceLine: null },
    { text: "I am in my third year at the Farmer School of Business.", addresses: "education", sourceLine: null },
  ],
  closing: "Sincerely,",
};

describe("resume .docx", () => {
  it("uses US Letter with half-inch margins, as the FSB original does", async () => {
    const xml = openDocx(await renderResumeDocx(RESUME, 1));
    expect(xml).toContain(`w:w="${LAYOUT.pageWidthTwips}"`);
    expect(xml).toContain(`w:h="${LAYOUT.pageHeightTwips}"`);
    // 720 twips is half an inch, which the user confirmed for undergraduates.
    expect(xml).toMatch(/w:top="720"/);
    expect(xml).toMatch(/w:left="720"/);
  });

  it("sets a right-aligned tab at the right margin rather than repeating tabs", async () => {
    // The FSB file aligns dates with many left tabs, a Google Docs artifact
    // that drifts as soon as a line is edited. One right tab renders the same
    // and survives editing, which is the point of shipping a .docx at all.
    const xml = openDocx(await renderResumeDocx(RESUME, 1));
    expect(xml).toContain(`w:val="right"`);
    expect(xml).toContain(`w:pos="${LAYOUT.contentWidthTwips}"`);
    expect(LAYOUT.contentWidthTwips).toBe(10_800); // 7.5 inches
  });

  it("carries the content the student will be judged on", async () => {
    const xml = openDocx(await renderResumeDocx(RESUME, 1));
    for (const expected of [
      "Kaitlin Jones",
      "joneskl@MiamiOH.edu",
      "Acme Logistics",
      "Data Analytics Intern",
      "Automated weekly operations reporting in SQL",
      "Expected Graduation 2027",
    ]) {
      expect(xml, expected).toContain(expected);
    }
  });

  it("uses Arial headings and a serif body for Standard 1", async () => {
    const xml = openDocx(await renderResumeDocx(RESUME, 1));
    expect(xml).toContain("Arial");
    expect(xml).toContain("Times New Roman");
  });

  it("uses Arial throughout for Standard 3", async () => {
    const xml = openDocx(await renderResumeDocx(RESUME, 3));
    expect(xml).not.toContain("Times New Roman");
  });

  it("omits the school suffix on Standard 1 and includes it on Standard 2", async () => {
    const one = openDocx(await renderResumeDocx(RESUME, 1));
    const two = openDocx(await renderResumeDocx(RESUME, 2));
    expect(one).not.toContain("Farmer School of Business");
    expect(two).toContain("Farmer School of Business");
  });

  it("leads with the organization on Standard 1 and the title on Standard 3", async () => {
    const one = openDocx(await renderResumeDocx(RESUME, 1));
    const three = openDocx(await renderResumeDocx(RESUME, 3));
    expect(one.indexOf("Acme Logistics")).toBeLessThan(
      one.indexOf("Data Analytics Intern"),
    );
    expect(three.indexOf("Data Analytics Intern")).toBeLessThan(
      three.indexOf("Acme Logistics"),
    );
  });

  it("omits an absent contact field rather than leaving an empty separator", async () => {
    const xml = openDocx(
      await renderResumeDocx(
        { ...RESUME, contact: { email: "a@MiamiOH.edu", phone: null, linkedin: null } },
        1,
      ),
    );
    expect(xml).not.toContain("| |");
  });
});

describe("cover letter .docx", () => {
  it("matches the resume letterhead so they read as a pair", async () => {
    const xml = openDocx(await renderCoverLetterDocx(LETTER));
    expect(xml).toContain("Kaitlin Jones");
    expect(xml).toContain("joneskl@MiamiOH.edu");
  });

  it("keeps the address block, salutation, body and closing in order", async () => {
    const xml = openDocx(await renderCoverLetterDocx(LETTER));
    const order = [
      "September 22, 2026",
      "Loretta Cooper",
      "XYZ Marketing",
      "Dear Ms. Cooper:",
      "Through Handshake",
      "Sincerely,",
    ].map((t) => xml.indexOf(t));
    expect(order.every((i) => i >= 0)).toBe(true);
    expect([...order]).toEqual([...order].sort((a, b) => a - b));
  });

  it("is set in Arial, as both cover letter templates are", async () => {
    const xml = openDocx(await renderCoverLetterDocx(LETTER));
    expect(xml).toContain("Arial");
    expect(xml).not.toContain("Times New Roman");
  });
});
