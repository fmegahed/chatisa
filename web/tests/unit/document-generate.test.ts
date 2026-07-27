import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  generateCoverLetter,
  generateTailoredResume,
  keepGpaOnlyIfAbove3,
  normaliseSalutation,
} from "@/lib/documents/generate";
import { COVER_LETTER_SHAPE } from "@/lib/prompts/fsb-standards";

const RESUME = `Kaitlin Jones
joneskl@MiamiOH.edu | (513) 555-5555

Data Analytics Intern, Acme Logistics, Summer 2025
Built weekly reports in Excel and SQL for the operations team
Cleaned shipment data and flagged duplicate records across three systems
Presented findings to five managers at the end of the internship

Treasurer, Business Analytics Club, 2024 to 2025
Managed a budget of $4,000 across eight events`;

const CONTACT = {
  email: "joneskl@MiamiOH.edu",
  phone: "(513) 555-5555",
  linkedin: null,
};

describe("tailored resume generation", () => {
  beforeEach(() => {
    process.env.CHATISA_MOCK_LLM = "1";
  });
  afterEach(() => {
    delete process.env.CHATISA_MOCK_LLM;
  });

  it("produces a resume and checks every bullet against the student's own", async () => {
    const result = await generateTailoredResume({
      modelId: "gpt-5.6-terra",
      template: 1,
      studentName: "Kaitlin Jones",
      contact: CONTACT,
      resumeText: RESUME,
      postingText: "Seeking an analyst comfortable with SQL and reporting.",
      company: "Northwind",
      positionTitle: "Analytics Intern",
    });

    expect(result.content.name).toBe("Kaitlin Jones");
    expect(result.content.sections.length).toBeGreaterThan(0);
    expect(result.grounding.checked).toBeGreaterThan(0);
  }, 60_000);

  it("flags an invented bullet rather than passing it through", async () => {
    // The mock deliberately includes "Directed a team of forty consultants
    // across three continents", which appears nowhere in the resume. If this
    // ever stops being flagged, the guard rail has failed silently, which is
    // the failure that matters most in this feature.
    const result = await generateTailoredResume({
      modelId: "gpt-5.6-terra",
      template: 1,
      studentName: "Kaitlin Jones",
      contact: CONTACT,
      resumeText: RESUME,
      postingText: null,
      company: "Northwind",
      positionTitle: "Analytics Intern",
    });

    expect(result.grounding.flagged.length).toBeGreaterThan(0);
    expect(
      result.grounding.flagged.some((f) => /forty consultants/i.test(f.text)),
    ).toBe(true);
    // Flagged, not deleted: the student decides, because the check is a
    // heuristic (user decision, 2026-07-21).
    const allBullets = result.content.sections
      .flatMap((s) => s.entries)
      .flatMap((e) => e.bullets)
      .map((b) => b.text);
    expect(allBullets.some((t) => /forty consultants/i.test(t))).toBe(true);
  }, 60_000);

  it("keeps bullets that only reword the student's own lines", async () => {
    const result = await generateTailoredResume({
      modelId: "gpt-5.6-terra",
      template: 1,
      studentName: "Kaitlin Jones",
      contact: CONTACT,
      resumeText: RESUME,
      postingText: null,
      company: "Northwind",
      positionTitle: "Analytics Intern",
    });
    expect(result.grounding.grounded).toBeGreaterThan(0);
  }, 60_000);
});

describe("cover letter generation", () => {
  beforeEach(() => {
    process.env.CHATISA_MOCK_LLM = "1";
  });
  afterEach(() => {
    delete process.env.CHATISA_MOCK_LLM;
  });

  it("produces the header, salutation and closing the standard requires", async () => {
    const result = await generateCoverLetter({
      modelId: "gpt-5.6-terra",
      studentName: "Kaitlin Jones",
      contact: CONTACT,
      resumeText: RESUME,
      postingText: "We need someone strong in SQL.",
      company: "Northwind",
      positionTitle: "Analytics Intern",
      recipientName: null,
      companyAddress: null,
      todayLabel: "September 22, 2026",
    });

    expect(result.content.recipient.company).toBe("Northwind");
    expect(result.content.salutation.endsWith(":")).toBe(true);
    expect(result.content.closing).toMatch(
      /sincerely|regards|warm regards|best regards/i,
    );
    expect(result.content.paragraphs.length).toBeGreaterThan(1);
  }, 60_000);

  it("checks only the paragraphs that claim experience", async () => {
    // The opening and the closing are about the company and the application,
    // so there is nothing in them to ground against a resume. Checking them
    // would produce noise that trains students to ignore the warnings.
    const result = await generateCoverLetter({
      modelId: "gpt-5.6-terra",
      studentName: "Kaitlin Jones",
      contact: CONTACT,
      resumeText: RESUME,
      postingText: null,
      company: "Northwind",
      positionTitle: "Analytics Intern",
      recipientName: "Ms. Cooper",
      companyAddress: null,
      todayLabel: "September 22, 2026",
    });
    expect(result.grounding.checked).toBeLessThan(
      result.content.paragraphs.length,
    );
  }, 60_000);
});

describe("rules enforced in code rather than trusted to the model", () => {
  it("drops a GPA of 3.0 or below, as the standard requires", () => {
    // A rule with a number in it is enforced here rather than requested.
    expect(keepGpaOnlyIfAbove3("3.6")).toBe("3.6");
    expect(keepGpaOnlyIfAbove3("3.01")).toBe("3.01");
    expect(keepGpaOnlyIfAbove3("3.0")).toBeNull();
    expect(keepGpaOnlyIfAbove3("2.8")).toBeNull();
    expect(keepGpaOnlyIfAbove3("GPA: 3.45")).toBe("GPA: 3.45");
    expect(keepGpaOnlyIfAbove3(null)).toBeNull();
    expect(keepGpaOnlyIfAbove3("not a number")).toBeNull();
  });

  it("ends the salutation the way the school's finished letters do", () => {
    // The annotated template shows a comma; both worked examples use a colon.
    expect(normaliseSalutation("Dear Ms. Cooper")).toBe("Dear Ms. Cooper:");
    expect(normaliseSalutation("Dear Ms. Cooper,")).toBe("Dear Ms. Cooper:");
    expect(normaliseSalutation("Dear Ms. Cooper:")).toBe("Dear Ms. Cooper:");
    expect(normaliseSalutation("")).toBe(
      `Dear Hiring Manager${COVER_LETTER_SHAPE.salutationPunctuation}`,
    );
  });
});
