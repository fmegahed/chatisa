import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import { extractEvidence, listSections, mergeEvidence } from "@/scripts/catalog/syllabi";

/**
 * Simple Syllabus evidence (v6.7.0). The fixture is ISA 225's real public
 * syllabus with the instructors' names and emails replaced, plus one stray
 * email planted in the calendar to prove extraction scrubs it.
 */
const doc = JSON.parse(
  readFileSync(path.join(process.cwd(), "tests", "fixtures", "syllabus", "isa-225.json"), "utf8"),
);

describe("listSections", () => {
  it("follows the library's pagination to the end, grouping sections by course", async () => {
    const pages = [
      { pagination: { total: 3, returned: 2, page: 0, page_size: 2 }, items: [
        { code: "a1", title: "ISA 225 A", term_name: "Fall Semester 2026-27" },
        { code: "a2", title: "ISA 225 B", term_name: "Fall Semester 2026-27" },
      ] },
      { pagination: { total: 3, returned: 1, page: 1, page_size: 2 }, items: [
        { code: "b1", title: "FIN 301 H A", term_name: "Winter Term 2026-27" },
      ] },
    ];
    const seen: string[] = [];
    const fetchJson = async (url: string) => {
      seen.push(url);
      return pages[Number(new URL(url).searchParams.get("page"))];
    };
    const sections = await listSections(fetchJson, 2);
    expect(seen).toHaveLength(2);
    expect(sections.get("ISA 225")).toEqual([
      { code: "a1", term: "Fall Semester 2026-27" },
      { code: "a2", term: "Fall Semester 2026-27" },
    ]);
    expect(sections.get("FIN 301")).toEqual([{ code: "b1", term: "Winter Term 2026-27" }]);
  });
});

describe("extractEvidence", () => {
  const ev = extractEvidence(doc);

  it("reads the weekly topics from the calendar", () => {
    expect(ev.topics).toEqual(expect.arrayContaining([
      "Multiple Linear Regression (MLR)",
      "Decision Trees",
    ]));
    expect(ev.topics.some((t) => t.startsWith("Market Basket Analysis"))).toBe(true);
  });

  it("never keeps an email address or anything from the instructor section", () => {
    const all = JSON.stringify(ev);
    expect(all).not.toMatch(/@/);
    expect(all).not.toMatch(/Instructor A|Instructor B|Ph\.D/);
  });

  it("does not keep reading assignments as topics", () => {
    expect(ev.topics.every((t) => !/^Ch\.\s*\d/.test(t))).toBe(true);
  });

  it("keeps textbook titles, never their authors or ISBNs", () => {
    expect(ev.readings).toEqual(["Business Statistics"]);
    const all = JSON.stringify(ev);
    expect(all).not.toMatch(/Sharpe|Velleman|9780134705217/);
  });

  it("does not mistake the verb 'excel' for the spreadsheet", () => {
    const doc2 = JSON.parse(JSON.stringify(doc));
    const cal = doc2.items[0].doc_data.components.find((c: { html?: string }) => (c.html ?? "").includes("Calendar"));
    cal.html = cal.html.replace("Decision Trees", "How to excel at Decision Trees");
    expect(extractEvidence(doc2).tools).not.toContain("Excel");
  });
});

describe("mergeEvidence", () => {
  it("combines sections: terms and tools unioned, topics deduplicated", () => {
    const merged = mergeEvidence([
      { term: "Fall Semester 2026-27", evidence: { topics: ["Decision Trees", "Regression"], tools: ["R"], outcomes: [], readings: ["R for Data Science"] } },
      { term: "Winter Term 2026-27", evidence: { topics: ["decision trees", "Clustering"], tools: ["Python"], outcomes: ["Build models"], readings: ["r for data science"] } },
    ]);
    expect(merged.terms).toEqual(["Fall Semester 2026-27", "Winter Term 2026-27"]);
    expect(merged.sections).toBe(2);
    expect(merged.tools).toEqual(["Python", "R"]);
    expect(merged.topics).toEqual(["Clustering", "Decision Trees", "Regression"]);
    expect(merged.outcomes).toEqual(["Build models"]);
    expect(merged.readings).toEqual(["R for Data Science"]);
  });
});
