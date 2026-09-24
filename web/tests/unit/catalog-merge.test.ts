import { describe, expect, it } from "vitest";
import { mergeCatalog, type CatalogCourse } from "@/scripts/catalog/merge";
import type { ParsedCourse } from "@/scripts/catalog/bulletin";

const course = (over: Partial<CatalogCourse> & { code: string }): CatalogCourse => ({
  altCodes: [], title: "T", previousTitles: [], credits: 3, prereq: [], prereqUncertain: false, retired: false,
  ...over,
});
const parsed = (over: Partial<ParsedCourse> & { code: string }): ParsedCourse => ({
  altCodes: [], title: "T", credits: 3, description: "d", prereqText: "", prereq: [], prereqUncertain: false,
  ...over,
});

describe("mergeCatalog", () => {
  it("never removes a course: one missing from the bulletin is marked retired and kept", () => {
    const { courses, report } = mergeCatalog([course({ code: "ISA 245", title: "Old Databases" })], [], new Set());
    expect(courses).toEqual([course({ code: "ISA 245", title: "Old Databases", retired: true })]);
    expect(report.retired).toEqual(["ISA 245"]);
  });

  it("keeps the previous title when a course is renamed, so search finds either", () => {
    const { courses, report } = mergeCatalog(
      [course({ code: "ISA 491", title: "Introduction to Data Mining in Business" })],
      [parsed({ code: "ISA 491", title: "Machine Learning" })],
      new Set(["ISA 491"]),
    );
    expect(courses[0].title).toBe("Machine Learning");
    expect(courses[0].previousTitles).toEqual(["Introduction to Data Mining in Business"]);
    expect(report.retitled).toEqual([{ code: "ISA 491", from: "Introduction to Data Mining in Business", to: "Machine Learning" }]);
  });

  it("flags a suspected renumbering: a new code with a retired course's title", () => {
    const { report } = mergeCatalog(
      [course({ code: "ISA 245", title: "Database Design" })],
      [parsed({ code: "ISA 345", title: "Database Design" })],
      new Set(["ISA 345"]),
    );
    expect(report.suspectedRenumbering).toEqual([{ from: "ISA 245", to: "ISA 345", title: "Database Design" }]);
  });

  it("adds courses that newly appear in a program, and only those", () => {
    const { courses, report } = mergeCatalog(
      [],
      [parsed({ code: "FIN 301", title: "Intro To Business Finance" }), parsed({ code: "FIN 999", title: "Not in any program" })],
      new Set(["FIN 301"]),
    );
    expect(courses.map((c) => c.code)).toEqual(["FIN 301"]);
    expect(report.added).toEqual(["FIN 301"]);
  });

  it("excludes Independent Studies (professor's decision, 2026-07-28)", () => {
    const { courses } = mergeCatalog([], [parsed({ code: "ESP 477", title: "Independent Studies" })], new Set(["ESP 477"]));
    expect(courses).toEqual([]);
  });

  it("marks topics, internships and contemporary-issues courses as freeform, keeping an existing flag", () => {
    const { courses } = mergeCatalog(
      [course({ code: "ISA 340", title: "Internship", special: "freeform" })],
      [parsed({ code: "ISA 340", title: "Internship" }), parsed({ code: "MGT 490", title: "Contemporary Issues" }), parsed({ code: "MKT 490", title: "Emerging Topics in Marketing" })],
      new Set(["ISA 340", "MGT 490", "MKT 490"]),
    );
    expect(courses.every((c) => c.special === "freeform")).toBe(true);
  });

  it("never re-flags an existing course as freeform because of a word in its title", () => {
    // ISA 621 "Enabling Technology Topics I" is a regular course with
    // approved links; the title heuristic is for courses new to the catalog.
    const { courses } = mergeCatalog(
      [course({ code: "ISA 621", title: "Enabling Technology Topics I" })],
      [parsed({ code: "ISA 621", title: "Enabling Technology Topics I" })],
      new Set(["ISA 621"]),
    );
    expect(courses[0].special).toBeUndefined();
  });

  it("seeding from the hand-written catalog updates titles without recording them as renames", () => {
    const { courses, report } = mergeCatalog(
      [course({ code: "ISA 211", title: "IT and Data Driven Decision Making" })],
      [parsed({ code: "ISA 211", title: "Information Technology and Data Driven Decision Making in Business" })],
      new Set(["ISA 211"]),
      { seeding: true },
    );
    expect(courses[0].title).toBe("Information Technology and Data Driven Decision Making in Business");
    expect(courses[0].previousTitles).toEqual([]);
    expect(report.retitled).toEqual([]);
    expect(report.prereqChanged).toEqual([]);
  });

  it("records prerequisite changes in the report", () => {
    const { report } = mergeCatalog(
      [course({ code: "ISA 401", prereq: [["ISA 245", "ISA 345"]] })],
      [parsed({ code: "ISA 401", prereq: [["ISA 345", "CSE 385"]] })],
      new Set(["ISA 401"]),
    );
    expect(report.prereqChanged).toEqual(["ISA 401"]);
  });

  it("sorts courses by code and their equivalents, so a rerun with no change produces no diff", () => {
    const a = mergeCatalog([], [parsed({ code: "MKT 291", altCodes: ["MKT 591", "BUS 591"] }), parsed({ code: "ACC 221" })], new Set(["MKT 291", "ACC 221"]));
    const b = mergeCatalog(a.courses, [parsed({ code: "ACC 221" }), parsed({ code: "MKT 291", altCodes: ["BUS 591", "MKT 591"] })], new Set(["MKT 291", "ACC 221"]));
    expect(a.courses.map((c) => c.code)).toEqual(["ACC 221", "MKT 291"]);
    expect(b.courses).toEqual(a.courses);
    expect(b.report.added).toEqual([]);
    expect(b.courses[1].altCodes).toEqual(["BUS 591", "MKT 591"]);
  });
});
