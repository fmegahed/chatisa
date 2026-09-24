import { describe, expect, it } from "vitest";
import { validateCourses } from "@/lib/scout/courses";
import { validatePrograms } from "@/lib/scout/programs";

/**
 * courses.json and programs.json are checked at load (review fix, v6.7.0),
 * like course-skills.json: a malformed file merged from a catalog PR fails
 * the PR's integrity step and startup, instead of breaking the checklist
 * for every student at the first click.
 */
const good = { code: "ISA 401", altCodes: [], title: "BI", previousTitles: [], credits: 3, prereq: [["ISA 345"]], prereqUncertain: false, retired: false };

describe("validateCourses", () => {
  it("accepts the catalog's own shape", () => {
    expect(validateCourses([good])).toEqual([good]);
  });
  it.each([
    ["a prereq that is not groups of codes", { ...good, prereq: "ISA 345" }],
    ["a prereq group that is not a list", { ...good, prereq: ["ISA 345"] }],
    ["a missing title", { ...good, title: undefined }],
    ["credits that are not a number", { ...good, credits: "3" }],
  ])("throws on %s", (_, bad) => {
    expect(() => validateCourses([bad])).toThrow(/catalog\/courses\.json entry 0/);
  });
});

describe("validatePrograms", () => {
  const program = { key: "x", name: "X", kind: "major", url: "https://bulletin.miamioh.edu/x", groups: [{ title: "Required", subtitle: null, instruction: null, items: [{ codes: ["ISA 401"] }], notes: [] }] };
  it("accepts the catalog's own shape", () => {
    expect(validatePrograms({ bulletinYear: "2026-27", programs: [program] }).programs).toHaveLength(1);
  });
  it("throws on an item without a list of codes", () => {
    const bad = { ...program, groups: [{ ...program.groups[0], items: [{ codes: "ISA 401" }] }] };
    expect(() => validatePrograms({ bulletinYear: "2026-27", programs: [bad] })).toThrow(/catalog\/programs\.json program 0/);
  });
});
