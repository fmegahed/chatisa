import { describe, expect, it } from "vitest";
import { inferPrereqs, type PrereqDeps } from "@/lib/scout/prereqs";
import type { ProfileCourse } from "@/lib/scout/profile-store";

/**
 * Automatic prerequisites (v6.7.0, professor's rules of 2026-09-24): a
 * course marked Done or Taking now brings its prerequisites in as Done,
 * labelled and removable. "Or" groups take the student's own path: an
 * option their programs require, then one from their major's department,
 * then the first listed. Permission routes and groups of four or more are
 * skipped, as are courses outside the catalog, and a removed prerequisite is
 * never added back. A small synthetic catalog keeps these stable as the
 * bulletin changes.
 */
const course = (code: string, prereq: string[][] = [], prereqUncertain = false) => ({
  code, altCodes: [] as string[], title: code, previousTitles: [], credits: 3, prereq, prereqUncertain, retired: false,
});
const catalog = [
  course("ISA 401", [["ISA 245", "ISA 345", "CSE 385"]]),
  course("ISA 345", [["ISA 235", "ISA 211"]]),
  course("ISA 235"),
  course("ISA 211"),
  course("CSE 385"),
  course("ISA 444", [["ECO 311", "ISA 291", "ISA 391", "STA 463"]]),
  course("ISA 414", [["ISA 381", "ISA 401"]], true),
  course("FIN 401", [["FIN 301", "FIN 311"], ["ACC 221"]]),
  course("FIN 301"),
  course("FIN 311"),
  course("ACC 221"),
  course("MKT 335", [["STA 125", "ISA 125"]]),
  course("ISA 125"),
];
const byCode = new Map(catalog.map((c) => [c.code, c]));
const deps = (programs: Record<string, string[]>): PrereqDeps => ({
  getCourse: (code) => byCode.get(code),
  programCodes: (key) => programs[key] ?? [],
});
const PROGRAMS = {
  "business-core": ["ISA 125", "ISA 235", "FIN 301", "ACC 221"],
  "business-analytics": ["ISA 345", "ISA 401", "ISA 444"],
  finance: ["FIN 401", "FIN 311"],
};
const done = (code: string, extra: Partial<ProfileCourse> = {}): ProfileCourse => ({ code, status: "done", ...extra });
const now = (code: string): ProfileCourse => ({ code, status: "now" });

describe("inferPrereqs", () => {
  it("follows the chain along the student's own path, labelled with the course that caused it", () => {
    const out = inferPrereqs({ programs: ["business-analytics"], courses: [now("ISA 401")], removedPrereqs: [] }, deps(PROGRAMS));
    // ISA 345 (required by the BA major) over ISA 245 (not in the catalog) and
    // CSE 385; then ISA 235 (in the business core) over ISA 211.
    expect(out).toEqual([
      now("ISA 401"),
      done("ISA 345", { addedBecause: "ISA 401" }),
      done("ISA 235", { addedBecause: "ISA 401" }),
    ]);
  });

  it("prefers the major's own department when no program names an option", () => {
    const out = inferPrereqs({ programs: ["finance"], courses: [now("FIN 401")], removedPrereqs: [] }, deps({ finance: ["FIN 401"] }));
    // FIN 301 or FIN 311: same department, neither required here, so the first
    // FIN option; ACC 221 is a single-option group.
    expect(out.map((c) => c.code)).toEqual(["FIN 401", "FIN 301", "ACC 221"]);
  });

  it("uses a program-required option before the department default", () => {
    const out = inferPrereqs({ programs: ["finance"], courses: [done("FIN 401")], removedPrereqs: [] }, deps(PROGRAMS));
    expect(out.map((c) => c.code)).toContain("FIN 311");
    expect(out.map((c) => c.code)).not.toContain("FIN 301");
  });

  it("skips a group of four or more options", () => {
    const out = inferPrereqs({ programs: ["business-analytics"], courses: [now("ISA 444")], removedPrereqs: [] }, deps(PROGRAMS));
    expect(out).toEqual([now("ISA 444")]);
  });

  it("skips a course whose prerequisite allows instructor permission", () => {
    const out = inferPrereqs({ programs: ["business-analytics"], courses: [now("ISA 414")], removedPrereqs: [] }, deps(PROGRAMS));
    expect(out).toEqual([now("ISA 414")]);
  });

  it("treats a group as met when the student already has any option", () => {
    const out = inferPrereqs({ programs: [], courses: [done("ISA 345"), done("ISA 211")], removedPrereqs: [] }, deps(PROGRAMS));
    expect(out.map((c) => c.code)).toEqual(["ISA 345", "ISA 211"]);
  });

  it("never re-adds a prerequisite the student removed, nor anything that came through it", () => {
    const out = inferPrereqs({ programs: ["business-analytics"], courses: [now("ISA 401")], removedPrereqs: ["ISA 345"] }, deps(PROGRAMS));
    expect(out).toEqual([now("ISA 401")]);
  });

  it("never changes a course the student set themselves", () => {
    const mine = done("ISA 345", { term: "Spring 2026" });
    const out = inferPrereqs({ programs: ["business-analytics"], courses: [now("ISA 401"), mine], removedPrereqs: [] }, deps(PROGRAMS));
    expect(out.find((c) => c.code === "ISA 345")).toEqual(mine);
  });

  it("chooses nothing outside the catalog", () => {
    const out = inferPrereqs({ programs: [], courses: [now("MKT 335")], removedPrereqs: [] }, deps({}));
    // STA 125 is not in this catalog; ISA 125 is the in-scope option.
    expect(out.map((c) => c.code)).toEqual(["MKT 335", "ISA 125"]);
  });
});
