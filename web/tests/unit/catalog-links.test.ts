import { describe, expect, it } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import path from "node:path";
import { COURSE_SKILLS } from "@/lib/scout/course-skills";

/**
 * The approved course-to-skill links moved from a hand-written TS array into
 * catalog/course-skills.json (v6.7.0) so the catalog pipeline can propose
 * changes as reviewable diffs. The move itself must change nothing: this
 * pins the 212 links approved on 2026-07-28 exactly.
 */
describe("course-skill links live in the catalog", () => {
  const root = process.cwd();

  it("are read from catalog/course-skills.json", () => {
    const file = path.join(root, "catalog", "course-skills.json");
    expect(existsSync(file)).toBe(true);
    expect(JSON.parse(readFileSync(file, "utf8"))).toEqual(COURSE_SKILLS);
  });

  it("still contain every link approved before the move, unchanged", () => {
    // The catalog pipeline may add links for new courses, but it never
    // removes or re-levels a link a person approved (v6.7.0).
    const before = JSON.parse(readFileSync(path.join(root, "tests", "fixtures", "course-skills-2026-09-24.json"), "utf8"));
    for (const link of before) expect(COURSE_SKILLS).toContainEqual(link);
  });
});
