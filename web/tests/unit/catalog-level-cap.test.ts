import { describe, expect, it } from "vitest";
import { COURSE_SKILLS, capLevel, levelCap } from "@/lib/scout/course-skills";

/**
 * Course level caps skill depth (professor's rule, 2026-09-24): 100-level
 * courses give exposure, 200-level at most applied, anchors start at 300.
 * The one exception he named: Excel in CSE 148 stays an anchor.
 */
describe("levelCap", () => {
  it.each([
    ["BUS 101", "business_acumen", "exposure"],
    ["ISA 125", "statistical_analysis", "exposure"],
    ["ISA 225", "data_analysis", "applied"],
    ["ACC 221", "financial_accounting", "applied"],
    ["ISA 345", "sql", "anchor"],
    ["ISA 495", "business_acumen", "anchor"],
    ["ISA 612", "machine_learning", "anchor"],
    ["CSE 148", "excel", "anchor"],
    ["CSE 148", "programming_fundamentals", "exposure"],
  ])("%s · %s tops out at %s", (course, skill, cap) => {
    expect(levelCap(course, skill)).toBe(cap);
  });

  it("lowers a level above the cap and leaves one at or below it", () => {
    expect(capLevel("ISA 225", "data_analysis", "anchor")).toBe("applied");
    expect(capLevel("ISA 225", "data_analysis", "exposure")).toBe("exposure");
    expect(capLevel("MTH 141", "mathematics", "applied")).toBe("exposure");
  });

  it("holds for every link in the catalog", () => {
    for (const l of COURSE_SKILLS) {
      expect(capLevel(l.course, l.skillId, l.level), `${l.course} ${l.skillId} is ${l.level}`).toBe(l.level);
    }
  });
});
