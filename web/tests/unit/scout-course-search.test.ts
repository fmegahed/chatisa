import { describe, expect, it } from "vitest";
import { matchesCourse, type CourseDef } from "@/lib/scout/courses";

/**
 * Course search (v6.7.0), shared by the checklist and the showcase picker:
 * code with or without the space, title, cross-listed codes, and previous
 * titles, so a renamed course is still found by the name students know.
 */
const isa491: CourseDef = {
  code: "ISA 491", altCodes: ["STA 491"], title: "Machine Learning", previousTitles: ["Introduction to Data Mining"],
  credits: 3, prereq: [], prereqUncertain: false, retired: false,
};

describe("matchesCourse", () => {
  it.each([["491"], ["isa491"], ["ISA 491"], ["machine"], ["sta 491"], ["data mining"]])("finds ISA 491 by %s", (q) => {
    expect(matchesCourse(isa491, q)).toBe(true);
  });
  it("does not match unrelated text", () => {
    expect(matchesCourse(isa491, "forecasting")).toBe(false);
  });
});
