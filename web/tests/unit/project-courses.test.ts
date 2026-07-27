// tests/unit/project-courses.test.ts
import { describe, expect, it } from "vitest";
import { ISA_COURSES, findCourse, courseLabel } from "@/lib/project/courses";

describe("ISA course catalog", () => {
  it("includes every catalog course with a code and title", () => {
    expect(ISA_COURSES.length).toBe(50);
    for (const c of ISA_COURSES) {
      expect(c.code).toMatch(/^\d{3}(\/\d{3})?$/);
      expect(c.title.length).toBeGreaterThan(0);
    }
  });

  it("keeps codes unique", () => {
    const codes = ISA_COURSES.map((c) => c.code);
    expect(new Set(codes).size).toBe(codes.length);
  });

  it("finds a course by code, including dual-listed codes", () => {
    expect(findCourse("401/501")?.title).toBe(
      "Business Intelligence and Data Visualization",
    );
    expect(findCourse("444/544")?.title).toBe("Business Forecasting");
    expect(findCourse("nope")).toBeUndefined();
  });

  it("labels a course with ISA prefix and a colon, no em dash", () => {
    const label = courseLabel(findCourse("444/544")!);
    expect(label).toBe("ISA 444/544: Business Forecasting");
    expect(label).not.toContain("—");
  });
});
