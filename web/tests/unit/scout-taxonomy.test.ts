/**
 * Integrity guardrails for the Job Scout skill data. These fail the build
 * when an id is duplicated or a mapping references a skill or course that
 * does not exist, which is exactly how careerbridge's free-text lists
 * drifted (design 2026-07-28 §2.1).
 */
import { describe, expect, it } from "vitest";
import { resolveSkillId, SKILLS, SKILL_IDS, getSkill } from "@/lib/scout/taxonomy";
import { COURSES, getCourse } from "@/lib/scout/courses";
import { COURSE_SKILLS } from "@/lib/scout/course-skills";

describe("taxonomy integrity", () => {
  it("has unique skill ids", () => {
    expect(new Set(SKILL_IDS).size).toBe(SKILLS.length);
  });

  it("every implies edge resolves to a real skill and never self-references", () => {
    for (const s of SKILLS) {
      for (const target of s.implies) {
        expect(getSkill(target), `${s.id} implies ${target}`).toBeDefined();
        expect(target).not.toBe(s.id);
      }
    }
  });

  it("resolveSkillId maps ids, labels, aliases, and near-misses; rejects junk", () => {
    // The wire schema is plain strings (Gemini rejects a 104-value enum,
    // 2026-07-28), so this resolver IS the vocabulary enforcement.
    expect(resolveSkillId("sql")).toBe("sql");
    expect(resolveSkillId("Power BI")).toBe("power_bi");
    expect(resolveSkillId("powerbi")).toBe("power_bi");
    expect(resolveSkillId("machine learning")).toBe("machine_learning");
    expect(resolveSkillId("Data Visualization")).toBe("data_visualization");
    expect(resolveSkillId("underwater basket weaving")).toBeNull();
    expect(resolveSkillId("")).toBeNull();
  });

  it("aliases are lowercase and never duplicate an id", () => {
    const ids = new Set(SKILL_IDS);
    for (const s of SKILLS) {
      for (const a of s.aliases) {
        expect(a).toBe(a.toLowerCase());
        expect(ids.has(a), `alias "${a}" collides with a skill id`).toBe(false);
      }
    }
  });
});

describe("course catalog integrity", () => {
  it("course codes and altCodes are globally unique", () => {
    const all = COURSES.flatMap((c) => [c.code, ...c.altCodes]);
    expect(new Set(all).size).toBe(all.length);
  });

  it("altCode lookup resolves cross-listed courses", () => {
    expect(getCourse("ISA 501")?.code).toBe("ISA 401");
    expect(getCourse("STA 365")?.code).toBe("ISA 365");
    expect(getCourse("BUS 645")?.code).toBe("ISA 645");
  });

  it("excludes Independent Studies per user instruction (2026-07-28)", () => {
    for (const code of ["ISA 177", "ISA 277", "ISA 377", "ISA 477", "ISA 677"]) {
      expect(getCourse(code)).toBeUndefined();
    }
  });
});

describe("course-skill mapping integrity", () => {
  it("every link references a real course (by primary code) and skill", () => {
    for (const link of COURSE_SKILLS) {
      const course = getCourse(link.course);
      expect(course, `unknown course ${link.course}`).toBeDefined();
      expect(course?.code, `${link.course} must use the primary code`).toBe(
        link.course,
      );
      expect(getSkill(link.skillId), `unknown skill ${link.skillId}`).toBeDefined();
    }
  });

  it("every non-freeform course has at least one anchor link", () => {
    for (const course of COURSES) {
      if (course.special) continue;
      const anchors = COURSE_SKILLS.filter(
        (l) => l.course === course.code && l.level === "anchor",
      );
      expect(anchors.length, `${course.code} needs an anchor`).toBeGreaterThan(0);
    }
  });

  it("freeform courses have no static mapping", () => {
    for (const course of COURSES.filter((c) => c.special)) {
      expect(
        COURSE_SKILLS.some((l) => l.course === course.code),
        `${course.code} is freeform and must not be statically mapped`,
      ).toBe(false);
    }
  });

  it("no duplicate course-skill pairs", () => {
    const keys = COURSE_SKILLS.map((l) => `${l.course}|${l.skillId}`);
    expect(new Set(keys).size).toBe(keys.length);
  });

  it("anchor links carry evidence phrases for grounded resume bullets", () => {
    for (const link of COURSE_SKILLS) {
      if (link.level !== "anchor") continue;
      expect(
        link.evidence && link.evidence.length > 10,
        `${link.course} ${link.skillId} anchor needs evidence`,
      ).toBeTruthy();
    }
  });
});
