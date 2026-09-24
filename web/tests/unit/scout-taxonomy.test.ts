/**
 * Integrity guardrails for the Job Scout skill data. These fail the build
 * when an id is duplicated or a mapping references a skill or course that
 * does not exist, which is exactly how careerbridge's free-text lists
 * drifted (design 2026-07-28 §2.1).
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import { resolveSkillId, SKILLS, SKILL_IDS, TAXONOMY_VERSION, getSkill } from "@/lib/scout/taxonomy";
import { COURSES, getCourse } from "@/lib/scout/courses";
import { COURSE_SKILLS, levelCap } from "@/lib/scout/course-skills";

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

  it("an alias belongs to exactly one skill (the resolver would silently keep the last)", () => {
    const owner = new Map<string, string>();
    for (const s of SKILLS) {
      for (const a of [s.label.toLowerCase(), ...s.aliases]) {
        const prev = owner.get(a);
        expect(prev === undefined || prev === s.id, `"${a}" is on both ${prev} and ${s.id}`).toBe(true);
        owner.set(a, s.id);
      }
    }
  });

  it("v2 adds 25 to 35 business-domain skills that resolve by label and alias (2026-09-24)", () => {
    expect(TAXONOMY_VERSION).toBe(2);
    const business = SKILLS.filter((s) => s.category === "business");
    expect(business.length).toBeGreaterThanOrEqual(25);
    expect(business.length).toBeLessThanOrEqual(35);
    expect(resolveSkillId("Financial Statement Analysis")).toBe("financial_analysis");
    expect(resolveSkillId("DCF")).toBe("financial_modeling");
    expect(resolveSkillId("six sigma")).toBe("process_improvement");
    expect(resolveSkillId("human resources")).toBe("human_capital_management");
    expect(resolveSkillId("GAAP")).toBe("financial_accounting");
    expect(resolveSkillId("Marketing Strategy")).toBe("marketing_strategy");
    // Existing ids keep resolving exactly as before.
    expect(resolveSkillId("logistics")).toBe("supply_chain");
    expect(resolveSkillId("audit analytics")).toBe("it_audit");
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

  it("every mapped course has an anchor; an unmapped one was seen by the pipeline and awaits review", () => {
    // v6.7.0: the catalog holds every FSB program course. A course either
    // has links including an anchor (unanchored partial links would
    // oversell), or has none and is recorded in catalog/mapping-state.json,
    // so no course is ever silently left unmapped.
    const state = JSON.parse(readFileSync(path.join(process.cwd(), "catalog", "mapping-state.json"), "utf8")) as Record<string, unknown>;
    for (const course of COURSES) {
      if (course.special || course.retired) continue;
      const links = COURSE_SKILLS.filter((l) => l.course === course.code);
      if (links.length === 0) {
        expect(state[course.code], `${course.code} has no links and was never mapped`).toBeDefined();
        continue;
      }
      // Below 300 the level cap (2026-09-24) rules out anchors, so those
      // courses need only a link with evidence.
      if (levelCap(course.code, "") === "anchor") {
        expect(links.some((l) => l.level === "anchor"), `${course.code} needs an anchor`).toBe(true);
      } else {
        expect(links.some((l) => (l.evidence?.length ?? 0) > 10), `${course.code} needs an evidenced link`).toBe(true);
      }
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
