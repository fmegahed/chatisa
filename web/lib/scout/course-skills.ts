/**
 * Course-to-skill mapping against lib/scout/taxonomy.ts.
 *
 * The approved links live in catalog/course-skills.json (v6.7.0). The
 * catalog pipeline (scripts/catalog/, run by the catalog-refresh GitHub
 * Action) proposes changes to that file as a pull request, and a person
 * approves them by merging; nothing is written to it automatically. The
 * first 212 links were authored by Claude Fable 5 in-session (2026-07-28)
 * from the bulletin descriptions and reviewed by the instructor
 * (docs/development/2026-07-28-course-skills-review.md).
 *
 * Levels (design §2.2): anchor = graded deliverables demonstrate it;
 * applied = used repeatedly as a working tool; exposure = introduced.
 * Evidence phrases are written in the student's voice fragments ("built...",
 * "designed...") because they feed grounded resume bullets downstream.
 *
 * Where the course material does not name a tool, none is claimed: a wrong
 * tool here becomes a wrong line on a resume. The deliberate exceptions
 * (Python in 242/381/419/630, R in 444/616) reflect how FSB actually
 * teaches those courses and were flagged for instructor review.
 */

import links from "@/catalog/course-skills.json";

export type CourseSkillLevel = "anchor" | "applied" | "exposure";

export interface CourseSkillLink {
  course: string;
  skillId: string;
  level: CourseSkillLevel;
  evidence?: string;
}

const LEVELS = new Set<string>(["anchor", "applied", "exposure"]);
const DEPTH: Record<CourseSkillLevel, number> = { exposure: 0, applied: 1, anchor: 2 };

/**
 * The professor's exceptions to the level rule below, by course and skill
 * (2026-09-24: "With the exception of excel in CSE 148 --> anchor").
 */
const LEVEL_CAP_EXCEPTIONS: Record<string, CourseSkillLevel> = {
  "CSE 148|excel": "anchor",
};

/**
 * The deepest level a course can give a skill (professor's rule,
 * 2026-09-24): 100-level courses introduce (exposure), 200-level courses
 * give at most applied, and anchors start at 300. Graduate courses are
 * uncapped. Keeps a first- or second-year student's profile from reading
 * Strong on introductory coursework.
 */
export function levelCap(course: string, skillId: string): CourseSkillLevel {
  const exception = LEVEL_CAP_EXCEPTIONS[`${course}|${skillId}`];
  if (exception) return exception;
  const hundreds = Number(/\d/.exec(course)?.[0] ?? 3);
  return hundreds <= 1 ? "exposure" : hundreds === 2 ? "applied" : "anchor";
}

/** `level`, lowered to the course's cap when it is above it. */
export function capLevel(course: string, skillId: string, level: CourseSkillLevel): CourseSkillLevel {
  const cap = levelCap(course, skillId);
  return DEPTH[level] > DEPTH[cap] ? cap : level;
}

/**
 * Checked once at load, so a hand edit or a bad merge fails loudly at
 * startup and in tests instead of silently skewing every student's
 * strengths. The integrity tests (tests/unit/scout-taxonomy.test.ts) check
 * that every course and skill id exists.
 */
function validated(raw: unknown): CourseSkillLink[] {
  if (!Array.isArray(raw)) throw new Error("catalog/course-skills.json must be an array");
  return raw.map((l, i) => {
    const link = l as Record<string, unknown>;
    if (typeof link.course !== "string" || typeof link.skillId !== "string" || !LEVELS.has(String(link.level))) {
      throw new Error(`catalog/course-skills.json entry ${i} is malformed`);
    }
    return {
      course: link.course,
      skillId: link.skillId,
      level: link.level as CourseSkillLevel,
      ...(typeof link.evidence === "string" ? { evidence: link.evidence } : {}),
    };
  });
}

export const COURSE_SKILLS: CourseSkillLink[] = validated(links);
