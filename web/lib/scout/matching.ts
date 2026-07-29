/**
 * Deterministic profile aggregation and job matching. Pure and
 * isomorphic on purpose: the browser computes every match so the server
 * never sees a student's profile (local-first decision, 2026-07-28).
 *
 * Numbers are design §2.3/§2.4 exactly; unit tests pin worked examples,
 * so a weight change here must change the tests knowingly.
 */

import { getSkill, SKILLS } from "./taxonomy";
import { getCourse } from "./courses";
import { COURSE_SKILLS, type CourseSkillLevel } from "./course-skills";

/** anchor: graded deliverables; applied: working tool; exposure: introduced. */
const LEVEL_WEIGHT: Record<CourseSkillLevel, number> = {
  anchor: 1.0,
  applied: 0.6,
  exposure: 0.25,
};

/** Partial credit across one implies edge, either direction. */
const IMPLIES_CREDIT = 0.6;
/** A skill counts as "covered" for the fraction display at this strength. */
const COVERED_THRESHOLD = 0.5;

export interface ExtraSkill {
  skillId: string;
  level: CourseSkillLevel;
}

export interface StrengthOverride {
  skillId: string;
  level: "strong" | "working" | "introduced";
}

/** What each self-set word means numerically; mirrors strengthWord's bands. */
const OVERRIDE_WEIGHT: Record<StrengthOverride["level"], number> = {
  strong: 1.0,
  working: 0.6,
  introduced: 0.25,
};

/**
 * Noisy-OR aggregation: each course or confirmed extra contributes
 * independently with weight level x credits/3, so three Python courses make
 * a strong signal that never exceeds 1. Extras (resume, internship lines)
 * are student-confirmed and weigh like a 3-credit course at their level.
 */
export function profileStrengths(
  courseCodes: string[],
  extras: ExtraSkill[],
  overrides: StrengthOverride[] = [],
): Map<string, number> {
  const complement = new Map<string, number>();
  const contribute = (skillId: string, w: number) => {
    const prev = complement.get(skillId) ?? 1;
    complement.set(skillId, prev * (1 - w));
  };

  for (const code of courseCodes) {
    const course = getCourse(code);
    if (!course || course.special) continue;
    const creditScale = Math.min(course.credits, 3) / 3;
    for (const link of COURSE_SKILLS) {
      if (link.course !== course.code) continue;
      contribute(link.skillId, LEVEL_WEIGHT[link.level] * creditScale);
    }
  }
  for (const extra of extras) {
    if (!getSkill(extra.skillId)) continue;
    contribute(extra.skillId, LEVEL_WEIGHT[extra.level]);
  }

  const strengths = new Map<string, number>();
  for (const [skillId, c] of complement) strengths.set(skillId, 1 - c);
  // The student's own word beats the computation, in both directions
  // (user feedback, 2026-07-29). An override on a skill nothing else
  // contributes still creates it.
  for (const o of overrides) {
    if (getSkill(o.skillId)) strengths.set(o.skillId, OVERRIDE_WEIGHT[o.level]);
  }
  return strengths;
}

/**
 * The student-facing word for a strength value. Words, never meters
 * (design system: no colour-alone, no progress bars).
 */
export function strengthWord(
  strength: number,
): "Strong" | "Working" | "Introduced" {
  return strength >= 0.8 ? "Strong" : strength >= 0.45 ? "Working" : "Introduced";
}

export type JobImportance = "required" | "preferred";

export interface JobSkill {
  skillId: string;
  importance: JobImportance;
}

export type MatchBand = "strong" | "good" | "stretch";

export interface SkillMatch {
  skillId: string;
  importance: JobImportance;
  /** Direct or best implies-credited strength, 0..1. */
  strength: number;
  covered: boolean;
  /** Set when coverage came through an implies edge, for honest display. */
  via?: string;
}

export interface JobMatch {
  score: number;
  band: MatchBand;
  coveredRequired: number;
  totalRequired: number;
  matched: SkillMatch[];
  /** Uncovered skills, required first then preferred, for the gap list. */
  gaps: SkillMatch[];
}

/** Neighbors one implies edge away, in either direction. */
function impliesNeighbors(skillId: string): string[] {
  const def = getSkill(skillId);
  const out = def ? [...def.implies] : [];
  for (const s of SKILLS) {
    if (s.implies.includes(skillId)) out.push(s.id);
  }
  return out;
}

/**
 * Requirement coverage, not Jaccard: how much of what the job asks for does
 * the student cover, weighted required 1.0 / preferred 0.5. Asymmetric on
 * purpose; extra student skills never penalize.
 *
 * Professional skills (kind "professional": teamwork, communication,
 * problem solving...) are demoted to preferred and never shown as gaps
 * (user decision, 2026-07-29, from a live card whose "gaps" were Teamwork
 * and Problem Solving): every posting lists them, no course checklist can
 * evidence them differently, and no portfolio project closes a "teamwork
 * gap". Covered ones still show in "You bring". Trade-off accepted: even
 * consulting-heavy roles score them at preferred weight.
 */
export function scoreJob(
  strengths: Map<string, number>,
  jobSkills: JobSkill[],
): JobMatch {
  const adjusted: JobSkill[] = jobSkills.map((js) =>
    getSkill(js.skillId)?.kind === "professional"
      ? { ...js, importance: "preferred" }
      : js,
  );
  const matches: SkillMatch[] = adjusted.map((js) => {
    let strength = strengths.get(js.skillId) ?? 0;
    let via: string | undefined;
    for (const neighbor of impliesNeighbors(js.skillId)) {
      const credited = (strengths.get(neighbor) ?? 0) * IMPLIES_CREDIT;
      if (credited > strength) {
        strength = credited;
        via = neighbor;
      }
    }
    return {
      skillId: js.skillId,
      importance: js.importance,
      strength,
      covered: strength >= COVERED_THRESHOLD,
      ...(via ? { via } : {}),
    };
  });

  let weightSum = 0;
  let weighted = 0;
  for (const m of matches) {
    const w = m.importance === "required" ? 1.0 : 0.5;
    weightSum += w;
    weighted += w * m.strength;
  }
  const score = weightSum === 0 ? 0 : weighted / weightSum;

  const required = matches.filter((m) => m.importance === "required");
  const band: MatchBand =
    score >= 0.7 ? "strong" : score >= 0.45 ? "good" : "stretch";

  const importanceOrder = (m: SkillMatch) => (m.importance === "required" ? 0 : 1);
  return {
    score,
    band,
    coveredRequired: required.filter((m) => m.covered).length,
    totalRequired: required.length,
    matched: matches.filter((m) => m.covered),
    gaps: matches
      .filter(
        (m) => !m.covered && getSkill(m.skillId)?.kind !== "professional",
      )
      .sort((a, b) => importanceOrder(a) - importanceOrder(b)),
  };
}
