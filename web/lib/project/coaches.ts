// lib/project/coaches.ts
/**
 * The five project coaches. Each produces one structured, editable deliverable.
 * Prompts and per-coach content schemas are defined in each coach's own slice;
 * this module carries only identity and display metadata so a project can store
 * which coaches its lead enabled.
 */
export type CoachType =
  | "scoping"
  | "premortem"
  | "team_structuring"
  | "devils_advocate"
  | "reflection";

export interface CoachMeta {
  type: CoachType;
  label: string;
  blurb: string;
  order: number;
}

export const COACHES: readonly CoachMeta[] = [
  {
    type: "scoping",
    label: "Project Scoping",
    blurb:
      "Turns a vague idea into a clear brief: the problem, goals, data, analysis, ethics, stakeholders, and how you will measure success.",
    order: 1,
  },
  {
    type: "premortem",
    label: "Premortem",
    blurb:
      "Imagines the project has already failed, then works backward to name the likely failures and how to avoid them.",
    order: 2,
  },
  {
    type: "team_structuring",
    label: "Team Structuring",
    blurb:
      "Maps each teammate's skills to the tasks that suit them, so the work is shared deliberately.",
    order: 3,
  },
  {
    type: "devils_advocate",
    label: "Devil's Advocate",
    blurb:
      "Pressure tests a key decision by arguing the other side, surfacing alternatives, risks, and mitigations.",
    order: 4,
  },
  {
    type: "reflection",
    label: "Reflection",
    blurb:
      "Helps the team look back on challenges, insights, and growth once the work is under way or done.",
    order: 5,
  },
];

const BY_TYPE = new Map(COACHES.map((c) => [c.type, c]));

export function isCoachType(x: string): x is CoachType {
  return BY_TYPE.has(x as CoachType);
}

export function coachLabel(type: CoachType): string {
  return BY_TYPE.get(type)?.label ?? type;
}
