/**
 * The Portfolio Builder wizard's draft: the one piece of state every step
 * reads and patches. It lives in lib rather than beside the component so the
 * step components can import the types without importing the shell that
 * renders them (and so lib never has to reach into components).
 */

import type { PreparedFile } from "./files";
import type { SiteContent } from "./content";
import type { ProjectOrigin } from "./origin";
import type { ChecklistState } from "@/lib/scout/checklist";

export type Step =
  | "mode" | "resume" | "classes" | "projects" | "details"
  | "course" | "files" | "story" | "review";

export const CAREER_STEPS: Step[] = ["mode", "resume", "classes", "projects", "details", "review"];
export const SHOWCASE_STEPS: Step[] = ["mode", "course", "files", "story", "review"];

export interface CareerProject {
  slug: string;
  title: string;
  externalUrl: string;
  files: PreparedFile[];
}

/** A course from another school as the guest typed it. */
export interface OtherCourse {
  name: string;
  school: string;
}

export interface Draft {
  siteId: string;
  mode: "career" | "showcase" | null;
  step: Step;
  // career
  resume: File | null;
  resumeLink: boolean;
  courses: string[];
  /** The subset of `courses` marked Taking now (v6.7.0); absent means none. */
  inProgress?: string[];
  /**
   * The checklist behind `courses` (v6.7.0): programs, statuses and removed
   * prerequisites, so returning to the step shows what the student chose.
   * Absent on drafts from before it; the step rebuilds it from `courses`.
   */
  coursePlan?: ChecklistState;
  /**
   * Courses from other schools, typed by guests (v6.6.0). Optional because
   * autosaves and stored sites from before v6.6.0 lack it: read as [].
   */
  otherCourses?: OtherCourse[];
  projects: CareerProject[];
  photo: { base64: string; bytes: number } | null;
  name: string;
  links: { label: string; url: string }[];
  // showcase
  /** Absent before v6.6.0, which means a Miami course: read via originOf. */
  origin?: ProjectOrigin;
  /** A Miami course code, or the typed "course, school" for origin "other". */
  course: string;
  semester: string;
  team: string[];
  files: PreparedFile[];
  prompts: { problem: string; hardest: string; next: string };
  // output
  content: SiteContent | null;
  readme: string | null;
  skillIds: string[];
  html: string;
}

export type Action =
  | { type: "patch"; patch: Partial<Draft> }
  | { type: "reset"; draft: Draft };

export function initialDraft(name: string, siteId: string): Draft {
  return {
    siteId, mode: null, step: "mode",
    resume: null, resumeLink: false, courses: [], otherCourses: [], projects: [],
    photo: null, name, links: [],
    origin: "miami", course: "", semester: "", team: [], files: [],
    prompts: { problem: "", hardest: "", next: "" },
    content: null, readme: null, skillIds: [], html: "",
  };
}

/** What every step component receives from the shell. */
export interface StepProps {
  draft: Draft;
  patch: (p: Partial<Draft>) => void;
  nav: { index: number; total: number; onBack: (() => void) | null; onNext: () => void };
}
