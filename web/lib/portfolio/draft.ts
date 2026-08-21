/**
 * The Portfolio Builder wizard's draft: the one piece of state every step
 * reads and patches. It lives in lib rather than beside the component so the
 * step components can import the types without importing the shell that
 * renders them (and so lib never has to reach into components).
 */

import type { PreparedFile } from "./files";
import type { SiteContent } from "./content";

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

export interface Draft {
  siteId: string;
  mode: "career" | "showcase" | null;
  step: Step;
  // career
  resume: File | null;
  resumeLink: boolean;
  courses: string[];
  projects: CareerProject[];
  photo: { base64: string; bytes: number } | null;
  name: string;
  links: { label: string; url: string }[];
  // showcase
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
    resume: null, resumeLink: false, courses: [], projects: [],
    photo: null, name, links: [],
    course: "", semester: "", team: [], files: [],
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
