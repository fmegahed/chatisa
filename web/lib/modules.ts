/**
 * The ChatISA modules. Single source for navigation, the home grid, and
 * per-module pages. Slugs are the kebab-case of the display name, so the URL a
 * student sees always matches the module they clicked; a display rename means a
 * slug rename plus a redirect from the old path (see next.config.ts). Modules
 * are grouped for the home page: coursework, jobs, general.
 */
export type ModuleGroup = "coursework" | "jobs" | "general";

export interface ModuleInfo {
  slug: string;
  name: string;
  /** Plain-language description of what the student can do. */
  description: string;
  group: ModuleGroup;
}

export const MODULES: ModuleInfo[] = [
  {
    slug: "coding-tutor",
    name: "Coding Tutor",
    description:
      "Ask for programming help in R and Python, explained at your level.",
    group: "coursework",
  },
  {
    slug: "coding-studio",
    name: "Coding Studio",
    description:
      "A workbench to write and run Python, R, and SQL on your data, with AI help.",
    group: "coursework",
  },
  {
    slug: "exam-prep",
    name: "Exam Prep",
    description:
      "Turn your course notes into a practice exam with graded feedback.",
    group: "coursework",
  },
  {
    slug: "project-assistant",
    name: "Project Assistant",
    description: "Plan and pressure-test team projects with coaching roles.",
    group: "coursework",
  },
  {
    slug: "job-scout",
    name: "Job Scout",
    description:
      "Browse this week's analytics and IS jobs, matched to the courses you have taken.",
    group: "jobs",
  },
  {
    slug: "jobapp-drafter",
    name: "JobApp Drafter",
    description:
      "Tailor your resume and cover letter to a job, to Farmer School standards.",
    group: "jobs",
  },
  {
    slug: "interview-mentor",
    name: "Interview Mentor",
    description:
      "Practice a spoken interview tailored to your resume and target job.",
    group: "jobs",
  },
  {
    slug: "ask-anything",
    name: "Ask Anything",
    description:
      "Chat with the frontier model of your choice. Your chats stay on your device.",
    group: "general",
  },
  {
    slug: "ai-comparison",
    name: "AI Comparison",
    description: "Ask several AI models at once and compare their answers.",
    group: "general",
  },
];

/** The groups in display order, with the heading each gets on the home page. */
export const MODULE_GROUPS: { group: ModuleGroup; heading: string }[] = [
  { group: "coursework", heading: "For your coursework" },
  { group: "jobs", heading: "For your job search" },
  { group: "general", heading: "General AI" },
];

export function getModule(slug: string): ModuleInfo | undefined {
  return MODULES.find((m) => m.slug === slug);
}
