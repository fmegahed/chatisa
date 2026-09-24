/**
 * Where a showcase project came from (v6.6.0). Every origin-dependent string
 * lives here so the page header, the README, the repository name, and the
 * model prompt can never disagree about the same project.
 *
 * Older drafts, autosaves and stored sites carry no origin; they were all
 * built for a Miami course, so a missing or unknown value means "miami".
 */

import { getCourse } from "@/lib/scout/courses";

export type ProjectOrigin = "miami" | "other" | "self" | "personal";

export const ORIGINS: ProjectOrigin[] = ["miami", "other", "self", "personal"];

export function originOf(value: unknown): ProjectOrigin {
  return typeof value === "string" && (ORIGINS as string[]).includes(value)
    ? (value as ProjectOrigin)
    : "miami";
}

/**
 * The page-header label. A Miami course reads "ISA 444 - Business
 * Forecasting", the convention the career Coursework adopted in 6.4.5.
 * Self-study and personal projects ignore any course text left over from
 * an origin the student switched away from.
 */
export function originLabel(origin: ProjectOrigin, course: string): string {
  const typed = course.trim();
  switch (origin) {
    case "self": return "Self-study project";
    case "personal": return "Personal project";
    case "other": return typed;
    case "miami": {
      if (!typed) return "";
      const title = getCourse(typed)?.title;
      return title ? `${typed} - ${title}` : typed;
    }
  }
}

/** The README's opening line when the model returns no README. */
export function originReadmeLine(origin: ProjectOrigin, course: string): string {
  const typed = course.trim();
  switch (origin) {
    case "self": return "A self-study project.";
    case "personal": return "A personal project.";
    default: return typed ? `Built for ${typed}.` : "";
  }
}

/** The first line of the showcase prompt. */
export function originPromptLine(origin: ProjectOrigin, course: string): string {
  const typed = course.trim();
  switch (origin) {
    case "self": return "A self-study project.";
    case "personal": return "A personal project.";
    case "other": return `Course (another school): ${typed}`;
    case "miami": return `Course: ${typed}`;
  }
}

/**
 * Guest-pass identities are guest-<n>@guest.chatisa. The constant lives in
 * lib/auth/guest.ts (GUEST_EMAIL_DOMAIN), which imports node:crypto and so
 * cannot reach browser code; a unit test keeps the two in step.
 */
const GUEST_DOMAIN = "guest.chatisa";

export function isGuestEmail(email: string | null | undefined): boolean {
  return typeof email === "string" && email.toLowerCase().endsWith(`@${GUEST_DOMAIN}`);
}
