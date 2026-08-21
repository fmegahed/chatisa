import { z } from "zod";

/**
 * Portfolio Builder content (2026-08-20). The model emits THIS and nothing
 * else; lib/portfolio/html.ts renders it deterministically. Two shapes:
 * the career portfolio (v2, migrated from Job Scout's v6.3.0 PortfolioContent)
 * and the project showcase (v1).
 */

/** Repo-safe slug: lowercase, digits, hyphens, 3 to 60 chars. */
export const SLUG = /^[a-z0-9][a-z0-9-]{2,59}$/;
/** Repo-relative path without traversal; spaces are already hyphenated. */
export const SAFE_PATH = /^[\w.-]+(\/[\w.-]+)*$/;

export const careerContentSchema = z.object({
  v: z.literal(2),
  siteTitle: z.string().min(1).max(80),
  headline: z.string().min(1).max(140),
  about: z.string().min(1).max(1500),
  skillGroups: z
    .array(z.object({ title: z.string().min(1).max(60), skills: z.array(z.string().min(1).max(60)).max(12) }))
    .max(6),
  projects: z
    .array(
      z.object({
        slug: z.string().regex(SLUG),
        title: z.string().min(1).max(80),
        blurb: z.string().min(1).max(600),
        skills: z.array(z.string().min(1).max(60)).max(8),
        externalUrl: z.string().max(300).nullable(),
      }),
    )
    .max(5),
  courses: z.array(z.object({ code: z.string().min(1).max(20), why: z.string().min(1).max(240) })).max(8),
  experience: z
    .array(
      z.object({
        org: z.string().min(1).max(100),
        role: z.string().min(1).max(100),
        dates: z.string().max(60),
        bullets: z.array(z.string().min(1).max(300)).max(5),
      }),
    )
    .max(6),
  education: z
    .array(z.object({ school: z.string().min(1).max(100), degree: z.string().max(120), dates: z.string().max(60) }))
    .max(3),
});
export type CareerContent = z.infer<typeof careerContentSchema>;

export const showcaseContentSchema = z.object({
  v: z.literal(1),
  title: z.string().min(1).max(100),
  tagline: z.string().min(1).max(160),
  problem: z.string().min(1).max(1500),
  data: z.string().min(1).max(1500),
  approach: z.string().min(1).max(2000),
  findings: z
    .array(
      z.object({
        heading: z.string().min(1).max(100),
        body: z.string().min(1).max(1200),
        /** A figures/<name> path from the uploaded set, or null. */
        figure: z.string().max(200).nullable(),
      }),
    )
    .max(6),
  deliverables: z.array(z.object({ label: z.string().min(1).max(80), path: z.string().max(200).regex(SAFE_PATH) })).max(12),
  skills: z.array(z.string().min(1).max(60)).max(10),
  nextSteps: z.string().max(1000),
});
export type ShowcaseContent = z.infer<typeof showcaseContentSchema>;

export type SiteContent =
  | { kind: "career"; content: CareerContent }
  | { kind: "showcase"; content: ShowcaseContent };

export function emptyCareer(): CareerContent {
  return {
    v: 2, siteTitle: "", headline: "", about: "", skillGroups: [], projects: [],
    courses: [], experience: [], education: [],
  };
}

const v1Schema = z.object({
  siteTitle: z.string(),
  headline: z.string(),
  about: z.string(),
  skillGroups: z.array(z.object({ title: z.string(), skills: z.array(z.string()) })),
  projectCards: z.array(
    z.object({ repoName: z.string(), title: z.string(), blurb: z.string(), skillLabels: z.array(z.string()), repoUrl: z.string() }),
  ),
  courseHighlights: z.array(z.object({ course: z.string(), why: z.string() })),
});

/** v6.3.0 PortfolioContent -> v2. New sections start empty; the student
 * fills them in the editor or regenerates. */
export function migrateCareerV1(old: unknown): CareerContent | null {
  const parsed = v1Schema.safeParse(old);
  if (!parsed.success) return null;
  const v1 = parsed.data;
  const migrated = {
    v: 2 as const,
    siteTitle: v1.siteTitle,
    headline: v1.headline,
    about: v1.about,
    skillGroups: v1.skillGroups,
    projects: v1.projectCards.map((c) => ({
      slug: c.repoName.toLowerCase().replace(/[^a-z0-9-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60) || "project",
      title: c.title,
      blurb: c.blurb,
      skills: c.skillLabels,
      externalUrl: c.repoUrl || null,
    })),
    courses: v1.courseHighlights.map((h) => ({ code: h.course, why: h.why })),
    experience: [],
    education: [],
  };
  const checked = careerContentSchema.safeParse(migrated);
  return checked.success ? checked.data : null;
}
