/**
 * Published work handed to JobApp Drafter (2026-08-20). The Portfolio
 * Builder keeps every published site in the student's own browser, so the
 * only way the drafter can cite one is if the student opts in and the
 * browser sends it along with the application. Nothing new is stored: the
 * block is folded into the resume text the applications route already keeps.
 */

import { z } from "zod";

const schema = z.array(z.object({
  title: z.string().min(1).max(120),
  summary: z.string().max(300),
  url: z.string().max(300).refine((u) => /^https?:\/\//i.test(u)),
  skills: z.array(z.string().max(60)).max(10),
})).max(6);

export type PublishedWorkItem = z.infer<typeof schema>[number];

/** Tolerant: a bad item is dropped, a bad payload is an empty list. */
export function parsePublishedWork(raw: string | null | undefined): PublishedWorkItem[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed.flatMap((item) => {
      const one = schema.element.safeParse(item);
      return one.success ? [one.data] : [];
    }).slice(0, 6);
  } catch {
    return [];
  }
}

export function publishedWorkBlock(items: PublishedWorkItem[]): string {
  if (items.length === 0) return "";
  return [
    "Published work (live links the candidate can share):",
    ...items.map((i) => `- ${i.title}: ${i.summary} (${i.url})${i.skills.length ? ` Skills: ${i.skills.join(", ")}` : ""}`),
  ].join("\n");
}
