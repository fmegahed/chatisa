/**
 * Course tiers: the popular-first grouping the course chips render from.
 * Lifted out of components/scout/ProfileTab.tsx (2026-08-20) so the Portfolio
 * Builder's CoursePicker and Job Scout's profile show the same tiers from one
 * definition. Pure data shaping over COURSES; no React, no browser APIs.
 */

import { COURSES, POPULAR_CODES, type CourseDef } from "@/lib/scout/courses";

export interface Tier {
  name: string;
  popular: CourseDef[];
  more: CourseDef[];
  collapsedByDefault?: boolean;
}

export function tierOf(code: string): number {
  return Number(code.replace(/\D/g, "").slice(0, 1));
}

export function buildTiers(): Tier[] {
  const byCode = new Map(COURSES.map((c) => [c.code, c]));
  const pick = (codes: string[]) =>
    codes.flatMap((c) => (byCode.has(c) ? [byCode.get(c)!] : []));
  const popularSet = new Set(Object.values(POPULAR_CODES).flat());
  const rest = (min: number, max: number) =>
    COURSES.filter((c) => {
      const n = tierOf(c.code);
      return n >= min && n <= max && !popularSet.has(c.code);
    });
  return [
    {
      name: "Foundations (100 and 200 level)",
      popular: pick(POPULAR_CODES.foundations),
      more: rest(1, 2),
    },
    {
      name: "Core (300 level)",
      popular: pick(POPULAR_CODES.core300),
      more: rest(3, 3),
    },
    {
      name: "Advanced (400 level)",
      popular: pick(POPULAR_CODES.advanced400),
      more: rest(4, 5),
    },
    {
      name: "Graduate (600 level)",
      popular: [],
      more: rest(6, 6),
      collapsedByDefault: true,
    },
  ];
}
