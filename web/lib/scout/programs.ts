/**
 * FSB programs as the Miami Bulletin lists them (v6.7.0), read from
 * catalog/programs.json, which the catalog pipeline builds. The course
 * picker shows each program's groups exactly as the bulletin arranges them;
 * it records what a student took and never audits a degree.
 */

import data from "@/catalog/programs.json";

export interface ProgramGroup {
  title: string;
  subtitle: string | null;
  instruction: string | null;
  /** One entry per requirement row; several codes mean an "or" row. */
  items: { codes: string[] }[];
  notes: string[];
}

export interface Program {
  key: string;
  name: string;
  kind: "core" | "major" | "comajor" | "minor";
  url: string;
  groups: ProgramGroup[];
}

const isStrings = (x: unknown): x is string[] => Array.isArray(x) && x.every((s) => typeof s === "string");
const KINDS = new Set(["core", "major", "comajor", "minor"]);

/** Checked once at load, like courses.json (review fix, v6.7.0). */
export function validatePrograms(raw: unknown): { bulletinYear: string; programs: Program[] } {
  const d = raw as { bulletinYear?: unknown; programs?: unknown };
  if (typeof d?.bulletinYear !== "string" || !Array.isArray(d.programs)) {
    throw new Error("catalog/programs.json must have bulletinYear and programs");
  }
  d.programs.forEach((p: Record<string, unknown>, i: number) => {
    const ok =
      typeof p.key === "string" && typeof p.name === "string" && KINDS.has(String(p.kind)) && typeof p.url === "string" &&
      Array.isArray(p.groups) &&
      p.groups.every((g: Record<string, unknown>) =>
        typeof g.title === "string" && Array.isArray(g.items) && isStrings(g.notes) &&
        g.items.every((it: { codes?: unknown }) => isStrings(it.codes) && it.codes.length > 0));
    if (!ok) throw new Error(`catalog/programs.json program ${i} is malformed`);
  });
  return d as { bulletinYear: string; programs: Program[] };
}

const catalog = validatePrograms(data);

/** "2026-27": the bulletin the groups follow, shown to students. */
export const BULLETIN_YEAR = catalog.bulletinYear;
export const PROGRAMS: Program[] = catalog.programs;

const byKey = new Map(PROGRAMS.map((p) => [p.key, p]));

export function getProgram(key: string): Program | undefined {
  return byKey.get(key);
}

/** Every course code a program lists, including every option of an "or" row. */
export function programCodes(key: string): string[] {
  return (byKey.get(key)?.groups ?? []).flatMap((g) => g.items.flatMap((i) => i.codes));
}
