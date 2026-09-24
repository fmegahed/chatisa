/**
 * The fixed rules applied to a model's skill proposals for one repository
 * (v6.8.0, professor-approved). They set the SUGGESTED level; the student
 * may still choose any level (spec decision 3). Lowering, never rejecting,
 * except for unknown skills and unproven tools, which are dropped.
 */

import type { CourseSkillLevel } from "./course-skills";
import { getSkill, mentionsSkill, resolveSkillId } from "./taxonomy";
import { isSubstantial, type RepoSummary } from "./github-summary";

export interface Proposal { skillId: string; level: CourseSkillLevel; evidence: string }
export interface RepoSuggestion { skillId: string; suggested: CourseSkillLevel; evidence: string }

const MAX_SKILLS = 8;
const MAX_ANCHORS = 3;
const EXT_LANGUAGE: Record<string, string> = { py: "Python", ipynb: "Python", r: "R", rmd: "R", qmd: "R", sql: "SQL", js: "JavaScript", ts: "JavaScript" };

/** Everything that proves a tool: dependency files, import lines, languages, file types. */
function proofText(s: RepoSummary): string {
  const imports = s.codeFiles
    .flatMap((f) => f.text.split("\n"))
    .filter((l) => /^\s*(import |from \S+ import|library\(|require\(|using |#include|SELECT |CREATE )/i.test(l));
  const exts = s.tree.map((t) => EXT_LANGUAGE[(t.path.split(".").pop() ?? "").toLowerCase()]).filter(Boolean);
  return [
    ...s.dependencyFiles.map((f) => f.text),
    ...imports,
    ...Object.keys(s.languages),
    ...exts,
  ].join("\n");
}

export function guardRepoSkills(
  summary: RepoSummary,
  proposals: Proposal[],
): { suggestions: RepoSuggestion[]; substantial: boolean; codeRead: boolean } {
  const substantial = isSubstantial(summary);
  // Dependency files prove tools, but they are not code or data: they can
  // neither make a repository count as "code read" nor carry an anchor
  // (review fix: a lone package.json let a repository reach anchor).
  const codeRead = summary.codeFiles.length > 0;
  const proof = proofText(summary);
  const readPaths = summary.codeFiles.map((f) => f.path);
  const seen = new Set<string>();
  const out: RepoSuggestion[] = [];
  let anchors = 0;
  for (const p of proposals) {
    if (out.length >= MAX_SKILLS) break;
    const id = resolveSkillId(p.skillId);
    if (!id || seen.has(id)) continue;
    const skill = getSkill(id);
    if (!skill) continue;
    if (id === "version_control") {
      if (summary.authorship.studentCommits < 10) continue;
    } else if (skill.kind === "tool" && !mentionsSkill(id, proof)) {
      continue;
    }
    const evidence = p.evidence.trim().slice(0, 200);
    let level: CourseSkillLevel = p.level;
    if (!codeRead) level = "exposure";
    if (level === "anchor") {
      const citesCode = readPaths.some((path) => evidence.includes(path) || evidence.includes(path.split("/").pop() ?? path));
      if (!substantial || !citesCode || anchors >= MAX_ANCHORS) level = "applied";
      else anchors++;
    }
    seen.add(id);
    out.push({ skillId: id, suggested: level, evidence });
  }
  return { suggestions: out, substantial, codeRead };
}
