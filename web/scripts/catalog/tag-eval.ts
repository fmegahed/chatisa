/**
 * Before/after check for the job tagger's v2 vocabulary (v6.7.0). Re-tags
 * the same stored postings with v1 (no business-domain skills) and v2, and
 * writes a report the professor spot-checks before the tagger change ships:
 *
 * - how much the existing analytics/IS tags moved (should be near zero);
 * - every new business-domain tag, with the posting text that triggered it,
 *   flagged when no wording in the posting supports it (the likely
 *   "tagged the employer's industry" mistake).
 *
 * Usage: npx tsx --conditions=react-server scripts/catalog/tag-eval.ts <scout.db> [sample=60]
 * Costs about $0.01 per posting (two tagging calls on Gemini 3.8 Flash).
 */

import { config as loadEnv } from "dotenv";
loadEnv({ path: ".env.local", quiet: true });
loadEnv({ path: ".env", quiet: true });

import { writeFileSync } from "node:fs";
import path from "node:path";
import Database from "better-sqlite3";
import { tagPosting, type TagResult } from "../../lib/scout/tag";
import { SKILLS, getSkill, mentionsSkill } from "../../lib/scout/taxonomy";
import type { RawPosting } from "../../lib/scout/sources/types";

const business = new Set(SKILLS.filter((s) => s.category === "business").map((s) => s.id));


async function main() {
  const [dbPath, sampleArg] = process.argv.slice(2);
  if (!dbPath) throw new Error("Usage: tag-eval.ts <scout.db> [sample]");
  const want = Number(sampleArg ?? 60);
  const db = new Database(dbPath, { readonly: true });
  const all = db.prepare("select id, title, company, description from scout_postings order by id").all() as
    { id: number; title: string; company: string; description: string }[];
  const step = Math.max(1, Math.floor(all.length / want));
  const sample = all.filter((_, i) => i % step === 0).slice(0, want);

  const rows: { p: (typeof sample)[number]; v1: TagResult; v1b: TagResult; v2: TagResult }[] = [];
  const failures = { v1: 0, v2: 0 };
  const failureNotes: string[] = [];
  let cost = 0;
  let next = 0;
  // A failed call is counted per vocabulary, not fatal: whether v2's longer
  // instructions fail more often is itself something this check must show.
  const tag = async (raw: RawPosting, vocab: "v1" | "v2") => {
    try {
      return await tagPosting(raw, vocab);
    } catch (err) {
      failures[vocab]++;
      failureNotes.push(`${vocab}: ${(err as Error).message.slice(0, 120)}`);
      return null;
    }
  };
  const worker = async () => {
    while (next < sample.length) {
      const p = sample[next++];
      const raw = { title: p.title, company: p.company, description: p.description } as RawPosting;
      // v1 twice gives the baseline: how much tags move with no change at all.
      const [v1, v1b, v2] = await Promise.all([tag(raw, "v1"), tag(raw, "v1"), tag(raw, "v2")]);
      cost += (v1?.costUsd ?? 0) + (v1b?.costUsd ?? 0) + (v2?.costUsd ?? 0);
      if (v1 && v1b && v2) rows.push({ p, v1, v1b, v2 });
    }
  };
  await Promise.all(Array.from({ length: 6 }, worker));
  rows.sort((a, b) => a.p.id - b.p.id);

  let kept = 0, dropped = 0, added = 0, baseKept = 0, baseDropped = 0;
  const domainTags: string[] = [];
  let unsupported = 0;
  for (const { p, v1, v1b, v2 } of rows) {
    const a = new Set(v1.skills.map((s) => s.skillId));
    const again = new Set(v1b.skills.map((s) => s.skillId));
    for (const id of a) {
      if (again.has(id)) baseKept++;
      else baseDropped++;
    }
    const b = new Set(v2.skills.filter((s) => !business.has(s.skillId)).map((s) => s.skillId));
    for (const id of a) {
      if (b.has(id)) kept++;
      else dropped++;
    }
    for (const id of b) if (!a.has(id)) added++;
    for (const s of v2.skills.filter((x) => business.has(x.skillId))) {
      const t = mentionsSkill(s.skillId, `${p.title}
${p.description}`);
      if (!t) unsupported++;
      domainTags.push(`| ${p.title.slice(0, 60)} | ${getSkill(s.skillId)!.label} | ${s.importance} | ${t ?? "**no supporting wording found**"} |`);
    }
  }
  const stable = kept / Math.max(1, kept + dropped);
  const baseline = baseKept / Math.max(1, baseKept + baseDropped);
  const md = [
    "# Job tagger v2: before/after check",
    "",
    `Sample: ${sample.length} stored postings from ${path.basename(dbPath)} (every ${step} of ${all.length}); ${rows.length} tagged by both. Cost: $${cost.toFixed(2)}.`,
    "",
    `Tagging failures: v1 ${failures.v1}, v2 ${failures.v2}.${failureNotes.length ? ` First: ${failureNotes[0]}` : ""}`,
    "",
    "## Existing analytics and IS tags",
    "",
    `- v1 vs v2: kept ${kept} (${(stable * 100).toFixed(0)}%), dropped ${dropped}, newly added ${added}.`,
    `- Baseline, v1 vs v1 again (no change at all): ${(baseline * 100).toFixed(0)}% kept. v2 is fine when it is close to this.`,
    "",
    `## New business-domain tags (${domainTags.length}; ${unsupported} with no supporting wording)`,
    "",
    "Spot-check: a tag is right when the role asks for that knowledge; wrong when it only reflects the employer's industry.",
    "",
    "| Posting | Skill | Importance | Text that triggered it |",
    "|---|---|---|---|",
    ...domainTags,
    "",
  ].join("\n");
  const out = path.join(process.cwd(), "..", "docs", "development", "2026-09-24-tag-eval.md");
  writeFileSync(out, md);
  console.log(`failures v1 ${failures.v1} v2 ${failures.v2}; tag eval: ${rows.length} postings, existing tags kept ${(stable * 100).toFixed(0)}% (baseline ${(baseline * 100).toFixed(0)}%), ${domainTags.length} domain tags (${unsupported} unsupported), $${cost.toFixed(2)} -> ${out}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
