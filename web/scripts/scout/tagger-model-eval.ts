/**
 * Tagger model comparison (v6.10.1): does a candidate model tag job postings
 * as well as the current one? Written when the weekly tagger moved from
 * Gemini 3.8 Flash to Gemini 4 Hydrogen.
 *
 * Each posting is tagged three times with the production instructions and
 * schema: by the BASELINE model twice (how much tags move with no change at
 * all) and by the CANDIDATE once. The candidate is acceptable when its
 * agreement with the baseline is close to the baseline's agreement with
 * itself, it fails no more often, and it never claims visa sponsorship the
 * posting does not state (a wrong "sponsors" wastes an international
 * student's application).
 *
 * Postings come fresh from USAJobs (free and keyed; Active Jobs quota is
 * shared with the production harvest, so it is not used here).
 *
 * Usage: npx tsx --conditions=react-server scripts/scout/tagger-model-eval.ts [sample=60]
 *        TAGGER_EVAL_MODELS=baseline,candidate overrides the pair.
 * Writes docs/development/<date>-tagger-model-eval.md. Costs about $1 for 60.
 */
import { config as loadEnv } from "dotenv";
loadEnv({ path: ".env.local", quiet: true });
loadEnv({ path: ".env", quiet: true });
import { writeFileSync } from "node:fs";
import path from "node:path";
import { generateObject } from "ai";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { z } from "zod";
import { buildTagInstructions, keepSupportedDomainTags, vocabularyIds } from "../../lib/scout/tag";
import { resolveSkillId } from "../../lib/scout/taxonomy";
import { searchUsajobs } from "../../lib/scout/sources/usajobs";
import { USAJOBS_KEYWORDS } from "../../lib/scout/queries";
import type { RawPosting } from "../../lib/scout/sources/types";

// Per million tokens, from Google's pricing page on 2026-11-16. 3.8 Flash is
// at its promotional rate (through 2026-12-31). Not read from the catalog,
// because the baseline may already have left it.
const PRICES: Record<string, [number, number]> = {
  "gemini-3.8-flash": [0.75, 3.75],
  "gemini-4-hydrogen": [1.5, 7.5],
};
const [BASELINE, CANDIDATE] = (process.env.TAGGER_EVAL_MODELS ?? "gemini-3.8-flash,gemini-4-hydrogen").split(",");
// Optional Gemini thinking level for the candidate only ("minimal", "low",
// ...). Thinking tokens are billed as output, so this moves cost directly.
const CANDIDATE_THINKING = process.env.TAGGER_EVAL_CANDIDATE_THINKING;

// Must match tagSchema in lib/scout/tag.ts (not exported there, and this
// script must not change what production sends).
const tagSchema = z.object({
  skills: z.array(z.object({ skillId: z.string().min(1).max(60), importance: z.enum(["required", "preferred"]) })).max(20),
  category: z.enum(["fulltime", "internship", "federal"]),
  seniorityOk: z.boolean(),
  visaSponsorship: z.enum(["sponsors", "no_sponsorship", "unknown"]),
});
type Tagged = { skills: Set<string>; category: string; seniorityOk: boolean; visa: string; usd: number; ms: number; thinking: number };

function fence(label: string, body: string, nonce: string): string {
  const cleaned = body.replaceAll(`</${label}`, `<\\/${label}`);
  return `<${label} nonce="${nonce}">\n${cleaned}\n</${label} nonce="${nonce}">`;
}

const google = createGoogleGenerativeAI({ apiKey: process.env.GOOGLE_API_KEY });
const allowed = new Set(vocabularyIds("v2"));

async function tag(modelId: string, p: RawPosting): Promise<Tagged> {
  const started = Date.now();
  const nonce = Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
  const { object, usage } = await generateObject({
    model: google(modelId),
    schema: tagSchema,
    instructions: buildTagInstructions("v2"),
    prompt: [`Title: ${p.title}`, `Company: ${p.company}`, fence("posting", p.description.slice(0, 12_000), nonce)].join("\n\n"),
    maxOutputTokens: 3_000,
    abortSignal: AbortSignal.timeout(120_000),
    ...(modelId === CANDIDATE && CANDIDATE_THINKING
      ? { providerOptions: { google: { thinkingConfig: { thinkingLevel: CANDIDATE_THINKING } } } }
      : {}),
  });
  const ids = new Set<string>();
  for (const s of object.skills) {
    const id = resolveSkillId(s.skillId);
    if (id && allowed.has(id)) ids.add(id);
  }
  const kept = keepSupportedDomainTags([...ids].map((skillId) => ({ skillId })), `${p.title}\n${p.description}`);
  const [pin, pout] = PRICES[modelId] ?? [0, 0];
  return {
    skills: new Set(kept.map((k) => k.skillId)),
    category: object.category,
    seniorityOk: object.seniorityOk,
    visa: object.visaSponsorship,
    usd: ((usage?.inputTokens ?? 0) * pin + (usage?.outputTokens ?? 0) * pout) / 1e6,
    ms: Date.now() - started,
    thinking: usage?.outputTokenDetails?.reasoningTokens ?? 0,
  };
}

const jaccard = (a: Set<string>, b: Set<string>) => {
  const union = new Set([...a, ...b]);
  if (union.size === 0) return 1;
  return [...a].filter((x) => b.has(x)).length / union.size;
};
const mean = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : 0);
const pct = (x: number) => `${Math.round(x * 100)}%`;

async function main() {
  const want = Number(process.argv[2] ?? 60);
  const seen = new Set<string>();
  const pool: RawPosting[] = [];
  for (const kw of USAJOBS_KEYWORDS) {
    const res = await searchUsajobs(kw);
    if (res.error) console.log(`  USAJobs "${kw}": ${res.error}`);
    for (const p of res.postings) {
      const key = `${p.title}|${p.company}`;
      if (seen.has(key) || p.description.length < 400) continue;
      seen.add(key);
      pool.push(p);
    }
  }
  pool.sort((a, b) => `${a.title}|${a.company}`.localeCompare(`${b.title}|${b.company}`));
  const step = Math.max(1, Math.floor(pool.length / want));
  const sample = pool.filter((_, i) => i % step === 0).slice(0, want);
  console.log(`Pool ${pool.length} postings; sample ${sample.length}. Baseline ${BASELINE}, candidate ${CANDIDATE}.`);

  type Row = { p: RawPosting; a1: Tagged | null; a2: Tagged | null; b: Tagged | null };
  const rows: Row[] = [];
  const failures: Record<string, string[]> = { [BASELINE]: [], [CANDIDATE]: [] };
  const safe = async (m: string, p: RawPosting) => {
    try {
      return await tag(m, p);
    } catch (err) {
      failures[m].push((err as Error).message.slice(0, 100));
      return null;
    }
  };
  let next = 0;
  const worker = async () => {
    while (next < sample.length) {
      const p = sample[next++];
      const [a1, a2, b] = await Promise.all([safe(BASELINE, p), safe(BASELINE, p), safe(CANDIDATE, p)]);
      rows.push({ p, a1, a2, b });
      process.stdout.write(".");
    }
  };
  await Promise.all(Array.from({ length: 4 }, worker));
  console.log("");

  const both = rows.filter((r) => r.a1 && r.a2 && r.b) as { p: RawPosting; a1: Tagged; a2: Tagged; b: Tagged }[];
  const base = both.map((r) => jaccard(r.a1.skills, r.a2.skills));
  const cand = both.map((r) => jaccard(r.a1.skills, r.b.skills));
  const agree = (f: (t: Tagged) => unknown, x: "a2" | "b") => both.filter((r) => f(r.a1) === f(r[x])).length / (both.length || 1);
  const newSponsors = both.filter((r) => r.b.visa === "sponsors" && r.a1.visa !== "sponsors" && r.a2.visa !== "sponsors");
  const cost = (k: "a1" | "b") => mean(rows.map((r) => r[k]?.usd ?? 0).filter((x) => x > 0));
  const ms = (k: "a1" | "b") => mean(rows.map((r) => r[k]?.ms ?? 0).filter((x) => x > 0));
  const skillsPer = (k: "a1" | "b") => mean(both.map((r) => r[k].skills.size));
  const thinkingPer = (k: "a1" | "b") => mean(both.map((r) => r[k].thinking));
  const label = CANDIDATE_THINKING ? `${CANDIDATE} (thinking ${CANDIDATE_THINKING})` : CANDIDATE;
  const total = rows.reduce((s, r) => s + (r.a1?.usd ?? 0) + (r.a2?.usd ?? 0) + (r.b?.usd ?? 0), 0);

  const lines = [
    `# Job tagger model check: ${BASELINE} vs ${label}`,
    "",
    `Sample: ${sample.length} fresh USAJobs postings (every ${step} of ${pool.length}, sorted by title), ${both.length} tagged by all three calls. Instructions and schema as in production (v2 vocabulary). Run cost: $${total.toFixed(2)}.`,
    "",
    "| Measure | Baseline vs itself | Baseline vs candidate |",
    "|---|---|---|",
    `| Skill sets, mean Jaccard | ${base.length ? mean(base).toFixed(2) : "-"} | ${cand.length ? mean(cand).toFixed(2) : "-"} |`,
    `| Category agrees | ${pct(agree((t) => t.category, "a2"))} | ${pct(agree((t) => t.category, "b"))} |`,
    `| Seniority flag agrees | ${pct(agree((t) => t.seniorityOk, "a2"))} | ${pct(agree((t) => t.seniorityOk, "b"))} |`,
    `| Visa sponsorship agrees | ${pct(agree((t) => t.visa, "a2"))} | ${pct(agree((t) => t.visa, "b"))} |`,
    "",
    "| Per model | Baseline | Candidate |",
    "|---|---|---|",
    `| Tagging failures | ${failures[BASELINE].length} of ${sample.length * 2} calls | ${failures[CANDIDATE].length} of ${sample.length} calls |`,
    `| Skills per posting, mean | ${skillsPer("a1").toFixed(1)} | ${skillsPer("b").toFixed(1)} |`,
    `| Cost per posting, mean | $${cost("a1").toFixed(4)} | $${cost("b").toFixed(4)} |`,
    `| Seconds per posting, mean | ${(ms("a1") / 1000).toFixed(1)} | ${(ms("b") / 1000).toFixed(1)} |`,
    `| Thinking tokens per posting, mean | ${Math.round(thinkingPer("a1"))} | ${Math.round(thinkingPer("b"))} |`,
    "",
    `Candidate claims "sponsors" where the baseline never did: ${newSponsors.length}.`,
    ...newSponsors.map((r) => `- ${r.p.title} (${r.p.company})`),
    "",
    "## Largest skill disagreements (candidate vs baseline run 1)",
    "",
    "| Posting | Only baseline | Only candidate |",
    "|---|---|---|",
    ...both
      .map((r) => ({ r, j: jaccard(r.a1.skills, r.b.skills) }))
      .sort((x, y) => x.j - y.j)
      .slice(0, 10)
      .map(({ r }) => {
        const onlyA = [...r.a1.skills].filter((s) => !r.b.skills.has(s)).join(", ") || "-";
        const onlyB = [...r.b.skills].filter((s) => !r.a1.skills.has(s)).join(", ") || "-";
        return `| ${r.p.title.slice(0, 60)} | ${onlyA} | ${onlyB} |`;
      }),
    "",
    ...(Object.values(failures).flat().length
      ? ["## Failures", "", ...Object.entries(failures).flatMap(([m, fs]) => fs.map((f) => `- ${m}: ${f}`))]
      : []),
  ];
  const date = new Date().toISOString().slice(0, 10);
  const suffix = CANDIDATE_THINKING ? `-thinking-${CANDIDATE_THINKING}` : "";
  const out = path.join(process.cwd(), "..", "docs", "development", `${date}-tagger-model-eval${suffix}.md`);
  writeFileSync(out, lines.join("\n") + "\n");
  console.log(lines.slice(0, 22).join("\n"));
  console.log(`\nReport: ${out}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
