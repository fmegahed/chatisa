/**
 * Luna vs Sol on real repositories (v6.8.0 model decision). Local only:
 *   npx tsx --conditions=react-server scripts/scout/repo-skills-eval.ts owner/repo [owner/repo ...]
 * Reads with the local `gh auth token`; writes a markdown report to stdout.
 */
import { execSync } from "node:child_process";
import { config as loadEnv } from "dotenv";
import { readRepo } from "../../lib/scout/github-read";
import { suggestRepoSkills } from "../../lib/scout/repo-skills";
import { getLanguageModel } from "../../lib/providers";
import { calculateCost } from "../../lib/config/models";
import { getSkill } from "../../lib/scout/taxonomy";

loadEnv({ path: ".env.local", quiet: true });
// Candidate first, reference second; override with EVAL_MODELS=a,b.
const MODELS = (process.env.EVAL_MODELS ?? "gpt-6-luna,gpt-6-sol").split(",") as [string, string];
const [CANDIDATE, REFERENCE] = MODELS;

async function main() {
  const repos = process.argv.slice(2);
  const token = execSync("gh auth token", { encoding: "utf8" }).trim();
  const login = execSync("gh api user --jq .login", { encoding: "utf8" }).trim();
  const conn = { v: 1 as const, token, login, connectedAt: "" };
  const lines: string[] = ["| Repository | Model | Anchors | Applied | Exposure | Failed | Cost |", "|---|---|---|---|---|---|---|"];
  const sets: Record<string, Record<string, Set<string>>> = {};
  const fails: Record<string, number> = { [CANDIDATE]: 0, [REFERENCE]: 0 };
  for (const fullName of repos) {
    // Older repositories use "master"; ask GitHub rather than assume.
    const defaultBranch = execSync(`gh api repos/${fullName} --jq .default_branch`, { encoding: "utf8" }).trim();
    const read = await readRepo(conn, { fullName, description: null, language: null, pushedAt: "", defaultBranch, htmlUrl: "" });
    if (!read.ok) { lines.push(`| ${fullName} | (not readable: ${read.error.kind}) | | | | | |`); continue; }
    sets[fullName] = {};
    for (const id of MODELS) {
      try {
        const out = await suggestRepoSkills(getLanguageModel(id), read.summary);
        const by = (l: string) => out.suggestions.filter((s) => s.suggested === l).map((s) => getSkill(s.skillId)?.label ?? s.skillId).join(", ");
        const cost = calculateCost(id, out.usage.inputTokens ?? 0, out.usage.outputTokens ?? 0) as { totalCost?: number };
        sets[fullName][id] = new Set(out.suggestions.filter((s) => s.suggested !== "exposure").map((s) => s.skillId));
        lines.push(`| ${fullName} | ${id} | ${by("anchor")} | ${by("applied")} | ${by("exposure")} | | $${(cost.totalCost ?? 0).toFixed(4)} |`);
      } catch (err) {
        fails[id]++;
        lines.push(`| ${fullName} | ${id} | | | | ${String(err).slice(0, 80)} | |`);
      }
    }
  }
  // Recall: of the skills Sol suggests at applied or anchor, how many Luna also suggests there.
  let hit = 0, total = 0;
  for (const s of Object.values(sets)) {
    if (!s[REFERENCE] || !s[CANDIDATE]) continue;
    for (const id of s[REFERENCE]) { total++; if (s[CANDIDATE].has(id)) hit++; }
  }
  console.log(`# Repository skills: ${CANDIDATE} vs ${REFERENCE}\n\nRepositories: ${repos.length}. Failures: ${CANDIDATE} ${fails[CANDIDATE]}, ${REFERENCE} ${fails[REFERENCE]}. ${CANDIDATE} covers ${hit} of ${REFERENCE}'s ${total} applied-or-anchor skills (${total ? Math.round((100 * hit) / total) : 0}%).\n\n${lines.join("\n")}`);
}
main().catch((e) => { console.error(e); process.exit(1); });
