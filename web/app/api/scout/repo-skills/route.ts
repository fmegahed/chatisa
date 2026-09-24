import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { getPageModels } from "@/lib/config/models";
import { getLanguageModel, isModelAvailable } from "@/lib/providers";
import { getMockModel } from "@/lib/providers/mock";
import { checkRateLimit, SCOUT_REPO_RATE_LIMIT } from "@/lib/ratelimit";
import { recordUsageEvent } from "@/lib/db";
import { logger } from "@/lib/log";
import { clipSummary, SUMMARY_LIMITS, type RepoSummary } from "@/lib/scout/github-summary";
import { suggestRepoSkills } from "@/lib/scout/repo-skills";

/**
 * Skills from a student's own public GitHub repositories (v6.8.0). The
 * browser reads GitHub with the student's token and sends size-capped
 * summaries; nothing about the repositories is stored or logged here. The
 * model proposes, the fixed guards decide the suggested level, and the
 * student confirms each skill at any level.
 */

// Loose on the wire; clipSummary enforces every cap (clip, never reject).
const bodySchema = z.object({
  modelId: z.string(),
  repos: z.array(z.record(z.string(), z.unknown())).min(1),
});

export async function POST(req: Request) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return NextResponse.json({ error: "Sign in required." }, { status: 401 });
  const limit = checkRateLimit(`scout-repo:${email}`, SCOUT_REPO_RATE_LIMIT);
  if (!limit.allowed) {
    return NextResponse.json({ error: `Give it a moment. Try again in ${limit.retryAfterSeconds} seconds.` }, { status: 429 });
  }
  let body: z.infer<typeof bodySchema>;
  try {
    body = bodySchema.parse(await req.json());
  } catch {
    return NextResponse.json({ error: "Send the repositories to read." }, { status: 400 });
  }
  if (!getPageModels("job_scout").includes(body.modelId)) {
    return NextResponse.json({ error: "That model is not offered here." }, { status: 400 });
  }
  if (process.env.CHATISA_MOCK_LLM !== "1" && !isModelAvailable(body.modelId)) {
    return NextResponse.json({ error: "That model is not configured on this server." }, { status: 400 });
  }
  const model = process.env.CHATISA_MOCK_LLM === "1" ? getMockModel() : getLanguageModel(body.modelId);
  let repos: RepoSummary[];
  try {
    repos = body.repos.slice(0, SUMMARY_LIMITS.reposPerRequest).map((r) => clipSummary(r as unknown as RepoSummary));
  } catch {
    // clipSummary coerces every field; this is a last line against a shape
    // it did not foresee, so a bad body is a 400, never a 500.
    return NextResponse.json({ error: "Send the repositories to read." }, { status: 400 });
  }

  const results = await Promise.all(repos.map(async (summary) => {
    const started = Date.now();
    try {
      const out = await suggestRepoSkills(model, summary);
      recordUsageEvent({
        userEmail: email, module: "job_scout", eventType: "repo_skills", modelId: body.modelId,
        inputTokens: out.usage.inputTokens, outputTokens: out.usage.outputTokens,
        latencyMs: Date.now() - started, promptChars: null, outcome: "ok",
      });
      return {
        fullName: summary.fullName, ok: true as const,
        suggestions: out.suggestions, substantial: out.substantial, codeRead: out.codeRead,
        authorship: summary.authorship,
      };
    } catch (err) {
      logger.error({ err: String(err) }, "scout repo skills failed");
      return { fullName: summary.fullName, ok: false as const, error: "Skill suggestions did not complete for this repository. Try again." };
    }
  }));
  return NextResponse.json({ results });
}
