import { NextResponse } from "next/server";
import { z } from "zod";
import { generateObject } from "ai";
import { auth } from "@/lib/auth";
import { calculateCost, getPageModels } from "@/lib/config/models";
import { getLanguageModel, isModelAvailable } from "@/lib/providers";
import { getMockModel } from "@/lib/providers/mock";
import { getSkill, SKILL_IDS } from "@/lib/scout/taxonomy";
import { checkRateLimit, SCOUT_PROJECT_RATE_LIMIT } from "@/lib/ratelimit";
import { recordUsageEvent } from "@/lib/db";
import { logger } from "@/lib/log";
import { outputTokenBudget } from "@/lib/chat/budget";

/**
 * Portfolio project scaffold generation. Job-agnostic by design (user
 * decision 2026-07-28): the input is skills, never an employer, so the
 * generated repo stays useful across every application. Stateless: the
 * client sends the selected skills and its own course evidence, the file
 * manifest comes back, nothing is persisted server-side.
 */

const requestSchema = z.object({
  modelId: z.string().min(1),
  skillIds: z
    .array(z.enum(SKILL_IDS as [string, ...string[]]))
    .min(1)
    .max(6),
  /** The student's own course evidence phrases, sent by the client. */
  evidence: z.array(z.string().max(200)).max(12).default([]),
});

const scaffoldSchema = z.object({
  repoName: z
    .string()
    .regex(/^[a-z0-9][a-z0-9-]{2,60}$/, "kebab-case repo name"),
  summary: z.string().max(400),
  readme: z.string().max(12_000),
  files: z
    .array(
      z.object({
        path: z
          .string()
          .regex(/^[\w.-]+(\/[\w.-]+)*$/, "relative path, no traversal"),
        contents: z.string().max(20_000),
      }),
    )
    .min(2)
    .max(12),
  /** Copy-paste shell steps: git init through first push. */
  instructions: z.array(z.string().max(300)).min(3).max(10),
  /** Draft resume bullets, placeholders for unmeasured numbers. */
  resumeBullets: z.array(z.string().max(250)).max(4),
});

const INSTRUCTIONS = `You design a portfolio project scaffold for an analytics or information systems student.

The project must demonstrate the requested skills through realistic business work with real, freely available public data (name actual datasets and where to get them; never invent URLs you are not confident exist). Never mention any specific employer or job posting: the project must stand on its own in any application.

Produce:
- A README.md that opens with what the project shows, maps each requested skill to where the work demonstrates it, and includes a milestone plan a student can follow in 2-4 weeks.
- Starter files: a sensible folder layout with stub code files (clearly marked TODO sections), a .gitignore appropriate to the stack, and a data/README pointing at the datasets. Keep stubs short; the student writes the real work.
- Shell instructions from git init to first push using GitHub CLI (gh repo create).
- Up to 4 draft resume bullets. Use bracketed placeholders like [X%] for any number the student has not measured yet; never invent metrics.

Where the student's own course evidence is provided, echo its vocabulary in the README's skill mapping so the claims stay grounded in work they have actually done.`;

export async function POST(req: Request) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) {
    return NextResponse.json({ error: "Sign in required." }, { status: 401 });
  }
  const limit = checkRateLimit(`scout-project:${email}`, SCOUT_PROJECT_RATE_LIMIT);
  if (!limit.allowed) {
    return NextResponse.json(
      {
        error: `Scaffold generation is limited. Try again in ${limit.retryAfterSeconds} seconds.`,
      },
      { status: 429 },
    );
  }

  let input: z.infer<typeof requestSchema>;
  try {
    input = requestSchema.parse(await req.json());
  } catch {
    return NextResponse.json(
      { error: "Pick 1 to 6 skills from the list and try again." },
      { status: 400 },
    );
  }
  if (!getPageModels("job_scout").includes(input.modelId)) {
    return NextResponse.json(
      { error: "That model is not offered here." },
      { status: 400 },
    );
  }
  if (process.env.CHATISA_MOCK_LLM !== "1" && !isModelAvailable(input.modelId)) {
    return NextResponse.json(
      { error: "That model is not configured on this server." },
      { status: 400 },
    );
  }

  const skillLines = input.skillIds
    .map((id) => `- ${id}: ${getSkill(id)?.label ?? id}`)
    .join("\n");
  const evidenceBlock =
    input.evidence.length > 0
      ? `\n\nThe student's own course evidence:\n${input.evidence
          .map((e) => `- ${e}`)
          .join("\n")}`
      : "";

  const model =
    process.env.CHATISA_MOCK_LLM === "1"
      ? getMockModel()
      : getLanguageModel(input.modelId);
  const started = Date.now();
  try {
    const { object, usage } = await generateObject({
      model,
      schema: scaffoldSchema,
      instructions: INSTRUCTIONS,
      prompt: `Skills to demonstrate:\n${skillLines}${evidenceBlock}`,
      maxOutputTokens: outputTokenBudget(input.modelId, 8_000),
    });
    const cost = calculateCost(
      input.modelId,
      usage?.inputTokens ?? 0,
      usage?.outputTokens ?? 0,
    );
    recordUsageEvent({
      userEmail: email,
      module: "job_scout",
      eventType: "project_generated",
      modelId: input.modelId,
      inputTokens: usage?.inputTokens ?? null,
      outputTokens: usage?.outputTokens ?? null,
      costUsd: "totalCost" in cost ? cost.totalCost : null,
      latencyMs: Date.now() - started,
      outcome: "ok",
    });
    return NextResponse.json({ scaffold: object });
  } catch (err) {
    logger.error({ err: String(err) }, "scout project generation failed");
    return NextResponse.json(
      { error: "The scaffold did not generate. Try again, or pick fewer skills." },
      { status: 502 },
    );
  }
}
