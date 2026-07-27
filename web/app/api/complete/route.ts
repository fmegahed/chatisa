import { NextResponse } from "next/server";
import { generateText } from "ai";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import { temperatureFor } from "@/lib/config/models";
import { getLanguageModel, isModelAvailable } from "@/lib/providers";
import { getMockModel } from "@/lib/providers/mock";
import { COMPLETION_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";
import {
  COMPLETION_DEFAULT_MODEL,
  COMPLETION_LANGUAGES,
  COMPLETION_MODELS,
  COMPLETION_SYSTEM_PROMPT,
  buildCompletionPrompt,
  parseCompletion,
} from "@/lib/sandbox/completion";

export const maxDuration = 30;

const requestSchema = z.object({
  language: z.enum(COMPLETION_LANGUAGES),
  prefix: z.string().max(8000),
  suffix: z.string().max(8000).optional(),
  model: z.string().max(128).optional(),
});

/**
 * Inline code completion for the Sandbox editor. Short, non-streaming. On any
 * soft failure (model unavailable, rate limited, model error) it returns an
 * empty completion with 200 rather than an error, so the editor silently shows
 * nothing instead of surfacing errors while a student types.
 */
export async function POST(req: Request) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return NextResponse.json({ completion: "" }, { status: 401 });

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ completion: "" });
  }

  const parsed = requestSchema.safeParse(body);
  if (!parsed.success) return NextResponse.json({ completion: "" });
  const { language, prefix, suffix = "", model } = parsed.data;

  // Nothing meaningful to complete from.
  if (prefix.trim().length === 0) return NextResponse.json({ completion: "" });

  const limit = checkRateLimit(`complete:${userEmail}`, COMPLETION_RATE_LIMIT);
  if (!limit.allowed) return NextResponse.json({ completion: "" });

  // Accept an in-allow-list override, else the fast default.
  const modelId =
    model && COMPLETION_MODELS.includes(model)
      ? model
      : COMPLETION_DEFAULT_MODEL;

  const mock = process.env.CHATISA_MOCK_LLM === "1";
  if (!mock && !isModelAvailable(modelId)) {
    return NextResponse.json({ completion: "" });
  }

  // gpt-oss is a reasoning model: at its default effort it spends most of its
  // output budget thinking before emitting a one-line completion, which is slow
  // and wasteful here. "low" keeps a completion terse and fast (verified against
  // the provider: ~32 output tokens vs ~197 unset, same result). "none" is not a
  // valid effort for gpt-oss on this route's provider, so "low" is the floor.
  const reasoningModel = modelId.includes("gpt-oss");

  try {
    const result = await generateText({
      model: mock ? getMockModel() : getLanguageModel(modelId),
      system: COMPLETION_SYSTEM_PROMPT,
      prompt: buildCompletionPrompt(language, prefix, suffix),
      temperature: temperatureFor(modelId, 0.2),
      maxOutputTokens: 256,
      abortSignal: req.signal,
      ...(reasoningModel && !mock
        ? { providerOptions: { huggingface: { reasoningEffort: "low" } } }
        : {}),
    });
    // The mock model does not produce code, so return a deterministic stand-in
    // completion for tests.
    const raw = mock ? "# ai suggestion" : result.text;
    return NextResponse.json({ completion: parseCompletion(raw) });
  } catch (err) {
    // A client aborts superseded requests as the student keeps typing; that is
    // expected, not a failure, so it is not logged. Anything else is.
    const aborted =
      req.signal.aborted ||
      (err instanceof Error && /abort/i.test(err.name + err.message));
    if (!aborted) logger.warn({ err: String(err) }, "completion failed");
    return NextResponse.json({ completion: "" });
  }
}
