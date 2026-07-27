import { NextResponse } from "next/server";
import {
  convertToModelMessages,
  streamText,
  type ModelMessage,
  type UIMessage,
} from "ai";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import {
  MODELS,
  calculateCost,
  getPageModels,
  temperatureFor,
} from "@/lib/config/models";
import {
  CHAT_MODULES,
  chatRequestSchema,
  textFromParts,
} from "@/lib/chat/config";
import {
  getLanguageModel,
  isModelAvailable,
  ProviderNotConfiguredError,
} from "@/lib/providers";
import { CHAT_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";
import { recordUsageEvent } from "@/lib/db";
import { getMockModel } from "@/lib/providers/mock";
import { classifyProviderFailure } from "@/lib/providers/errors";
import {
  TRUNCATION_NOTICE,
  describeEmptyResponse,
  outputTokenBudget,
} from "@/lib/chat/budget";

export const maxDuration = 120;

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

/**
 * Streaming chat endpoint. Every privileged call happens here on the server:
 * the browser never sees a provider key. The client sends a model *id* only;
 * this route decides whether that id is allowed for the module.
 */
export async function POST(req: Request) {
  const requestId = crypto.randomUUID();
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Request body must be JSON.");
  }

  const parsed = chatRequestSchema.safeParse(body);
  if (!parsed.success) {
    return errorResponse(400, "That request wasn't valid.", {
      fields: [...new Set(parsed.error.issues.map((i) => i.path.join(".")))],
    });
  }
  const {
    module: moduleKey,
    modelId,
    messages,
    context,
  } = parsed.data;

  const moduleConfig = CHAT_MODULES[moduleKey];
  if (!moduleConfig) return errorResponse(400, "Unknown module.");

  // Server-authoritative model policy: the module's allow-list wins.
  if (!getPageModels(moduleKey).includes(modelId)) {
    return errorResponse(400, "That model isn't available for this module.");
  }
  if (!isModelAvailable(modelId)) {
    return errorResponse(
      503,
      "That model isn't configured on this server right now. Pick another model.",
    );
  }

  const limit = checkRateLimit(`chat:${userEmail}`, CHAT_RATE_LIMIT);
  if (!limit.allowed) {
    return errorResponse(
      429,
      "You've sent a lot of messages in a short time. Wait a moment and try again.",
      { retryAfterSeconds: limit.retryAfterSeconds },
    );
  }

  // The last user turn's text is used only for the content-free usage event
  // (its length), never stored.
  const lastUserText = textFromParts(
    [...messages].reverse().find((m) => m.role === "user")?.parts,
  );

  const modelConfig = MODELS[modelId];
  const startedAt = Date.now();

  try {
    const uiMessages = messages as unknown as UIMessage[];
    // The legacy app seeded a fixed opening user turn so the tutor introduces
    // itself; preserved here for response parity. The system prompt travels
    // in `instructions` (AI SDK v7 rejects system roles inside `messages`).
    const modelMessages: ModelMessage[] = [
      ...(moduleConfig.openingUserMessage
        ? ([
            { role: "user", content: moduleConfig.openingUserMessage },
          ] as ModelMessage[])
        : []),
      ...(await convertToModelMessages(uiMessages)),
    ];

    // Set in onFinish and read after the stream ends, so an answer that turns
    // out to be empty can be explained rather than rendered as a blank bubble.
    let emptyExplanation: string | null = null;

    // Ephemeral context (the Sandbox's current script, last result and
    // variables) rides in the system instructions for this turn only. It is
    // never written to the conversation, so it does not accumulate or persist.
    const instructions = context
      ? `${moduleConfig.systemPrompt}\n\n--- The student's current work ---\n${context}`
      : moduleConfig.systemPrompt;

    const result = streamText({
      model:
        process.env.CHATISA_MOCK_LLM === "1"
          ? getMockModel()
          : getLanguageModel(modelId),
      instructions,
      messages: modelMessages,
      temperature: temperatureFor(modelId, moduleConfig.temperature),
      // A runaway guard, not a spend control, and widened for reasoning models
      // whose hidden thinking competes with the visible answer for the same
      // allowance. Clamped to what the model can actually emit.
      maxOutputTokens: outputTokenBudget(modelId, moduleConfig.maxOutputTokens),
      abortSignal: req.signal,
      onFinish({ text, usage, finishReason }) {
        const inputTokens = usage?.inputTokens ?? null;
        const outputTokens = usage?.outputTokens ?? null;
        const cost =
          inputTokens != null && outputTokens != null
            ? calculateCost(modelId, inputTokens, outputTokens)
            : null;
        const costUsd = cost && "totalCost" in cost ? cost.totalCost : null;

        // A response with nothing readable in it must not be stored as though
        // it were an answer. Previously an empty assistant message was
        // persisted and rendered as a blank bubble with no explanation, which
        // is what a reasoning model hitting the old 1000-token cap produced.
        const visible = text.trim();
        if (visible === "") {
          emptyExplanation = describeEmptyResponse(
            finishReason === "length"
              ? "truncated_before_text"
              : "no_text_returned",
          );
          logger.warn(
            {
              requestId,
              module: moduleKey,
              modelId,
              finishReason,
              outputTokens,
            },
            "model returned no visible text",
          );
        }
        recordUsageEvent({
          userEmail,
          module: moduleKey,
          eventType: "chat_completion",
          modelId,
          provider: modelConfig.provider,
          inputTokens,
          outputTokens,
          costUsd,
          latencyMs: Date.now() - startedAt,
          promptChars: lastUserText.length,
          responseChars: text.length,
          outcome: finishReason ?? "stop",
        });
        // Lengths and ids only: never prompt or response text.
        logger.info(
          {
            requestId,
            module: moduleKey,
            modelId,
            inputTokens,
            outputTokens,
            latencyMs: Date.now() - startedAt,
          },
          "chat completion",
        );
      },
      onError({ error }) {
        const failure = classifyProviderFailure(error);
        logger.error(
          {
            requestId,
            module: moduleKey,
            modelId,
            failureKind: failure.kind,
            // Surfaces in log search so an exhausted account or a withdrawn
            // model is findable, rather than buried among ordinary blips.
            operatorAction: failure.operatorAction,
            err: String(error),
          },
          failure.operatorAction
            ? "chat stream failed: needs operator attention"
            : "chat stream failed",
        );
        recordUsageEvent({
          userEmail,
          module: moduleKey,
          eventType: "chat_error",
          modelId,
          provider: modelConfig.provider,
          latencyMs: Date.now() - startedAt,
          promptChars: lastUserText.length,
          outcome: failure.kind,
        });
      },
    });

    return result.toUIMessageStreamResponse({
      // Attaches an explanation to the finished message when the model
      // produced nothing readable, or was cut off with text already delivered.
      // Without this the student sees a blank bubble, or an answer that stops
      // mid-sentence, with nothing telling them what happened.
      messageMetadata: ({ part }) => {
        if (part.type !== "finish") return undefined;
        if (emptyExplanation) return { notice: emptyExplanation };
        if (part.finishReason === "length") return { notice: TRUNCATION_NOTICE };
        return undefined;
      },
      // Client-visible message: no provider internals, no keys. Says whether
      // retrying can actually work, because telling a student to retry an
      // exhausted account sends them into a loop that never succeeds.
      onError: (error) => classifyProviderFailure(error).message,
    });
  } catch (err) {
    if (err instanceof ProviderNotConfiguredError) {
      logger.error({ requestId, provider: err.provider }, "provider missing key");
      return errorResponse(
        503,
        "That model isn't configured on this server right now. Pick another model.",
      );
    }
    logger.error({ requestId, err: String(err) }, "chat request failed");
    return errorResponse(
      500,
      "Something went wrong starting that response. Your message was kept, so you can try again.",
    );
  }
}
