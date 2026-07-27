import { NextResponse } from "next/server";
import {
  convertToModelMessages,
  stepCountIs,
  streamText,
  type ModelMessage,
  type UIMessage,
} from "ai";
import { forwardAnthropicContainerIdFromLastStep } from "@ai-sdk/anthropic";
import { awaitsBrowserTool } from "@/lib/ask/stop-conditions";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import {
  MODELS,
  calculateCost,
  getPageModels,
  temperatureFor,
} from "@/lib/config/models";
import { CHAT_MODULES, textFromParts } from "@/lib/chat/config";
import { askRequestSchema } from "@/lib/ask/tools";
import { askServerToolDefs } from "@/lib/ask/server-tools";
import {
  anthropicTemplateMessage,
  hostedToolsFor,
  templateFileId,
  wantsGeneratedFile,
  type HostedProvider,
} from "@/lib/ask/hosted";
import {
  attachmentBlockText,
  type AttachmentData,
} from "@/lib/files/attachments";
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

/** The module is fixed: this route exists to attach Ask Anything's tools. */
const MODULE_KEY = "ask_anything";

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

/**
 * Ask Anything's streaming endpoint: the tools-bearing sibling of /api/chat
 * (design 2026-07-24). Tools are declared WITHOUT execute, so the model's tool
 * calls stream to the browser, which runs them on the WASM runtimes and posts
 * the results back as the next request; each request here is one hop of that
 * loop. Same auth, model policy, rate limiting, and content-free usage events
 * as /api/chat; conversation content is never stored (ADR-022).
 */
export async function POST(req: Request) {
  const requestId = crypto.randomUUID();
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  // Attachments ride inline (data URLs), so requests can be large; this guard
  // bounds them. 30 MB covers the 25 MB per-message attachment cap plus the
  // JSON around it, and stays inside Anthropic's own 32 MB request limit.
  const declaredBytes = Number(req.headers.get("content-length") ?? "0");
  if (Number.isFinite(declaredBytes) && declaredBytes > 30 * 1024 * 1024) {
    return errorResponse(
      413,
      "That message is too large. Attach files under 25 MB.",
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Request body must be JSON.");
  }

  const parsed = askRequestSchema.safeParse(body);
  if (!parsed.success) {
    return errorResponse(400, "That request wasn't valid.", {
      fields: [...new Set(parsed.error.issues.map((i) => i.path.join(".")))],
    });
  }
  const { modelId, messages } = parsed.data;
  const moduleConfig = CHAT_MODULES[MODULE_KEY];

  // Server-authoritative model policy: the module's allow-list wins.
  if (!getPageModels(MODULE_KEY).includes(modelId)) {
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
    // Hosted execution (slice E): the chat provider's own sandbox tool rides
    // along with the browser and research tools. Roster policy already
    // guarantees an Anthropic or OpenAI model here. The OpenAI template file
    // attaches to the interpreter container's file list (materialized only if
    // the tool runs); Anthropic's template injection happens below, gated,
    // because its container_upload block forces a container into existence.
    const hostedProvider = modelConfig.provider as HostedProvider;
    const openaiTemplateId =
      hostedProvider === "openai" ? await templateFileId("openai") : null;
    const tools = {
      ...askServerToolDefs(),
      ...hostedToolsFor(hostedProvider, openaiTemplateId),
    };
    // The tool set must ride along: without it the converter cannot serialize
    // the tool-run_python parts a continuation request carries, and the whole
    // request fails after the browser has already executed the code.
    // convertDataPart renders attachment parts (extracted Office text, dataset
    // announcements) into the labeled text blocks the model reads; file parts
    // (images, PDFs) convert natively.
    const modelMessages: ModelMessage[] = await convertToModelMessages(uiMessages, {
      tools,
      convertDataPart: (part: { type: string; data?: unknown }) =>
        part.type === "data-attachment"
          ? {
              type: "text" as const,
              text: attachmentBlockText(part.data as AttachmentData),
            }
          : undefined,
    });

    // Miami deck template for Anthropic: injected just before the student's
    // message, only when they asked for a generated file (see lib/ask/hosted).
    if (
      hostedProvider === "anthropic" &&
      wantsGeneratedFile(lastUserText) &&
      modelMessages.length > 0
    ) {
      const anthropicTemplate = await templateFileId("anthropic");
      if (anthropicTemplate) {
        modelMessages.splice(
          modelMessages.length - 1,
          0,
          anthropicTemplateMessage(anthropicTemplate),
        );
      }
    }

    // Anthropic caches only up to an explicit breakpoint; marking the last
    // message caches the whole prefix (system, attached PDFs, earlier turns),
    // which is what makes native PDF attachments affordable across a chat's
    // turns. OpenAI caches automatically, no marker needed.
    if (modelConfig.provider === "anthropic" && modelMessages.length > 0) {
      const last = modelMessages[modelMessages.length - 1];
      last.providerOptions = {
        ...last.providerOptions,
        anthropic: { cacheControl: { type: "ephemeral" } },
      };
    }

    let emptyExplanation: string | null = null;

    const result = streamText({
      model:
        process.env.CHATISA_MOCK_LLM === "1"
          ? getMockModel()
          : getLanguageModel(modelId),
      instructions: moduleConfig.systemPrompt,
      messages: modelMessages,
      tools,
      // Two conditions, and both are load-bearing.
      //
      // stepCountIs bounds server- and provider-executed steps.
      // awaitsBrowserTool ends the turn the moment a run_python / run_r /
      // run_sql call is waiting on the student's browser, which the step count
      // alone cannot express. Without it, a turn that resolved every call in an
      // earlier step kept stepping and then failed with MissingToolResults on
      // the browser call it could never resolve server-side (measured on Claude
      // Opus 5, 2026-07-26; see lib/ask/stop-conditions).
      stopWhen: [stepCountIs(10), awaitsBrowserTool()],
      // Carry Anthropic's code-execution container across steps. Without this
      // every step gets a FRESH container, so the sandbox filesystem resets
      // between tool calls: the uploaded template has to be re-copied, its path
      // changes each time, and the model wastes its whole step budget redoing
      // setup. Measured 2026-07-25 on "make me a 5-slide deck": 25+ code calls
      // and still unfinished at 15 minutes, with the model itself reporting
      // "/tmp didn't persist across separate tool calls" and "the INPUT_DIR path
      // changes between calls". The helper returns undefined when no container
      // id is in the prior steps, so the OpenAI and mock paths are unaffected.
      prepareStep: forwardAnthropicContainerIdFromLastStep,
      temperature: temperatureFor(modelId, moduleConfig.temperature),
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

        // A turn that ends in tool calls legitimately has no text: the browser
        // executes and the loop continues. Only a terminal turn with nothing
        // readable warrants the empty-response note.
        const visible = text.trim();
        if (visible === "" && finishReason !== "tool-calls") {
          emptyExplanation = describeEmptyResponse(
            finishReason === "length"
              ? "truncated_before_text"
              : "no_text_returned",
          );
          logger.warn(
            { requestId, module: MODULE_KEY, modelId, finishReason, outputTokens },
            "model returned no visible text",
          );
        }
        recordUsageEvent({
          userEmail,
          module: MODULE_KEY,
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
        logger.info(
          {
            requestId,
            module: MODULE_KEY,
            modelId,
            inputTokens,
            outputTokens,
            latencyMs: Date.now() - startedAt,
          },
          "ask-anything completion",
        );
      },
      onError({ error }) {
        const failure = classifyProviderFailure(error);
        logger.error(
          {
            requestId,
            module: MODULE_KEY,
            modelId,
            failureKind: failure.kind,
            operatorAction: failure.operatorAction,
            err: String(error),
          },
          failure.operatorAction
            ? "ask-anything stream failed: needs operator attention"
            : "ask-anything stream failed",
        );
        recordUsageEvent({
          userEmail,
          module: MODULE_KEY,
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
      messageMetadata: ({ part }) => {
        if (part.type !== "finish") return undefined;
        if (emptyExplanation) return { notice: emptyExplanation };
        if (part.finishReason === "length") return { notice: TRUNCATION_NOTICE };
        return undefined;
      },
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
    logger.error({ requestId, err: String(err) }, "ask-anything request failed");
    return errorResponse(
      500,
      "Something went wrong starting that response. Your message was kept, so you can try again.",
    );
  }
}
