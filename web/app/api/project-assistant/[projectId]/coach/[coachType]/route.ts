// app/api/project-assistant/[projectId]/coach/[coachType]/route.ts
import { z } from "zod";
import {
  convertToModelMessages,
  stepCountIs,
  streamText,
  tool,
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
import { getLanguageModel, isModelAvailable } from "@/lib/providers";
import { getMockModel } from "@/lib/providers/mock";
import { classifyProviderFailure } from "@/lib/providers/errors";
import { outputTokenBudget } from "@/lib/chat/budget";
import { CHAT_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";
import { recordUsageEvent } from "@/lib/db";
import {
  getAccessibleProject,
  getDeliverable,
  getOrCreateDeliverable,
  saveDeliverableContent,
} from "@/lib/db/projects";
import { getCoachEngine, type GenericOp } from "@/lib/project/coach-engine";

export const runtime = "nodejs";
export const maxDuration = 120;

const MODULE = "project_coach";

function jsonError(status: number, message: string) {
  return Response.json({ error: message }, { status });
}

const bodySchema = z.object({
  modelId: z.string().min(1).max(128),
  messages: z.array(z.any()).min(1).max(400),
});

export async function POST(
  req: Request,
  { params }: { params: Promise<{ projectId: string; coachType: string }> },
) {
  const requestId = crypto.randomUUID();
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return jsonError(401, "Sign in to continue.");

  const { projectId, coachType } = await params;
  const engine = getCoachEngine(coachType);
  if (!engine) return jsonError(400, "That coach is not available.");

  const project = getAccessibleProject(projectId, email);
  if (!project) return jsonError(404, "That project could not be found.");

  let raw: unknown;
  try {
    raw = await req.json();
  } catch {
    return jsonError(400, "Request body must be JSON.");
  }
  const parsed = bodySchema.safeParse(raw);
  if (!parsed.success) return jsonError(400, "That request wasn't valid.");

  const { modelId, messages } = parsed.data;
  if (!getPageModels(MODULE).includes(modelId)) {
    return jsonError(400, "That model isn't available for this coach.");
  }
  if (!isModelAvailable(modelId)) {
    return jsonError(503, "That model isn't configured right now. Pick another.");
  }

  const limit = checkRateLimit(`coach:${email}`, CHAT_RATE_LIMIT);
  if (!limit.allowed) {
    return jsonError(429, "You've sent a lot of messages. Wait a moment and try again.");
  }

  getOrCreateDeliverable(projectId, coachType);
  const updatedBy = session.user?.name ?? email;
  const modelConfig = MODELS[modelId];
  const startedAt = Date.now();

  const readContent = () => {
    const row = getDeliverable(projectId, coachType);
    return row ? engine.parseContent(row.contentJson) : engine.emptyContent();
  };
  const applyAndSave = (op: GenericOp) => {
    const next = engine.applyOp(readContent(), op);
    saveDeliverableContent({
      projectId,
      coachType,
      contentJson: JSON.stringify(next),
      updatedBy,
    });
  };

  const tools = {
    setField: tool({
      description: "Record a single settled field in the worksheet.",
      inputSchema: z.object({ path: z.string(), value: z.string() }),
      execute: async ({ path, value }) => {
        applyAndSave({ kind: "setField", path, value });
        return { ok: true, path };
      },
    }),
    addRow: tool({
      description: "Add an empty row to a worksheet table before filling it.",
      inputSchema: z.object({ table: z.string() }),
      execute: async ({ table }) => {
        applyAndSave({ kind: "addRow", table });
        return { ok: true, table };
      },
    }),
    setRow: tool({
      description: "Set the fields of an existing worksheet table row by index.",
      inputSchema: z.object({
        table: z.string(),
        index: z.number().int().min(0),
        row: z.record(z.string(), z.string()),
      }),
      execute: async ({ table, index, row }) => {
        applyAndSave({ kind: "setRow", table, index, row });
        return { ok: true, table, index };
      },
    }),
  };

  const instructions = `${engine.systemPrompt}\n\n--- Worksheet so far (JSON) ---\n${engine.serializeForPrompt(readContent())}`;

  try {
    const model =
      process.env.CHATISA_MOCK_LLM === "1"
        ? getMockModel()
        : getLanguageModel(modelId);

    const result = streamText({
      model,
      instructions,
      messages: await convertToModelMessages(messages as unknown as UIMessage[]),
      tools,
      toolChoice: "auto",
      // The coach may take several steps: ask, record a tool call, then reply.
      stopWhen: stepCountIs(8),
      temperature: temperatureFor(modelId, 0.3),
      maxOutputTokens: outputTokenBudget(modelId, 1200),
      abortSignal: req.signal,
      onFinish({ text, usage, finishReason }) {
        const inputTokens = usage?.inputTokens ?? null;
        const outputTokens = usage?.outputTokens ?? null;
        const cost =
          inputTokens != null && outputTokens != null
            ? calculateCost(modelId, inputTokens, outputTokens)
            : null;
        recordUsageEvent({
          userEmail: email,
          module: MODULE,
          eventType: "coach_completion",
          modelId,
          provider: modelConfig.provider,
          inputTokens,
          outputTokens,
          costUsd: cost && "totalCost" in cost ? cost.totalCost : null,
          latencyMs: Date.now() - startedAt,
          responseChars: text.length,
          outcome: finishReason ?? "stop",
        });
      },
      onError({ error }) {
        logger.error(
          { requestId, module: MODULE, modelId, err: String(error) },
          "coach stream failed",
        );
        recordUsageEvent({
          userEmail: email,
          module: MODULE,
          eventType: "coach_error",
          modelId,
          provider: modelConfig.provider,
          latencyMs: Date.now() - startedAt,
          outcome: classifyProviderFailure(error).kind,
        });
      },
    });

    return result.toUIMessageStreamResponse({
      onError: (error) => classifyProviderFailure(error).message,
    });
  } catch (err) {
    logger.error({ requestId, err: String(err) }, "coach route failed");
    return jsonError(503, "That model isn't configured right now. Pick another.");
  }
}
