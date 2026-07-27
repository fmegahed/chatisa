import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import {
  createExamWithQuestions,
  getDocumentPages,
  getOwnedExamDocument,
  listExams,
  purgeExpiredDocumentText,
  recordUsageEvent,
} from "@/lib/db";
import { getPageModels } from "@/lib/config/models";
import { isModelAvailable } from "@/lib/providers";
import { describeCoverage } from "@/lib/exam/chunking";
import { describeShortfall, generateExam } from "@/lib/exam/generate";
import { QUESTION_TYPES } from "@/lib/exam/schemas";
import { EXAM_GENERATE_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";

export const maxDuration = 300;

const MODULE = "exam_ally";

const createExamSchema = z.object({
  documentId: z.uuid(),
  modelId: z.string().min(1).max(128),
  questionType: z.enum(QUESTION_TYPES),
  count: z.number().int().min(1).max(20),
  examMode: z.enum(["practice", "exam"]),
  fromPage: z.number().int().min(1).max(1500).optional(),
  toPage: z.number().int().min(1).max(1500).optional(),
});

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

/** The signed-in student's exams, newest first. */
export async function GET() {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  return NextResponse.json({
    exams: listExams(userEmail).map((e) => ({
      id: e.id,
      documentId: e.documentId,
      status: e.status,
      examMode: e.examMode,
      questionType: e.questionType,
      requestedCount: e.requestedCount,
      deliveredCount: e.deliveredCount,
      currentPosition: e.currentPosition,
      updatedAt: e.updatedAt,
    })),
  });
}

/** Builds an exam from a document the student already uploaded. */
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
  const parsed = createExamSchema.safeParse(body);
  if (!parsed.success) {
    return errorResponse(400, "That request wasn't valid.", {
      fields: [...new Set(parsed.error.issues.map((i) => i.path.join(".")))],
    });
  }
  const input = parsed.data;

  // The module's allow-list decides, not the client.
  if (!getPageModels(MODULE).includes(input.modelId)) {
    return errorResponse(400, "That model isn't available for this module.");
  }
  if (!isModelAvailable(input.modelId)) {
    return errorResponse(
      503,
      "That model isn't configured on this server right now. Pick another model.",
    );
  }

  const limit = checkRateLimit(`exam-generate:${userEmail}`, EXAM_GENERATE_RATE_LIMIT);
  if (!limit.allowed) {
    return errorResponse(
      429,
      "You've generated several exams in a short time. Wait a moment and try again.",
      { retryAfterSeconds: limit.retryAfterSeconds },
    );
  }

  // Opportunistic sweep of documents whose text was never cleaned up
  // (ADR-015): the safety net only works if something calls it.
  purgeExpiredDocumentText();

  const document = getOwnedExamDocument(input.documentId, userEmail);
  if (!document) {
    return errorResponse(404, "That document could not be found.");
  }

  const pages = getDocumentPages(input.documentId).map((p) => ({
    pageNumber: p.pageNumber,
    text: p.text,
    charCount: p.charCount,
    source: p.source,
  }));
  if (pages.length === 0) {
    return errorResponse(
      410,
      "The text for that document is no longer stored. Upload the file again to build a new exam.",
      { code: "TEXT_PURGED" },
    );
  }

  const fromPage = input.fromPage ?? 1;
  const toPage = input.toPage ?? document.pageCount;
  if (fromPage > toPage) {
    return errorResponse(400, "The start page must not be after the end page.");
  }

  const startedAt = Date.now();
  try {
    const result = await generateExam({
      modelId: input.modelId,
      questionType: input.questionType,
      count: input.count,
      pages,
      fromPage,
      toPage,
    });

    const droppedCount = result.dropped.reduce((sum, d) => sum + d.count, 0);

    if (result.failed) {
      recordUsageEvent({
        userEmail,
        module: MODULE,
        eventType: "exam_generation_failed",
        modelId: input.modelId,
        latencyMs: Date.now() - startedAt,
        outcome: "grounding_failed",
      });
      return errorResponse(
        422,
        "We couldn't build questions we can trace back to your document. Try a narrower page range, or a different model. This often happens with slide decks or documents that are mostly charts and images.",
        { code: "GROUNDING_FAILED", dropped: result.dropped },
      );
    }

    const examId = createExamWithQuestions({
      userEmail,
      documentId: input.documentId,
      modelId: input.modelId,
      examMode: input.examMode,
      questionType: input.questionType,
      requestedCount: input.count,
      droppedCount,
      scopeFromPage: fromPage,
      scopeToPage: toPage,
      coverage: result.coverage,
      questions: result.questions.map((q) => ({
        type: q.type,
        stem: q.stem,
        options: q.options,
        correctIndex: q.correctIndex,
        modelAnswer: q.modelAnswer,
        rubric: q.rubric,
        explanation: q.explanation,
        topic: q.topic,
        bloom: q.bloom,
        sourceQuote: q.sourceQuote,
        sourcePage: q.sourcePage,
        groundingStatus: q.groundingStatus,
        pointsPossible: q.pointsPossible,
      })),
    });

    recordUsageEvent({
      userEmail,
      module: MODULE,
      eventType: "exam_generation",
      modelId: input.modelId,
      latencyMs: Date.now() - startedAt,
      outcome: "ready",
    });
    logger.info(
      {
        requestId,
        module: MODULE,
        modelId: input.modelId,
        delivered: result.questions.length,
        dropped: droppedCount,
        latencyMs: Date.now() - startedAt,
      },
      "exam generated",
    );

    return NextResponse.json(
      {
        examId,
        deliveredCount: result.questions.length,
        requestedCount: input.count,
        droppedCount,
        coverage: describeCoverage(result.coverage, document.pageCount),
        shortfall: describeShortfall(input.count, result.questions.length),
      },
      { status: 201 },
    );
  } catch (err) {
    logger.error(
      { requestId, module: MODULE, err: String(err) },
      "exam generation failed",
    );
    recordUsageEvent({
      userEmail,
      module: MODULE,
      eventType: "exam_generation_error",
      modelId: input.modelId,
      latencyMs: Date.now() - startedAt,
      outcome: "error",
    });
    return errorResponse(
      500,
      "Something went wrong building that exam. Your document is still here, so you can try again.",
    );
  }
}
