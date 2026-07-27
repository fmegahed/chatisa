import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import {
  createExamWithQuestions,
  getDocumentPages,
  getExamQuestions,
  getOwnedExam,
  getOwnedExamDocument,
  recordUsageEvent,
} from "@/lib/db";
import { describeCoverage } from "@/lib/exam/chunking";
import { describeShortfall, generateExam } from "@/lib/exam/generate";
import { QUESTION_TYPES, type QuestionType } from "@/lib/exam/schemas";
import { EXAM_GENERATE_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";

export const maxDuration = 300;

const MODULE = "exam_ally";

const retrySchema = z.object({
  topics: z.array(z.string().min(1).max(60)).min(1).max(10),
  count: z.number().int().min(1).max(20).optional(),
  questionType: z.enum(QUESTION_TYPES).optional(),
});

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

/**
 * Builds a fresh exam concentrating on the topics a student found hard, from
 * the same document. Previously asked questions are excluded, so this is new
 * practice rather than the same questions again.
 */
export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const source = getOwnedExam((await params).id, userEmail);
  if (!source) return errorResponse(404, "That exam could not be found.");

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Request body must be JSON.");
  }
  const parsed = retrySchema.safeParse(body);
  if (!parsed.success) {
    return errorResponse(400, "That request wasn't valid.", {
      fields: [...new Set(parsed.error.issues.map((i) => i.path.join(".")))],
    });
  }
  const input = parsed.data;

  const limit = checkRateLimit(`exam-generate:${userEmail}`, EXAM_GENERATE_RATE_LIMIT);
  if (!limit.allowed) {
    return errorResponse(
      429,
      "You've generated several exams in a short time. Wait a moment and try again.",
      { retryAfterSeconds: limit.retryAfterSeconds },
    );
  }

  const document = getOwnedExamDocument(source.documentId, userEmail);
  if (!document) return errorResponse(404, "That document could not be found.");

  const pages = getDocumentPages(source.documentId).map((p) => ({
    pageNumber: p.pageNumber,
    text: p.text,
    charCount: p.charCount,
    source: p.source,
  }));
  if (pages.length === 0) {
    return errorResponse(
      410,
      "The text for that document is no longer stored, so we can't write new questions from it. Upload the file again to keep practising.",
      { code: "TEXT_PURGED" },
    );
  }

  const previousStems = getExamQuestions(source.id).map((q) => q.stem);
  const count = input.count ?? Math.min(source.deliveredCount, 5);
  const questionType = (input.questionType ??
    source.questionType) as QuestionType;
  const startedAt = Date.now();

  try {
    const result = await generateExam({
      modelId: source.modelId,
      questionType,
      count,
      pages,
      fromPage: source.scopeFromPage,
      toPage: source.scopeToPage,
      focusTopics: input.topics,
      excludeStems: previousStems,
    });

    if (result.failed) {
      return errorResponse(
        422,
        "We couldn't write new questions on those topics from this document. Try practising the whole document again instead.",
        { code: "GROUNDING_FAILED" },
      );
    }

    const examId = createExamWithQuestions({
      userEmail,
      documentId: source.documentId,
      modelId: source.modelId,
      examMode: source.examMode,
      questionType,
      requestedCount: count,
      droppedCount: result.dropped.reduce((sum, d) => sum + d.count, 0),
      scopeFromPage: source.scopeFromPage,
      scopeToPage: source.scopeToPage,
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
      eventType: "exam_retry_topics",
      modelId: source.modelId,
      latencyMs: Date.now() - startedAt,
      outcome: "ready",
    });

    return NextResponse.json(
      {
        examId,
        deliveredCount: result.questions.length,
        requestedCount: count,
        coverage: describeCoverage(result.coverage, document.pageCount),
        shortfall: describeShortfall(count, result.questions.length),
      },
      { status: 201 },
    );
  } catch (err) {
    logger.error({ module: MODULE, err: String(err) }, "topic retry failed");
    return errorResponse(
      500,
      "Something went wrong building that practice set. Try again.",
    );
  }
}
