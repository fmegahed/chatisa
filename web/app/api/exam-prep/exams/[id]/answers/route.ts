import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import {
  advanceExam,
  getAnswerForQuestion,
  getExamQuestions,
  getOwnedExam,
  recordUsageEvent,
  saveAnswer,
} from "@/lib/db";
import {
  bandFor,
  gradeMultipleChoice,
  gradeWrittenAnswer,
  type GradedAnswer,
} from "@/lib/exam/grade";
import { CHAT_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";

export const maxDuration = 120;

const MODULE = "exam_ally";

const answerSchema = z.object({
  questionId: z.uuid(),
  selectedIndex: z.number().int().min(0).max(9).nullable().optional(),
  responseText: z.string().max(8_000).nullable().optional(),
  confidence: z.enum(["guessing", "fairly_sure", "confident"]).nullable().optional(),
});

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const exam = getOwnedExam((await params).id, userEmail);
  if (!exam) return errorResponse(404, "That exam could not be found.");
  if (exam.status === "completed") {
    return errorResponse(409, "This exam was already submitted.");
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Request body must be JSON.");
  }
  const parsed = answerSchema.safeParse(body);
  if (!parsed.success) {
    return errorResponse(400, "That request wasn't valid.", {
      fields: [...new Set(parsed.error.issues.map((i) => i.path.join(".")))],
    });
  }
  const input = parsed.data;

  // Questions are only ever reached through an exam the caller owns.
  const questions = getExamQuestions(exam.id);
  const question = questions.find((q) => q.id === input.questionId);
  if (!question) return errorResponse(404, "That question could not be found.");

  // Re-submitting returns what was already recorded, rather than grading
  // again: cheaper, and the student sees one consistent result.
  const existing = getAnswerForQuestion(question.id);
  if (existing?.gradedBy) {
    return NextResponse.json(buildResponse(exam.examMode, question, existing));
  }

  const isMcq = question.type === "multiple_choice";
  if (isMcq && (input.selectedIndex === null || input.selectedIndex === undefined)) {
    return errorResponse(400, "Choose an answer before submitting.");
  }
  if (!isMcq && !input.responseText?.trim()) {
    return errorResponse(400, "Write an answer before submitting.");
  }

  const rubric = JSON.parse(question.rubricJson) as {
    criterion: string;
    points: number;
  }[];

  let graded: GradedAnswer;
  if (isMcq) {
    graded = gradeMultipleChoice(
      input.selectedIndex ?? null,
      question.correctIndex,
      question.pointsPossible,
    );
  } else {
    const limit = checkRateLimit(`exam-grade:${userEmail}`, CHAT_RATE_LIMIT);
    if (!limit.allowed) {
      return errorResponse(
        429,
        "You've submitted a lot of answers very quickly. Wait a moment and try again.",
        { retryAfterSeconds: limit.retryAfterSeconds },
      );
    }
    try {
      graded = await gradeWrittenAnswer({
        modelId: exam.modelId,
        stem: question.stem,
        modelAnswer: question.modelAnswer,
        rubric,
        responseText: input.responseText ?? "",
        pointsPossible: question.pointsPossible,
      });
    } catch (err) {
      // A grading failure is ours, not the student's. The answer is kept and
      // the question is excluded from the score rather than marked zero.
      logger.error({ module: MODULE, err: String(err) }, "grading failed");
      graded = {
        gradedBy: "failed",
        isCorrect: null,
        pointsAwarded: null,
        criteria: [],
        feedback: "",
      };
    }
  }

  saveAnswer({
    examId: exam.id,
    questionId: question.id,
    selectedIndex: input.selectedIndex ?? null,
    responseText: input.responseText ?? null,
    confidence: input.confidence ?? null,
    gradedBy: graded.gradedBy,
    graderModelId: graded.gradedBy === "model" ? exam.modelId : null,
    isCorrect: graded.isCorrect,
    pointsAwarded: graded.pointsAwarded,
    criteria: graded.criteria.length > 0 ? graded.criteria : null,
    feedback: graded.feedback || null,
  });

  const answeredCount = questions.filter(
    (q) => q.id === question.id || getAnswerForQuestion(q.id),
  ).length;
  const complete = answeredCount >= questions.length;
  advanceExam({
    examId: exam.id,
    position: Math.min(question.position + 1, questions.length - 1),
    complete,
  });

  recordUsageEvent({
    userEmail,
    module: MODULE,
    eventType: isMcq ? "answer_graded_local" : "answer_graded_model",
    modelId: isMcq ? null : exam.modelId,
    outcome: graded.gradedBy,
  });

  const stored = getAnswerForQuestion(question.id);
  return NextResponse.json({
    ...buildResponse(exam.examMode, question, stored),
    complete,
  });
}

type StoredQuestion = ReturnType<typeof getExamQuestions>[number];
type StoredAnswer = NonNullable<ReturnType<typeof getAnswerForQuestion>>;

/**
 * Exam mode records the answer and says nothing more. Practice mode returns
 * the result, which is the whole point of answering one question at a time.
 */
function buildResponse(
  examMode: string,
  question: StoredQuestion,
  answer: StoredAnswer | undefined,
) {
  if (!answer) return { recorded: true };
  if (examMode === "exam") {
    return { recorded: true, questionId: question.id };
  }
  const criteria = answer.criteriaJson
    ? (JSON.parse(answer.criteriaJson) as unknown[])
    : [];
  return {
    recorded: true,
    questionId: question.id,
    gradedBy: answer.gradedBy,
    isCorrect: answer.isCorrect === null ? null : answer.isCorrect === 1,
    // Written answers report a band rather than a percentage (ADR-016).
    band:
      question.type === "multiple_choice" || answer.pointsAwarded === null
        ? null
        : bandFor(answer.pointsAwarded, question.pointsPossible),
    pointsAwarded:
      question.type === "multiple_choice" ? answer.pointsAwarded : null,
    pointsPossible:
      question.type === "multiple_choice" ? question.pointsPossible : null,
    criteria,
    feedback: answer.feedback,
    explanation: question.explanation,
    modelAnswer: question.modelAnswer,
    correctIndex: question.correctIndex,
    sourcePage: question.sourcePage,
    sourceQuote: question.sourceQuote,
  };
}
