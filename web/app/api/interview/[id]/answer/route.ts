import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import { MODELS } from "@/lib/config/models";
import { classifyProviderFailure } from "@/lib/providers/errors";
import {
  appendInterviewQuestion,
  completeInterview,
  getInterviewTurns,
  getOwnedInterview,
  recordUsageEvent,
  saveInterviewAnswer,
} from "@/lib/db";
import { buildSummary, judgeAnswer, nextQuestion } from "@/lib/interview/engine";
import { projectInterview } from "@/lib/interview/projection";
import { isEmptyAnswer } from "@/lib/interview/scoring";
import type { InterviewType } from "@/lib/prompts/interview-mentor";

export const runtime = "nodejs";
export const maxDuration = 120;

const MODULE = "interview_mentor";

const answerSchema = z.object({
  answerText: z.string().max(20_000),
  answerSource: z.enum(["spoken", "typed", "skipped"]),
  answerSeconds: z.number().int().min(0).max(3_600).nullable().optional(),
});

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

/**
 * Records an answer, judges it, and asks the next question.
 *
 * The judgement is stored but not returned while the interview is running:
 * telling a student mid-interview that their last answer was weak changes how
 * they answer the rest, which makes the practice less like the real thing. It
 * is released in full the moment they finish.
 */
export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const { id } = await params;
  const interview = getOwnedInterview(id, userEmail);
  if (!interview) return errorResponse(404, "That interview could not be found.");
  if (interview.status === "completed") {
    return errorResponse(409, "This interview is already finished.");
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Request body must be JSON.");
  }
  const parsed = answerSchema.safeParse(body);
  if (!parsed.success) return errorResponse(400, "That request wasn't valid.");
  const input = parsed.data;

  const turns = getInterviewTurns(id);
  const current = turns.find((t) => t.answeredAt === null);
  if (!current) {
    return errorResponse(409, "There is no question waiting for an answer.");
  }

  const startedAt = Date.now();
  const skipped = input.answerSource === "skipped" || isEmptyAnswer(input.answerText);

  try {
    // An empty or skipped answer is recorded as such without a model call.
    // Asking a model to judge nothing wastes money and invites it to invent
    // merit that is not there.
    if (skipped) {
      saveInterviewAnswer({
        turnId: current.id,
        answerText: input.answerText.trim() || null,
        answerSource: "skipped",
        answerSeconds: input.answerSeconds ?? null,
        // Null, not an all-not-met judgement. Storing verdicts here would make
        // the rollup treat a skip as an answered question that failed every
        // criterion, which would misreport a student who simply ran out of time.
        criteriaJson: null,
        strength: null,
        improvement: null,
      });
    } else {
      const judgement = await judgeAnswer({
        modelId: interview.modelId,
        question: current.question,
        answer: input.answerText,
      });
      saveInterviewAnswer({
        turnId: current.id,
        answerText: input.answerText,
        answerSource: input.answerSource,
        answerSeconds: input.answerSeconds ?? null,
        criteriaJson: JSON.stringify(judgement.criteria),
        strength: judgement.strength,
        improvement: judgement.improvement,
      });
    }

    const answeredHistory = getInterviewTurns(id).map((t) => ({
      question: t.question,
      answer: t.answerText,
    }));

    const isLast = interview.askedCount >= interview.plannedQuestions;

    if (isLast) {
      const summary = await buildSummary({
        modelId: interview.modelId,
        jobTitle: interview.jobTitle,
        history: answeredHistory,
      });
      completeInterview(id, JSON.stringify(summary));
    } else {
      const question = await nextQuestion({
        modelId: interview.modelId,
        interviewType: interview.interviewType as InterviewType,
        jobTitle: interview.jobTitle,
        roleBrief: interview.roleBrief,
        candidateBrief: interview.candidateBrief,
        gradeLevel: interview.gradeLevel,
        major: interview.major,
        plannedQuestions: interview.plannedQuestions,
        history: answeredHistory,
      });
      appendInterviewQuestion({
        interviewId: id,
        ordinal: interview.askedCount + 1,
        question: question.question,
        topic: question.topic,
      });
    }

    recordUsageEvent({
      userEmail,
      module: MODULE,
      eventType: isLast ? "interview_completed" : "interview_answer",
      modelId: interview.modelId,
      provider: MODELS[interview.modelId]?.provider ?? null,
      latencyMs: Date.now() - startedAt,
      responseChars: input.answerText.length,
      outcome: skipped ? "skipped" : "ok",
    });

    const fresh = getOwnedInterview(id, userEmail)!;
    return NextResponse.json({
      interview: projectInterview(fresh, getInterviewTurns(id)),
    });
  } catch (err) {
    const failure = classifyProviderFailure(err);
    logger.error(
      {
        err: String(err),
        failureKind: failure.kind,
        operatorAction: failure.operatorAction,
      },
      failure.operatorAction
        ? "interview answer failed: needs operator attention"
        : "interview answer failed",
    );
    // The answer may already be saved. Say so, because a student who thinks
    // their answer was lost will retype it and get a duplicate.
    return errorResponse(502, failure.message, { answerSaved: true });
  }
}
