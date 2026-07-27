import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import {
  advanceExam,
  getExamAnswers,
  getExamQuestions,
  getOwnedExam,
  purgeDocumentText,
} from "@/lib/db";
import { bandFor } from "@/lib/exam/grade";

/**
 * Finishes an exam and reports results.
 *
 * Questions we failed to grade are excluded from the total rather than scored
 * zero: a provider outage is not the student's mistake. Written answers are
 * reported as bands and per-criterion detail, never a percentage (ADR-016).
 */
export async function POST(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) {
    return NextResponse.json({ error: "Sign in to continue." }, { status: 401 });
  }

  const exam = getOwnedExam((await params).id, userEmail);
  if (!exam) {
    return NextResponse.json(
      { error: "That exam could not be found." },
      { status: 404 },
    );
  }

  const questions = getExamQuestions(exam.id);
  const alreadyFinished = exam.status === "completed";
  advanceExam({
    examId: exam.id,
    position: Math.max(0, questions.length - 1),
    complete: true,
  });

  // The student is done with this material, so its text is dropped now
  // (ADR-015). Retrying weak topics happens before this point, while the
  // text is still available.
  if (!alreadyFinished) purgeDocumentText(exam.documentId);

  const answers = new Map(
    getExamAnswers(exam.id).map((a) => [a.questionId, a]),
  );

  let pointsAwarded = 0;
  let pointsPossible = 0;
  let ungraded = 0;
  const byTopic = new Map<string, { earned: number; possible: number }>();

  const detail = questions.map((question) => {
    const answer = answers.get(question.id);
    const gradable = answer?.gradedBy === "local" || answer?.gradedBy === "model";
    if (!gradable) ungraded += 1;

    if (gradable && answer?.pointsAwarded !== null && answer?.pointsAwarded !== undefined) {
      pointsAwarded += answer.pointsAwarded;
      pointsPossible += question.pointsPossible;
      const topic = byTopic.get(question.topic) ?? { earned: 0, possible: 0 };
      topic.earned += answer.pointsAwarded;
      topic.possible += question.pointsPossible;
      byTopic.set(question.topic, topic);
    }

    return {
      questionId: question.id,
      position: question.position,
      type: question.type,
      stem: question.stem,
      topic: question.topic,
      options: question.optionsJson
        ? (JSON.parse(question.optionsJson) as string[])
        : null,
      correctIndex: question.correctIndex,
      explanation: question.explanation,
      modelAnswer: question.modelAnswer,
      sourcePage: question.sourcePage,
      sourceQuote: question.sourceQuote,
      answered: Boolean(answer),
      gradedBy: answer?.gradedBy ?? null,
      selectedIndex: answer?.selectedIndex ?? null,
      responseText: answer?.responseText ?? null,
      confidence: answer?.confidence ?? null,
      isCorrect: answer?.isCorrect === null || answer?.isCorrect === undefined
        ? null
        : answer.isCorrect === 1,
      band:
        question.type === "multiple_choice" ||
        answer?.pointsAwarded === null ||
        answer?.pointsAwarded === undefined
          ? null
          : bandFor(answer.pointsAwarded, question.pointsPossible),
      criteria: answer?.criteriaJson
        ? (JSON.parse(answer.criteriaJson) as unknown[])
        : [],
      feedback: answer?.feedback ?? null,
    };
  });

  // Weakest topics first: this is what the student should go and review.
  const topics = [...byTopic.entries()]
    .map(([topic, t]) => ({
      topic,
      earned: t.earned,
      possible: t.possible,
      band: bandFor(t.earned, t.possible),
    }))
    .sort((a, b) => a.earned / a.possible - b.earned / b.possible);

  const studyPlan = topics
    .filter((t) => t.band !== "strong")
    .map((t) => {
      const pages = detail
        .filter((d) => d.topic === t.topic)
        .map((d) => d.sourcePage);
      const unique = [...new Set(pages)].sort((a, b) => a - b);
      return {
        topic: t.topic,
        band: t.band,
        pages: unique,
      };
    });

  return NextResponse.json({
    id: exam.id,
    questionType: exam.questionType,
    examMode: exam.examMode,
    // A multiple choice exam has an exact mark; a written one does not.
    exactScore:
      exam.questionType === "multiple_choice" && pointsPossible > 0
        ? { pointsAwarded, pointsPossible }
        : null,
    overallBand: pointsPossible > 0 ? bandFor(pointsAwarded, pointsPossible) : null,
    ungradedCount: ungraded,
    topics,
    studyPlan,
    questions: detail,
  });
}
