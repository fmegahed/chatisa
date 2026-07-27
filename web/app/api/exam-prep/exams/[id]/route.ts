import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { deleteExam, getExamQuestions, getOwnedExam } from "@/lib/db";
import { toClientQuestion } from "@/lib/exam/projection";

/**
 * An exam the student owns. Questions are served only up to the point they
 * have reached, and answer keys only once the exam is finished, so nothing can
 * be read ahead of time.
 */
export async function GET(
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

  const finished = exam.status === "completed";
  const questions = getExamQuestions(exam.id)
    // Never serialize questions the student has not reached.
    .filter((q) => finished || q.position <= exam.currentPosition)
    .map((q) => toClientQuestion(q, finished));

  return NextResponse.json({
    id: exam.id,
    documentId: exam.documentId,
    status: exam.status,
    examMode: exam.examMode,
    questionType: exam.questionType,
    requestedCount: exam.requestedCount,
    deliveredCount: exam.deliveredCount,
    droppedCount: exam.droppedCount,
    currentPosition: exam.currentPosition,
    coverage: JSON.parse(exam.coverageJson) as unknown,
    questions,
  });
}

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) {
    return NextResponse.json({ error: "Sign in to continue." }, { status: 401 });
  }
  const removed = deleteExam((await params).id, userEmail);
  if (!removed) {
    return NextResponse.json(
      { error: "That exam could not be found." },
      { status: 404 },
    );
  }
  return new NextResponse(null, { status: 204 });
}
