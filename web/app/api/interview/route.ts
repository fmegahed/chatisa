import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import { getPageModels, MODELS } from "@/lib/config/models";
import { isModelAvailable } from "@/lib/providers";
import { classifyProviderFailure } from "@/lib/providers/errors";
import {
  appendInterviewQuestion,
  createInterview,
  listInterviews,
  recordUsageEvent,
} from "@/lib/db";
import { buildBriefs, nextQuestion } from "@/lib/interview/engine";
import { fetchJobPosting } from "@/lib/jobs/fetch-posting";
import { readResumePdf } from "@/lib/jobs/read-resume";
import { NoReadableTextError, UploadTooLargeError } from "@/lib/exam/upload";
import { PdfError } from "@/lib/exam/pdf";
import { PdfBusyError } from "@/lib/exam/pdf-pool";
import { EXAM_GENERATE_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";

export const runtime = "nodejs";
export const maxDuration = 120;

const MODULE = "interview_mentor";
const MODULE_MODELS = "interview_mentor_transcription";

const startSchema = z.object({
  modelId: z.string().min(1),
  interviewType: z.enum(["behavioral", "technical", "case", "mixed"]),
  company: z.string().min(1).max(160),
  jobTitle: z.string().min(2).max(120),
  jobUrl: z.string().max(2_000).nullable().optional(),
  postingText: z.string().max(60_000).nullable().optional(),
  // Year and major stay optional (user decision, 2026-07-21).
  gradeLevel: z.string().max(60).nullable().optional(),
  major: z.string().max(120).nullable().optional(),
  questionCount: z.coerce.number().int().min(3).max(10),
});

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

/** Past interviews, so an unfinished one can be resumed. */
export async function GET() {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  return NextResponse.json({
    interviews: listInterviews(userEmail).map((row) => ({
      id: row.id,
      jobTitle: row.jobTitle,
      interviewType: row.interviewType,
      status: row.status,
      plannedQuestions: row.plannedQuestions,
      askedCount: row.askedCount,
      createdAt: row.createdAt,
      completedAt: row.completedAt,
    })),
  });
}

/** Starts an interview and asks the first question. */
export async function POST(req: Request) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return errorResponse(400, "That request wasn't valid.");
  }

  const parsed = startSchema.safeParse({
    modelId: form.get("modelId"),
    interviewType: form.get("interviewType"),
    company: form.get("company"),
    jobTitle: form.get("jobTitle"),
    jobUrl: (form.get("jobUrl") as string) || null,
    postingText: (form.get("postingText") as string) || null,
    gradeLevel: (form.get("gradeLevel") as string) || null,
    major: (form.get("major") as string) || null,
    questionCount: form.get("questionCount"),
  });
  if (!parsed.success) {
    return errorResponse(400, "Fill in the company, the job title, and the number of questions.", {
      fields: [...new Set(parsed.error.issues.map((i) => i.path.join(".")))],
    });
  }
  const input = parsed.data;

  // Server-authoritative model policy, as everywhere else.
  if (!getPageModels(MODULE_MODELS).includes(input.modelId)) {
    return errorResponse(400, "That model isn't available for this module.");
  }
  if (!isModelAvailable(input.modelId)) {
    return errorResponse(
      503,
      "That model isn't configured on this server right now. Pick another model.",
    );
  }

  const limit = checkRateLimit(`interview:${userEmail}`, EXAM_GENERATE_RATE_LIMIT);
  if (!limit.allowed) {
    return errorResponse(
      429,
      "You've started a lot of interviews in a short time. Wait a moment and try again.",
      { retryAfterSeconds: limit.retryAfterSeconds },
    );
  }

  // The resume is required and arrives as a PDF: same reader as JobApp Assistant.
  const resumeFile = form.get("resume");
  if (!(resumeFile instanceof File) || resumeFile.size === 0) {
    return errorResponse(
      400,
      "Upload your resume as a PDF. The interview is built around your real background.",
    );
  }
  let resumeText: string;
  try {
    const bytes = new Uint8Array(await resumeFile.arrayBuffer());
    resumeText = (await readResumePdf({ filename: resumeFile.name, bytes })).text;
  } catch (err) {
    if (err instanceof UploadTooLargeError || err instanceof NoReadableTextError) {
      return errorResponse(400, err.message);
    }
    if (err instanceof PdfBusyError) {
      return errorResponse(503, "The server is busy reading other documents. Try again shortly.");
    }
    if (err instanceof PdfError) return errorResponse(400, err.message);
    logger.error({ err: String(err) }, "interview resume read failed");
    return errorResponse(500, "That resume could not be read. Try a different PDF.");
  }

  // The job is required too: pasted text wins over a link, and if the link
  // could not be read the student is told to paste (same logic as JobApp).
  let postingText = input.postingText?.trim() || null;
  let postingMessage: string | null = null;
  if (!postingText && input.jobUrl) {
    const fetched = await fetchJobPosting(input.jobUrl);
    postingMessage = fetched.message;
    if (fetched.text) postingText = fetched.text;
  }
  if (!postingText) {
    return errorResponse(
      400,
      postingMessage ??
        "Add the job description, by link or by pasting it. The questions are built around the real posting.",
    );
  }

  const startedAt = Date.now();
  try {
    // The resume and posting are condensed here and then discarded. Only the
    // short briefs are stored, so no standing copy of a student's personal
    // history accumulates on the server.
    const briefs = await buildBriefs({
      modelId: input.modelId,
      resumeText,
      jobDescription: postingText,
    });

    const interviewId = createInterview({
      userEmail,
      modelId: input.modelId,
      interviewType: input.interviewType,
      jobTitle: input.jobTitle,
      roleBrief: briefs.roleBrief,
      candidateBrief: briefs.candidateBrief,
      gradeLevel: input.gradeLevel ?? null,
      major: input.major ?? null,
      plannedQuestions: input.questionCount,
    });

    const first = await nextQuestion({
      modelId: input.modelId,
      interviewType: input.interviewType,
      jobTitle: input.jobTitle,
      roleBrief: briefs.roleBrief,
      candidateBrief: briefs.candidateBrief,
      gradeLevel: input.gradeLevel ?? null,
      major: input.major ?? null,
      plannedQuestions: input.questionCount,
      history: [],
    });

    appendInterviewQuestion({
      interviewId,
      ordinal: 1,
      question: first.question,
      topic: first.topic,
    });

    recordUsageEvent({
      userEmail,
      module: MODULE,
      eventType: "interview_started",
      modelId: input.modelId,
      provider: MODELS[input.modelId]?.provider ?? null,
      latencyMs: Date.now() - startedAt,
      outcome: "ok",
    });

    return NextResponse.json({
      interviewId,
      briefUsed: Boolean(briefs.roleBrief || briefs.candidateBrief),
      // Resume and posting are always present now, so a brief was always
      // requested; briefUsed says whether it could actually be summarised.
      briefRequested: true,
    });
  } catch (err) {
    const failure = classifyProviderFailure(err);
    logger.error(
      { err: String(err), failureKind: failure.kind, operatorAction: failure.operatorAction },
      failure.operatorAction
        ? "interview start failed: needs operator attention"
        : "interview start failed",
    );
    recordUsageEvent({
      userEmail,
      module: MODULE,
      eventType: "interview_error",
      modelId: input.modelId,
      latencyMs: Date.now() - startedAt,
      outcome: failure.kind,
    });
    return errorResponse(502, failure.message);
  }
}
