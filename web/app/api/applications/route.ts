import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import {
  createJobApplication,
  listApplications,
  recordUsageEvent,
} from "@/lib/db";
import { fetchJobPosting } from "@/lib/jobs/fetch-posting";
import { readResumePdf } from "@/lib/jobs/read-resume";
import {
  parsePublishedWork,
  publishedWorkBlock,
} from "@/lib/jobs/published-work";
import { NoReadableTextError, UploadTooLargeError } from "@/lib/exam/upload";
import { PdfError } from "@/lib/exam/pdf";
import { PdfBusyError } from "@/lib/exam/pdf-pool";
import { EXAM_UPLOAD_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";

export const runtime = "nodejs";
export const maxDuration = 300;

const MODULE = "jobapp_assistant";

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

export async function GET() {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  return NextResponse.json({
    applications: listApplications(userEmail).map((a) => ({
      id: a.id,
      company: a.company,
      positionTitle: a.positionTitle,
      jobUrl: a.jobUrl,
      descriptionSource: a.descriptionSource,
      hasResume: a.resumeText !== null,
      createdAt: a.createdAt,
    })),
  });
}

const createSchema = z.object({
  company: z.string().min(1).max(160),
  positionTitle: z.string().min(2).max(160),
  jobUrl: z.string().max(2_000).nullable().optional(),
  postingText: z.string().max(60_000).nullable().optional(),
});

/**
 * Creates a job application from a multipart form: the job details, an optional
 * resume PDF, and an optional link to fetch.
 */
export async function POST(req: Request) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const limit = checkRateLimit(`application:${userEmail}`, EXAM_UPLOAD_RATE_LIMIT);
  if (!limit.allowed) {
    return errorResponse(429, "Too many uploads in a short time. Wait a moment.", {
      retryAfterSeconds: limit.retryAfterSeconds,
    });
  }

  let form: FormData;
  try {
    form = await req.formData();
  } catch {
    return errorResponse(400, "That request wasn't valid.");
  }

  const parsed = createSchema.safeParse({
    company: form.get("company"),
    positionTitle: form.get("positionTitle"),
    jobUrl: (form.get("jobUrl") as string) || null,
    postingText: (form.get("postingText") as string) || null,
  });
  if (!parsed.success) {
    return errorResponse(400, "Fill in the company and the position title.");
  }
  const input = parsed.data;

  // The posting: pasted text always wins over a link, because the student
  // pasting it means they have seen it and we have not.
  let postingText = input.postingText?.trim() || null;
  let descriptionSource = postingText ? "pasted" : "none";
  // Job Scout prefills carry their provenance; honored only when posting
  // text actually came along, so a client cannot label thin air (2026-07-28).
  if (postingText && form.get("postingSource") === "job_scout") {
    descriptionSource = "job_scout";
  }
  let postingMessage: string | null = null;

  if (!postingText && input.jobUrl) {
    const fetched = await fetchJobPosting(input.jobUrl);
    postingMessage = fetched.message;
    if (fetched.text) {
      postingText = fetched.text;
      descriptionSource = "fetched";
    }
  }

  // The resume.
  let resumeText: string | null = null;
  let resumeFilename: string | null = null;
  let resumeNote: string | null = null;
  const file = form.get("resume");

  if (file instanceof File && file.size > 0) {
    try {
      const bytes = new Uint8Array(await file.arrayBuffer());
      const read = await readResumePdf({ filename: file.name, bytes });
      resumeText = read.text;
      resumeFilename = read.filename;
      if (read.visionPageCount > 0) {
        resumeNote = `Your resume looked like a scan, so ${read.visionPageCount === 1 ? "one page was" : `${read.visionPageCount} pages were`} read as an image. Check the wording carefully before you send anything.`;
      }
    } catch (err) {
      if (err instanceof UploadTooLargeError || err instanceof NoReadableTextError) {
        return errorResponse(400, err.message);
      }
      if (err instanceof PdfBusyError) {
        return errorResponse(503, "The server is busy reading other documents. Try again shortly.");
      }
      if (err instanceof PdfError) return errorResponse(400, err.message);
      logger.error({ err: String(err) }, "resume read failed");
      return errorResponse(500, "That resume could not be read. Try a different PDF.");
    }
  }

  // Opt-in only, and only ever what the browser sent: published sites live
  // in the student's browser, so this is the one place they reach a draft.
  // It rides along with the resume text rather than becoming a new record,
  // and it goes FIRST: lib/documents/generate.ts fences the resume text at
  // 20,000 characters, and a long resume would otherwise cut the block away.
  const published = parsePublishedWork(form.get("publishedWork") as string | null);
  if (published.length > 0) {
    const block = publishedWorkBlock(published);
    resumeText = resumeText ? `${block}\n\n${resumeText}` : block;
    if (!resumeFilename) resumeFilename = "published-work.txt";
  }

  const id = createJobApplication({
    userEmail,
    company: input.company.trim(),
    positionTitle: input.positionTitle.trim(),
    jobUrl: input.jobUrl?.trim() || null,
    descriptionSource,
    postingText,
    resumeText,
    resumeFilename,
    roleBrief: null,
    candidateBrief: null,
  });

  recordUsageEvent({
    userEmail,
    module: MODULE,
    eventType: "application_created",
    outcome: descriptionSource,
  });

  return NextResponse.json({
    applicationId: id,
    descriptionSource,
    postingMessage,
    resumeNote,
    hasPosting: postingText !== null,
    hasResume: resumeText !== null,
  });
}
