import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import { listExamDocuments, recordUsageEvent } from "@/lib/db";
import { PdfBusyError } from "@/lib/exam/pdf-pool";
import { PdfError } from "@/lib/exam/pdf";
import {
  MAX_UPLOAD_BYTES,
  NoReadableTextError,
  UploadTooLargeError,
  formatBytes,
  ingestUpload,
} from "@/lib/exam/upload";
import { EXAM_UPLOAD_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";

/** Reading and transcribing a scanned document can take a while. */
export const maxDuration = 300;

const MODULE = "exam_ally";

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

/** The signed-in student's uploaded documents, newest first. */
export async function GET() {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  return NextResponse.json({
    documents: listExamDocuments(userEmail).map((d) => ({
      id: d.id,
      filename: d.filename,
      pageCount: d.pageCount,
      textPageCount: d.textPageCount,
      visionPageCount: d.visionPageCount,
      charCount: d.charCount,
      classification: d.classification,
      createdAt: d.createdAt,
      textAvailable: d.textPurgedAt === null,
    })),
  });
}

/**
 * Accepts a PDF, reads it off the request thread, transcribes any scanned
 * pages, and stores the text needed to build an exam. The uploaded file is
 * never written to disk.
 */
export async function POST(req: Request) {
  const requestId = crypto.randomUUID();
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const limit = checkRateLimit(`exam-upload:${userEmail}`, EXAM_UPLOAD_RATE_LIMIT);
  if (!limit.allowed) {
    return errorResponse(
      429,
      "You've uploaded several files in a short time. Wait a moment and try again.",
      { retryAfterSeconds: limit.retryAfterSeconds },
    );
  }

  // Content-Length is client controlled, so it is only an early exit. The
  // real size is checked again after reading.
  const declared = Number(req.headers.get("content-length") ?? "0");
  if (Number.isFinite(declared) && declared > MAX_UPLOAD_BYTES * 1.05) {
    return errorResponse(
      413,
      `That file is larger than ${formatBytes(MAX_UPLOAD_BYTES)}. Upload a smaller PDF, or split the chapters you need.`,
    );
  }

  let file: File | null = null;
  try {
    const form = await req.formData();
    const value = form.get("file");
    if (value instanceof File) file = value;
  } catch {
    return errorResponse(400, "That upload could not be read. Try again.");
  }
  if (!file) return errorResponse(400, "Choose a PDF file to upload.");
  if (file.size > MAX_UPLOAD_BYTES) {
    return errorResponse(413, new UploadTooLargeError().message);
  }

  const startedAt = Date.now();
  try {
    const bytes = new Uint8Array(await file.arrayBuffer());
    const outcome = await ingestUpload({
      userEmail,
      filename: file.name,
      bytes,
    });

    recordUsageEvent({
      userEmail,
      module: MODULE,
      eventType: "document_upload",
      latencyMs: Date.now() - startedAt,
      promptChars: outcome.charCount,
      outcome: outcome.classification,
    });
    // Counts and ids only: never document text.
    logger.info(
      {
        requestId,
        module: MODULE,
        pageCount: outcome.pageCount,
        visionPageCount: outcome.visionPageCount,
        latencyMs: Date.now() - startedAt,
      },
      "study document ingested",
    );

    return NextResponse.json(outcome, { status: 201 });
  } catch (err) {
    if (err instanceof PdfBusyError) {
      // Saturated worker pool: ask for a retry rather than queueing forever.
      return errorResponse(503, err.message, { retryAfterSeconds: 15 });
    }
    if (err instanceof UploadTooLargeError) {
      return errorResponse(413, err.message);
    }
    if (err instanceof NoReadableTextError) {
      return errorResponse(422, err.message, { code: "NO_READABLE_TEXT" });
    }
    if (err instanceof PdfError) {
      return errorResponse(422, err.message, { code: err.code });
    }
    logger.error(
      { requestId, module: MODULE, err: String(err) },
      "study document upload failed",
    );
    recordUsageEvent({
      userEmail,
      module: MODULE,
      eventType: "document_upload_error",
      latencyMs: Date.now() - startedAt,
      outcome: "error",
    });
    return errorResponse(
      500,
      "Something went wrong reading that file. Try again.",
    );
  }
}
