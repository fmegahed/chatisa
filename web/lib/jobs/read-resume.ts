import "server-only";
import {
  MAX_UPLOAD_BYTES,
  NoReadableTextError,
  UploadTooLargeError,
  safeFilename,
} from "@/lib/exam/upload";
import { PdfError, pagesNeedingVision, readPdf } from "@/lib/exam/pdf";
import {
  mergeTranscriptions,
  transcribeScannedPages,
} from "@/lib/exam/vision";

/**
 * Reading a student's resume PDF.
 *
 * Reuses Exam Ally's extraction primitives directly rather than its
 * `ingestUpload`, because that one persists an exam document: a resume is not
 * course material and should not appear in the student's Exam Ally library. The
 * expensive parts, the worker pool and the automatic vision routing for scanned
 * pages, are shared, so a scanned resume works here for free.
 *
 * A resume is short, so the vision cap is small. A file that needs 40 pages
 * transcribed is not a resume.
 */

const MAX_RESUME_VISION_PAGES = 4;

export interface ResumeReadResult {
  filename: string;
  text: string;
  pageCount: number;
  /** Pages that had to be read as images, so the UI can say so. */
  visionPageCount: number;
  charCount: number;
}

export async function readResumePdf(params: {
  filename: string;
  bytes: Uint8Array;
}): Promise<ResumeReadResult> {
  if (params.bytes.byteLength > MAX_UPLOAD_BYTES) throw new UploadTooLargeError();
  if (params.bytes.byteLength === 0) {
    throw new PdfError("EMPTY_FILE", "That file is empty.");
  }

  const extracted = await readPdf(params.bytes, {
    maxVisionPages: MAX_RESUME_VISION_PAGES,
  });

  let merged = extracted;
  const needVision = pagesNeedingVision(extracted);
  if (needVision.length > 0 && extracted.images.length > 0) {
    const transcription = await transcribeScannedPages({ extracted });
    merged = mergeTranscriptions(extracted, transcription);
  }

  const usable = merged.pages.filter((p) => p.text.length > 0);
  if (usable.length === 0) {
    throw new NoReadableTextError(needVision.length > 0);
  }

  return {
    filename: safeFilename(params.filename),
    // Line structure is preserved because grounding matches generated bullets
    // against the student's own lines, so the lines have to survive.
    text: usable.map((p) => p.text).join("\n"),
    pageCount: merged.pageCount,
    visionPageCount: merged.pages.filter((p) => p.source === "vision").length,
    charCount: usable.reduce((sum, p) => sum + p.charCount, 0),
  };
}
