/**
 * Upload handling for study documents: validate, read, transcribe scanned
 * pages, persist. Kept out of the route handler so the rules are unit
 * testable and the route stays a thin HTTP wrapper.
 */
import { MAX_VISION_PAGES, mergeTranscriptions, transcribeScannedPages } from "./vision";
import { PdfError, pagesNeedingVision, readPdf, type ExtractedPdf } from "./pdf";
import { createExamDocument } from "@/lib/db";

/** Largest upload accepted, before reading the body into memory. */
export const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

export interface UploadOutcome {
  documentId: string;
  filename: string;
  pageCount: number;
  textPageCount: number;
  visionPageCount: number;
  charCount: number;
  classification: ExtractedPdf["classification"];
  warnings: string[];
  /** Pages transcribed by a vision model, so the UI can label them. */
  transcribedPages: number[];
  /** Pages that needed transcription but were past the per-document cap. */
  skippedPages: number[];
  /** Pages that were too unclear to read. */
  illegiblePages: number[];
}

/**
 * Reduces a client-supplied filename to something safe to store and show.
 * It is display only: nothing ever uses it as a path.
 */
export function safeFilename(raw: string): string {
  const base = raw.split(/[\\/]/).pop() ?? "document.pdf";
  const cleaned = base
    // Strip control characters only; spaces and punctuation are kept.
    .replace(/[\u0000-\u001f\u007f]/g, "")
    .replace(/\s+/g, " ")
    .trim();
  const limited = cleaned.slice(0, 120);
  return limited.length > 0 ? limited : "document.pdf";
}

/** Human-readable size, used in the too-large message. */
export function formatBytes(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(0)} MB`;
}

export class UploadTooLargeError extends Error {
  constructor() {
    super(
      `That file is larger than ${formatBytes(MAX_UPLOAD_BYTES)}. Upload a smaller PDF, or split the chapters you need.`,
    );
    this.name = "UploadTooLargeError";
  }
}

export class NoReadableTextError extends Error {
  constructor(public readonly hadVisionAttempt: boolean) {
    super(
      hadVisionAttempt
        ? "We could not read any text from that PDF, even from the page images. If it is a photo or scan, try a clearer copy."
        : "That PDF has no readable text. If it is a scan, upload a clearer copy so the pages can be read.",
    );
    this.name = "NoReadableTextError";
  }
}

/**
 * Full upload pipeline. Parsing and rasterization run in the worker pool, so
 * this never blocks the request thread with CPU work.
 */
export async function ingestUpload(params: {
  userEmail: string;
  filename: string;
  bytes: Uint8Array;
  maxVisionPages?: number;
}): Promise<UploadOutcome> {
  if (params.bytes.byteLength > MAX_UPLOAD_BYTES) {
    throw new UploadTooLargeError();
  }
  if (params.bytes.byteLength === 0) {
    throw new PdfError("EMPTY_FILE", "That file is empty.");
  }

  const extracted = await readPdf(params.bytes, {
    maxVisionPages: params.maxVisionPages ?? MAX_VISION_PAGES,
  });

  // Transcribe any pages the PDF itself could not supply text for.
  const needVision = pagesNeedingVision(extracted);
  let merged = extracted;
  let transcribedPages: number[] = [];
  let skippedPages: number[] = [...extracted.skippedVisionPages];
  let illegiblePages: number[] = [];

  if (needVision.length > 0 && extracted.images.length > 0) {
    const transcription = await transcribeScannedPages({ extracted });
    merged = mergeTranscriptions(extracted, transcription);
    transcribedPages = transcription.pages
      .filter((p) => p.text.length > 0)
      .map((p) => p.pageNumber);
    skippedPages = transcription.skippedPages;
    illegiblePages = transcription.illegiblePages;
  }

  const usable = merged.pages.filter((p) => p.text.length > 0);
  if (usable.length === 0) {
    throw new NoReadableTextError(needVision.length > 0);
  }

  const textPageCount = merged.pages.filter((p) => p.source === "text").length;
  const visionPageCount = merged.pages.filter(
    (p) => p.source === "vision",
  ).length;
  const charCount = usable.reduce((sum, p) => sum + p.charCount, 0);
  const filename = safeFilename(params.filename);

  const documentId = createExamDocument({
    userEmail: params.userEmail,
    filename,
    sizeBytes: params.bytes.byteLength,
    pageCount: merged.pageCount,
    textPageCount,
    visionPageCount,
    charCount,
    classification: merged.classification,
    warnings: merged.warnings,
    pages: usable.map((p) => ({
      pageNumber: p.pageNumber,
      text: p.text,
      charCount: p.charCount,
      source: p.source,
    })),
  });

  return {
    documentId,
    filename,
    pageCount: merged.pageCount,
    textPageCount,
    visionPageCount,
    charCount,
    classification: merged.classification,
    warnings: merged.warnings,
    transcribedPages,
    skippedPages,
    illegiblePages,
  };
}
