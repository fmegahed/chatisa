/**
 * Server-side PDF reading for Exam Ally.
 *
 * Two reading paths, chosen automatically per page (ADR-014):
 *  - pages with usable selectable text are read directly;
 *  - pages without are rendered and transcribed by a vision model.
 *
 * All parsing and rasterization runs in a worker pool, never on the request
 * thread, because both block whichever thread they run on and this app is
 * meant to serve many students at once. The uploaded file is never written to
 * disk (ADR-015).
 */
import {
  MIN_CHARS_FOR_TEXT_PAGE,
  classifyDocument,
  classifyPage,
  looksLikePdf,
  normalizePageText,
  type PageSource,
  type PdfPage,
} from "./pdf-core";
import {
  PdfBusyError,
  PdfWorkerError,
  processPdfInWorker,
  type ProcessedPdf,
} from "./pdf-pool";

export {
  MIN_CHARS_FOR_TEXT_PAGE,
  classifyDocument,
  classifyPage,
  looksLikePdf,
  normalizePageText,
  PdfBusyError,
  PdfWorkerError,
  processPdfInWorker,
};
export type { PageSource, PdfPage, ProcessedPdf };

export type ExtractionWarning =
  | "TRUNCATED_PAGES"
  | "TRUNCATED_CHARS"
  | "DEADLINE_REACHED"
  | "NONSTANDARD_ENCODING";

export interface ExtractedPdf {
  pageCount: number;
  pages: PdfPage[];
  classification: "text" | "mixed" | "scanned";
  warnings: string[];
  /** Pre-rendered images for pages awaiting transcription. */
  images: { pageNumber: number; png: Uint8Array }[];
  /** Pages needing vision that exceeded the per-document cap. */
  skippedVisionPages: number[];
}

export class PdfError extends Error {
  constructor(
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "PdfError";
  }
}

/**
 * Reads a PDF off the request thread. Rejects with PdfError for input the
 * student needs to fix, and with PdfBusyError when the server is saturated.
 */
export async function readPdf(
  bytes: Uint8Array,
  opts: { maxVisionPages?: number; deadlineMs?: number } = {},
): Promise<ExtractedPdf> {
  if (!looksLikePdf(bytes)) {
    throw new PdfError("NOT_A_PDF", "That file is not a PDF.");
  }
  try {
    return await processPdfInWorker({
      bytes,
      maxVisionPages: opts.maxVisionPages,
      deadlineMs: opts.deadlineMs,
    });
  } catch (err) {
    if (err instanceof PdfWorkerError) {
      throw new PdfError(err.code, err.message);
    }
    throw err;
  }
}

/** Pages still awaiting transcription, in order. */
export function pagesNeedingVision(extracted: {
  pages: PdfPage[];
}): number[] {
  return extracted.pages
    .filter((p) => p.source === "needs_vision")
    .map((p) => p.pageNumber);
}
