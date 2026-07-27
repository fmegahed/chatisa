export type PageSource = "text" | "needs_vision" | "vision";

export interface PdfPage {
  pageNumber: number;
  text: string;
  charCount: number;
  source: PageSource;
}

export const MIN_CHARS_FOR_TEXT_PAGE: number;
export function normalizePageText(raw: string): string;
export function classifyPage(pageNumber: number, raw: string): PdfPage;
export function classifyDocument(
  pages: PdfPage[],
): "text" | "mixed" | "scanned";
export function looksLikePdf(bytes: Uint8Array): boolean;
