/**
 * Pure PDF text helpers with no dependencies, shared by the main thread and
 * the PDF worker. Kept as plain ESM so the worker (which runs outside the
 * bundler) and the typed application code use one implementation, not two.
 */

/** Below this many characters, a page is assumed to be image-only. */
export const MIN_CHARS_FOR_TEXT_PAGE = 50;

/**
 * Tidies extracted text without changing its meaning. Grounding compares
 * extracted text against extracted text, so both sides get this treatment.
 */
export function normalizePageText(raw) {
  return raw
    .replace(/\r\n?/g, "\n")
    .replace(/­/g, "") // soft hyphen
    .replace(/ﬁ/g, "fi")
    .replace(/ﬂ/g, "fl")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

/** Turns one raw page string into a classified page record. */
export function classifyPage(pageNumber, raw) {
  const cleaned = normalizePageText(raw ?? "");
  const usable = cleaned.length >= MIN_CHARS_FOR_TEXT_PAGE;
  return {
    pageNumber,
    text: usable ? cleaned : "",
    charCount: usable ? cleaned.length : 0,
    source: usable ? "text" : "needs_vision",
  };
}

/** How the document as a whole reads. */
export function classifyDocument(pages) {
  if (pages.length === 0) return "scanned";
  const textPages = pages.filter((p) => p.source === "text").length;
  if (textPages === 0) return "scanned";
  if (textPages === pages.length) return "text";
  return "mixed";
}

/** True when the bytes start with the PDF magic number. */
export function looksLikePdf(bytes) {
  return (
    bytes.length >= 5 &&
    bytes[0] === 0x25 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x44 &&
    bytes[3] === 0x46 &&
    bytes[4] === 0x2d
  );
}
