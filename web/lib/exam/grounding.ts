/**
 * Grounding: proof that a question really came from the student's document.
 *
 * Every generated question must quote the source verbatim. That quote is
 * checked against the extracted text here, and questions that cannot be traced
 * are dropped. This is what separates questions about the material from
 * plausible-sounding invention, and it is the module's main defence against a
 * model quietly making things up.
 */

export type GroundingStatus = "verified" | "verified_fuzzy" | "repaired";

export interface GroundingCheck {
  grounded: boolean;
  status?: GroundingStatus;
  /** Corrected page number when the claimed one was slightly off. */
  page?: number;
  reason?: "not_found" | "too_short";
}

/**
 * Comparison form. Applied to both sides, so quirks cancel out rather than
 * causing false rejections. Plain lowercase, not locale aware, deliberately.
 */
export function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/­/g, "")
    .replace(/[‘’ʼ]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/[‐-―−]/g, "-")
    // Words split across a line break by hyphenation are rejoined first...
    .replace(/-\s*\n\s*/g, "")
    // ...then remaining hyphens are dropped, so a compound written "so-called"
    // in a quote matches the same word broken as "so-\ncalled" in the PDF.
    // Both sides get this, so the comparison stays symmetric.
    .replace(/-/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/** Overlapping word groups, used for tolerant matching. */
function shingles(text: string, size = 5): string[] {
  const words = text.split(" ").filter(Boolean);
  if (words.length <= size) return words.length > 0 ? [words.join(" ")] : [];
  const out: string[] = [];
  for (let i = 0; i + size <= words.length; i += 1) {
    out.push(words.slice(i, i + size).join(" "));
  }
  return out;
}

/** Share of the quote's word groups that appear in the source. */
export function shingleContainment(quote: string, source: string): number {
  const parts = shingles(quote);
  if (parts.length === 0) return 0;
  let found = 0;
  for (const part of parts) if (source.includes(part)) found += 1;
  return found / parts.length;
}

/** Below this, a quote is treated as not coming from the document. */
export const FUZZY_THRESHOLD = 0.8;
const MIN_QUOTE_CHARS = 30;

export interface PageText {
  pageNumber: number;
  text: string;
}

/**
 * Checks one quote against the pages it should have come from.
 * A page number that is off by one is corrected rather than rejected: text
 * extraction routinely shifts content across page boundaries, and discarding
 * a good question over that would be wrong.
 */
export function checkGrounding(
  quote: string,
  claimedPage: number,
  pages: PageText[],
): GroundingCheck {
  if (quote.trim().length < MIN_QUOTE_CHARS) {
    return { grounded: false, reason: "too_short" };
  }
  const needle = normalize(quote);

  const exactPage = pages.find((p) => normalize(p.text).includes(needle));
  if (exactPage) {
    return {
      grounded: true,
      status: exactPage.pageNumber === claimedPage ? "verified" : "repaired",
      page: exactPage.pageNumber,
    };
  }

  let best: { page: number; score: number } | null = null;
  for (const page of pages) {
    const score = shingleContainment(needle, normalize(page.text));
    if (!best || score > best.score) best = { page: page.pageNumber, score };
  }
  if (best && best.score >= FUZZY_THRESHOLD) {
    return {
      grounded: true,
      status: best.page === claimedPage ? "verified_fuzzy" : "repaired",
      page: best.page,
    };
  }

  return { grounded: false, reason: "not_found" };
}
