/**
 * Checking that a tailored document says only what the student's resume says.
 *
 * This is the guard rail that decides whether the feature is a learning tool or
 * a fabrication machine. A model asked to tailor a resume for a specific job
 * will, unprompted, add plausible experience the student never had: a
 * technology they did not use, a number they never measured, a leadership role
 * they did not hold. Those are the exact claims that fall apart in an
 * interview, which is the one place they must not.
 *
 * The same approach as Exam Ally's question grounding: the model claims a
 * source, and we verify the claim rather than trusting it. Rewording is
 * expected and fine, so the check is on content words rather than exact text.
 */

const STOPWORDS = new Set([
  "a","an","and","are","as","at","be","been","by","for","from","had","has","have",
  "in","into","is","it","its","of","on","or","that","the","their","to","was","were",
  "with","which","this","these","those","using","used","use","other","also","than",
  "then","them","they","we","our","i","my","me","you","your","he","she","his","her",
]);

/** Content words only, lowercased and de-punctuated. */
export function contentWords(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9%$.\s-]/g, " ")
    .split(/\s+/)
    .map((w) => w.replace(/^[.-]+|[.-]+$/g, ""))
    .filter((w) => w.length > 2 && !STOPWORDS.has(w));
}

/** Fraction of the claim's content words that appear in the source. */
export function overlap(claim: string, source: string): number {
  const claimWords = new Set(contentWords(claim));
  if (claimWords.size === 0) return 0;
  const sourceWords = new Set(contentWords(source));
  let hits = 0;
  for (const word of claimWords) if (sourceWords.has(word)) hits += 1;
  return hits / claimWords.size;
}

/**
 * Tailoring is supposed to reword, reorder and sharpen, so this cannot demand
 * close similarity. But a permissive threshold is easy to slip past: "Managed a
 * team of forty consultants" shares "managed" and "team" with almost any
 * resume while inventing the entire substance. Generic verbs are cheap overlap.
 *
 * Set where genuine rewording still passes comfortably (measured at 0.67 for
 * real examples) and generic-word coincidence does not. The error cost is
 * deliberately asymmetric: a false flag asks the student to check a line they
 * wrote, while a false pass sends a fabricated claim to an employer, so this
 * errs toward flagging.
 */
export const OVERLAP_THRESHOLD = 0.55;

/**
 * Numbers get their own check, because an invented number is the most damaging
 * and most common fabrication: "increased efficiency by 40%" reads well and is
 * indefensible if the student never measured it.
 */
export function numbersIn(text: string): string[] {
  return (text.match(/\$?\d[\d,.]*%?/g) ?? [])
    .map((n) => n.replace(/[,$]/g, "").replace(/\.$/, ""))
    .filter((n) => n !== "" && n !== ".");
}

export function hasInventedNumbers(claim: string, source: string): boolean {
  const sourceNumbers = new Set(numbersIn(source));
  return numbersIn(claim).some((n) => !sourceNumbers.has(n));
}

export type GroundingVerdict = "grounded" | "unsupported" | "invented_numbers";

export interface CheckedClaim {
  text: string;
  sourceLine: string | null;
  verdict: GroundingVerdict;
  /** Why, in words a student can act on. */
  note: string | null;
}

/**
 * Checks one generated claim against the student's resume.
 *
 * `resumeText` is the whole resume, used as a fallback when the model named no
 * source line: a bullet may legitimately draw on several lines at once.
 */
export function checkClaim(
  text: string,
  sourceLine: string | null,
  resumeText: string,
): CheckedClaim {
  const source = sourceLine && sourceLine.trim() !== "" ? sourceLine : resumeText;

  if (overlap(text, source) < OVERLAP_THRESHOLD) {
    return {
      text,
      sourceLine,
      verdict: "unsupported",
      note: "We could not match this to anything in your resume. Edit it so it describes something you actually did, or remove it.",
    };
  }

  if (hasInventedNumbers(text, resumeText)) {
    return {
      text,
      sourceLine,
      verdict: "invented_numbers",
      note: "This includes a figure that is not in your resume. Replace it with a number you can stand behind, or take the number out.",
    };
  }

  return { text, sourceLine, verdict: "grounded", note: null };
}

export interface GroundingReport {
  checked: number;
  grounded: number;
  flagged: CheckedClaim[];
}

export function checkClaims(
  claims: { text: string; sourceLine: string | null }[],
  resumeText: string,
): GroundingReport {
  const results = claims.map((c) => checkClaim(c.text, c.sourceLine, resumeText));
  const flagged = results.filter((r) => r.verdict !== "grounded");
  return {
    checked: results.length,
    grounded: results.length - flagged.length,
    flagged,
  };
}

/** Plain sentence for the student about what needs their attention. */
export function describeGrounding(report: GroundingReport): string | null {
  if (report.flagged.length === 0) return null;
  const n = report.flagged.length;
  return `${n === 1 ? "One line needs" : `${n} lines need`} your attention before you send this. ${n === 1 ? "It" : "They"} could not be traced back to your resume, so ${n === 1 ? "it may be" : "they may be"} something the model added rather than something you did.`;
}
