/**
 * Choosing which parts of a document the questions come from.
 *
 * The legacy app silently kept the first thirty pages, so a student uploading
 * a textbook was examined on its preface. Here, when a document is larger than
 * the model can hold, excerpts are sampled across the whole chosen range and
 * the student is told exactly which pages contributed.
 */

export interface SourcePage {
  pageNumber: number;
  text: string;
  charCount: number;
  source?: string;
}

export interface Excerpt {
  fromPage: number;
  toPage: number;
  text: string;
}

export interface CoverageReport {
  strategy: "full" | "sampled";
  pagesUsed: number[];
  pagesSkipped: number[];
  /** Pages with too little text to be worth sampling. */
  pagesWithLittleText: number[];
  charsUsed: number;
  charsAvailable: number;
  excerptCount: number;
}

export interface SelectionResult {
  excerpts: Excerpt[];
  coverage: CoverageReport;
}

/** Pages below this are treated as decoration rather than content. */
const MIN_PAGE_CHARS = 200;

export function selectExcerpts(params: {
  pages: SourcePage[];
  fromPage: number;
  toPage: number;
  charBudget: number;
  questionCount: number;
}): SelectionResult {
  const inScope = params.pages
    .filter(
      (p) => p.pageNumber >= params.fromPage && p.pageNumber <= params.toPage,
    )
    .sort((a, b) => a.pageNumber - b.pageNumber);

  const substantial = inScope.filter((p) => p.charCount >= MIN_PAGE_CHARS);

  // Very short documents have no substantial pages; use what there is rather
  // than refusing, so a one-page handout still works.
  const usable = substantial.length > 0 ? substantial : inScope;

  // Only pages actually left out are reported as too thin. Otherwise a short
  // document would claim its pages were both used and skipped.
  const usableNumbers = new Set(usable.map((p) => p.pageNumber));
  const thin = inScope
    .filter((p) => !usableNumbers.has(p.pageNumber))
    .map((p) => p.pageNumber);
  const charsAvailable = usable.reduce((sum, p) => sum + p.charCount, 0);

  if (usable.length === 0) {
    return {
      excerpts: [],
      coverage: {
        strategy: "full",
        pagesUsed: [],
        pagesSkipped: [],
        pagesWithLittleText: thin,
        charsUsed: 0,
        charsAvailable: 0,
        excerptCount: 0,
      },
    };
  }

  // Everything fits: use the whole range, contiguously.
  if (charsAvailable <= params.charBudget) {
    return {
      excerpts: [
        {
          fromPage: usable[0].pageNumber,
          toPage: usable[usable.length - 1].pageNumber,
          text: usable.map((p) => `[page ${p.pageNumber}]\n${p.text}`).join("\n\n"),
        },
      ],
      coverage: {
        strategy: "full",
        pagesUsed: usable.map((p) => p.pageNumber),
        pagesSkipped: [],
        pagesWithLittleText: thin,
        charsUsed: charsAvailable,
        charsAvailable,
        excerptCount: 1,
      },
    };
  }

  // Too large: take evenly spaced windows so coverage spans the whole range
  // rather than stopping partway through.
  const targetExcerpts = Math.max(
    1,
    Math.min(params.questionCount, Math.floor(params.charBudget / 3_000)),
  );
  const stride = usable.length / targetExcerpts;
  const perExcerptBudget = Math.floor(params.charBudget / targetExcerpts);

  const excerpts: Excerpt[] = [];
  const pagesUsed: number[] = [];
  let charsUsed = 0;

  for (let i = 0; i < targetExcerpts; i += 1) {
    const start = Math.floor(i * stride);
    if (start >= usable.length) break;
    const window: SourcePage[] = [];
    let windowChars = 0;
    for (let j = start; j < usable.length; j += 1) {
      const page = usable[j];
      if (windowChars > 0 && windowChars + page.charCount > perExcerptBudget) {
        break;
      }
      window.push(page);
      windowChars += page.charCount;
      if (windowChars >= perExcerptBudget) break;
    }
    if (window.length === 0) continue;
    excerpts.push({
      fromPage: window[0].pageNumber,
      toPage: window[window.length - 1].pageNumber,
      text: window.map((p) => `[page ${p.pageNumber}]\n${p.text}`).join("\n\n"),
    });
    for (const p of window) pagesUsed.push(p.pageNumber);
    charsUsed += windowChars;
  }

  const used = new Set(pagesUsed);
  return {
    excerpts,
    coverage: {
      strategy: "sampled",
      pagesUsed: [...used].sort((a, b) => a - b),
      pagesSkipped: usable
        .map((p) => p.pageNumber)
        .filter((n) => !used.has(n)),
      pagesWithLittleText: thin,
      charsUsed,
      charsAvailable,
      excerptCount: excerpts.length,
    },
  };
}

/** One sentence a student can read, describing what the exam covered. */
export function describeCoverage(
  coverage: CoverageReport,
  totalPages: number,
): string {
  if (coverage.pagesUsed.length === 0) return "No readable pages were used.";
  const ranges: string[] = [];
  let start = coverage.pagesUsed[0];
  let prev = start;
  for (const page of coverage.pagesUsed.slice(1)) {
    if (page !== prev + 1) {
      ranges.push(start === prev ? `${start}` : `${start} to ${prev}`);
      start = page;
    }
    prev = page;
  }
  ranges.push(start === prev ? `${start}` : `${start} to ${prev}`);

  const base =
    coverage.strategy === "full"
      ? `Questions were drawn from pages ${ranges.join(", ")} of ${totalPages}.`
      : `Questions were drawn from ${coverage.excerptCount} sections sampled across pages ${ranges.join(", ")} of ${totalPages}.`;

  if (coverage.pagesWithLittleText.length === 0) return base;
  return `${base} Pages ${coverage.pagesWithLittleText.join(", ")} had too little text to use.`;
}
