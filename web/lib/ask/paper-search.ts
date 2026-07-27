import "server-only";
import { logger } from "@/lib/log";
import { checkRateLimit } from "@/lib/ratelimit";

/**
 * Academic search for Ask Anything (slice C): arXiv + Semantic Scholar +
 * OpenAlex fanned out in parallel, plus Wikipedia for background, all keyless
 * public APIs on fixed hosts (no SSRF surface: the student never supplies a
 * URL here). Raw responses are DISTILLED before the model sees them:
 * normalized to compact records, deduped across sources, ranked, and capped,
 * so a search costs a few thousand input tokens instead of a few hundred
 * thousand, and there is no per-search provider fee.
 *
 * Every normalizer is pure and unit-tested against recorded fixtures. The
 * fetchers take an injectable fetch so tests never touch the network, and a
 * source that fails or is rate-limited degrades to the others with a note
 * rather than failing the tool.
 */

export interface PaperRecord {
  title: string;
  authors: string[];
  year: number | null;
  venue: string | null;
  citations: number | null;
  doi: string | null;
  url: string | null;
  /** Which databases returned it ("arxiv+openalex" marks corroboration). */
  source: string;
  abstract: string | null;
  /**
   * 0-based position in each source list that returned this paper. Internal
   * ranking input only: distill() strips it before the model sees the result.
   * Every source ranks by its own relevance, and preserving that ordering is
   * the strongest signal available here (see rankPapers).
   */
  positions?: number[];
}

export interface BackgroundRecord {
  title: string;
  extract: string;
  url: string;
}

export interface PaperSearchResult {
  papers: PaperRecord[];
  background: BackgroundRecord[];
  /** Sources that could not contribute this time, for honesty in the result. */
  unavailable: string[];
}

/** Same budget as the code tools: a search can never flood the context. */
export const PAPER_OUTPUT_MAX = 8_000;
export const ABSTRACT_MAX = 1_200;
const SOURCE_TIMEOUT_MS = 8_000;
const MAX_LIMIT = 10;

/** Per-source courtesy limits, per server process. arXiv asks for one request
 * per three seconds; Semantic Scholar's keyless pool is shared across all
 * unauthenticated users, so it is treated gently too. */
const SOURCE_LIMITS: Record<string, { limit: number; windowMs: number }> = {
  arxiv: { limit: 18, windowMs: 60_000 },
  semanticscholar: { limit: 25, windowMs: 60_000 },
  openalex: { limit: 60, windowMs: 60_000 },
  wikipedia: { limit: 60, windowMs: 60_000 },
  crossref: { limit: 40, windowMs: 60_000 },
};

type Fetcher = typeof fetch;

function decodeXmlEntities(s: string): string {
  return s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
    .replace(/&amp;/g, "&");
}

function collapse(s: string): string {
  return s.replace(/\s+/g, " ").trim();
}

/** Strips inline JATS/HTML tags Crossref and Wikipedia embed in text. */
function stripTags(s: string): string {
  return collapse(s.replace(/<[^>]+>/g, " "));
}

// ---------------------------------------------------------------------------
// Normalizers (pure; one per source)
// ---------------------------------------------------------------------------

/** arXiv Atom XML -> records. Regex-parsed on purpose: Node has no DOMParser,
 * and the feed's entry structure is stable and flat. */
export function normalizeArxiv(atomXml: string): PaperRecord[] {
  const entries = atomXml.split(/<entry>/).slice(1);
  return entries.map((entry, position) => {
    const field = (tag: string) =>
      decodeXmlEntities(
        collapse(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`).exec(entry)?.[1] ?? ""),
      );
    const authors = [...entry.matchAll(/<name>([\s\S]*?)<\/name>/g)].map((m) =>
      decodeXmlEntities(collapse(m[1])),
    );
    const idUrl = field("id");
    const published = field("published");
    const doiMatch = /<arxiv:doi[^>]*>([\s\S]*?)<\/arxiv:doi>/.exec(entry);
    return {
      title: field("title"),
      authors,
      year: published ? Number(published.slice(0, 4)) || null : null,
      venue: "arXiv",
      citations: null,
      doi: doiMatch ? collapse(doiMatch[1]).toLowerCase() : null,
      url: idUrl || null,
      source: "arxiv",
      abstract: field("summary") || null,
      positions: [position],
    };
  });
}

interface S2Paper {
  title?: string;
  authors?: { name?: string }[];
  year?: number;
  venue?: string;
  citationCount?: number;
  externalIds?: { DOI?: string; ArXiv?: string };
  url?: string;
  abstract?: string | null;
  tldr?: { text?: string } | null;
  referenceCount?: number;
}

export function normalizeSemanticScholar(json: unknown): PaperRecord[] {
  const data = (json as { data?: S2Paper[] })?.data ?? [];
  return data.map((p, position) => ({
    title: collapse(p.title ?? ""),
    authors: (p.authors ?? []).map((a) => a.name ?? "").filter(Boolean),
    year: p.year ?? null,
    venue: p.venue || null,
    citations: p.citationCount ?? null,
    doi: p.externalIds?.DOI?.toLowerCase() ?? null,
    url:
      p.url ??
      (p.externalIds?.ArXiv ? `https://arxiv.org/abs/${p.externalIds.ArXiv}` : null),
    source: "semanticscholar",
    abstract: p.abstract ? collapse(p.abstract) : null,
    positions: [position],
  }));
}

interface OpenAlexWork {
  title?: string;
  display_name?: string;
  authorships?: { author?: { display_name?: string } }[];
  publication_year?: number;
  primary_location?: { source?: { display_name?: string } };
  cited_by_count?: number;
  doi?: string;
  id?: string;
  abstract_inverted_index?: Record<string, number[]>;
}

/** OpenAlex stores abstracts as an inverted index; rebuild the text. */
export function abstractFromInvertedIndex(
  index: Record<string, number[]> | undefined,
): string | null {
  if (!index) return null;
  const words: string[] = [];
  for (const [word, positions] of Object.entries(index)) {
    for (const pos of positions) words[pos] = word;
  }
  const text = collapse(words.join(" "));
  return text || null;
}

export function normalizeOpenAlex(json: unknown): PaperRecord[] {
  const results = (json as { results?: OpenAlexWork[] })?.results ?? [];
  return results.map((w, position) => ({
    title: collapse(w.title ?? w.display_name ?? ""),
    authors: (w.authorships ?? [])
      .map((a) => a.author?.display_name ?? "")
      .filter(Boolean),
    year: w.publication_year ?? null,
    venue: w.primary_location?.source?.display_name ?? null,
    citations: w.cited_by_count ?? null,
    doi: w.doi ? w.doi.replace(/^https?:\/\/doi\.org\//, "").toLowerCase() : null,
    url: w.doi ?? w.id ?? null,
    source: "openalex",
    abstract: abstractFromInvertedIndex(w.abstract_inverted_index),
    positions: [position],
  }));
}

interface WikiSearchPage {
  title?: string;
  key?: string;
  excerpt?: string;
}

export function normalizeWikipedia(json: unknown): BackgroundRecord[] {
  const pages = (json as { pages?: WikiSearchPage[] })?.pages ?? [];
  return pages
    .filter((p) => p.title && p.excerpt)
    .map((p) => ({
      title: p.title!,
      extract: stripTags(p.excerpt!),
      url: `https://en.wikipedia.org/wiki/${encodeURIComponent(p.key ?? p.title!)}`,
    }));
}

// ---------------------------------------------------------------------------
// Merge, rank, distill (pure)
// ---------------------------------------------------------------------------

/** Dedupe key: DOI when present, else the normalized title. */
export function paperKey(p: PaperRecord): string {
  if (p.doi) return `doi:${p.doi}`;
  return `title:${p.title.toLowerCase().replace(/[^a-z0-9]+/g, "")}`;
}

function normalizedTitle(title: string): string {
  return title.toLowerCase().replace(/[^a-z0-9]+/g, "");
}

/**
 * Merges per-source result lists: one record per paper, sources joined, the
 * richest fields kept (max citations, first abstract, any DOI). Indexed by
 * DOI AND by normalized title, because the common real-world case is an arXiv
 * record without a DOI meeting the same paper from Semantic Scholar or
 * OpenAlex with one; a single-key dedupe would keep both.
 */
export function mergePapers(lists: PaperRecord[][]): PaperRecord[] {
  const records: PaperRecord[] = [];
  const byDoi = new Map<string, PaperRecord>();
  const byTitle = new Map<string, PaperRecord>();
  for (const list of lists) {
    for (const p of list) {
      if (!p.title) continue;
      const titleKey = normalizedTitle(p.title);
      const existing =
        (p.doi ? byDoi.get(p.doi) : undefined) ?? byTitle.get(titleKey);
      if (!existing) {
        const copy = { ...p };
        records.push(copy);
        if (copy.doi) byDoi.set(copy.doi, copy);
        byTitle.set(titleKey, copy);
        continue;
      }
      existing.source = existing.source.includes(p.source)
        ? existing.source
        : `${existing.source}+${p.source}`;
      existing.citations = Math.max(existing.citations ?? 0, p.citations ?? 0);
      existing.abstract = existing.abstract ?? p.abstract;
      existing.doi = existing.doi ?? p.doi;
      existing.url = existing.url ?? p.url;
      existing.venue = existing.venue ?? p.venue;
      existing.year = existing.year ?? p.year;
      // Keep every source's placement: agreeing sources compound in rankPapers.
      if (p.positions?.length) {
        existing.positions = [...(existing.positions ?? []), ...p.positions];
      }
      if (p.authors.length > existing.authors.length) existing.authors = p.authors;
      if (existing.doi) byDoi.set(existing.doi, existing);
      byTitle.set(titleKey, existing);
    }
  }
  return records;
}

/** How far relevance outweighs the quality signals. High on purpose: a source
 * putting a paper first is a much better answer to "which paper did the
 * student mean" than a citation count is. */
const RELEVANCE_WEIGHT = 3;

/**
 * Reciprocal-rank fusion over the placements the sources gave a paper. Each
 * source ranks by its own relevance, and for a narrow query (a specific title,
 * a named system) that ordering is the only thing that finds the right paper:
 * it comes back FIRST with no citations, while broad full-text matches come
 * back mid-list with a handful. Summing 1/(1+position) also rewards agreement
 * between sources, so it subsumes the old separate corroboration term.
 */
function fusionScore(p: PaperRecord): number {
  if (p.positions?.length) {
    return p.positions.reduce((sum, position) => sum + 1 / (1 + position), 0);
  }
  // Records assembled by hand (fixtures, mock mode) carry no positions; the
  // number of agreeing sources is the only signal of that kind left.
  return p.source.split("+").length;
}

/**
 * Relevance blend: each source's own ranking first (fused), then citations
 * (log-damped) and recency as tiebreakers among comparably relevant hits.
 *
 * Ranking on citations and recency ALONE was a real defect (found 2026-07-25):
 * a search for "ChatISA" put the paper of that name, returned first by both
 * arXiv and OpenAlex, below unrelated papers that merely mentioned it in their
 * full text and happened to have a few more citations.
 */
export function rankPapers(papers: PaperRecord[], nowYear: number): PaperRecord[] {
  const score = (p: PaperRecord) => {
    const cites = Math.log10(1 + (p.citations ?? 0));
    const age = p.year ? Math.max(0, nowYear - p.year) : 10;
    const recency = age <= 1 ? 2 : age <= 3 ? 1 : age <= 6 ? 0.3 : 0;
    return RELEVANCE_WEIGHT * fusionScore(p) + cites + recency;
  };
  return [...papers].sort((a, b) => score(b) - score(a));
}

/** Drops the internal ranking metadata: it would cost the model tokens and
 * mean nothing to it. */
function forModel(p: PaperRecord): PaperRecord {
  const out = { ...p };
  delete out.positions;
  return out;
}

/** Final cut: cap the list, trim authors and abstracts, drop internal ranking
 * fields, and shrink until the serialized payload fits the output budget. */
export function distill(
  result: PaperSearchResult,
  limit: number,
): PaperSearchResult {
  const papers = result.papers.slice(0, Math.min(limit, MAX_LIMIT)).map((p) => ({
    ...forModel(p),
    authors:
      p.authors.length > 6
        ? [...p.authors.slice(0, 6), `and ${p.authors.length - 6} more`]
        : p.authors,
    abstract:
      p.abstract && p.abstract.length > ABSTRACT_MAX
        ? `${p.abstract.slice(0, ABSTRACT_MAX)}...`
        : p.abstract,
  }));
  const background = result.background.slice(0, 2).map((b) => ({
    ...b,
    extract: b.extract.length > 500 ? `${b.extract.slice(0, 500)}...` : b.extract,
  }));
  let out: PaperSearchResult = { papers, background, unavailable: result.unavailable };
  // Shrink abstracts first, then drop tail papers, until the payload fits.
  let abstractCap = ABSTRACT_MAX;
  while (JSON.stringify(out).length > PAPER_OUTPUT_MAX) {
    if (abstractCap > 300) {
      abstractCap = Math.floor(abstractCap / 2);
      out = {
        ...out,
        papers: out.papers.map((p) => ({
          ...p,
          abstract:
            p.abstract && p.abstract.length > abstractCap
              ? `${p.abstract.slice(0, abstractCap)}...`
              : p.abstract,
        })),
      };
    } else if (out.papers.length > 3) {
      out = { ...out, papers: out.papers.slice(0, out.papers.length - 1) };
    } else {
      out = { ...out, papers: out.papers.map((p) => ({ ...p, abstract: null })) };
      break;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Fetchers
// ---------------------------------------------------------------------------

/** Carries the status through trySource, which needs to tell "this paper does
 * not exist" (404) apart from "the database is refusing us" (429, 5xx). */
class HttpError extends Error {
  constructor(readonly status: number) {
    super(`HTTP ${status}`);
    this.name = "HttpError";
  }
}

async function fetchJson(
  url: string,
  fetcher: Fetcher,
  headers?: Record<string, string>,
): Promise<unknown> {
  const res = await fetcher(url, {
    headers: {
      "user-agent": "ChatISA/1.0 (Miami University; educational use)",
      ...headers,
    },
    signal: AbortSignal.timeout(SOURCE_TIMEOUT_MS),
  });
  if (!res.ok) throw new HttpError(res.status);
  return res.json();
}

async function fetchText(url: string, fetcher: Fetcher): Promise<string> {
  const res = await fetcher(url, {
    headers: { "user-agent": "ChatISA/1.0 (Miami University; educational use)" },
    signal: AbortSignal.timeout(SOURCE_TIMEOUT_MS),
  });
  if (!res.ok) throw new HttpError(res.status);
  return res.text();
}

/** Why a source did not answer. "not-found" is a real answer about the paper;
 * the other two are answers about the database. */
export type SourceFailure = "rate-limited" | "not-found" | "unavailable";

type SourceOutcome<T> =
  | { ok: true; value: T }
  | { ok: false; reason: SourceFailure };

/**
 * One source's contribution, guarded by its courtesy limit. Failures are
 * CLASSIFIED and logged rather than swallowed: a bare catch used to turn 404,
 * 429, timeout, and malformed JSON into the same silent null, so get_paper told
 * students the service was down when the paper simply did not exist, and left
 * operators nothing to read (found 2026-07-25).
 */
async function trySource<T>(
  name: string,
  work: () => Promise<T>,
): Promise<SourceOutcome<T>> {
  const limits = SOURCE_LIMITS[name];
  if (!checkRateLimit(`paper:${name}`, limits).allowed) {
    logger.warn({ source: name }, "paper source skipped: own courtesy limit");
    return { ok: false, reason: "rate-limited" };
  }
  try {
    return { ok: true, value: await work() };
  } catch (err) {
    const status = err instanceof HttpError ? err.status : null;
    const reason: SourceFailure =
      status === 404
        ? "not-found"
        : status === 429
          ? "rate-limited"
          : "unavailable";
    logger.warn(
      { source: name, status, reason, err: String(err) },
      "paper source failed",
    );
    return { ok: false, reason };
  }
}

function valueOr<T>(outcome: SourceOutcome<T>, fallback: T): T {
  return outcome.ok ? outcome.value : fallback;
}

/**
 * The search behind the search_papers tool. In mock-LLM mode it returns
 * fixtures so the e2e loop is deterministic and offline.
 */
export async function searchPapers(
  query: string,
  limit: number,
  opts: { fetcher?: Fetcher; nowYear?: number } = {},
): Promise<PaperSearchResult> {
  if (process.env.CHATISA_MOCK_LLM === "1") return mockSearchResult(query);
  const fetcher = opts.fetcher ?? fetch;
  const n = Math.min(Math.max(1, limit), MAX_LIMIT);
  const q = encodeURIComponent(query);

  const s2Key = process.env.SEMANTIC_SCHOLAR_API_KEY;
  const mailto = process.env.OPENALEX_MAILTO;

  const [arxiv, s2, openalex, wiki] = await Promise.all([
    trySource("arxiv", async () =>
      normalizeArxiv(
        await fetchText(
          `https://export.arxiv.org/api/query?search_query=all:${q}&max_results=${n}&sortBy=relevance`,
          fetcher,
        ),
      ),
    ),
    trySource("semanticscholar", async () =>
      normalizeSemanticScholar(
        await fetchJson(
          `https://api.semanticscholar.org/graph/v1/paper/search?query=${q}&limit=${n}&fields=title,authors,year,venue,citationCount,externalIds,url,abstract`,
          fetcher,
          s2Key ? { "x-api-key": s2Key } : undefined,
        ),
      ),
    ),
    trySource("openalex", async () =>
      normalizeOpenAlex(
        await fetchJson(
          `https://api.openalex.org/works?search=${q}&per_page=${n}${mailto ? `&mailto=${encodeURIComponent(mailto)}` : ""}`,
          fetcher,
        ),
      ),
    ),
    trySource("wikipedia", async () =>
      normalizeWikipedia(
        await fetchJson(
          `https://en.wikipedia.org/w/rest.php/v1/search/page?q=${q}&limit=2`,
          fetcher,
        ),
      ),
    ),
  ]);

  const unavailable = [
    !arxiv.ok ? "arXiv" : null,
    !s2.ok ? "Semantic Scholar" : null,
    !openalex.ok ? "OpenAlex" : null,
  ].filter((s): s is string => s !== null);

  const merged = mergePapers([
    valueOr(arxiv, []),
    valueOr(s2, []),
    valueOr(openalex, []),
  ]);
  const ranked = rankPapers(merged, opts.nowYear ?? new Date().getFullYear());
  return distill(
    { papers: ranked, background: valueOr(wiki, []), unavailable },
    n,
  );
}

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/**
 * The bare arXiv id from whatever the model passes: search results carry
 * "http://arxiv.org/abs/2407.15010v2", and neither the URL wrapper nor the
 * version suffix is a valid id at any database. Returns null for anything that
 * is not an arXiv id, so an unusable value never reaches a request path.
 */
export function normalizeArxivId(raw: string): string | null {
  const trimmed = raw.trim();
  const fromUrl =
    /arxiv\.org\/(?:abs|pdf)\/([^\s?#]+)/i.exec(trimmed)?.[1] ?? trimmed;
  const bare = fromUrl
    .replace(/^arxiv:/i, "")
    .replace(/\.pdf$/i, "")
    .replace(/v\d+$/i, "");
  // Modern ids are 2407.15010; pre-2007 ids are math.GT/0309136.
  const modern = /^\d{4}\.\d{4,5}$/.test(bare);
  const legacy = /^[a-z-]+(\.[A-Z]{2})?\/\d{7}$/i.test(bare);
  return modern || legacy ? bare : null;
}

/** The bare DOI from a raw string or a resolver URL. Returns null unless the
 * result is DOI-shaped and free of characters that could reshape a request. */
export function normalizeDoi(raw: string): string | null {
  const bare = raw
    .trim()
    .replace(/^https?:\/\/(dx\.)?doi\.org\//i, "")
    .replace(/^doi:/i, "")
    .toLowerCase();
  if (!/^10\.\d{4,9}\/[^\s?#]+$/.test(bare)) return null;
  if (bare.includes("..")) return null;
  return bare;
}

/** The get_paper output shape, uniform across whichever source answered. */
function detailFromS2(json: unknown): Record<string, unknown> | null {
  const p = json as S2Paper & { tldr?: { text?: string } };
  if (!p?.title) return null;
  return {
    title: p.title,
    authors: (p.authors ?? []).map((a) => a.name).filter(Boolean),
    year: p.year ?? null,
    venue: p.venue ?? null,
    citations: p.citationCount ?? null,
    references: p.referenceCount ?? null,
    doi: p.externalIds?.DOI ?? null,
    url: p.url ?? null,
    tldr: p.tldr?.text ?? null,
    abstract: p.abstract ?? null,
    source: "semanticscholar",
  };
}

function detailFromRecord(p: PaperRecord): Record<string, unknown> | null {
  if (!p?.title) return null;
  return {
    title: p.title,
    authors: p.authors,
    year: p.year,
    venue: p.venue,
    citations: p.citations,
    references: null,
    doi: p.doi,
    url: p.url,
    tldr: null,
    abstract: p.abstract,
    source: p.source,
  };
}

function capDetail(out: Record<string, unknown>): Record<string, unknown> {
  return JSON.stringify(out).length > PAPER_OUTPUT_MAX
    ? { ...out, abstract: String(out.abstract ?? "").slice(0, ABSTRACT_MAX) }
    : out;
}

/**
 * The lookup behind the get_paper tool: one paper in depth, by DOI or arXiv id.
 *
 * Tries Semantic Scholar, then arXiv, then OpenAlex, and returns the first
 * source that has the paper. It was Semantic Scholar ONLY until 2026-07-25,
 * which made the tool useless in practice: the keyless Semantic Scholar pool is
 * shared across every unauthenticated caller and answers 429 to nearly every
 * request, so a tool with no second source failed every time even though arXiv
 * and OpenAlex both had the paper. Set SEMANTIC_SCHOLAR_API_KEY to get the
 * richest source (it alone carries tldr and reference counts) back as the
 * first hop rather than a hop that usually fails.
 */
export async function getPaper(
  id: { doi?: string; arxivId?: string },
  opts: { fetcher?: Fetcher } = {},
): Promise<Record<string, unknown> | { error: string }> {
  if (process.env.CHATISA_MOCK_LLM === "1") return mockPaperDetail();
  const fetcher = opts.fetcher ?? fetch;
  if (!id.doi && !id.arxivId) {
    return { error: "Provide a doi or an arxivId." };
  }
  const doi = id.doi ? normalizeDoi(id.doi) : null;
  const arxivId = id.arxivId ? normalizeArxivId(id.arxivId) : null;
  if (!doi && !arxivId) {
    return {
      error:
        "That id was not a recognizable DOI or arXiv id. Pass a DOI like 10.1000/xyz123 or an arXiv id like 2407.15010, or call search_papers to find the paper first.",
    };
  }

  const s2Key = process.env.SEMANTIC_SCHOLAR_API_KEY;
  const mailto = process.env.OPENALEX_MAILTO;
  const attempts: {
    name: string;
    work: () => Promise<Record<string, unknown> | null>;
  }[] = [
    {
      name: "semanticscholar",
      work: async () =>
        detailFromS2(
          await fetchJson(
            `https://api.semanticscholar.org/graph/v1/paper/${encodeURIComponent(
              doi ? `DOI:${doi}` : `arXiv:${arxivId}`,
            )}?fields=title,authors,year,venue,citationCount,referenceCount,externalIds,url,abstract,tldr`,
            fetcher,
            s2Key ? { "x-api-key": s2Key } : undefined,
          ),
        ),
    },
  ];
  if (arxivId) {
    attempts.push({
      name: "arxiv",
      work: async () =>
        detailFromRecord(
          normalizeArxiv(
            await fetchText(
              `https://export.arxiv.org/api/query?id_list=${encodeURIComponent(arxivId)}`,
              fetcher,
            ),
          )[0],
        ),
    });
  }
  attempts.push({
    name: "openalex",
    work: async () => {
      // OpenAlex indexes arXiv preprints under their registered DOI.
      const key = doi ?? `10.48550/arXiv.${arxivId}`;
      const work = await fetchJson(
        `https://api.openalex.org/works/doi:${encodeURIComponent(key)}${
          mailto ? `?mailto=${encodeURIComponent(mailto)}` : ""
        }`,
        fetcher,
      );
      return detailFromRecord(normalizeOpenAlex({ results: [work] })[0]);
    },
  });

  const reasons: SourceFailure[] = [];
  for (const attempt of attempts) {
    const outcome = await trySource(attempt.name, attempt.work);
    if (outcome.ok) {
      if (outcome.value) return capDetail(outcome.value);
      // A 200 with nothing in it is the source saying it has no such paper.
      reasons.push("not-found");
      continue;
    }
    reasons.push(outcome.reason);
  }

  logger.warn(
    { doi, arxivId, reasons },
    "get_paper found no source with the paper",
  );
  return {
    error: reasons.every((r) => r === "not-found")
      ? "No record of that paper in Semantic Scholar, arXiv, or OpenAlex. Check the id, or ask the student to attach the PDF."
      : "The paper databases could not be reached just now. Use the abstract from search_papers, or ask the student to attach the PDF.",
  };
}

// ---------------------------------------------------------------------------
// Mock fixtures (deterministic e2e without network)
// ---------------------------------------------------------------------------

function mockSearchResult(query: string): PaperSearchResult {
  return {
    papers: [
      {
        title: `Conformal Prediction Methods: a Mock Survey of ${query}`,
        authors: ["A. Scholar", "B. Researcher"],
        year: 2025,
        venue: "arXiv",
        citations: 214,
        doi: "10.0000/mock.1",
        url: "https://arxiv.org/abs/0000.00001",
        source: "arxiv+openalex",
        abstract:
          "MOCK_ABSTRACT This fixture paper is returned in test mode so the search loop is deterministic and offline.",
      },
      {
        title: "A Second Mock Paper",
        authors: ["C. Author"],
        year: 2024,
        venue: "Mock Journal",
        citations: 12,
        doi: "10.0000/mock.2",
        url: "https://doi.org/10.0000/mock.2",
        source: "semanticscholar",
        abstract: "A smaller fixture entry.",
      },
    ],
    background: [
      {
        title: "Mock Topic",
        extract: "Background fixture from the encyclopedia source.",
        url: "https://en.wikipedia.org/wiki/Mock_Topic",
      },
    ],
    unavailable: [],
  };
}

function mockPaperDetail(): Record<string, unknown> {
  return {
    title: "Conformal Prediction Methods: a Mock Survey",
    authors: ["A. Scholar", "B. Researcher"],
    year: 2025,
    venue: "arXiv",
    citations: 214,
    references: 58,
    doi: "10.0000/mock.1",
    url: "https://arxiv.org/abs/0000.00001",
    tldr: "MOCK_TLDR A fixture summary.",
    abstract: "MOCK_ABSTRACT The full fixture abstract for detail lookups.",
  };
}
