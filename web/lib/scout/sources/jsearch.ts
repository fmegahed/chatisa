import "server-only";
import { toStateCode, type Fetcher, type RawPosting, type SourceResult } from "./types";

/**
 * JSearch (RapidAPI) source. One request per query, `date_posted=week` so a
 * weekly harvest picks up exactly the new week (design §4.2; user decision
 * 2026-07-28: JSearch and USAJobs only).
 *
 * Payload shape verified against a REAL /search-v2 response on 2026-07-28,
 * after the first live run burned 160 quota requests parsing the wrong
 * envelope: v2 nests results under `data.jobs`, sends full state names
 * ("Ohio"), and lists employment types per job. The legacy `data: [...]`
 * array shape is still accepted defensively.
 */

const ENDPOINT = "https://jsearch.p.rapidapi.com/search-v2";
const TIMEOUT_MS = 12_000;

/** Payload rows are the aggregator's, so every field is treated as absent-able. */
interface JsearchJob {
  job_id?: string;
  job_title?: string;
  employer_name?: string;
  job_city?: string | null;
  job_state?: string | null;
  job_is_remote?: boolean | null;
  job_apply_link?: string | null;
  job_description?: string | null;
  job_posted_at_datetime_utc?: string | null;
  job_employment_types?: string[] | null;
}

/**
 * Aggregators stuff company and location into titles ("Cyber Security
 * Intern - Summer 2025 at Park Place Technologies Cleveland, OH"), which
 * the card then repeats in its own byline (user screenshot, 2026-07-29).
 * Cut the title at " at <company>" when the company matches.
 */
export function cleanTitle(raw: string, company: string): string {
  let title = raw.replace(/\s+/g, " ").trim();
  const marker = ` at ${company.trim().toLowerCase()}`;
  const idx = title.toLowerCase().lastIndexOf(marker);
  if (idx > 0) title = title.slice(0, idx);
  // Tidy any separator the cut left dangling ("Data Analyst -").
  return title.replace(/[\s\-–|,]+$/, "").trim() || raw.trim();
}

/** Pure; unit-tested against a real captured payload. */
export function normalizeJsearch(
  payload: unknown,
  category: "fulltime" | "internship",
): RawPosting[] {
  const envelope = payload as
    | { data?: { jobs?: unknown[] } | unknown[] }
    | null
    | undefined;
  const jobs: JsearchJob[] = Array.isArray(envelope?.data)
    ? (envelope.data as JsearchJob[])
    : Array.isArray((envelope?.data as { jobs?: unknown[] })?.jobs)
      ? ((envelope!.data as { jobs: unknown[] }).jobs as JsearchJob[])
      : [];
  const out: RawPosting[] = [];
  for (const job of jobs) {
    if (!job.job_id || !job.job_title || !job.employer_name) continue;
    const applyUrl = job.job_apply_link ?? "";
    const description = (job.job_description ?? "").trim();
    if (!applyUrl || description.length < 100) continue;
    const types = job.job_employment_types ?? [];
    // Part-time-only listings are out of scope (entry-level FT, internships,
    // and contract-to-hire stay in).
    if (types.length > 0 && types.every((t) => t === "PARTTIME")) continue;
    out.push({
      source: "jsearch",
      externalId: job.job_id,
      title: cleanTitle(job.job_title, job.employer_name),
      company: job.employer_name.trim(),
      locationCity: job.job_city?.trim() || null,
      locationState: toStateCode(job.job_state),
      remote: Boolean(job.job_is_remote),
      // The ad's own word beats which query pass surfaced it.
      category: types.includes("INTERN") ? "internship" : category,
      applyUrl,
      description,
      postedAt: job.job_posted_at_datetime_utc
        ? job.job_posted_at_datetime_utc.slice(0, 10)
        : null,
    });
  }
  return out;
}

export async function searchJsearch(
  params: { query: string; category: "fulltime" | "internship" },
  fetcher: Fetcher = fetch,
): Promise<SourceResult & { quotaExhausted?: boolean }> {
  const key = process.env.RAPIDAPI_KEY;
  if (!key) {
    return { postings: [], requests: 0, error: "RAPIDAPI_KEY is not set" };
  }
  const url = new URL(ENDPOINT);
  url.searchParams.set("query", params.query);
  url.searchParams.set("num_pages", "1");
  url.searchParams.set("country", "us");
  url.searchParams.set("date_posted", "week");
  if (params.category === "internship") {
    url.searchParams.set("employment_types", "INTERN");
  }
  try {
    const res = await fetcher(url.toString(), {
      headers: {
        "x-rapidapi-key": key,
        "x-rapidapi-host": "jsearch.p.rapidapi.com",
      },
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    if (res.status === 429) {
      // Plan quota. The caller must STOP querying: every further request
      // burns the same exhausted quota (learned live, 2026-07-28).
      return {
        postings: [],
        requests: 1,
        error: "JSearch answered 429 (plan quota exhausted)",
        quotaExhausted: true,
      };
    }
    if (!res.ok) {
      return {
        postings: [],
        requests: 1,
        error: `JSearch answered ${res.status}`,
      };
    }
    return {
      postings: normalizeJsearch(await res.json(), params.category),
      requests: 1,
    };
  } catch (err) {
    // The cause chain names the real network failure; "request failed"
    // alone made the 2026-07-28 run undiagnosable.
    const cause = (err as { cause?: { code?: string } }).cause?.code;
    const name = err instanceof Error ? err.name : "Error";
    return {
      postings: [],
      requests: 1,
      error: `JSearch request failed (${cause ?? name})`,
    };
  }
}
