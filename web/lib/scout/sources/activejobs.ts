import "server-only";
import { toStateCode, type Fetcher, type RawPosting, type SourceResult } from "./types";

/**
 * Active Jobs DB (fantastic.jobs on RapidAPI) source, replacing JSearch
 * (user decision 2026-07-29, ADR-027): it indexes employer career sites and
 * 54 ATS platforms directly, so `url` is the employer's own application
 * page, never an aggregator. Shapes verified against a real /active-ats
 * response captured 2026-07-29 (tests/fixtures/scout/activejobs-page.json).
 *
 * Quota reality (from the plan's rate-limit headers, not the marketing
 * page): requests AND returned jobs are metered per month, and jobs are the
 * scarce unit. The caller budgets jobs per run; one 429 means the month's
 * pool is gone, so the query loop must stop.
 */

const ENDPOINT = "https://active-jobs-db.p.rapidapi.com/active-ats";
/** JSearch's 12s timed out 23/160 requests on 2026-07-28; large limits here
 * return hundreds of rows in one response, so give it real headroom. */
const TIMEOUT_MS = 25_000;

/** Rows come from thousands of employer boards; every field is absent-able. */
interface ActiveJob {
  id?: number | string;
  title?: string;
  organization?: string;
  url?: string | null;
  description_text?: string | null;
  date_posted?: string | null;
  cities_derived?: (string | null)[] | null;
  regions_derived?: (string | null)[] | null;
  employment_type?: string[] | null;
  ai_work_arrangement?: string | null;
  ai_experience_level?: string | null;
  ats_duplicate?: boolean | null;
}

/** Clearly-senior bands the board never wants; "2-5" stays in because many
 * new-grad-friendly ads claim it, and the tagging pass judges those. */
const SENIOR_LEVELS = new Set(["5-10", "10+"]);

/**
 * Pure; unit-tested against the captured payload. `preferStates` picks which
 * of a multi-state posting's locations to pin the card to (Parsons lists
 * Virginia, Missouri, and Ohio on one row; an Ohio student should see the
 * Ohio one). Falls back to the first listed location.
 */
export function normalizeActiveJobs(
  payload: unknown,
  category: "fulltime" | "internship",
  preferStates?: Set<string>,
): RawPosting[] {
  const jobs: ActiveJob[] = Array.isArray(payload) ? payload : [];
  const out: RawPosting[] = [];
  for (const job of jobs) {
    if (job.id === undefined || job.id === null || !job.title || !job.organization) continue;
    if (job.ats_duplicate) continue;
    const applyUrl = job.url ?? "";
    const description = (job.description_text ?? "").trim();
    if (!applyUrl || description.length < 100) continue;

    const types = job.employment_type ?? [];
    if (
      types.length > 0 &&
      types.every((t) => /part[\s_-]?time/i.test(t))
    ) {
      continue;
    }
    if (job.ai_experience_level && SENIOR_LEVELS.has(job.ai_experience_level)) {
      continue;
    }

    // Parallel arrays of city/region per advertised location.
    const regions = (job.regions_derived ?? []).map((r) => toStateCode(r));
    const cities = job.cities_derived ?? [];
    let pick = 0;
    if (preferStates) {
      const preferred = regions.findIndex((r) => r !== null && preferStates.has(r));
      if (preferred >= 0) pick = preferred;
    }

    const title = job.title.replace(/\s+/g, " ").trim();
    const isIntern =
      /\bintern(ship)?\b/i.test(title) ||
      types.some((t) => /intern/i.test(t));

    out.push({
      source: "activejobs",
      externalId: String(job.id),
      title,
      company: job.organization.trim(),
      locationCity: cities[pick]?.trim() || null,
      locationState: regions[pick] ?? null,
      remote: Boolean(job.ai_work_arrangement?.startsWith("Remote")),
      category: isIntern ? "internship" : category,
      applyUrl,
      description,
      postedAt: job.date_posted ? job.date_posted.slice(0, 10) : null,
    });
  }
  return out;
}

export async function searchActiveJobs(
  params: {
    /** OR-expression over quoted phrases, e.g. `"Data Analyst" OR "BI Analyst"`. */
    title: string;
    /** OR-expression over quoted state/country names; omit for nationwide. */
    location?: string;
    timeFrame: "24h" | "7d";
    limit: number;
    offset?: number;
    category: "fulltime" | "internship";
    preferStates?: Set<string>;
    /** Comma list for ai_experience_level ("0-2,2-5"). Server-side filtering
     * here is quota strategy: every returned job is metered, so senior rows
     * must never be paid for and then thrown away (user, 2026-07-29). */
    experienceLevels?: string;
    /** Comma list for ai_employment_type ("INTERN"). */
    employmentTypes?: string;
  },
  fetcher: Fetcher = fetch,
): Promise<SourceResult & { quotaExhausted?: boolean }> {
  const key = process.env.RAPIDAPI_KEY;
  if (!key) {
    return { postings: [], requests: 0, error: "RAPIDAPI_KEY is not set" };
  }
  const url = new URL(ENDPOINT);
  url.searchParams.set("time_frame", params.timeFrame);
  url.searchParams.set("title", params.title);
  if (params.location) url.searchParams.set("location", params.location);
  url.searchParams.set("limit", String(params.limit));
  if (params.offset) url.searchParams.set("offset", String(params.offset));
  if (params.experienceLevels)
    url.searchParams.set("ai_experience_level", params.experienceLevels);
  if (params.employmentTypes)
    url.searchParams.set("ai_employment_type", params.employmentTypes);
  url.searchParams.set("description_format", "text");
  try {
    const res = await fetcher(url.toString(), {
      headers: {
        "x-rapidapi-key": key,
        "x-rapidapi-host": "active-jobs-db.p.rapidapi.com",
      },
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    if (res.status === 429) {
      // Monthly job/request pool exhausted; further requests only burn it.
      return {
        postings: [],
        requests: 1,
        error: "Active Jobs DB answered 429 (plan quota exhausted)",
        quotaExhausted: true,
      };
    }
    if (!res.ok) {
      return {
        postings: [],
        requests: 1,
        error: `Active Jobs DB answered ${res.status}`,
      };
    }
    return {
      postings: normalizeActiveJobs(
        await res.json(),
        params.category,
        params.preferStates,
      ),
      requests: 1,
    };
  } catch (err) {
    const cause = (err as { cause?: { code?: string } }).cause?.code;
    const name = err instanceof Error ? err.name : "Error";
    return {
      postings: [],
      requests: 1,
      error: `Active Jobs DB request failed (${cause ?? name})`,
    };
  }
}
