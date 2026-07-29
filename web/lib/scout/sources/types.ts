/**
 * Shared shapes for Job Scout's harvest sources. Normalizers are pure
 * functions over each API's payload and fetchers are injectable, so unit
 * tests never touch the network (the lib/ask/paper-search.ts pattern).
 */

export type Fetcher = typeof fetch;

export interface RawPosting {
  source: "activejobs" | "usajobs";
  externalId: string;
  title: string;
  company: string;
  locationCity: string | null;
  locationState: string | null;
  remote: boolean;
  category: "fulltime" | "internship" | "federal";
  applyUrl: string;
  description: string;
  /** ISO date (YYYY-MM-DD) when the source provides one. */
  postedAt: string | null;
}

/** One source's contribution to a run, with an honest failure note. */
export interface SourceResult {
  postings: RawPosting[];
  requests: number;
  /** Student/operator-readable reason the source came up short, if it did. */
  error?: string;
}

/**
 * Full state names → postal codes. Active Jobs DB's regions_derived and
 * USAJobs both send full names ("Ohio", "District of Columbia"); slicing two
 * characters corrupted Kentucky to "KE" (found 2026-07-28 on the first real
 * JSearch payload), so every source normalizes through this table.
 */
const STATE_CODES: Record<string, string> = {
  alabama: "AL", alaska: "AK", arizona: "AZ", arkansas: "AR",
  california: "CA", colorado: "CO", connecticut: "CT", delaware: "DE",
  "district of columbia": "DC", florida: "FL", georgia: "GA", hawaii: "HI",
  idaho: "ID", illinois: "IL", indiana: "IN", iowa: "IA", kansas: "KS",
  kentucky: "KY", louisiana: "LA", maine: "ME", maryland: "MD",
  massachusetts: "MA", michigan: "MI", minnesota: "MN", mississippi: "MS",
  missouri: "MO", montana: "MT", nebraska: "NE", nevada: "NV",
  "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
  "new york": "NY", "north carolina": "NC", "north dakota": "ND",
  ohio: "OH", oklahoma: "OK", oregon: "OR", pennsylvania: "PA",
  "rhode island": "RI", "south carolina": "SC", "south dakota": "SD",
  tennessee: "TN", texas: "TX", utah: "UT", vermont: "VT", virginia: "VA",
  washington: "WA", "west virginia": "WV", wisconsin: "WI", wyoming: "WY",
};

const VALID_CODES = new Set(Object.values(STATE_CODES));

/** Accepts "OH", "oh", "Ohio", " District of Columbia "; null otherwise. */
export function toStateCode(raw: string | null | undefined): string | null {
  if (!raw) return null;
  const trimmed = raw.trim();
  const upper = trimmed.toUpperCase();
  if (VALID_CODES.has(upper)) return upper;
  return STATE_CODES[trimmed.toLowerCase()] ?? null;
}

/** lower(company)|lower(title)|state — cross-source duplicate detection. */
export function fingerprintOf(p: {
  company: string;
  title: string;
  locationState: string | null;
}): string {
  return [
    p.company.trim().toLowerCase(),
    p.title.trim().toLowerCase(),
    p.locationState ?? "",
  ].join("|");
}
