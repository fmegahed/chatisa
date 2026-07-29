import "server-only";
import { toStateCode, type Fetcher, type RawPosting, type SourceResult } from "./types";

/**
 * USAJobs source: the federal track (user decision 2026-07-28). Free API;
 * requires a registered key plus the account email in the User-Agent header
 * per data.usajobs.gov's terms.
 */

const ENDPOINT = "https://data.usajobs.gov/api/search";
const TIMEOUT_MS = 12_000;

interface UsajobsItem {
  MatchedObjectId?: string;
  MatchedObjectDescriptor?: {
    PositionTitle?: string;
    OrganizationName?: string;
    PositionURI?: string;
    ApplyURI?: string[];
    PublicationStartDate?: string;
    PositionLocation?: { CityName?: string; CountrySubDivisionCode?: string }[];
    UserArea?: { Details?: { JobSummary?: string; MajorDuties?: string[] | string } };
    QualificationSummary?: string;
  };
}

/** Pure; unit-tested against a fixture payload. */
export function normalizeUsajobs(payload: unknown): RawPosting[] {
  const items =
    (payload as { SearchResult?: { SearchResultItems?: UsajobsItem[] } })
      ?.SearchResult?.SearchResultItems ?? [];
  const out: RawPosting[] = [];
  for (const item of items) {
    const d = item.MatchedObjectDescriptor;
    if (!item.MatchedObjectId || !d?.PositionTitle || !d.OrganizationName) {
      continue;
    }
    const applyUrl = d.ApplyURI?.[0] ?? d.PositionURI ?? "";
    const duties = Array.isArray(d.UserArea?.Details?.MajorDuties)
      ? d.UserArea.Details.MajorDuties.join("\n")
      : (d.UserArea?.Details?.MajorDuties ?? "");
    const description = [
      d.UserArea?.Details?.JobSummary ?? "",
      duties,
      d.QualificationSummary ?? "",
    ]
      .filter(Boolean)
      .join("\n\n")
      .trim();
    if (!applyUrl || description.length < 100) continue;
    const loc = d.PositionLocation?.[0];
    // CityName arrives as "Washington, District of Columbia".
    const city = loc?.CityName?.split(",")[0]?.trim() || null;
    const subdivision =
      loc?.CountrySubDivisionCode ?? loc?.CityName?.split(",")[1] ?? "";
    out.push({
      source: "usajobs",
      externalId: item.MatchedObjectId,
      title: d.PositionTitle.trim(),
      company: d.OrganizationName.trim(),
      locationCity: city,
      locationState: toStateCode(subdivision),
      remote: false,
      category: "federal",
      applyUrl,
      description,
      postedAt: d.PublicationStartDate
        ? d.PublicationStartDate.slice(0, 10)
        : null,
    });
  }
  return out;
}

export async function searchUsajobs(
  keyword: string,
  fetcher: Fetcher = fetch,
): Promise<SourceResult> {
  const key = process.env.USAJOBS_API_KEY;
  const email = process.env.USAJOBS_EMAIL;
  if (!key || !email) {
    return {
      postings: [],
      requests: 0,
      error: "USAJOBS_API_KEY or USAJOBS_EMAIL is not set",
    };
  }
  const url = new URL(ENDPOINT);
  url.searchParams.set("Keyword", keyword);
  url.searchParams.set("ResultsPerPage", "100");
  url.searchParams.set("SortField", "opendate");
  url.searchParams.set("SortDirection", "desc");
  try {
    const res = await fetcher(url.toString(), {
      headers: {
        "authorization-key": key,
        "user-agent": email,
        accept: "application/json",
      },
      signal: AbortSignal.timeout(TIMEOUT_MS),
    });
    if (!res.ok) {
      return {
        postings: [],
        requests: 1,
        error: `USAJobs answered ${res.status}`,
      };
    }
    return { postings: normalizeUsajobs(await res.json()), requests: 1 };
  } catch {
    return { postings: [], requests: 1, error: "USAJobs request failed" };
  }
}
