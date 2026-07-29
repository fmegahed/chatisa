/**
 * The weekly harvest's curated query matrix (design §4.2). Data, not code,
 * so coverage changes are diffs a reviewer can read, and a unit test bounds
 * the request count well under the quota (~4,600/week available).
 *
 * Role families target ISA career paths: business analytics, information
 * systems, information security, and business/technology consulting
 * (user scope decision, 2026-07-28).
 */

export interface HarvestQuery {
  query: string;
  category: "fulltime" | "internship";
}

const ROLE_FAMILIES = [
  "entry level business analyst",
  "entry level data analyst",
  "business intelligence analyst",
  "entry level data scientist",
  "entry level data engineer",
  "information systems analyst",
  "entry level IT auditor",
  "information security analyst",
  "entry level cybersecurity analyst",
  "entry level technology consultant",
];

/** Ohio metros first, then regional hubs, then national + remote. */
const FULLTIME_MARKETS = [
  "Cincinnati, OH",
  "Columbus, OH",
  "Cleveland, OH",
  "Dayton, OH",
  "Chicago, IL",
  "Indianapolis, IN",
  "Louisville, KY",
  "Pittsburgh, PA",
  "New York, NY",
  "Charlotte, NC",
  "Dallas, TX",
  "Atlanta, GA",
  // California added at user request (2026-07-29).
  "San Francisco, CA",
  "Los Angeles, CA",
  "remote",
];

/** Internships hire nationally; a compact market list keeps requests sane. */
const INTERN_MARKETS = [
  "Cincinnati, OH",
  "Columbus, OH",
  "Cleveland, OH",
  "Chicago, IL",
  "remote",
];

const INTERN_ROLES = [
  "business analytics intern",
  "data analyst intern",
  "data science intern",
  "information systems intern",
  "cybersecurity intern",
  "IT audit intern",
];

export const HARVEST_QUERIES: HarvestQuery[] = [
  ...ROLE_FAMILIES.flatMap((role) =>
    FULLTIME_MARKETS.map((market) => ({
      query: `${role} in ${market}`,
      category: "fulltime" as const,
    })),
  ),
  ...INTERN_ROLES.flatMap((role) =>
    INTERN_MARKETS.map((market) => ({
      query: `${role} in ${market}`,
      category: "internship" as const,
    })),
  ),
];

/** One USAJobs request each (100 results/page covers a week comfortably). */
export const USAJOBS_KEYWORDS = [
  "data analytics",
  "data scientist",
  "management analyst",
  "information technology specialist",
  "cybersecurity",
  "information security",
  "statistician",
  "auditor information systems",
];

/**
 * Deterministic relevance gate before any model spend: a title must look
 * like an ISA career and must not be senior. Cheap and unit-tested; the
 * tagging model still sees only what passes.
 */
const RELEVANT_TITLE =
  /(analy|data|information|cyber|security|consult|intelligence|scientist|engineer|system|technolog|statistic|audit|forecast|database|machine learning|\bBI\b|\bIT\b)/i;

const EXCLUDED_TITLE =
  /(senior|\bsr\.?\b|principal|\bstaff\b|director|vice president|\bvp\b|chief|head of|\biii\b|\biv\b|architect|nurse|physician|therapist|driver|warehouse|retail|cashier|custodian|cook|mechanic|electrician)/i;

export function isRelevantTitle(title: string): boolean {
  return RELEVANT_TITLE.test(title) && !EXCLUDED_TITLE.test(title);
}
