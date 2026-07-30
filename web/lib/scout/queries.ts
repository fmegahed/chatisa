/**
 * The weekly harvest's curated query matrix (design §4.2; reworked for
 * Active Jobs DB, ADR-027). Data, not code, so coverage changes are diffs a
 * reviewer can read.
 *
 * Economics changed with the source: Active Jobs DB meters RETURNED JOBS
 * per month (5,000 on the current plan), not just requests, and one request
 * can return hundreds of rows filtered by an OR-expression over titles. So
 * the matrix is a handful of role CLUSTERS with per-cluster job limits, and
 * a unit test pins the per-run job budget under a quarter of the monthly
 * pool (four weekly runs + probe headroom).
 *
 * Role families target ISA career paths: business analytics, information
 * systems, information security, and business/technology consulting
 * (user scope decision, 2026-07-28).
 */

export interface ActiveJobsQuery {
  /** OR-expression over quoted title phrases (verified live 2026-07-29). */
  title: string;
  category: "fulltime" | "internship";
  /** Max jobs this cluster may return; the metered unit. */
  limit: number;
  /** ai_experience_level comma list; the fresh-grad band is "0-2,2-5"
   * ("2-5" stays because ads routinely overstate; the tagging pass judges
   * those; measured 24h counts 2026-07-29: 0-2 alone kept 14% of Data
   * Analyst rows, 0-2,2-5 kept 68%, seniors and unclassified out). */
  experienceLevels?: string;
  /** ai_employment_type comma list, e.g. "INTERN". */
  employmentTypes?: string;
}

/** The states the board serves: Ohio + regional hubs + the coasts
 * (California at user request, 2026-07-29). Location filtering happens in
 * the API query; multi-state postings are pinned to the first of these. */
export const TARGET_STATE_CODES = new Set([
  "OH", "IL", "IN", "KY", "PA", "NY", "NC", "TX", "GA", "CA",
]);

export const ACTIVEJOBS_LOCATION = [
  "Ohio", "Illinois", "Indiana", "Kentucky", "Pennsylvania", "New York",
  "North Carolina", "Texas", "Georgia", "California",
]
  .map((s) => `"${s}"`)
  .join(" OR ");

/** The fresh-grad experience band every full-time cluster is filtered to
 * SERVER-SIDE, so no senior posting is ever paid for (user, 2026-07-29:
 * "strategic, tailored to ISA fresh grads or internships"). */
export const ENTRY_LEVELS = "0-2,2-5";

export const ACTIVEJOBS_QUERIES: ActiveJobsQuery[] = [
  {
    title:
      '"Data Analyst" OR "Business Analyst" OR "Business Intelligence" OR "Analytics Consultant"',
    category: "fulltime",
    limit: 250,
    experienceLevels: ENTRY_LEVELS,
  },
  {
    title: '"Data Scientist" OR "Data Engineer" OR "Machine Learning"',
    category: "fulltime",
    limit: 150,
    experienceLevels: ENTRY_LEVELS,
  },
  {
    title:
      '"Information Systems" OR "Systems Analyst" OR "IT Analyst" OR "Technology Consultant"',
    category: "fulltime",
    limit: 150,
    experienceLevels: ENTRY_LEVELS,
  },
  {
    title:
      '"Cybersecurity Analyst" OR "Information Security" OR "Security Analyst" OR "IT Auditor" OR "IT Audit"',
    category: "fulltime",
    limit: 150,
    experienceLevels: ENTRY_LEVELS,
  },
  // Internships get the bigger share of the remaining budget: they are the
  // half of the board every ISA year can use, and ai_employment_type=INTERN
  // isolates them server-side better than title matching alone.
  {
    title:
      '"Data Analyst Intern" OR "Business Analytics Intern" OR "Data Science Intern" OR "Business Intelligence Intern" OR "Analytics Intern"',
    category: "internship",
    limit: 125,
    employmentTypes: "INTERN",
  },
  {
    title:
      '"Cybersecurity Intern" OR "Information Systems Intern" OR "IT Audit Intern" OR "IT Intern" OR "Information Technology Intern"',
    category: "internship",
    limit: 125,
    employmentTypes: "INTERN",
  },
];

/** Jobs one weekly run may consume; the unit test holds the matrix to it. */
export const ACTIVEJOBS_RUN_JOB_BUDGET = 1_100;

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

// manager/lead/supervisor added 2026-07-29: the first Active Jobs DB smoke
// surfaced "Business Intelligence Manager" twice, and people-management
// roles are never fresh-grad postings. \blead\b leaves "Leadership
// Development Program" alone. Academic posts (professor, faculty, adjunct)
// added the same day after "Professor, Geospatial Information Systems"
// topped the filmed board: undergrads and MSBA students do not apply to
// professorships.
const EXCLUDED_TITLE =
  /(senior|\bsr\.?\b|principal|\bstaff\b|director|vice president|\bvp\b|chief|head of|\biii\b|\biv\b|architect|\bmanager\b|\blead\b|supervisor|professor|faculty|lecturer|instructor|adjunct|postdoc|\bdean\b|nurse|physician|therapist|driver|warehouse|retail|cashier|custodian|cook|mechanic|electrician)/i;

export function isRelevantTitle(title: string): boolean {
  return RELEVANT_TITLE.test(title) && !EXCLUDED_TITLE.test(title);
}
