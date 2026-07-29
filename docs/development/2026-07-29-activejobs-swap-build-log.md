# Build log: Active Jobs DB replaces JSearch; scout.db split (ADR-027)

Date: 2026-07-29. User decisions: "Only Use the Active Jobs DB" (LinkedIn
API declined as primary; USAJobs stays), fresh database for the new feed,
old JSearch-era rows preserved but disconnected.

## Live probes that shaped the design (all with the existing RAPIDAPI_KEY)

- Upgraded plan headers: **2,500 requests/month, 5,000 returned JOBS/month**
  (jobs are the metered unit; the user believed 5,000 requests). `limit`
  accepts values past 1,000; an empty result burns no jobs.
- `/active-ats` requires `time_frame` (1h|24h|7d|6m); `title` and
  `location` accept OR-expressions over quoted phrases (verified live);
  `title_advanced` does NOT exist on this endpoint (400).
- Sampled Ohio results: University Hospitals on taleo.net, Parsons on
  myworkdayjobs.com, Metronet on ultipro.com — employer-direct throughout.
- The LinkedIn sibling API's `direct_apply=only` means LinkedIn EASY APPLY;
  urls stay on linkedin.com. Declined as primary on the user's
  employer-direct preference.
- Fixture captured from the real endpoint (3 records, 30 KB):
  `tests/fixtures/scout/activejobs-page.json`; `jsearch-page.json` deleted.

## Changes

- `lib/db/index.ts`: new `getScoutDb()` opening `<dataDir>/scout.db` with
  inline DDL (postings + runs + indexes); every scout function repointed;
  `closeDb()` closes both. Run stats columns renamed
  `activejobs_requests/found`. `chatisa.db`'s scout tables (still created
  by migrations 0007/0008) are never touched again — that is where the
  user's preserved JSearch-era data lives on existing installs.
- `lib/db/schema.ts`: `scoutRuns` columns renamed; source comment updated.
- NEW `lib/scout/sources/activejobs.ts`: normalizer (drops
  `ats_duplicate`, part-time-only, `ai_experience_level` 5-10/10+;
  multi-state postings pinned to the first TARGET state via parallel
  regions/cities arrays; remote from `ai_work_arrangement`; intern
  override from title/type) + `searchActiveJobs` (25s timeout vs
  JSearch's 12s that timed out 23/160; 429 → `quotaExhausted` stops the
  loop). `sources/jsearch.ts` DELETED (git history keeps it).
- `lib/scout/queries.ts`: matrix reworked to six role-cluster queries with
  per-cluster job limits summing to 950 (`ACTIVEJOBS_RUN_JOB_BUDGET`
  pins ≤1,100 = under a quarter of the monthly pool);
  `ACTIVEJOBS_LOCATION` OR-expression + `TARGET_STATE_CODES` (the same 10
  states as before, CA included). USAJobs keywords unchanged.
- `lib/scout/harvest.ts`: collect stage swapped; summary/run fields
  renamed activejobs*; everything downstream (dedupe, 30-day drop,
  tagging, cost cap, retirement) unchanged.
- Copy: job-scout page and JobFeed status line now say "employer career
  sites and USAJobs"; feed-types sourceErrors key renamed; mock fixture
  postings' source → "activejobs"; chatisa.env.example comment updated.
- Tests: `scout-sources.test.ts` rewritten against the captured payload
  (includes an assertion that no apply URL matches
  linkedin|indeed|ziprecruiter, and the location-expression ↔ state-set
  consistency check); `scout-harvest.test.ts` rewritten (Date frozen at
  2026-07-30 to match the new fixture's capture date);
  `scout-db.test.ts` gains the ADR-027 split guarantee (scout.db exists,
  legacy tables stay empty) and stale-run insert moved to the scout DB;
  routes/e2e source strings updated.

## Notes

- The `ai_key_skills` enrichment could seed or replace part of the Gemini
  tagging pass (cost cut); deliberately NOT done in this change. Logged.
- Quota math: weekly run ≤950 jobs + USAJobs; 4-5 Sunday runs/month =
  3,800-4,750 of 5,000. Dev seed harvests spend the same pool — seed
  sparingly.
- "Sr Clinical Data Analyst" in the fixture documents an interaction:
  the source keeps it (2-5 years) but the title gate (\bsr\b) drops it at
  harvest. Working as designed for an entry-level board.

## Strategy pass (user, 2026-07-29: "strategic, tailored to ISA fresh grads or internships")

- Probed live: `ai_experience_level` and `ai_employment_type` ARE
  server-side filters on /active-ats (the 400 for a wrong name even
  suggests them). Comma lists work. Measured on 24h Data Analyst rows:
  no filter 165, `0-2` 23, `0-2,2-5` 113 (seniors + unclassified out).
  The count endpoint takes 1h|24h|1m|6m, NOT 7d (search takes 7d).
- Every full-time cluster now carries `ai_experience_level=0-2,2-5`
  ("2-5" retained: ads overstate, tagging judges those) and every intern
  cluster carries `ai_employment_type=INTERN` — so no senior posting is
  ever PAID FOR under the jobs-metered quota. Unit test pins this per
  cluster.
- Budget rebalanced toward internships (analytics 250; intern clusters
  125 each; total 950 unchanged), intern title clusters widened
  ("Analytics Intern", "Information Technology Intern").
- `EXCLUDED_TITLE` gains manager|lead|supervisor (the first smoke
  surfaced "Business Intelligence Manager" twice); a test guards that
  "Leadership Development Program" titles survive.

## Verification

- `npx vitest run`: 832 tests, 86 files, all green after the strategy
  pass (harvest suite frozen at 2026-07-30 against the new fixture).
  One suite-level flake appeared once mid-run and never reproduced in
  isolation or on the confirmation run; same signature as the phantom
  during v6.1.1 verification. Watch item, not a blocker.
- `npx tsc --noEmit` clean; `npm run lint` clean.
- `npx playwright test job-scout`: 17/17, desktop + mobile-320 (the mock
  fixture feed now seeds into scout.db at boot, so the split is exercised
  by the running app, not just unit tests).
- Live smoke (5-job limit, one request): OR title + states location
  accepted; returned postings normalized with employer-ATS apply domains
  (recruiting.paylocity.com, uhg.taleo.net, workforcenow.adp.com) and
  correct state pins.
- Released as v6.2.0 (2026-07-29): bundle self-test passed on the fresh
  scout.db path ("scout no harvest yet" allowlisted), 305 MB zip, Node
  24/ABI 137. The server's board refills automatically: the boot tick
  sees an empty scout_runs table and harvests immediately after deploy.
  The user runs the production seed/deploy themselves.
