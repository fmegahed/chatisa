# Job Scout — build log (2026-07-28)

Running record of every change made while implementing
`2026-07-28-job-scout-design.md`. Per user instruction: **nothing is
committed**; this log is the change record. Purely additive — existing
modules must not change behaviour.

## Session changes

- `docs/development/2026-07-28-job-scout-design.md` — revised per user:
  tagging model `gemini-3.6-flash` (was claude-sonnet-5), cost cap default
  10 USD (was 75), §6 portfolio generator rewritten as job-agnostic
  (skill-seeded, README never names an employer).
- `docs/development/interaction-log.md` — appended the 2026-07-28 entry.
- `docs/development/2026-07-28-job-scout-build-log.md` — this file.
- `docs/development/2026-07-28-job-scout-plan.md` — implementation plan
  (tasks A-H).

### Task A — skill data layer (complete; 21 unit tests green)

- `web/lib/scout/taxonomy.ts` — new; ~104-id closed vocabulary
  (TAXONOMY_VERSION 1) with kind/category/aliases/one-deep implies edges.
- `web/lib/scout/courses.ts` — new; bulletin snapshot (2026-07-28), 45
  courses (42 mapped + 340/480/481 freeform), cross-listed altCodes,
  credit hours; Independent Studies excluded.
- `web/lib/scout/course-skills.ts` — new; ~190 leveled links
  (anchor/applied/exposure) with evidence phrases on anchors; authored by
  Fable 5, pending instructor review.
- `web/lib/scout/matching.ts` — new; noisy-OR profileStrengths (anchor 1.0 /
  applied 0.6 / exposure 0.25, credits/3 scale) + requirement-coverage
  scoreJob (required 1.0 / preferred 0.5, implies credit 0.6, covered ≥ 0.5,
  bands 0.70/0.45).
- `web/tests/unit/scout-taxonomy.test.ts`,
  `web/tests/unit/scout-matching.test.ts` — new; integrity + hand-computed
  worked examples. `npx vitest run` on both: 21/21 pass.
- `docs/development/2026-07-28-course-skills-review.md` — new; instructor
  review table incl. flagged tool inferences (Python/R/Tableau/Spark).

### Task B — database layer (complete; full suite 764/764 green)

- `web/lib/db/schema.ts` — appended `scout_postings` (18 cols, unique
  (source, external_id), index (active, category), no student columns by
  design) and `scout_runs` (scheduler memory + cost record).
- `web/drizzle/0007_boring_sentinels.sql` — generated migration
  (applies at boot per house pattern).
- `web/lib/db/index.ts` — appended job scout helpers: upsertScoutPosting
  (id stable across weekly re-harvests), scoutFingerprintExists,
  listScoutPostings (description excluded from feed), countScoutPostings,
  getScoutPosting, retireScoutPostings, createScoutRun, finishScoutRun,
  latestSuccessfulScoutRun, scoutRunInProgress.
- `web/tests/unit/scout-db.test.ts` — new; 5 tests (temp-dir DB pattern).
- Full `npx vitest run`: 79 files / 764 tests pass — existing modules
  unaffected.

### Task C — harvest pipeline (complete; 13 unit tests green)

- `web/lib/scout/sources/types.ts` — new; RawPosting/SourceResult/Fetcher +
  fingerprintOf.
- `web/lib/scout/sources/jsearch.ts` — new; /search-v2 client
  (date_posted=week, num_pages=1, INTERN pass) + pure normalizeJsearch
  (drops missing-id/apply-link and <100-char descriptions).
- `web/lib/scout/sources/usajobs.ts` — new; data.usajobs.gov client + pure
  normalizeUsajobs (state-name → postal code, summary+duties+quals join).
- `web/lib/scout/queries.ts` — new; query matrix as data (130 full-time +
  30 intern JSearch queries + 8 USAJobs keywords = 168 requests/run,
  test-bounded ≤600) + isRelevantTitle deterministic gate.
- `web/lib/scout/tag.ts` — new; gemini-3.6-flash generateObject tagging
  (skillId enum-forced, nonce-fenced posting text, required/preferred
  importance, seniorityOk), calculateCost per call, maxRunUsd (default 10).
- `web/lib/scout/harvest.ts` — new; orchestrator: collect → dedupe
  (external id + cross-source fingerprint) → title gate → tag (concurrency
  4, cost cap → partial + droppedByCap recorded, per-posting failures
  swallowed) → upsert → retire (unseen 13 d inactive, 90 d purge). Model's
  category call overrides the query's. Deviations from spec noted: cap-hit
  postings are dropped (not stored inactive) with the count recorded; the
  35-day postedAt retirement was dropped because a re-seen posting is still
  open by definition.
- `web/tests/fixtures/scout/jsearch-page.json`, `usajobs-page.json` — new.
- `web/tests/unit/scout-sources.test.ts` (8),
  `web/tests/unit/scout-harvest.test.ts` (4, incl. deterministic cost-cap
  concurrency case) — 13/13 pass.

### Task D — scheduler, env, health, manual trigger (complete)

- `web/lib/scout/scheduler.ts` — new; hourly tick, due = "no success since
  the last Sunday 2 AM America/New_York boundary" via Intl wall-clock
  labels; `hour >= 2` (not ===) so spring-forward Sundays with no 2 AM
  still run; future-timestamp clock-skew guard; catch-up at boot; no-op
  under test/mock/no-keys; setTimeout unref'd.
- `web/lib/config/env.ts` — added optional RAPIDAPI_KEY, USAJOBS_API_KEY,
  USAJOBS_EMAIL, CHATISA_SCOUT_ADMINS, CHATISA_SCOUT_MAX_RUN_USD with the
  Ask-Anything-style rationale comment (degrade, never block boot).
- `web/instrumentation.ts` — registers startScoutScheduler() after getDb().
- `web/app/api/health/route.ts` — deep checks gain a `scout` line
  (freshness + active posting count), informational like speech's
  not-configured semantics: never turns the server red.
- `web/app/api/scout/refresh/route.ts` — new; POST, auth + admin allowlist,
  409 when running, 202 fire-and-forget.
- `web/.env.example` — Job Scout section (names + comments only).
- `web/tests/unit/scout-scheduler.test.ts` — new; 11 tests incl. both 2026
  DST weekends. Scheduler + env + health suites: 20/20 pass.

### Task E — registration + API routes (complete; 7 route tests green)

- `web/lib/modules.ts` — Job Scout inserted before jobapp-drafter in the
  jobs group (nav, home card, and placeholder route follow automatically).
- `web/lib/config/models.ts` — ModuleKey "job_scout"; PAGE_MODELS rule
  (structured output + 64k context, mirroring jobapp_assistant);
  DEFAULT_MODELS gpt-5.6-terra.
- `web/app/api/scout/feed/route.ts` — new; auth'd feed with
  category/state/remote filters + freshness; content-free feed_view event.
- `web/app/api/scout/postings/[id]/route.ts` — new; full detail, plain 404.
- `web/app/api/scout/resume-skills/route.ts` — new; transient resume/free-
  text → confirmable {skillId, level, evidence} suggestions; nothing
  persisted; 10/min rate limit.
- `web/app/api/scout/project/route.ts` — new; job-agnostic scaffold
  generation (skills + client-sent evidence in, file manifest out, path
  regex blocks traversal); 4/min rate limit; cost recorded content-free.
- `web/lib/providers/mock.ts` — three new mock branches (tagging keyed on
  seniorityOk, extraction keyed on skills-only, scaffold keyed on repoName).
- `web/tests/unit/scout-routes.test.ts` — new; 7 tests (401, filters,
  detail 404, unknown-skill 400, mock scaffold, admin gate 403/202).

### Task F — UI (complete; lint + tsc clean; full suite 798/798 twice)

- `web/package.json` / `package-lock.json` — added fflate (client-side zip).
- `web/app/(app)/job-scout/page.tsx` — new; house server-page template with
  the privacy statement panel.
- `web/lib/scout/profile-store.ts` — new; js-profile-v1 / js-saved-v1
  localStorage stores, corrupt-JSON degrade.
- `web/lib/scout/use-scout-store.ts` — new; useSyncExternalStore bindings
  (the PauseDial pattern; avoids set-state-in-effect lint).
- `web/components/scout/JobScout.tsx` — new; returning-user-first root:
  setup wizard only when no profile; profile bar + feed + portfolio panel
  after. PortfolioPanel keyed on seed skills (fresh mount instead of a
  setState effect).
- `web/components/scout/ProfileSetup.tsx` — new; numbered stages (courses
  by level group with cross-listed codes, optional resume/free-text skill
  suggestions with confirm-and-level radios, save gate).
- `web/components/scout/JobFeed.tsx` — new; browser-side matching, serif
  coverage fraction + text band (never colour alone), filters, inline
  disclosure detail with provenance and gap-to-portfolio jump, save/hide,
  honest freshness/degradation line.
- `web/components/scout/PortfolioPanel.tsx` — new; skill chips (≤6),
  ModelChooser, scaffold preview, fflate zip download, gh CLI instructions,
  placeholder-bullet honesty note.
- `web/tests/unit/scout-profile-store.test.ts` — new; 3 tests.
- One transient full-suite failure on first run did not reproduce across
  two subsequent full runs and isolated scout-suite runs.

### Task G — handoffs (complete; tsc + lint clean; 85 files / 799 tests)

- `web/app/(app)/jobapp-drafter/page.tsx` — accepts `?job=<postingId>`;
  loads the public posting and passes `initialJob` (no ownership check
  needed for public content; invalid/stale ids prefill nothing).
- `web/components/jobs/JobAppAssistant.tsx` — optional `initialJob` prop
  seeds the four job fields; "Loaded from Job Scout" status line; sends
  `postingSource=job_scout` only while the posting text is unedited; after
  documents generate, a "Practice the interview for this job" link carries
  `?application=<id>` to Interview Mentor. No behaviour change without the
  prop.
- `web/app/api/applications/route.ts` — honors `postingSource=job_scout`
  only when posting text is present (descriptionSource union extended
  additively).
- `web/app/(app)/interview-mentor/page.tsx` — accepts `?application=<id>`,
  ownership-checked via getOwnedApplication, prefills InterviewMentor.
- `web/components/interview/InterviewMentor.tsx` — optional `initialJob`
  prop seeds company/title/url/posting; status line; sends `applicationId`
  only while the title is unedited.
- `web/app/api/interview/route.ts` + `web/lib/db/index.ts createInterview`
  — accept optional applicationId (re-ownership-checked server-side; stale
  ids dropped, never blocking), finally setting the schema's reserved
  `interviews.applicationId` FK. No migration needed.
- `web/tests/unit/interview-application-link.test.ts` — new; 1 test.

### Task H — fixtures, e2e, script, docs

- `web/lib/scout/mock-fixtures.ts` — new; six deterministic postings seeded
  at boot in mock mode only (instrumentation.ts hook).
- `web/tests/e2e/job-scout.spec.ts` — new; profile setup → matched feed →
  filters/disclosure → JobApp handoff prefill → scaffold generation →
  axe WCAG A/AA on both views → unauthenticated 401/redirect block.
  First run: 11/13 (strict-mode locator on the scaffold heading); fixed and
  re-run.
- `web/scripts/generate-course-skills.ts` + `package.json` script
  `scout:course-skills` — new; two-model (claude-sonnet-5 + gpt-5.6-terra)
  regeneration producing an AGREE/ONLY review markdown; dev-only, never
  bundled, never overwrites course-skills.ts automatically.
- `web/scripts/chatisa-server.mjs` — launcher note for missing scout keys.
- `web/scripts/make-deploy-bundle.mjs` — chatisa.env.example gains the Job
  Scout section (names + comments only).
- `docs/development/decision-log.md` — ADR-025 (cached public postings +
  in-process scheduler, local-first privacy), ADR-026 (flash tagging with
  cost cap, revision history).
- `docs/CHANGELOG.md` — Unreleased section for Job Scout.
- `PROJECT_MEMORY.md` — build entry incl. the needs-a-human list (live
  harvest never run; instructor mapping review pending; first real Sunday
  firing unverified).

## Final verification (2026-07-28)

- `npx vitest run`: 85 files / 799 tests pass (three consecutive full runs).
- `npx tsc --noEmit`: exit 0. `npm run lint`: clean.
- `npx playwright test job-scout`: 13/13 across desktop + mobile-320
  (after fixing one strict-mode locator in the spec itself).
- `git log`: no commits made after the spec commit 8364257; all
  implementation work sits uncommitted in the working tree (43 paths:
  22 modified, 21 new), per user instruction.
- Not exercised and honestly outstanding: a live harvest against the real
  JSearch/USAJobs keys, the instructor review of the course-skill mapping,
  and the first scheduled Sunday 2 AM run on the production server.

## Follow-up session (2026-07-28, same day): visa extraction, 30-day rule, seed run

Three user requests: name the env keys, seed the feed with a manual run
today, drop jobs older than a month; plus "are we extracting visa
sponsorship?" (we were not; now we are).

- `web/.env.local` (gitignored) — RAPIDAPI_KEY set with the user's JSearch
  key; USAJOBS_API_KEY / USAJOBS_EMAIL added blank (no USAJobs account
  yet); CHATISA_SCOUT_ADMINS=megahefm@miamioh.edu.
- **Visa sponsorship extraction** (user request; careerbridge had it, the
  redesign had dropped it): `scout_postings.visa_sponsorship` column
  (migration `0008_previous_stryfe.sql`, default "unknown"); tagging schema
  + prompt classify `sponsors | no_sponsorship | unknown` from the
  posting's own words only; feed passthrough; card badge ("Posting
  mentions/says no visa sponsorship", silent when unknown) and a
  "Hide 'no visa sponsorship' postings" filter; mock + fixtures updated
  (fix-002 sponsors, fix-003 no_sponsorship).
- **30-day age rule** (user decision): harvest drops dated postings older
  than 30 days before any model spend; `retireScoutPostings` gains
  `postedBeforeIso` so already-stored postings age out weekly; spec §4.3
  updated (replaces the earlier dropped-35-day note).
- `web/tests/unit/scout-harvest.test.ts` — Date frozen at 2026-07-28
  (Date-only fake timers) so fixture postedAt values never age past the
  30-day rule and rot the suite; new age-drop test. `scout-db.test.ts` —
  postedBefore retirement test. Suite: 85 files / 801 tests green; tsc and
  lint clean.
- `web/scripts/run-harvest.ts` + npm script `scout:harvest` — command-line
  harvest against CHATISA_DATA_DIR (`NODE_OPTIONS=--conditions=react-server`).
- **Seed run executed** against C:\chatisa-data per user instruction
  ("manual run ... today as an exception"); results recorded below when
  complete.

### Seed run post-mortem (run 59db5b15, 2026-07-29T00:45Z): FAILED, three real bugs found

The run made 160 JSearch requests in 14.7 minutes and stored nothing.
Diagnosis against a live captured payload found, in order:

1. **Wrong response envelope.** /search-v2 nests results under
   `data.jobs`, not `data: [...]` (my fixture was modeled on the classic
   /search shape). Every 200 normalized to zero postings silently. Fixed;
   fixture rewritten from the real captured payload; legacy shape still
   accepted defensively.
2. **The key is on the BASIC plan: 200 requests/month, not the 20,000 the
   user believes they have.** The run itself consumed the remaining quota
   (429: "exceeded the MONTHLY quota ... plan, BASIC"); resets ~47 h after
   2026-07-29T01:05Z (≈ Thu Jul 30, evening ET). User action needed:
   upgrade this key to the 20k plan on RapidAPI, or supply the key that
   actually carries the 20k subscription.
3. **Full state names.** Real payloads send "Ohio"/"Kentucky"; the
   two-char slice would have corrupted KY→"KE", PA→"PE". Shared
   `toStateCode()` (types.ts) now maps names and codes for both sources;
   USAJobs' local table removed.

Hardening shipped with the fixes: JSearch 429 stops the query loop after
ONE request (never again 160 burns against a dead quota; unit-pinned);
source errors now keep the LAST message plus a failure count (first-error-
wins had hidden the dominant failure); catch-path errors include the
undici cause code; part-time-only listings dropped; per-ad INTERN
employment type overrides the query pass's category.

Verification after fixes: 85 files / 804 unit tests, tsc, lint all clean.
Seed retry pending quota (or USAJobs registration, which would seed the
federal track independently today).

### Seed retry round (2026-07-29, ~01:10Z onward): keys fixed, two more bugs found live

- User supplied a working JSearch key (10,000 requests/month plan; the old
  key stays dead at BASIC/200) and USAJobs credentials
  (megahefm@miamioh.edu). All three set in `.env.local`.
- Run 536511e1: **652 postings collected (envelope fix works; 140/160
  queries ok, 20 hit the 12 s timeout), 424 after dedupe/gates — and
  tagged 0 at $0.** Diagnosis: Gemini rejects a 104-value enum in its
  response schema (`INVALID_ARGUMENT`, isolated by schema bisection:
  3-value enum OK, 104-value enum fails). The per-posting catch swallowed
  all 424 failures silently.
- Fixes: wire schemas (tagging + resume extraction) now take plain skill
  strings; `resolveSkillId()` (taxonomy.ts) enforces the closed vocabulary
  after generation via id → label/alias → underscore normalization, with
  unknowns dropped and duplicate resolutions keeping the stronger
  importance. Tagging failures are now counted and reported in
  `sourceErrors.tagging` with the last error message, and a run that tags
  nothing despite candidates is `partial`, never `completed`. Also:
  `scoutRunInProgress()` ignores "running" rows older than 2 h (a killed
  process would otherwise block every future harvest).
- Live verification: single tagPosting against real Gemini returns correct
  skills, importance split, and caught an explicit "sponsorship is not
  available" phrase as no_sponsorship. ~$0.004/posting.
- USAJobs live probe through searchUsajobs: 100 postings, one request,
  clean normalization.
- 85 files / 806 unit tests, tsc, lint clean. Full seed harvest relaunched
  (JSearch + USAJobs + working tagging); results below.

### Seed complete (run a0295ae0, finished 2026-07-29T01:50Z): 418 active postings

- Collected: 608 JSearch + 628 USAJobs = 1,236; 696 after dedupe and the
  title/age gates; **418 tagged and stored**; cost **$5.49** (under the $10
  cap; higher than the $0.004/posting probe because federal descriptions
  run long).
- Feed shape: 220 full-time / 132 federal / 66 internships; OH leads named
  states (60), 69 remote; visa stance: 6 explicit sponsors, 49 explicit
  no-sponsorship, 363 unknown (honest — most ads say nothing). Top skills
  look sane: communication, data_analysis, problem_solving, sql (88),
  python (75), cybersecurity (107).
- Known tuning items for a later pass, not blocking: 23 of 160 JSearch
  queries hit the 12 s timeout (consider 20 s or one retry); 125 of 696
  tagging calls failed with "could not parse the response" (flash being
  flash — a retry-once would recover most); seniority-rejected postings
  (~153) are skipped without a counter of their own.
- `.env.local` AUTH_URL switched to http://localhost:3000 for local
  browsing (production chatisa.env unaffected).

## Flow & usability iteration (2026-07-29, user feedback after browsing the live feed)

Plan approved in plan mode (four tabs with My Projects before the feed;
projects feed the profile once a repo URL exists; resume persists on the
student's device). Changes, all client-state or additive API:

- `web/components/scout/JobScout.tsx` — rewritten as the tab shell
  (role=tablist, arrow-key nav, ?tab= via replaceState); fetches the feed
  index once and shares it across tabs; strengths now include built-project
  contributions.
- `web/components/scout/ProfileTab.tsx` (+`SkillsPanel.tsx`, `FilePick.tsx`)
  — replaces ProfileSetup: popular-first course chips per tier
  (instructor's subsets; disclosures for the rest; graduate tier collapsed),
  live "Skills you are building" panel (category groups, Strong/Working/
  Introduced words, source provenance), "This week's most-wanted skills"
  demand comparison, manual add-a-skill, styled resume button (the unclear
  native input from the screenshot), accept/dismiss suggestion cards, and
  device-resume storage with a remove control.
- `web/lib/scout/device-files.ts` — new; best-effort IndexedDB
  (js-files-v1) for the resume PDF and scaffold JSON, modeled on
  lib/ask/file-store.ts. Server storage unchanged: none.
- `web/components/scout/ProjectsTab.tsx` — replaces PortfolioPanel:
  generator plus persistent artifact cards (localStorage js-projects-v1 +
  scaffold JSON in IndexedDB for re-download), "I pushed it" repo link, and
  the built-project skill contribution (applied level, gated on repoUrl —
  `projectExtras()` in profile-store). README-polish button deferred: the
  project route has no polish mode yet; regenerating would masquerade as
  polishing.
- `web/components/scout/JobFeed.tsx` — props-driven (index from parent),
  multi-state checkbox filter with per-state counts, saved toggle now
  records a snapshot, handoff button text "Draft my resume and cover
  letter", savedOnly toggle replaced by the Saved tab.
- `web/components/scout/SavedTab.tsx` — new; active saves as full cards,
  retired saves as honestly-labelled snapshot rows.
- `web/lib/scout/profile-store.ts` — saved store v2 (snapshots + v1
  migration), projects store, `projectExtras()`; `use-scout-store.ts`
  gains the projects hook.
- `web/lib/scout/feed-types.ts` — shared FeedPosting/FeedIndex +
  `demandRanking()`.
- `web/lib/scout/courses.ts` — POPULAR_CODES (instructor's lists);
  `matching.ts` — `strengthWord()`.
- `web/app/api/scout/feed/route.ts` + `lib/db/index.ts` — `shape=index`
  one-shot feed (listAllScoutPostings; ~600 KB pre-gzip at 2,000 postings).
- `web/components/scout/DeviceResumeOffer.tsx` — new; "Use it here" offer
  rendered in JobApp Drafter and Interview Mentor resume sections (both
  files touched additively; no change without a device resume).
- `web/app/(app)/job-scout/page.tsx` — intro softened to "Draft a
  customized resume and cover letter with JobApp Drafter" (user wording).
- Tests: scout-profile-store rewritten (v2 migration, projects gating,
  POPULAR_CODES integrity); scout-routes gains shape=index;
  `tests/e2e/job-scout.spec.ts` rewritten to the tab flow incl. the
  device-resume handoff. Unit suite 85 files / 812 tests green; tsc and
  lint clean (tsc noise from the RUNNING dev server's .next generated
  types only).

### State filter follow-up (2026-07-29, user discussion)

- User questioned the 50-checkbox state filter and floated a Power BI
  slicer; pushed back on Ctrl-click multi-select (hidden affordance, fails
  touch and screen readers) and shipped the alternative the user approved:
  top states BY POSTING COUNT as one-click chips, tail behind "More
  states" with a type-ahead over full state names ("ken" finds Kentucky),
  selected tail states surfacing as removable chips. Same popular-first
  pattern as the course picker.
- Ranking documented at user request: code comment at the sort (score
  desc, postedAt tiebreak, filters never reorder) plus a one-line
  student-facing note under the feed heading.
- California added to the harvest markets (San Francisco + Los Angeles;
  +20 requests/run, 188 total, still test-bounded ≤600).
- e2e state test updated to the chip UI. Unit suite 812/812; tsc + lint
  clean. The prior e2e background run is disregarded: it died in auth
  setup because mid-run source edits kept its dev server recompiling
  (environmental, not a defect); rerun launched after edits settled.
- **e2e rerun: 15/15 pass** (desktop + mobile-320): live skills panel,
  chip state filter, saved-jobs home, project artifacts with repo-link
  gating, device-resume forwarding into JobApp Drafter, axe WCAG A/AA on
  the profile and jobs tabs, unauthenticated 401 block. The flow iteration
  is fully verified; still zero commits.

## Card fixes + "Polish a project I already built" (2026-07-29, user screenshot + discussion)

- **Title cleanup**: `cleanTitle()` in jsearch.ts strips aggregator-stuffed
  " at <company> <location>" suffixes (the Park Place Technologies card);
  existing rows self-heal on the next harvest's upsert refresh.
- **Professional skills demoted in matching** (user decision from the live
  card whose gaps were Teamwork and Problem Solving): kind "professional"
  job skills score at preferred weight, never count in the required
  fraction, and never appear as gaps; covered ones still show in "You
  bring". Trade-off (consulting-ish roles) documented at scoreJob.
- **Polish mode, the new primary project path** (user direction: the
  careerbridge upload-your-materials idea, done better; "organize +
  suggest" depth chosen): `POST /api/scout/polish` takes the student's
  real files (text content transiently, binaries by name), returns an
  organization plan — repo layout mapping EVERY upload (deterministic
  server guard), grounded README with a Suggested-improvements section,
  .gitignore, publish-exclusions with reasons, honesty-ruled resume
  bullets, taxonomy-resolved skills. Their code ships verbatim; the zip is
  assembled client-side from their own originals. Artifact recorded
  immediately and — unlike scaffolds — its skills count in the profile at
  once (`projectExtras`: polished OR repoUrl), because the work already
  exists. Re-download rebuilds from stored text originals and names any
  binaries the student must re-add.
- ProjectsTab gains the mode toggle (polish default; scaffold = "Start
  something new"); mock model gains a polish branch that maps real
  filenames so the zip path is genuinely exercised.
- Tests: cleanTitle + professional-demotion units, polish route placement
  guard, polished-counts-immediately store test, new e2e polish flow.
  Unit suite 85 files / 818 tests green; tsc + lint clean.
- **Final e2e sweep: 17/17 pass** (desktop + mobile-320), after two
  strict-mode locator fixes in the new polish test (the repo name and the
  "Suggested improvements" phrase each legitimately render twice — result
  pane plus artifact card, heading plus README preview). Feature state at
  end of 2026-07-29: complete and fully verified; zero commits.

## Feedback round (2026-07-29 continued): heading, level overrides, feed retry, logo

- "Your courses" → "Your ISA courses" (ProfileTab + e2e assertions).
- **Skill level overrides** (user request): every row in the skills panel
  is now a select — the computed word stays as "X (from your courses)" and
  the student can set Strong/Working/Introduced, in either direction, or
  revert to Auto. Stored in the local profile (`overrides` on
  js-profile-v1, backward compatible), applied last over the noisy-OR in
  `profileStrengths` (strong 1.0 / working 0.6 / introduced 0.25), so
  matching, demand standing, and the fraction all honor the student's own
  word. Sources note "set by you". Unit-tested both directions plus
  override-creates-skill.
- **Feed failure made recoverable**: the alert now carries a "Try again"
  button (reload nonce, no page refresh). Root-cause investigation: the
  identical query serializes cleanly against the real seeded DB (418 rows,
  361 KB) and the e2e suite loads the same endpoint on every fresh boot,
  so the user's live failure is attributed to their long-running dev
  server sitting through ~40 source edits (the same churn that corrupted
  .next/dev/types mid-write). Remedy: restart `npm run dev`.
- next/image logo warning: `w-auto` added beside `h-auto` on the login and
  guest page marks.
- 818/818 unit, tsc + lint clean; full e2e sweep re-run after changes
  (17/17).

## Release v6.1.0 (2026-07-29, user lifted the no-commit freeze)

- Version 6.0.0 → 6.1.0 (package.json + lock); CHANGELOG's Unreleased
  section became "v6.1.0 - July 29, 2026" with the polish/overrides/
  matching items added; release notes at `docs/releases/v6.1.0.md`.
- Root `.gitignore` gains `new_feature/` (the careerbridge prototype
  currently lives OUTSIDE the repo at the workspace root; the entry covers
  it if ever moved inside).
- First bundle build FAILED its own self-test — correctly: the new `scout`
  deep-health field reports "no harvest yet" in the self-test's fresh data
  dir, and the OPTIONAL_DEEP allowlist is deliberately narrow (the exact
  failure speech hit on 2026-07-26). Allowlisted `scout: /^no harvest
  yet$/` only; any other non-ok scout value still fails a bundle.
- Footer "Updated" date stamps automatically during the bundle build (the
  v6.0.0 pipeline behavior, unchanged).
- Pre-commit sweep: real key values appear nowhere outside gitignored
  .env.local; no .db, guest-link, or deploy files staged.
