# Job Scout Implementation Plan (2026-07-28)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Build the Job Scout module per `2026-07-28-job-scout-design.md`:
weekly JSearch+USAJobs harvest, flash-tagged postings, local-first coursework
matching, JobApp/Interview handoffs, job-agnostic portfolio scaffolds.

**Architecture:** Additive Next.js module. Server: two new SQLite tables, an
in-process weekly scheduler, pure normalizers + a flash tagging pass. Client:
localStorage profile, deterministic matching, inline-disclosure feed.

**Tech Stack:** Next 16 App Router, Drizzle + better-sqlite3, AI SDK v7
(`generateObject`), Zod v4, Vitest, Playwright. New dep: `fflate` (client zip).

## Global Constraints

- **No git commits. Every change appended to
  `docs/development/2026-07-28-job-scout-build-log.md`** (user, 2026-07-28).
- Purely additive: existing modules keep identical behaviour; touched shared
  files (`modules.ts`, `models.ts`, `env.ts`, `schema.ts`, `instrumentation.ts`,
  health route, JobApp/Interview pages) get new branches only.
- Tagging model: `gemini-3.6-flash`. Cost cap env `CHATISA_SCOUT_MAX_RUN_USD`
  default 10.
- Sources: JSearch + USAJobs only. Key env names: `RAPIDAPI_KEY`,
  `USAJOBS_API_KEY`, `USAJOBS_EMAIL`. **Key values never enter the repo.**
- Local-first: no student-keyed server tables; profile/saved lists in
  `localStorage` (`js-profile-v1`, `js-saved-v1`).
- All LLM call sites branch on `CHATISA_MOCK_LLM === "1"` → `getMockModel()`.
- Server model policy per route: `getPageModels("job_scout").includes(modelId)`.
- Untrusted harvested/pasted text is nonce-fenced (reuse
  `lib/documents/generate.ts` fence pattern).
- Comments cite decisions/dates per house convention; student copy is plain
  second person; no em dashes in user-facing text; axe-clean a11y.
- Miami tokens only (`bg-paper`, `border-medium-tan`, `text-miami-red`,
  `rounded-card`, `.ribbon` once per view, coverage fraction not meters).

---

### Task A: Skill data layer (taxonomy, catalog, mapping, matching)

**Files:**
- Create: `lib/scout/taxonomy.ts` — `TAXONOMY_VERSION = 1`;
  `interface SkillDef { id: string; label: string; kind: "tool"|"method"|"domain"|"professional"; category: string; aliases: string[]; implies: string[] }`;
  `export const SKILLS: SkillDef[]` (~120–140 ids per spec §2.1, incl.
  generative_ai, prompt_engineering, version_control, snowflake, databricks);
  `export const SKILL_IDS`, `getSkill(id)`.
- Create: `lib/scout/courses.ts` — bulletin snapshot (fetched 2026-07-28):
  `interface CourseDef { code: string; altCodes: string[]; title: string; credits: number; description: string; special?: "freeform" }`;
  ~44 mapped courses + ISA 340/480/481 as `special: "freeform"`;
  Independent Studies excluded.
- Create: `lib/scout/course-skills.ts` —
  `interface CourseSkillLink { course: string; skillId: string; level: "anchor"|"applied"|"exposure"; evidence?: string }`;
  `export const COURSE_SKILLS: CourseSkillLink[]`. Initial mapping authored
  in-session by Fable 5 against the taxonomy (instructor reviews
  `docs/development/2026-07-28-course-skills-review.md`, generated in Task A);
  `scripts/generate-course-skills.mjs` ships later for regeneration (Task H).
- Create: `lib/scout/matching.ts` — pure, isomorphic:
  - `profileStrengths(courses: string[], extras: {skillId, level}[]): Map<string, number>`
    — noisy-OR: `1 - Π(1 - w)`, w = levelWeight × credits/3;
    anchor 1.0, applied 0.6, exposure 0.25 (extras: credits treated as 3).
  - `scoreJob(strengths, jobSkills: {skillId, importance}[]): { score, band, covered, gaps }`
    — spec §2.4 formula; implies credit ×0.6; bands ≥0.70 Strong, ≥0.45 Good.
- Test: `tests/unit/scout-taxonomy.test.ts` — ids unique; every `implies`
  and `COURSE_SKILLS.skillId` resolves; every mapped course has ≥1 anchor;
  every course code unique incl. altCodes.
- Test: `tests/unit/scout-matching.test.ts` — worked examples:
  single anchor 3-credit course → strength 1.0-capped noisy-OR ⇒ 0.999… no:
  strength = 1 - (1-1.0) = 1.0 exactly for one anchor; 241 (1.5cr anchor)
  ⇒ 0.5; two applied 3-cr courses ⇒ 1-(0.4)² = 0.84; scoreJob with
  2 required (one covered), 1 preferred (implied) ⇒
  (1.0×1 + 0 + 0.5×0.6×s)/2.5 asserted numerically; band edges 0.70/0.45.

**Interfaces produced:** `SKILLS`, `SKILL_IDS`, `SkillDef`, `CourseDef`,
`COURSES`, `COURSE_SKILLS`, `profileStrengths`, `scoreJob`,
`TAXONOMY_VERSION`.

- [ ] Write taxonomy + integrity tests; run `npx vitest run tests/unit/scout-taxonomy.test.ts` (fail → implement → pass)
- [ ] Write matching tests with hand-computed values; implement; pass
- [ ] Author course-skills mapping + review markdown; integrity tests pass
- [ ] Append files to build log

### Task B: Database tables and helpers

**Files:**
- Modify: `lib/db/schema.ts` — append `scoutPostings`, `scoutRuns` tables
  exactly as spec §7 (snake_case columns, ISO string timestamps, `*Json`
  text columns, unique `(source, external_id)`, index `(active, category)`).
- Create: `drizzle/00NN_*.sql` via `npx drizzle-kit generate`.
- Modify: `lib/db/index.ts` — append sync helpers:
  `insertScoutPostings(rows)`, `upsertScoutPosting(row)` (on conflict update
  lastSeenAt/skillsJson), `listScoutPostings({category?, state?, remote?, skillId?, limit, offset})`,
  `getScoutPosting(id)`, `deactivateStalePostings(cutoffIso, missedRunIds)`,
  `purgeOldPostings(cutoffIso)`, `createScoutRun(trigger)`,
  `finishScoutRun(id, patch)`, `latestSuccessfulScoutRun()`.
- Test: `tests/unit/scout-db.test.ts` — in-temp-dir DB (existing test
  pattern): upsert dedupes on (source, externalId); list filters by
  category; deactivate/purge respect cutoffs; run lifecycle.

**Interfaces produced:** the helper names above; `ScoutPostingRow`,
`ScoutRunRow` types exported from `lib/db/schema.ts`.

- [ ] Schema + generate migration; unit tests fail → helpers → pass
- [ ] `npx vitest run` full suite still green (no breaking change)
- [ ] Build log

### Task C: Harvest pipeline (normalizers, dedupe, tagging)

**Files:**
- Create: `lib/scout/sources/types.ts` —
  `interface RawPosting { source: "jsearch"|"usajobs"; externalId: string; title: string; company: string; locationCity: string|null; locationState: string|null; remote: boolean; category: "fulltime"|"internship"|"federal"; applyUrl: string; description: string; postedAt: string|null }`;
  `type Fetcher = typeof fetch`.
- Create: `lib/scout/sources/jsearch.ts` — `"server-only"`;
  `searchJsearch(query, opts, fetcher): Promise<RawPosting[]>` hitting
  `https://jsearch.p.rapidapi.com/search-v2` with `X-Rapidapi-Key` from env,
  `date_posted=week`, `country=us`; `normalizeJsearch(payload)` pure export
  for tests; 12 s timeout; honest per-source error result.
- Create: `lib/scout/sources/usajobs.ts` — same shape against
  `https://data.usajobs.gov/api/search` (`Authorization-Key`, `User-Agent`
  email); `normalizeUsajobs(payload)` pure.
- Create: `lib/scout/queries.ts` — `HARVEST_QUERIES`: role families ×
  geographies + intern pass per spec §4.2, as data (unit-testable count
  ≤ 600 requests/run).
- Create: `lib/scout/harvest.ts` — orchestrator:
  `runHarvest({trigger}): Promise<ScoutRunSummary>` = createScoutRun →
  fan through queries with bounded concurrency (4) → normalize → fingerprint
  dedupe (`fingerprint = lower(company)|lower(title)|state`) → deterministic
  relevance filter (title keyword allowlist/blocklist) → tag → upsert →
  deactivate/purge → finishScoutRun. Per-source failure degrades, recorded.
- Create: `lib/scout/tag.ts` — `tagPosting(posting, model): Promise<{skills, category, seniorityOk}>`
  via `generateObject` with skillId enum from `SKILL_IDS`, nonce-fenced
  description, `gemini-3.6-flash` via `getLanguageModel`, mock-mode branch;
  running cost accumulator + `CHATISA_SCOUT_MAX_RUN_USD` cap (default 10) →
  `partial` run.
- Test: `tests/unit/scout-sources.test.ts` — fixture JSON payloads
  (`tests/fixtures/scout/jsearch-page.json`, `usajobs-page.json`) →
  normalizer output asserted field-by-field; injectable fetcher never hits
  network; query matrix size bound.
- Test: `tests/unit/scout-harvest.test.ts` — fake fetcher + mock model:
  dedupe across sources by fingerprint; relevance filter drops junk title;
  cost cap flips run to `partial`; one source failing yields `completed`
  run with that source's error recorded.

**Interfaces consumed:** Task A `SKILL_IDS`; Task B helpers.
**Interfaces produced:** `runHarvest`, `ScoutRunSummary`
`{ runId, status, jsearchFound, usajobsFound, tagged, costUsd, sourceErrors: {jsearch?: string, usajobs?: string} }`.

- [ ] Fixtures + normalizer tests → implement → pass
- [ ] Harvest orchestrator tests (fake fetcher, mock model) → implement → pass
- [ ] Build log

### Task D: Scheduler, env, health, admin trigger

**Files:**
- Modify: `lib/config/env.ts` — add optional keys after the Ask Anything
  block with matching rationale comment: `RAPIDAPI_KEY`, `USAJOBS_API_KEY`,
  `USAJOBS_EMAIL`, `CHATISA_SCOUT_ADMINS`, `CHATISA_SCOUT_MAX_RUN_USD`
  (string, parsed float at use).
- Create: `lib/scout/scheduler.ts` — `"server-only"`:
  `nextDueTime(lastSuccessIso: string|null, now: Date): Date` pure (Sunday
  02:00 America/New_York via `Intl.DateTimeFormat` parts, DST-aware);
  `isHarvestDue(now): boolean`; `startScoutScheduler()` — hourly setTimeout
  chain, in-memory `running` flag, skips when `NODE_ENV==="test"`,
  `CHATISA_MOCK_LLM==="1"`, or no `RAPIDAPI_KEY`+`USAJOBS_API_KEY` at all.
- Modify: `instrumentation.ts` — after `getDb()`:
  `const { startScoutScheduler } = await import("./lib/scout/scheduler"); startScoutScheduler();`
- Modify: `app/api/health/route.ts` — deep section gains
  `scout: { lastRun, postingCount, stale }` (stale = >8 days).
- Create: `app/api/scout/refresh/route.ts` — POST; `auth()` +
  `CHATISA_SCOUT_ADMINS` (comma-separated, lowercased) membership; 202 +
  fire-and-forget `runHarvest({trigger:"manual"})`; 409 when already running.
- Modify: `.env.example` — names + one-line comments (no values).
- Test: `tests/unit/scout-scheduler.test.ts` — `nextDueTime` around DST
  (2026-03-08, 2026-11-01), catch-up when lastSuccess 8 days old, no
  double-fire when lastSuccess is this Sunday.

**Interfaces produced:** `startScoutScheduler`, `isHarvestDue`, `nextDueTime`.

- [ ] Scheduler pure-logic tests → implement → pass
- [ ] Wire instrumentation + health + refresh route; full vitest green
- [ ] Build log

### Task E: Module registration and API routes

**Files:**
- Modify: `lib/modules.ts` — insert Job Scout before `jobapp-drafter`:
  slug `job-scout`, name `Job Scout`, description
  "Browse this week's analytics and IS jobs, matched to the courses you have taken.",
  group `jobs`.
- Modify: `lib/config/models.ts` — `ModuleKey` + `"job_scout"`;
  `PAGE_MODELS.job_scout = { includeAll: true, requireStructuredOutput: true, minContextWindow: 64000 }`;
  `DEFAULT_MODELS.job_scout = "gpt-5.6-terra"` (matches jobapp).
- Create: `app/api/scout/feed/route.ts` — GET, auth'd; query params
  category/state/remote/skill/limit/offset; returns postings (id, title,
  company, location, remote, category, applyUrl, postedAt, skillsJson,
  taxonomyVersion) + freshness `{updatedAt, total, sourceErrors}`; no
  student data touched; `recordUsageEvent({module:"job_scout", eventType:"feed_view"})`.
- Create: `app/api/scout/postings/[id]/route.ts` — GET, auth'd; full row
  incl. description; 404 JSON otherwise.
- Create: `app/api/scout/resume-skills/route.ts` — POST multipart (PDF ≤ 8 MB);
  reuse `lib/jobs/read-resume.ts` transiently; `generateObject` (model from
  request, policy-checked; mock-aware) → `{skills: [{skillId, level, evidence}]}`
  against `SKILL_IDS` enum; nothing persisted; rate limit via existing
  `checkRateLimit` (10/min per user).
- Create: `app/api/scout/project/route.ts` — POST JSON
  `{modelId, skillIds (≤6, validated against SKILL_IDS), evidence: string[] (client-sent, ≤2000 chars total)}` →
  `generateObject` file-manifest schema
  `{repoName, readme, files: [{path, contents}] (≤12 files, ≤200 KB total), instructions}`;
  job-agnostic system prompt (never name an employer); mock-aware;
  `recordUsageEvent` with tokens/cost; rate limit 4/min.
- Test: `tests/unit/scout-routes.test.ts` — feed filter passthrough +
  401 unauthenticated (existing route-test pattern with mocked auth);
  project route rejects unknown skillId and oversize evidence.

**Interfaces produced:** route shapes above (client consumes in Task F).

- [ ] models/modules registration; typecheck + existing tests green
- [ ] Routes with tests → pass
- [ ] Build log

### Task F: UI (page + components + local stores)

**Files:**
- Create: `app/(app)/job-scout/page.tsx` — server page per house template
  (auth redirect, `filterAvailableModels(getPageModels("job_scout"))`,
  `recordUsageEvent module_open`, ribbon "Module", h1 "Job Scout", honest
  intro line naming sources and the weekly cadence, NoModels panel copy).
- Create: `lib/scout/profile-store.ts` — client-safe localStorage wrapper:
  `loadProfile(): ScoutProfile|null`, `saveProfile`, `loadSaved(): SavedState`,
  `toggleSaved(id)`, `hidePosting(id)`;
  `interface ScoutProfile { v: 1; courses: string[]; extras: {skillId, level, source: "resume"|"freeform"}[] }`.
- Create: `components/scout/JobScout.tsx` — top-level client component:
  first-run (no profile) renders `ProfileSetup`; else `ProfileBar` +
  freshness status + `FeedFilters` + `JobFeed` + `PortfolioPanel`.
- Create: `components/scout/ProfileSetup.tsx` — numbered stages 1–3:
  course checklist grouped by level (100/200/300/400/600) with cross-listed
  alt codes shown; freeform one-liners for 340/480/481; optional resume
  upload → `/api/scout/resume-skills` → confirm-and-level list (radio per
  skill: anchor/applied/exposure defaults applied); "See your matches".
- Create: `components/scout/ProfileBar.tsx` — "`N` courses · `M` skills ·
  Edit profile" (button, aria-expanded, reopens setup inline).
- Create: `components/scout/FeedFilters.tsx` — fieldset radios for
  category (All / Full-time / Internships / Federal), state select (from
  feed data), remote checkbox, "Saved only" toggle; all controlled props.
- Create: `components/scout/JobFeed.tsx` + `components/scout/JobCard.tsx` —
  fetches `/api/scout/feed`, client-side `scoreJob` per posting, sort by
  score; card = serif coverage fraction "5/6 required skills" + text band +
  title/company/location/category/postedAt; expand-in-place (aria-expanded,
  aria-controls) → description (fetch detail on first expand), matched
  skills with course provenance, gaps sorted by importance, action row:
  Apply on employer site (external link, rel noopener), Prepare with JobApp
  Drafter (`/jobapp-drafter?job=<id>`), Save toggle, "Build these skills"
  (scrolls to PortfolioPanel, pre-checks gap skills). "Load more" paging.
- Create: `components/scout/PortfolioPanel.tsx` — skill chip multi-select
  (checkbox list, pre-checkable), ModelChooser, Generate → renders README
  preview (pre) + file tree + Download zip (fflate `zipSync` client-side) +
  copy-paste `git init`/`gh repo create` block + "paste repo URL" polish
  input (calls project route with `mode:"polish"`? NO — YAGNI: polish is a
  second generate call with the README + repo URL in evidence; same route).
- Modify: `package.json` — add `fflate` dependency.
- Test: `tests/unit/scout-profile-store.test.ts` — round-trip, versioning,
  corrupt JSON → null (jsdom localStorage).

**Interfaces consumed:** Task A matching, Task E routes.

- [ ] profile-store tests → implement → pass
- [ ] Page + components; `npm run lint` + typecheck green
- [ ] Manual smoke via dev server + mock mode
- [ ] Build log

### Task G: Handoffs (JobApp prefill, Interview loop)

**Files:**
- Modify: `app/(app)/jobapp-drafter/page.tsx` — accept
  `searchParams: Promise<{job?: string}>`; when present and posting exists,
  pass `initialJob={{company, positionTitle, applyUrl, postingText: description}}`
  into `JobAppAssistant`. Public content; no ownership check needed (any
  student may prepare for any posting).
- Modify: `components/jobs/JobAppAssistant.tsx` — new optional prop
  `initialJob?: { company: string; positionTitle: string; applyUrl: string; postingText: string }`
  seeding existing useState initials only (no behaviour change when absent);
  a `role="status"` line "Loaded from Job Scout. Edit anything." when present.
- Modify: `app/api/applications/route.ts` — accept optional
  `descriptionSource: "job_scout"` from the form when postingText came from
  a scout posting (client sends flag; server validates enum), extending the
  pasted/fetched/none union additively.
- Modify: `lib/db/schema.ts` comment only if needed (descriptionSource is
  free text — verify; no migration expected).
- Modify: `app/(app)/interview-mentor/page.tsx` + `components/interview/InterviewMentor.tsx`
  — accept `searchParams {application?: string}`; server loads
  `getOwnedApplication(id, email)`; prefill company/jobTitle/postingText and
  pass `applicationId` through.
- Modify: `app/api/interview/route.ts` + `lib/db/index.ts createInterview`
  — accept optional `applicationId`, ownership-checked, sets the FK
  (schema field already exists; no migration).
- Modify: `app/(app)/jobapp-drafter/page.tsx` stage nav gains
  "1. Find a job (Job Scout)" link additively renumbered? NO — keep existing
  numbering untouched; add a plain "Found this job in Job Scout?" link only
  when `?job=` absent is NOT needed. Skip nav changes (YAGNI).
- Test: extend `tests/e2e/job-search.spec.ts`? No — new spec in Task H;
  unit test for createInterview applicationId ownership in
  `tests/unit/interview-application-link.test.ts`.

**Interfaces consumed:** Task B `getScoutPosting`; existing
`getOwnedApplication`, `createInterview`.

- [ ] createInterview applicationId unit test → implement → pass
- [ ] Page prefills; existing jobapp/interview unit+e2e suites still green
- [ ] Build log

### Task H: Fixtures, e2e, docs, script

**Files:**
- Create: `lib/scout/mock-fixtures.ts` + seed hook — when
  `CHATISA_MOCK_LLM==="1"` and scout tables empty, seed 12 deterministic
  postings (mix of categories/states/skills) at boot so e2e has a feed.
- Create: `tests/e2e/job-scout.spec.ts` — profile setup → feed renders with
  coverage fractions → filter → expand card → JobApp handoff asserts
  prefilled company + `descriptionSource "job_scout"` → axe WCAG A/AA sweep
  → unauthenticated 401/redirect block.
- Create: `scripts/generate-course-skills.mjs` — dev-only regeneration
  script per spec §2.2 (two models, agreement/disagreement markdown);
  reads bulletin snapshot JSON; not bundled.
- Modify: `docs/development/decision-log.md` — ADR-025 (cached public
  postings + in-process scheduler), ADR-026 (flash tagging + cost cap,
  revision history).
- Modify: `docs/CHANGELOG.md`, `PROJECT_MEMORY.md` — entries.
- Modify: `scripts/make-deploy-bundle.mjs` RECOMMENDED env list +
  `chatisa.env.example` names (via its generator) — names only.
- Finalize build log.

- [ ] Fixtures + e2e spec pass under mock mode (`npx playwright test job-scout`)
- [ ] Full `npx vitest run` + `npm run lint` + `npx tsc --noEmit` green
- [ ] Docs + ADRs written; build log finalized

## Self-review

- Spec coverage: §1→F, §2→A(+H script), §3→A/E/F (no student tables), §4→C/D,
  §5→G, §6→E/F, §7→B, §8→D/E, §9→A–H tests, §10→H. Gaps: none found.
- Placeholders: none (PortfolioPanel "polish" resolved inline as second
  generate call; nav renumbering explicitly rejected).
- Type consistency: `RawPosting` fields match schema columns; `scoreJob`
  band thresholds 0.70/0.45 match spec; `initialJob.postingText` feeds
  existing `postingText` state.
