# Job Scout — design (2026-07-28)

A new module in the "For your job search" group, listed before JobApp Drafter.
Job Scout keeps a weekly-refreshed board of ISA-relevant job postings, matches
them against a student's coursework and resume, hands a chosen job to JobApp
Drafter (and onward to Interview Mentor) without retyping, and generates
portfolio project scaffolds the student pushes to their own GitHub.

This is the native redesign of the `new_feature/careerbridge/` prototype.
Careerbridge's faculty workflows (company registry, crawl management, approval
queues, portfolio exams) are deliberately dropped: Job Scout is student
self-service, and the one human-review step (the course-to-skill mapping)
collapses into a one-time diff review by the instructor.

## Decisions locked (user, 2026-07-28)

- **Local-first privacy.** Saved jobs, courses taken, resume-derived skills,
  and match preferences live in `localStorage` on the student's device. The
  server never learns which jobs a student viewed or saved. A server record
  appears only when the student clicks "Prepare with JobApp Drafter," which
  creates a `job_applications` row exactly as pasting a posting does today.
- **Sources: JSearch (RapidAPI) and USAJobs only.** RemoteOK explicitly
  excluded (low-quality listings).
- **Harvest scope:** entry-level full-time across Ohio metros + regional hubs,
  remote + major national markets, internships/co-ops, and a federal track
  via USAJobs.
- **Schedule:** Sundays 2:00 AM America/New_York, in-process scheduler.
- **Models: frontier everywhere.** Weekly posting tagging uses a frontier
  model (claude-sonnet-5), not a budget tier. Student-facing generation uses
  the standard catalog via `ModelChooser`.
- **GitHub: scaffold download, no OAuth.** No tokens ever held.
- **Skill profile: courses + optional resume**, resume processed transiently.
- **Course catalog: live bulletin** (`bulletin.miamioh.edu/courses-instruction/isa/`),
  not the stale careerbridge JSON. Independent Studies excluded.
- **Name:** Job Scout, slug `job-scout`, `ModuleKey` `job_scout`.

Considered and rejected: server-side saved-job lists (behavioural data, new
privacy category); GitHub OAuth device flow (token custody, failure surface —
possible later slice); grade/recency factors in matching (self-report friction
and privacy cost for little signal); RemoteOK (user exclusion).

## 1. Student flow

One page, numbered stages in the house style (`ribbon`, `max-w-4xl` shell).

1. **Your profile.** Checklist of ISA courses taken (bundled catalog, §2).
   Optional resume upload: the server extracts taxonomy skills from it via a
   frontier model and returns them for the student to confirm and level
   (a skill used at an internship may outrank a course). Confirmed profile is
   written to `localStorage` (`js-profile-v1`); the resume itself is processed
   in memory (existing JobApp PDF pipeline) and never stored — ADR-015 rules.
   Internship (ISA 340) and Topics (480/481) entries prompt one line of
   "what did you work on?"; a model maps that line into taxonomy skills,
   stored locally like resume skills.
2. **This week's jobs.** The pre-populated feed with freshness banner
   ("Updated Sunday, July 26 · 1,842 postings"). Filters: category
   (full-time / internship / federal), state/metro, remote, skill. Each card
   shows a match band (Strong / Good / Stretch), match %, matched skills, and
   gap skills — computed deterministically in the browser (§3), so the profile
   never leaves the device. Save/hide is local (`js-saved-v1`). A failed
   source degrades to the other with an honest note, never a broken page.
3. **Act on a job.** Detail view: full description, prominent
   **"Apply on employer site"** link (original posting URL), **"Prepare with
   JobApp Drafter"** (§5), and **"Build a portfolio project"** (§6).
4. **Portfolio project.** Scaffold generation and GitHub instructions (§6).

## 2. Skill layer

### 2.1 Taxonomy (`lib/scout/taxonomy.ts`)

One closed vocabulary (~120–140 ids) shared by course mapping, resume
extraction, and job tagging, so matching is exact set arithmetic. Rebuilt, not
inherited: careerbridge's YAML filed linear algebra under soft skills, gave
`defi` its own id while folding AWS/Azure/GCP into one, and predates GenAI.

- An id exists only if employers name the skill in postings **and** a student
  can evidence it as a unit. Named tools (Tableau, Power BI, SQL, Python, R,
  Excel, AWS, Azure, Snowflake, Salesforce, SAP, Git/GitHub…) get ids;
  everything else is an alias.
- Each id: `label`, `kind` (`tool | method | domain | professional`),
  `category`, `aliases[]` (deterministic text matching + highlighting),
  shallow `implies[]` edges (`power_bi → data_visualization`,
  `arima → forecasting`) for specific↔general partial credit. One level deep,
  no ontology.
- New coverage: generative AI / LLM applications, prompt engineering, version
  control, data governance, modern data stack (Snowflake, Databricks, dbt as
  ids or aliases per the naming rule).
- `TAXONOMY_VERSION` constant; job tags record the version they were made with.

### 2.2 Course catalog and mapping (`lib/scout/courses.ts`, `lib/scout/course-skills.ts`)

- **Source:** the live bulletin page, scraped once by the generation script
  into a committed JSON snapshot with the fetch date recorded. Re-run only
  when the curriculum changes. Excluded: Independent Studies
  (177/277/377/477/677). Special checklist entries (free-text, no static
  mapping): Internship 340, Topics 480/481. Cross-listed pairs (401/501,
  414/514, 444/544, 491/591) are single entries answering to both codes;
  cross-department listings (STA 125/250/333/365, ACC 305, BUS 645) recorded
  so a student can check the code they took. Roughly 44 mapped courses,
  including the new ISA 336 Generative AI in Business.
- **Levels per course-skill link:** `anchor` (graded deliverables demonstrate
  it), `applied` (used repeatedly as a working tool), `exposure` (introduced).
- **Evidence phrase per anchor/applied link** ("built and evaluated ARIMA and
  exponential-smoothing forecasts on business data") — grounded material for
  the project generator and JobApp's resume bullets, solving hallucination at
  the data layer.
- **Generation script** (`scripts/generate-course-skills.mjs`, dev-machine
  only, never shipped): two frontier models (claude-sonnet-5 and a GPT-5.6
  tier) independently map each course description against the taxonomy via
  structured output (`{skillId, level, evidence}`, skillId enum-forced).
  Agreements auto-accept; disagreements are flagged in a review markdown with
  both models' choices. The instructor reviews the table once (~20 minutes,
  ideally in-session), edits, commits. Credit hours are carried per course so
  1.5-credit courses (241, 242, 628, 629) weigh less than 3-credit anchors.
  Accepts an optional syllabi folder later for richer evidence.

### 2.3 Student profile aggregation (client-side, `lib/scout/matching.ts` — shared pure code)

Per-skill strength aggregates course links, credit weight, and confirmed
resume/internship skills with **noisy-OR** (diminishing returns: each source
contributes independently; three Python courses ≈ strong but bounded).
Level weights: anchor 1.0, applied 0.6, exposure 0.25, scaled by
credits/3. Output: strength 0–1 per skillId, plus provenance ("Python: built
across 4 courses, including ML applications") for display and for grounded
resume bullets.

### 2.4 Matching formula (deterministic, in-browser, unit-tested)

Job tags carry `importance: required | preferred` (§4.3). Score is
requirement coverage, not Jaccard:

```
score = Σ over job skills ( importanceWeight × studentStrength* ) / Σ importanceWeight
  importanceWeight: required 1.0, preferred 0.5
  studentStrength*: direct strength, or best implies-edge strength × 0.6
```

Asymmetric on purpose: extra student skills never penalize. Displayed as
bands — Strong (score ≥ 0.70), Good (≥ 0.45), Stretch (below) — with the
breakdown
spelled out ("You cover 5 of 6 required skills; gap: cloud computing"). The
gap list, sorted by importance, feeds the project generator.

## 3. Privacy model

- Server-side: **public employer content only** (postings + tags) and
  content-free `usage_events` (module `job_scout`: `module_open`,
  `feed_view`, `project_generated`, `handoff` — never which posting).
- Client-side: `js-profile-v1` (courses, confirmed skills with levels),
  `js-saved-v1` (saved/hidden posting ids). Both survive only in the browser.
- Resume and free-text internship lines: request-scoped, in-memory, returned
  and discarded — nothing persisted, nothing logged (pino redaction already
  covers emails; prompt/response bodies are never logged).
- The feed API is authenticated (standard three-layer auth) but requests are
  not associated with postings in any table; access logs are the standard
  content-free pino lines.
- New ADR records the one genuinely new category: server-cached public job
  postings.

## 4. Weekly harvest (server)

### 4.1 Scheduler (`lib/scout/scheduler.ts`, registered in `instrumentation.ts`)

- Hourly `setTimeout` chain; on each tick, compute "is a Sunday 2:00 AM
  America/New_York run due that has not happened" from `scout_runs` (DST-aware
  via `Intl.DateTimeFormat` parts, no date library). Catch-up at boot if the
  server was down at 2 AM (the launcher restarts the child on crash, so timers
  must be derived from persisted state, never assumed).
- Guards: `NEXT_RUNTIME === "nodejs"` only; skipped when `NODE_ENV === "test"`
  or `CHATISA_MOCK_LLM=1`; a run already `running` blocks a second (single
  process, plain in-memory flag + DB status).
- Manual trigger: `POST /api/scout/refresh`, `auth()`-gated **and** restricted
  to emails in `CHATISA_SCOUT_ADMINS` (comma-separated env). Returns 202 and
  streams nothing; progress is visible in `scout_runs`.
- Freshness in `GET /api/health?deep=1`: last successful run time, posting
  count, per-source status. Stale feed (> 8 days) surfaces as a warning.

### 4.2 Sources and query plan

- **JSearch** (`jsearch.p.rapidapi.com/search-v2`, headers `X-Rapidapi-Key`,
  `X-Rapidapi-Host`): curated query matrix — ~10 role families (business
  analyst, data analyst, BI analyst/developer, data scientist entry, data
  engineer entry, information systems analyst, IT auditor, information
  security analyst, cybersecurity analyst, business/technology consultant)
  × geography passes (Ohio metros: Cincinnati, Columbus, Cleveland, Dayton;
  regional hubs: Chicago, Indianapolis, Louisville, Pittsburgh; national
  markets + remote) with `date_posted=week` (each harvest picks up the new
  week), `country=us`, and a separate `employment_types=INTERN` pass for the
  internship track. ~300–600 requests/run against ~4,600/week available
  (20k/month plan). Per-source courtesy rate limit and 8–12 s timeouts,
  matching the `paper-search.ts` pattern.
- **USAJobs** (`data.usajobs.gov/api/search`, headers `Authorization-Key`,
  `User-Agent: <email>`): keyword + occupational-series searches for federal
  analytics/IT/audit/cyber roles; tagged `category = federal`.
- Normalizers are pure functions with an injectable `Fetcher`, unit-tested on
  fixture payloads; a failing source degrades the run to the other and records
  an honest per-source status.

### 4.3 Pipeline

normalize → dedupe → relevance filter → tag → store.

- **Dedupe:** `(source, externalId)` unique, plus a
  company+title+location fingerprint across sources (JSearch aggregates
  boards; the same role appears twice).
- **Relevance filter:** cheap deterministic pass (title/keyword) drops the
  obviously off-target before any model spend.
- **Tagging (frontier, claude-sonnet-5):** `generateObject` per posting with
  the posting text nonce-fenced (the `lib/documents/generate.ts` fence —
  harvested text is untrusted input), emitting
  `{skills: [{skillId (enum), importance: required|preferred}], category, seniorityOk}`.
  Bounded concurrency, retry-once, and a **per-run cost cap**
  (`CHATISA_SCOUT_MAX_RUN_USD`, default 75): beyond the cap the run stores
  untagged postings as inactive and marks itself `partial`. Expected cost
  ~$25–60/run at Sonnet pricing; actual cost recorded per run.
- **Retention:** a posting missing from two consecutive harvests or older
  than 35 days goes `active = false`; rows deleted after 90 days. Public
  content, but no reason to hoard it.

## 5. Handoffs

- **Job Scout → JobApp Drafter:** "Prepare with JobApp Drafter" navigates to
  `/jobapp-drafter?job=<postingId>` (Next 16 async `searchParams`, the
  login-page precedent). The server page loads the public posting and passes
  `initialJob` (company, title, applyUrl, postingText) into `JobAppAssistant`;
  `descriptionSource` gains the value `"job_scout"` — the provenance field was
  built for exactly this. From there the existing flow is unchanged, and the
  application row is created only when the student continues, preserving
  local-first semantics.
- **JobApp Drafter → Interview Mentor:** close the long-dangling loop. After
  documents generate, "Practice this interview" navigates to
  `/interview-mentor?application=<id>`; the server page verifies ownership
  (`getOwnedApplication`) and prefills InterviewMentor, finally setting the
  never-used `interviews.applicationId` FK. Students go feed → documents →
  mock interview without retyping a job.
- **Apply:** always the original posting URL, opened externally. Job Scout
  never proxies applications.

## 6. Portfolio project generator + GitHub

From a job's detail view (seeded by its gap skills and evidence phrases from
the student's own courses — sent explicitly by the client in the request, not
looked up server-side):

- A frontier model (student-picked via `ModelChooser`) generates a project
  brief via `generateObject`: README mapping the project to the job's named
  skills, milestone plan, pointers to real public datasets, starter file
  stubs (folder layout, notebook/R script skeletons, `.gitignore`), suggested
  repo name.
- Delivered as a **zip assembled client-side** (`fflate`, small and
  tree-shakeable) from the returned file manifest, plus copy-paste
  `git init` / `gh repo create` instructions. No OAuth, no tokens, nothing
  persisted server-side; the request is stateless and rate-limited.
- Optional: paste the finished repo URL back for README polish and grounded
  resume bullets (reusing JobApp's honesty rule + grounding check).

## 7. Data model (new tables, Drizzle `lib/db/schema.ts`)

```
scout_postings: id (uuid text PK), source ("jsearch"|"usajobs"), externalId,
  fingerprint, title, company, locationCity, locationState, remote (bool),
  category ("fulltime"|"internship"|"federal"), applyUrl, description,
  postedAt, harvestedAt, lastSeenAt, expiresAt, skillsJson
  ([{skillId, importance}]), taxonomyVersion, active (bool)
  unique (source, externalId); index (active, category)

scout_runs: id, startedAt, finishedAt,
  status ("running"|"completed"|"partial"|"failed"), trigger ("schedule"|"manual"),
  jsearchRequests, jsearchFound, usajobsRequests, usajobsFound,
  taggedCount, dedupedCount, costUsd, error
```

No student-keyed tables. Migration via `drizzle-kit generate`, applied at boot.

## 8. Module registration & config

- `lib/modules.ts`: Job Scout inserted before `jobapp-drafter` in the jobs
  group. Slug `job-scout`; page `app/(app)/job-scout/page.tsx`.
- `lib/config/models.ts`: `ModuleKey` `job_scout`;
  `PAGE_MODELS.job_scout = { includeAll: true, requireStructuredOutput: true, minContextWindow: 64000 }`;
  a `DEFAULT_MODELS` entry. Server routes enforce
  `getPageModels("job_scout").includes(modelId)`.
- Env (`lib/config/env.ts`, optional feature keys — the
  `SEMANTIC_SCHOLAR_API_KEY` precedent; missing keys degrade the feed, never
  block boot): `RAPIDAPI_KEY`, `USAJOBS_API_KEY`, `USAJOBS_EMAIL`,
  `CHATISA_SCOUT_ADMINS`, `CHATISA_SCOUT_MAX_RUN_USD`. Names added to
  `.env.example`, `chatisa.env.example`, and the launcher's RECOMMENDED list.
  **Key values live only in `.env.local` / `chatisa.env` — never in the repo,
  docs, or memory files.**
- API routes (all `auth()`-gated, mock-mode aware):
  `GET /api/scout/feed` (filters + pagination), `GET /api/scout/postings/[id]`,
  `POST /api/scout/resume-skills`, `POST /api/scout/project`,
  `POST /api/scout/refresh` (admin-gated).

## 9. Testing & ops

- **Unit:** normalizers (fixture payloads per source), dedupe/fingerprint,
  scheduler due-logic (DST boundaries: the 2 AM run around March/November
  transitions), noisy-OR aggregation, matching formula bands, taxonomy
  integrity (ids unique, implies/aliases resolve, course-skills reference
  real ids).
- **e2e** (`tests/e2e/job-scout.spec.ts`): profile → feed → detail → handoff
  path against seeded fixture postings under mock mode; the standard axe
  WCAG A/AA sweep; unauthenticated 401/redirect block; JobApp prefill
  assertion (`descriptionSource === "job_scout"`).
- **Mock mode:** harvest never runs; a fixture seed provides a deterministic
  feed; all `generateObject` call sites branch to `getMockModel()`.
- **Ops:** run history and cost in `scout_runs`; freshness in deep health;
  provider failures classified via `classifyProviderFailure`, never leaked raw.

## 10. Records to write alongside implementation

- ADR: server-cached public job postings + in-process weekly scheduler
  (first server-side background job; first server-cached external content).
- ADR: frontier-model tagging with per-run cost cap (user decision,
  2026-07-28).
- `docs/CHANGELOG.md` and `PROJECT_MEMORY.md` entries per house convention;
  roadmap entry removed/marked done if one is added.

## Out of scope (recorded so they stay out)

- Faculty dashboards, curriculum analytics, approval queues (careerbridge's
  F1–F8) — the mapping review replaces them.
- GitHub OAuth / direct repo creation — possible later slice.
- Cross-device profile sync — would require server-side student data.
- Employer-page crawling — JSearch/USAJobs only, per user decision.
- Email alerts/digests — the app has no outbound email today.
