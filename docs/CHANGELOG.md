# ChatISA Changelog

All notable changes to ChatISA are documented in this file.

---

## v6.4.5 - August 23, 2026

**Portfolio Builder: coursework lines show "ISA 444 - Business
Forecasting" in bold.** Full notes: `docs/releases/v6.4.5.md`.

---

## v6.4.4 - August 23, 2026

**Portfolio Builder: coursework lines show the course title; course codes
are normalized against what the student listed.** Full notes:
`docs/releases/v6.4.4.md`.

---

## v6.4.3 - August 23, 2026

**Portfolio Builder: the showcase preview shows uploaded figures, and the
file meter explains the three files every site carries.** Full notes:
`docs/releases/v6.4.3.md`.

---

## v6.4.2 - August 23, 2026

**Portfolio Builder: the preview shows the student's photo instead of a
broken image, and the mock GitHub login name is overridable for demos.**
Full notes: `docs/releases/v6.4.2.md`.

---

## v6.4.1 - August 23, 2026

**Portfolio Builder: generation no longer fails on ordinary uploads, the
publish limits are raised, and the module is hardened.** Full notes:
`docs/releases/v6.4.1.md`. "The request was malformed" came from the
generate route rejecting any file whose text ran past 30,000 characters (most
real .Rmd and .R files), plus unbounded names, semesters, and teammate names.
The route now clips those to its limits instead of refusing the request. The
push caps rise from 400 KB per file and 2 MB per site to 25 MB per file and
100 MB per site (60 files); text files above 400 KB and notebooks above 5 MB
are published as-is but not read for the page text.

Hardening in the same pass: the wizard autosaves to IndexedDB and offers an
unfinished site back on the next visit (a reload no longer loses uploads);
the upload steps state the limits up front; base64 is encoded natively to
keep large files from crashing a small browser; the GitHub push shows
per-file progress, can be cancelled, retries a dropped upload once, and
reads GitHub's secondary rate limit as "wait, then retry"; the generate
route bounds its request body and the browser trims file text before
sending; a test proves an uploaded file cannot close its own prompt fence.

---

## v6.4.0 - August 21, 2026

**Portfolio Builder: publish a portfolio site or a project showcase to GitHub
Pages, with a preview you edit first.** Full notes:
`docs/releases/v6.4.0.md`.

### Added
- **Portfolio Builder** at `/portfolio`, listed first under "For your job
  search", in two modes: a **career portfolio** (resume, classes, one to five
  projects, optional photo, links) published to a single `portfolio`
  repository, and a **project showcase** (one course project told as problem,
  data, approach, findings, deliverables, team) published to its own
  `<course>-<title>` repository.
- A review step that shows the generated page in a sandboxed preview beside a
  field-by-field editor; nothing reaches GitHub until Publish, and
  republishing updates the same repository.
- Browser-only storage: site records in localStorage, drafts and files in
  IndexedDB. The server reads uploads transiently and keeps nothing; the only
  server records are the existing content-free usage events.
- Publishing safeguards: data files start unpublished with a reminder that
  course datasets are often licensed or instructor-provided; the photo is
  resized in the browser to a 512 px JPEG, which strips EXIF; the resume PDF
  is published only behind an opt-in checkbox that warns the phone number and
  address on it become public; and the push limits (60 files, 400 KB per
  file, 2 MB total) are metered and enforced before generation, not at push.
- New routes `POST /api/portfolio/generate` (both modes; the model emits
  content JSON only, and the server re-validates slugs, figures, deliverables
  and skills against the material that was actually submitted) and
  `POST /api/portfolio/event` (a content-free publish count).
- JobApp Drafter: an opt-in **"Include my published work"** toggle, off by
  default, that adds portfolio and showcase links and summaries to the
  candidate material.

### Changed
- The browser push engine (`lib/scout/github.ts`) uploads binary files as
  base64 blobs, so photos, PDFs, figures, notebooks and Office documents
  survive a push.
- Job Scout loses the **Portfolio Site** tab and the **Polish a project I
  already built** pane; both live in the Portfolio Builder now, and link
  cards point at the mode that replaced them. `/job-scout?tab=portfolio` and
  `/job-scout/github-connected` redirect, and the connect landing page is
  module-neutral at `/portfolio/github-connected`.
- A published site's skills count toward the Job Scout profile exactly as a
  built project's do. Projects polished under v6.3.0 keep their records and
  their skills but can no longer be re-pushed.

### Fixed
- GitHub connect failed in production with "redirect_uri is not associated
  with this application": the OAuth redirect was built from the internal
  request origin behind the TLS relay, and is now built from `AUTH_URL`.
- Zip downloads and the command-line instructions are removed from Job
  Scout's projects. GitHub is the only destination.

## v6.3.0 - August 20, 2026

Full notes: `docs/releases/v6.3.0.md`.

## v6.2.1 - July 30, 2026

**Job board quality and resume export fixes**, all found while filming the
v6.2.0 demo videos.

### Fixed
- Exported resumes no longer print the Education or Skills sections twice
  (the renderer drew its structured blocks and the model's own sections),
  and the school name no longer leaks into the degree line.
- Academic postings (professor, faculty, lecturer, instructor, adjunct)
  no longer pass the harvest relevance gate; they are not jobs students
  apply to.
- Feed ranking now shrinks by tag evidence, so a posting tagged with a
  single skill can no longer score a perfect 1/1 and outrank a broad 6/7
  match. Displayed scores and coverage are unchanged; only ordering.

## v6.2.0 - July 29, 2026

**Job Scout feed rebuilt on employer-direct sources.** Full notes:
`docs/releases/v6.2.0.md`.

### Changed
- Job Scout's board is now sourced from **Active Jobs DB** (employer career
  sites and 54 ATS platforms, direct apply links) plus USAJobs; JSearch is
  removed after one production week showed its aggregator-heavy results
  (ADR-027). The harvest query matrix becomes six role-cluster requests
  budgeted against the plan's monthly returned-jobs quota.
- Every search is filtered to ISA fresh grads and internships at the API
  itself: full-time clusters carry the source's 0-2/2-5 years experience
  band, internship clusters its INTERN employment type, so senior postings
  are never fetched. People-management titles (manager, lead, supervisor)
  no longer pass the relevance gate; leadership development programs still
  do.
- Scout postings move to their own database file (`scout.db`); the
  aggregator-era rows remain in `chatisa.db`, preserved but no longer read.
  The board refills automatically on the first boot after deploy.
- Clearly senior postings and source-flagged duplicates are dropped before
  any model spend.

---

## v6.1.1 - July 29, 2026

**Code and notebook files as first-class uploads.**

### Added
- Ask Anything accepts `.py`, `.R`, `.Rmd`, `.qmd`, `.ipynb`, and `.html`
  attachments. Code files ride as text regardless of the (often blank)
  MIME type the browser reports; Jupyter notebooks are converted
  client-side to their markdown and code cells with capped text outputs,
  and up to four plot outputs are attached as real images the model can
  see.
- My Projects polish reads notebooks properly: cells are extracted before
  the per-file text cap (previously the cap usually landed inside the
  first base64 plot blob, so the model never saw the code), and the raw
  size allowance for notebooks rises to 5 MB. The zip still ships the
  student's original bytes untouched.
- The polish prompt knows R and Python coursework ecosystems: rendered
  HTML, `_files/`, `.Rproj.user`, `.ipynb_checkpoints`, `__pycache__`,
  `.venv`, and `renv/library` are steered to exclude/.gitignore while
  lockfiles stay.
- Polish now says so when a file is too large to read or when more than
  15 files are chosen, instead of silently degrading.

### Fixed
- A polish request whose repo layout echoed a filename with a space
  ("Final Project.ipynb") failed entirely; paths are now hyphenated
  deterministically instead.
- Re-downloading a polished project zip now writes the full original text
  files, not the truncated copies sent to the model.

## v6.1.0 - July 29, 2026

**Job Scout: a weekly job board matched to the ISA curriculum.** Full notes:
`docs/releases/v6.1.0.md`.

### Added
- **Job Scout**, a ninth module leading the job-search group: a weekly board
  of ISA-relevant postings (JSearch + USAJobs, harvested Sundays 2 AM
  Eastern), matched in the browser against the courses a student has taken
  and the skills they confirm from their resume. Profiles and saved jobs
  never leave the device (ADR-025); posting tagging runs on a flash-tier
  model with a per-run cost cap (ADR-026).
- One-click handoffs: a Job Scout posting prefills JobApp Drafter
  (`descriptionSource: "job_scout"`), and finished JobApp documents link to
  Interview Mentor with the same job, finally wiring the
  `interviews.applicationId` column reserved for it.
- A job-agnostic portfolio project generator: pick skills (a job's gap list
  pre-checks them), get a scaffold with README, starter files, dataset
  pointers, and GitHub CLI instructions as a zip assembled in the browser.
- The course-to-skill mapping is data under test: ~190 leveled links over
  the live bulletin catalog, with an instructor review table and a two-model
  regeneration script (`npm run scout:course-skills`).
- Visa sponsorship stance extracted per posting from the ad's own words
  (`sponsors` / `no_sponsorship` / `unknown`), shown only when the posting
  stated one, with a filter to hide explicit "no sponsorship" listings.
  Postings older than 30 days are never stored and age out of the feed.
- Job Scout reorganized into four tabs (My Profile, My Projects, This
  Week's Jobs, Saved Jobs): popular-first course chips with a live "skills
  you are building" panel and weekly demand comparison; generated projects
  persist as artifacts whose skills count in the profile once a repo link
  is added; saved jobs survive posting retirement; multi-state filtering
  (top states as chips, the rest behind a type-ahead); and the profile
  resume stays on the student's device so JobApp Drafter and Interview
  Mentor can reuse it with one click.
- "Polish a project I already built": upload real coursework files and get
  a repo layout, grounded README, publish-exclusion warnings, and honest
  resume bullets back, with the student's code shipped verbatim. The
  primary project path; polished projects count in the skill profile
  immediately.
- Students can override any computed skill level (Strong / Working /
  Introduced) in the skills panel, in either direction.

### Changed
- Matching no longer counts professional skills (teamwork, communication)
  as required or lists them as gaps; they weigh in at preferred only.
- Aggregator-stuffed job titles ("... at Company City, ST") are cleaned at
  harvest.
- Postings are ranked by requirement-coverage match, newest first on ties,
  and the feed states this.

## v6.0.0 - July 26, 2026

**The Streamlit application is replaced by a full-stack Next.js platform.** Full
notes: `docs/releases/v6.0.0.md`.

### Added
- Eight modules: Coding Tutor, Coding Studio, Exam Prep, Project Assistant,
  JobApp Drafter, Interview Mentor, Ask Anything, AI Comparison
- Google sign-in restricted to verified `@miamioh.edu` accounts
- Python, R, and SQL executed in the student's own browser (Pyodide, WebR,
  SQLite WASM), self-hosted with a mirrored R package repository
- Spoken interview practice on Deepgram, with live captions and a feedback report
- A house chart style shared across every module that plots
- A self-verifying deploy bundle, and a live-model test harness

### Changed
- 18 models across 4 providers, each verified against the provider's live listing
- All provider calls server-side; the legacy app embedded a client secret in page
  HTML and in an iframe URL fragment
- Analytics start clean; the legacy JSONL activity log is not migrated
- The Streamlit app moves to `legacy/` and still runs

### Removed
- Chat conversation content is no longer stored anywhere (ADR-022)
- Uploaded documents are never persisted; only sampled excerpts (ADR-015)

### Testing
- 735 unit tests and 17 end-to-end suites, from a starting point of none,
  including 50 characterization tests pinning the port against the legacy Python

### Note
- The transition from Streamlit to this platform was carried out with extensive
  use of Claude Code, under a staged plan-then-implement workflow. See
  "How this was built" in `docs/releases/v6.0.0.md`.

---
## v5.0.3 - December 18, 2025

### Bug Fix
- Added missing `pdf4llm` package to requirements.txt

---

## v5.0.2 - December 18, 2025

### Bug Fix
- Fixed Gemini models returning list instead of string causing TypeError

---

## v5.0.1 - December 18, 2025

### Changes
- Cleaned up `requirements.txt` to include only direct dependencies with latest versions
- Replaced deprecated `use_container_width` with `width="stretch"` (Streamlit 1.52+)
- Removed unused LangChain import causing errors with LangChain 1.2.0
- Suppressed Pydantic V1 compatibility warning for Python 3.14+

---

## [v5.0.0](releases/v5.0.0.md) - December 18, 2025

### Highlights
- **SOTA Models**: Added latest models from OpenAI (GPT-5.2, GPT-5 Mini), Anthropic (Claude Sonnet 4.5), and Google (Gemini 3 Pro/Flash Preview)
- **UI/UX Overhaul**: Consistent sidebar design with Miami colors across all six modules
- **Navigation Emojis**: Added matching emojis to all navigation elements
- **AI Sandbox Enhancements**: Dynamic model selection, configurable via tags
- **Clear Conversation**: Added button to Coding Companion, Project Coach, and Exam Ally

### Bug Fixes
- Fixed Gemini 3 Pro empty response issue (reasoning tokens)
- Fixed GPT-5 Mini temperature parameter error
- Fixed AI Sandbox image download issue

[Full release notes](releases/v5.0.0.md)

---

## Version Format

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible changes or significant new features
- **MINOR**: New functionality in a backward-compatible manner
- **PATCH**: Backward-compatible bug fixes
