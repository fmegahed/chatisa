# ChatISA Changelog

All notable changes to ChatISA are documented in this file.

---

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
