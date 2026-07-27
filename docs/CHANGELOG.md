# ChatISA Changelog

All notable changes to ChatISA are documented in this file.

---

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
