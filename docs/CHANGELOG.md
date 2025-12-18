# ChatISA Changelog

All notable changes to ChatISA are documented in this file.

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
