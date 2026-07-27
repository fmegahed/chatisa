# ChatISA roadmap (deferred and planned)

A living list of features intentionally deferred so the current work can ship,
plus larger planned modules. Ordered roughly by group. Update as items ship.

## AI Sandbox (Coding Studio)

### Data viewer (shipped: scroll + row/column counts + paging + refresh)

Deferred enhancements, in rough priority order:

- **Column sort** by clicking a header (asc/desc), like RStudio.
- **Filter bar** per column (RStudio's Filter row), for numeric ranges and text.
- **Auto-refresh** an open data tab when its frame changes on the next Run
  (today the student clicks Refresh).
- **Cell formatting**: right-align numbers, show NA/NULL distinctly, truncate
  very long strings with a tooltip.
- **Column resize** and a sticky first (row-number / key) column.
- **Jump to page / page size** control for very large frames.
- **Export** the current view (CSV download).

### Other Sandbox items

- **Import Dataset menu** (RStudio-style, above the Environment): per-language file
  upload into the session, so a student can load their own CSV/Excel into a data
  frame (Python `pd.read_csv`, R `read.csv`, SQL `CREATE TABLE` from a CSV). This
  is the file-upload feature; next planned Coding Studio piece.
- **SQL plots**: a grammar-of-graphics default example for SQL (noted by the user
  as alpha/speculative); explore whether an in-browser plotting path for SQL
  results is worth a default sample.
- **Insert code from the chat into the editor** (deferred by choice; today the
  chat is read-only advice with Copy).
- **R highlighting**: a full lezer R grammar so function calls are highlighted
  distinctly from variables (today R uses a legacy stream grammar that tags all
  identifiers the same, so functions and variables share one colour).
- **Completions**: a completion-model picker in the toolbar (the route already
  accepts an allow-list override; default is a fast groq model); an option for a
  fill-in-the-middle-tuned model; smarter multi-line preview.
- **R web scraping (rvest / httr2 / curl)** — PLANNED, being specced next. The
  real enabler is cross-origin isolation (COOP/COEP), which gives WebR the
  SharedArrayBuffer channel and therefore synchronous networking; R's libcurl
  then tunnels through a ws-proxy (SOCKS5 over WebSocket), so CORS never applies.
  Confirmed by a spike on 2026-07-23: `rvest::read_html` scraped the FSB
  directory (which sends no CORS headers) in our own WebR 0.6.0 once isolated.
  This is NOT a CORS/HTTP proxy (the earlier note here was wrong). Plan: enable
  cross-origin isolation, set `ALL_PROXY` to a swappable target (ship on the
  public `ws.r-universe.dev`), bundle rvest/httr2/curl/xml2, and fix the
  Limitations notice (it currently states the opposite of the truth).
- **Self-hosted ws-proxy** — DEFERRED to the Next.js production deployment.
  Production is a Windows VM (Conda + Streamlit today, bound to 443 with its own
  cert via `webapp/chatisa.bat`; no Docker, no reverse proxy), so the stock Linux
  container `ghcr.io/r-wasm/ws-proxy` does not fit. Chosen direction: a small
  native Node WebSocket-to-SOCKS5 relay co-located with the app, reusing the
  cert, egress-filtered (block RFC1918 / loopback / link-local / cloud metadata)
  and gated to authenticated ChatISA sessions so it is not an open relay. Design
  it as part of solving Next.js production TLS termination. See the SSRF-guarded
  job fetcher for reusable egress-guard patterns.
- **Python web scraping (requests / beautifulsoup4)** — DEFERRED, and harder
  than R. Pyodide's `requests` is fetch-based and CORS-bound; it does not get R's
  socket tunnel. `beautifulsoup4` parses fine once you have the HTML. Needs its
  own HTTP CORS-proxy or a Pyodide socket shim; its own slice.

### Coding Studio GUI polish (A to F) — professor spec 2026-07-23

A large, multi-feature epic to decompose into slices. Full requirements in
`2026-07-23-coding-studio-polish-requirements.md`. Summary: (A) Clear Console
button; (B) Ctrl/Cmd+Click context docs in a HELP tab next to PLOTS, dialect-aware
for SQL; (C) scrollable long scripts; (D) Ctrl/Cmd+Enter runs the complete logical
statement (language-aware R pipes/ggplot/multi-line calls, Python indentation
blocks + continuation, SQL statements/CTEs/subqueries/multi-statement/procedural);
(E) continuation auto-indent + language-aware linting; (F) export variables and
workspaces (per-language formats, import/restore, serialization security, never
leak credentials). Suggested slices: (1) A+C quick wins; (2) D; (3) E; (4) B;
(5) F. Decompose and plan per slice before building.

## General AI (new module, PLAN MODE requested)

- **General ChatISA window** (working title; needs a name, e.g. "Ask Anything")
  as tab 7: a Claude.ai / ChatGPT-style chat. Professor scope (2026-07-23): a
  clone of popular chat sites with two caveats: (a) any chat history / multiple
  chats must live on the USER side for privacy (consistent with the no-server-chat
  retention decision), and (b) the student can choose from any of our LLM
  providers, with a good default. Figure out ALL needed features in PLAN MODE:
  tool access, web search (via the providers' own tools, not a separate service),
  running code results, producing outputs, downloadable documents, file inputs,
  etc. Large, multi-slice; this one is an interactive plan-mode design, not a
  parallel build.

- **AI Comparison** (tab 8): blind side-by-side comparison of two LLMs. Anonymous
  (time-seeded random pair) or pick-models; n prompts (default 1, max 5); one
  prompt at a time, two answers side by side, student votes left/right; final
  report reveals both models, highlights the winner, and shows each model's votes.
  Ephemeral (no server persistence). Spec:
  `2026-07-23-ai-comparison-spec.md`. Build-ready; implementation plan being
  written.

## Cross-cutting / naming (next slice)

- **Tab rename + reorder** into three groups (class 1-4, jobs 5-6, general 7-8):
  1 Coding Tutor, 2 Coding Studio, 3 Exam Prep, 4 Project Assistant,
  5 JobApp Drafter, 6 Interview Mentor, 7 General window, 8 AI Comparison.
- **Home page**: update the module groups, names, and count to match.
- **Author/version**: single author (Fadel Megahed); version and update date
  (target release July 27, 2026) shown in the footer.
- **Favicon**: Miami beveled-M.

## Infrastructure (later)

- **Slice 10 CSP**: must include `script-src 'wasm-unsafe-eval' 'self'`,
  `worker-src 'self' blob:`, `connect-src 'self'` plus the WebR package repo host
  (`https://repo.r-wasm.org`) for user-installed R packages, or execution and R
  installs break.
