# Ask Anything (tab 7, slug `general-chat`) — Design

Approved by the professor 2026-07-24 (architecture, name, roster, file scope, web hierarchy, runtime constraints). Supersedes the roadmap's "General ChatISA window" sketch.

## What it is

A Claude.ai / ChatGPT-style assistant, provider-agnostic, with a real code-interpreter loop that runs in the student's own browser. The old app rented OpenAI's server-side virtual environment, so only OpenAI models could "create stuff." Here, any tool-capable model writes Python/R/SQL, the student's tab executes it on the Coding Studio WASM runtimes, results (including plots) stream back to the model, and it iterates. Private (data never leaves the tab), provider-agnostic, zero server compute.

## Product shape

Two-pane page at `/general-chat`, display name **Ask Anything**:

- **Left sidebar:** New chat, list of past chats (title + relative time), delete per chat. Collapsible on mobile.
- **Conversation pane:** messages render through the existing chat Markdown (code blocks keep Copy / Run / Customize). Tool activity renders as collapsible inline cards: "Ran Python, 2.3s" with code, output, and any plot; "Read example.com"; "Searched the web"; "Created document, Download".
- **Composer:** attach button, message box, model picker, send/stop.

## Chats live on the device (privacy)

Consistent with ADR-022 (chat content is never persisted server-side): the chat list and full message history live in localStorage (`aa-chats-v1`, versioned), the same pattern as the Coding Studio drafts. Titles derive client-side from the first user message. Delete is local; there is nothing to delete on the server. The server records only the existing content-free `usage_events` row per call. localStorage is ~5MB: oldest-chat trimming with a visible note when near quota; plot images stored as compressed data URLs; a chat's runtime session (Python variables etc.) is ephemeral and NOT persisted, only the transcript.

## Model roster

Curated 8, all **vision + text + tools + structured output** (the professor's filter, so every model in the picker can see attached images and drive tools):

GPT-5.6 Sol, GPT-5.6 Terra, GPT-5.6 Luna, Claude Opus 4.8, **Claude Sonnet 5 (default)**, Gemini 3.1 Pro, Gemini 3.6 Flash, Kimi K2.7 Code. (GLM-5.2 and DeepSeek-V4-Pro are excluded: tools+structured but no vision.)

Registered as `ask_anything` in `CHAT_MODULES` and `PAGE_MODELS` (`specificModels`, explicit list). The picker badges which models have provider-native web search/fetch.

## The agentic loop

- Server route `/api/ask-anything` (sibling of `/api/chat`, sharing auth, rate limiting, error classification, and usage events) runs `streamText` with tool definitions **without `execute`** and `stopWhen: stepCountIs(10)`.
- Tool calls stream to the browser; `useChat`'s `onToolCall` executes them in the tab; `addToolResult` + `sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls` returns results so the model continues. (All present in our AI SDK: verified.)
- Per-chat runtime sessions (`createSession` per language, lazily): variables persist across turns within a chat; New chat = fresh sessions.
- Tool results sent to the model are truncated (~8k chars); the student's card shows the full output. Plots return to the model as "a plot was produced" plus render fully in the card (image bytes stay client-side; vision models MAY receive the plot as an image part, a slice-B decision knob, default off to control cost).

## Tools

| Tool | Runs | Via |
|---|---|---|
| `run_python` | browser | Pyodide worker (pandas, matplotlib, sklearn, ...) |
| `run_r` | browser | WebR worker (tidyverse, ggplot) |
| `run_sql` | browser | SQLite worker |
| `create_document` | browser | existing docx renderer, branded .docx download (bounded, see Outputs) |
| `read_url` | server | SSRF-guarded fetch (reuses `lib/jobs/fetch-posting.ts` guards + `htmlToText`), modes: cleaned text / raw HTML |
| web search + fetch | provider | native tools on commercial models (OpenAI webSearch; Anthropic webSearch/webFetch; Google search/urlContext) |
| hosted code interpreter | provider | OpenAI `codeInterpreter` (container files), Anthropic `codeExecution_20260120` (file outputs), Google `codeExecution` (inline results only) — for what the browser cannot do; see Execution routing |

## Execution routing: browser first, hosted when needed (professor's revision)

Two code-execution families coexist in one agentic loop, and the **system prompt is the "ask filter"**: the model is taught to route each job rather than us building a separate classifier.

- **Browser tools first** (`run_python`/`run_r`/`run_sql`): private (data never leaves the tab), free, instant, sessionful. The default for data analysis, plots, CSV/xlsx outputs, .docx, and anything the bundled runtimes cover.
- **Provider-hosted interpreter** when the task needs what WASM lacks: PowerPoint generation (python-pptx), compiled packages (statsforecast, prophet...), heavy/long compute, or file formats our runtimes cannot write. Available on OpenAI, Anthropic, and Google models (inline-only on Google); Kimi has browser tools only.
- **Privacy is visible and biased toward the browser:** a hosted-execution card is labeled "Ran on {provider}'s servers." The prompt directs the model not to send the student's uploaded data to the hosted sandbox when the browser can do the job, and to say so when it must.
- **Generated files** (.pptx, .xlsx, images) from OpenAI/Anthropic sandboxes are fetched server-side via the provider file APIs (`/api/ask-anything/files/...`, auth-guarded, streamed to the browser as a download). Verified: our SDK exposes `codeInterpreter` + container file ids (OpenAI), `codeExecution_20260120` + `file_id` (Anthropic), `codeExecution` (Google).
- **Cost honesty:** hosted interpreter sessions bill outside token usage; the usage event for such turns records the event with the model's token cost and an `outcome` marker noting hosted execution, and a per-turn cap (1 hosted session) bounds spend.

## Outputs the model can create (bounded, professor's revision)

The tool contract promises only what the browser can genuinely produce well, so the model never advertises capability it lacks:

- **Word documents (.docx):** the existing `docx` JS renderer (proven in Project Assistant; light, not WASM), Miami-branded, downloadable.
- **Data files:** CSV (pandas/readr) and .xlsx (openpyxl) written by `run_python` / `run_r`, surfaced as downloads from the tool card.
- **Plots:** PNG images from matplotlib/ggplot, rendered inline and downloadable.
- **Beyond the browser's power** (professor's revision): PowerPoint generation, PDFs, compiled-package work, and other heavy outputs are NOT faked in WASM; they route to the provider-hosted interpreter on OpenAI/Anthropic models (python-pptx and friends in a real sandbox), and the generated file is fetched server-side for download. On models without file-producing hosted execution (Google inline-only, Kimi), the model says so plainly and offers .docx or Markdown instead. Reading pptx/docx as input stays in-browser (cheap jszip text extraction).

## Web content hierarchy (professor's revision)

Taught to the model in the system prompt, in priority order:

1. **The model's own provider tools first** (commercial models): native web search and native URL fetch (Anthropic `webFetch`, Google `urlContext`). Best quality, zero infrastructure of ours.
2. **`read_url`** (every model, incl. Kimi): our server fetches the page (SSRF-guarded, egress-filtered), returning cleaned text for reading or raw HTML for parsing.
3. **Structured extraction:** hand raw HTML to `run_python` (beautifulsoup4 / lxml / `pandas.read_html`) or scrape directly in `run_r` (rvest/httr2 through the ws-proxy; acknowledged as not the strongest, hence the bs4 path). Python code itself can never fetch (CORS); R can.

## Runtime constraints are part of the tool contract

The system prompt and tool descriptions state the environment precisely, so the model never guesses:

- **Python (Pyodide):** preloaded numpy, pandas, matplotlib, scipy, scikit-learn, statsmodels, pyarrow, polars, seaborn, openpyxl, **+ (new bundling below) beautifulsoup4, lxml, html5lib, requests**. Only Pyodide-built or pure-Python packages can be added; compiled packages (statsforecast, pyreadr, tensorflow) can never install. **No network from Python code** (CORS): use `read_url` and feed content in. Single-threaded, no subprocess, ~60s timeout, plots captured automatically.
- **Import-failure enrichment:** when a `run_python` error is an import failure, the tool result appends the `classifyPythonPackage` verdict (from `pyodide-lock.json`), so the model gets "statsforecast cannot be installed here (needs compiling); statsmodels is preloaded" instead of a bare traceback, and self-corrects in one step.
- **R (WebR):** bundled tidyverse/readxl/janitor/httr2/rvest mirror; most of CRAN installable from the webR WASM repo. R is the language that CAN reach the web (ws-proxy). First `run_r` in a session takes ~30s warm (IndexedDB library restore); the tool description warns the model and the card shows "Preparing R (first use)...".
- **SQL:** SQLite dialect only, in-memory per-chat DB.

## New bundling (required, professor-approved)

The self-hosted Pyodide mirror currently holds only the 23-wheel closure of the preload list, so lock-listed packages like beautifulsoup4 are not actually importable from our origin (also a latent accuracy gap in the existing "What can I install?" copy and checker). Extend `PYTHON_PACKAGES` in `scripts/setup-runtimes.mjs` with `beautifulsoup4`, `lxml`, `html5lib`, `requests` (closure pulls soupsieve, urllib3, charset-normalizer, idna, certifi) and re-run the mirror build. R needs no new packages (rvest/xml2/httr2/jsonlite already bundled). The package checker's "available on import" wording is corrected to distinguish mirrored (import works) from Pyodide-built-but-unmirrored (micropip from PyPI for pure wheels; otherwise unavailable).

## Files in

Attach via the composer; extraction happens client- or server-side before send, so the model receives text/data parts, never raw binaries. ~25MB cap (Coding Studio convention).

- **Images** -> AI SDK image parts (whole roster is vision).
- **PDF** -> existing Exam Ally extractor (incl. scanned-page vision routing) -> text context.
- **csv / xlsx** -> loaded into the chat's Python session as a DataFrame (announced to the model: name, shape, columns) so `run_python` can analyze immediately.
- **docx / pptx** -> jszip XML text extraction (new small `lib/files/office-text.ts`) -> text context.
- **json / txt** -> text context (or into the Python session when large/tabular).
- **URLs** are not attachments; the model calls `read_url` (or its provider fetch) when the student pastes a link.

## Error handling

- Tool execution errors return as tool results (the model sees and recovers); the card shows the error; repeated failure (3x same tool) short-circuits with a student-visible note.
- Runtime-unsupported browser (no WASM): tools are omitted from the request and the UI says code execution is unavailable; plain chat still works.
- Provider errors reuse `/api/chat`'s student-safe error classification.
- localStorage quota: warn at 80%, trim oldest chats with a visible notice, never silently drop the active chat.

## Testing

- **Unit:** chat store (create/switch/delete/trim/versioning), tool schemas, output truncation, import-failure enrichment, office-text extraction, read_url guard reuse (SSRF cases), roster filter.
- **e2e (mock LLM):** scripted tool-call responses added to mock mode so the loop is deterministic and free. Chat shell CRUD + persistence across reload; a scripted `run_python` call executes on the real Pyodide worker and the result round-trips; attachment flows; axe on the page and dialogs at desktop + 320px.
- **Opt-in live:** one real Sonnet 5 run that produces a plot from an uploaded CSV (CHATISA_LIVE_NET gate).

## Build slices

- **A. Chat shell:** page + sidebar + localStorage store + model picker + streaming via new route (no tools yet). Replaces the placeholder module.
- **B. Agentic code tools:** run_python/r/sql loop, tool cards, plots, import-failure enrichment, mirror bundling expansion.
- **C. Files in:** images, PDF, csv/xlsx-to-runtime, docx/pptx/json extraction.
- **D. Create + web:** create_document, read_url, provider search/fetch wiring + picker badges, checker wording fix.
- **E. Hosted interpreter:** provider code-execution tools (OpenAI/Anthropic/Google), the browser-vs-hosted routing prompt, "Ran on provider's servers" cards, server file retrieval for generated .pptx/.xlsx, per-turn session cap.

Each slice lands with its tests green (typecheck, lint, unit, e2e incl. axe) before the next begins; migration-log entry per slice.

---

## Revision v3 (2026-07-24, approved in discussion)

Decisions made after slice B shipped, superseding parts of the design above:

1. **Providers: Anthropic + OpenAI only.** Roster is GPT-5.6 Sol/Terra/Luna,
   Opus 4.8, Sonnet 5 (default unchanged). Rationale: both providers accept
   the same native file parts (images, PDFs) so chats with attachments stay
   model-switchable; both have hosted code execution, removing slice E's
   cross-provider bridge. Gemini and Kimi remain in AI Comparison (whose
   includeAll-minus-speech roster is untouched). Re-adding a third provider
   later re-opens the capability matrix and the bridge; that cost is accepted
   knowingly.
2. **Files: native first.** Images and PDFs go to the model as file parts
   (figures, layout, and scans included); PDF caps 25 MB / about 100 pages.
   Word/PowerPoint are extracted client-side (Anthropic requires plain text);
   csv/xlsx load into the chat's Python session as DataFrames. Attachment
   payloads live in IndexedDB (aa-files-v1) with references in the
   localStorage chat, so chats with files survive reloads on-device (ADR-022).
3. **Web strategy: academic-first, no provider search fees.** All queries are
   expected to be academic, so provider webSearch is NOT wired. Instead:
   search_papers (arXiv + Semantic Scholar + OpenAlex, keyless, distilled and
   deduplicated server-side, Wikipedia background), get_paper (depth via
   Semantic Scholar), and read_url (general page reader reusing the JobApp
   SSRF guards). Provider webSearch remains a one-line addition if freshness
   needs appear.
4. **Miami-themed output.** Brand assets distilled from the professor's
   figure set and deck template live in web/assets/brand/ (canonical red
   standardized to #C41230). A get_miami_style tool serves the TikZ preamble,
   gantt exemplar, palette, and LaTeX-doc styling on demand. In slice E, deck
   generation injects the template .pptx into the provider container so
   hosted PowerPoints come out Miami-branded.
5. **Slice D is dissolved.** Web/academic search moved into slice C;
   create_document is cut (hosted generation in slice E covers real documents,
   PowerPoints included, with your rule-based routing and cost policy). The
   module completes with slice E.
