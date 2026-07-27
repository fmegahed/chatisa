# Ask Anything slice C: files in + web search + academic search

Second revision of 2026-07-24: (1) the roster narrows to Anthropic and OpenAI;
(2) web search and academic (arXiv and friends) search move INTO this slice
from slice D, per the professor's direction. Students attach files in the composer: images, PDFs, Word,
PowerPoint, spreadsheets, csv, json, txt. Both roster providers accept images
and PDFs directly in messages, so those go through natively (the model sees
charts, layout, and scanned pages). File types no provider reads in messages
(Word, PowerPoint, datasets) are converted client-side. Attachments never
touch the ChatISA server, and nothing is persisted server-side (ADR-022).

## Roster change (part of this slice)

Ask Anything's model list narrows to Anthropic + OpenAI: GPT-5.6 Sol, Terra,
Luna; Claude Opus 4.8; Claude Sonnet 5 (default unchanged). Gemini and Kimi
remain available in AI Comparison. Rationale: both roster providers accept the
same native file parts, so any chat, including one holding a PDF, can still
switch models mid-conversation; and both have hosted code execution for slice
E's routing rule.

## Per-type handling

| File type | Handling | What the model sees |
| --- | --- | --- |
| Images (png/jpeg/webp/gif) | Downscaled client-side (canvas, max 1600px) | Native image part |
| PDF | Sent natively as a file part (both providers render pages as images + text) | The actual document, figures included |
| csv / xlsx | Imported into the chat's Python session as a DataFrame via the Coding Studio worker | An announcement (variable name, shape, columns); `run_python` computes on the real data |
| docx / pptx | Client-side zip + XML text extraction (`jszip`, new `lib/files/office-text.ts`) | `[Attached file: name]` text block |
| json / txt / md | Read as text, capped at 60k chars with a truncation note | `[Attached file: name]` text block |

Caps: 25 MB per file and per message (inside Anthropic's 32 MB request limit);
a client-side page-count heuristic warns when a PDF likely exceeds Anthropic's
100-page cap; provider errors are caught and explained in student language.
Native PDF pages are re-sent each turn, so both providers' prompt caching is
enabled, and the cap keeps worst-case cost bounded.

## Device-side persistence (new: IndexedDB for payloads)

Chat transcripts stay in localStorage (`aa-chats-v1`). Raw attachment bytes
(PDFs, images) cannot live there (~5 MB quota), so a new IndexedDB store
(`aa-files-v1`) holds them: one record per attachment `{ id, chatId, name,
mediaType, bytes }`. The persisted message keeps a `aa-file:<id>` reference in
place of the data URL; loading a chat rehydrates references back into file
parts before sending. Deleting a chat deletes its attachment records. Result:
chats with files survive reloads, stay model-switchable, and never leave the
device. If IndexedDB is unavailable (private mode), attachments still work for
the live session and the chat notes they will not survive a reload.

## Mechanics (verified against the installed AI SDK)

- Composer sends `sendMessage({ parts: [...attachmentParts, text] })`.
- Images and PDFs ride as standard `FileUIPart`s (data URLs);
  `convertToModelMessages` maps them to provider-native content for both
  Anthropic and OpenAI.
- Extracted text and dataset announcements ride as
  `{ type: "data-attachment", data: { kind, name, detail, text, truncated } }`
  parts; the server converts them with `convertToModelMessages`'s
  `convertDataPart` option into labeled text blocks. Unknown data parts are
  ignored by the SDK, so other modules are unaffected.

## Work items

1. **Roster**: `PAGE_MODELS.ask_anything` to the five Anthropic/OpenAI models;
   design doc revision note; roster unit tests updated.
2. **`lib/ask/file-store.ts`**: the IndexedDB attachment store (put/get/delete
   by chat, corruption-safe, storage-unavailable fallback), unit-tested against
   a fake IndexedDB.
3. **`lib/files/attachments.ts`**: pure helpers, unit-tested — classification
   by extension/MIME, size caps, the PDF page-count heuristic, dataset variable
   naming, announcement and attachment-block builders, part shapes.
4. **`lib/files/office-text.ts`**: docx/pptx XML-to-text (DOMParser), pure and
   unit-tested; thin `officeTextFromFile` opens the zip with lazily imported
   `jszip` (new dependency, no transitive deps).
5. **`lib/files/image.ts`**: client-side downscale; the pure decision helper is
   unit-tested.
6. **`components/chat/Chat.tsx`**: optional `attachments` prop
   (`{ accept, prepare(file) }`). Chat owns the composer UI: Attach button,
   pending chips (reading/ready/error, remove, screen-reader announcements),
   send includes ready parts, send disabled while reading. `ChatMessage`
   renders image file parts as thumbnails, PDF file parts and data-attachment
   parts as labeled chips.
7. **`components/ask/AskAnything.tsx`**: implements `prepare` (image →
   downscale; pdf → file part + page heuristic; csv/xlsx → session DataFrame;
   docx/pptx → office text; json/txt → text), stores payloads in the file
   store, rehydrates on chat load, deletes them with the chat.
8. **Server route**: `convertDataPart` mapping; body-size guard (413 over
   30 MB); prompt caching enabled for both providers; system prompt gains the
   attachments contract (attached blocks, dataset variables, "if the dataset
   variable is gone after a reload, re-run the import or ask the student to
   re-attach").
9. **Mock model**: `FILE_ACK <excerpt>` for attached-file text blocks,
   `PDF_ACK`/`IMAGE_ACK` when a file part is present, and a scripted
   `run_python` printing the dataset's shape for "describe the dataset".
10. **Web search wiring**: provider-tool declaration by model provider in the
    ask route; source-chip rendering in ChatMessage; system-prompt web
    contract; mock source part.
11. **Academic search wiring**: `lib/ask/paper-search.ts` (fetchers,
    normalizer, dedupe/rank, caps, rate limiting, mock-mode fixtures), the
    two tools added to `askToolDefs` with execute, ToolCard rendering for
    paper results.
12. **Tests**: unit for 2-5, the convertDataPart mapping, and the
    paper-search normalizer/merger/ranker against recorded API fixtures; e2e
    for txt attach (chip, FILE_ACK, persists across reload), csv attach +
    real Pyodide shape round-trip, image attach (thumbnail), PDF attach as
    native part (PDF_ACK, survives reload via IndexedDB), oversize rejection,
    web-source chips from a scripted source part, an academic search
    round-trip on fixture data, and axe on the composer with chips at desktop
    and 320px.

## Web search (provider-native, both providers)

Every roster model gains web access through its own provider's tools, declared
server-side per request based on the chosen model's provider:

- OpenAI models: `webSearch`.
- Anthropic models: `webSearch_20260209` and `webFetch_20260209`.

These execute on the provider's servers (billed per search, roughly $10 per
1,000 calls on both providers, plus fetched-content tokens), so no ChatISA
server code runs a search. The UI renders the stream's source parts as
citation chips under the reply ("Sources: ..."), and the system prompt gains
the web contract: search when freshness or facts beyond training are needed,
cite what you used, prefer the student's attached material when it answers the
question. The mock model emits a scripted source part so the chip rendering is
e2e-testable without network.

## Academic search (arXiv + Semantic Scholar + OpenAlex, plain TypeScript)

Decision 2026-07-24 (superseding the MCP approach): a pure-TypeScript,
server-executed toolset in `lib/ask/paper-search.ts`. No Python, no MCP, no
new deployment requirements; the production box stays Node-only.

- **Tools (with server-side `execute`, so the whole search-read-synthesize
  loop happens inside one streaming request):**
  - `search_papers({ query, limit, yearFrom? })`: parallel fan-out to arXiv
    (Atom XML), Semantic Scholar Graph API, and OpenAlex (both JSON; all
    keyless public APIs on fixed hosts). Results are DISTILLED before the
    model sees them: normalized to compact records (title, authors, year,
    venue, citations, doi, url, source, abstract trimmed to ~1,200 chars),
    deduped across sources by DOI then normalized title (cross-source hits
    are marked corroborated), ranked by citations + recency, capped at the
    top N and 8k characters total.
  - `get_paper({ doi | arxivId })`: one paper's full abstract, references
    count, and citation context from Semantic Scholar, same 8k cap.
- **Cost:** distilled results are ordinary input tokens; unlike provider web
  search there is no per-search fee.
- **Etiquette and resilience:** a shared per-source in-process rate limiter
  honoring arXiv's courtesy rules and Semantic Scholar's shared pool; an
  optional `SEMANTIC_SCHOLAR_API_KEY` env raises that pool's limits;
  per-source timeouts, and a failed source degrades to the others with a
  note in the result rather than failing the tool.
- **UI:** results render in the existing ToolCard ("Searched arXiv, Semantic
  Scholar, OpenAlex") with linked titles; the model cites URLs in prose.
- **Tests:** the normalizer/merger/ranker are pure functions unit-tested
  against recorded fixture responses (no network in CI); in mock-LLM mode
  `execute` returns canned fixture papers so the e2e loop (scripted
  search_papers call, card, PAPER_ACK continuation) runs offline.

## Miami-branded outputs (assets provided 2026-07-24 in webapp/assets/)

A `web/assets/brand/` directory (server-readable, not public) built from the
professor's assets: the 12-slide template-by-example deck
(`miami_university_powerpoint_template_with_original_logo.pptx`), a distilled
`miami-tikz` style preamble (canonical colors miamired #C41230 [standardized;
the source figures used #C3142D], agentblue #1D5FAD, evalgold #EFDB72; helvet
sans; rounded ultra-thick tinted boxes; Stealth arrows; white-backed edge
labels; legend row), two exemplar figures, and short brand notes.

- **This slice (C):** a server-executed `get_miami_style({ kind: "tikz" |
  "colors" | "latex-doc" })` tool returns the style assets on demand; the
  system prompt directs the model to fetch it before producing Miami-styled
  figures or LaTeX. Output is ready-to-compile .tex in a code block (students
  compile in Overleaf; no TeX toolchain exists in the browser or the provider
  sandboxes, so source is the deliverable). Token cost is paid only when used.
- **Slice E:** deck-generation sessions get the template .pptx injected into
  the provider container (uploaded once per provider, file_id cached) with
  instructions to build from its exemplar slides via python-pptx, so hosted
  PowerPoints come out Miami-branded. The same mechanism extends to a Word
  report template whenever one is provided.

## Simplifications vs. the earlier draft

- No server extraction route, no Exam Prep pipeline refactor, no server-side
  vision transcription: PDFs go to the providers as-is.
- Web search moved INTO this slice (provider-native on both providers); the
  server `read_url` tool is dropped from the plan (provider fetch tools and
  R scraping cover it).
- Slice E needs no per-provider capability fallbacks: both roster providers
  have hosted code execution, where the rule-based routing and the
  cheaper/mid-tier executor-model cost policy apply.

## Out of scope (later slices)

- `create_document` and the package-checker wording fix (slice D, now small).
- Hosted code execution, generated-file retrieval (slice E).

## Untouched by this slice (verified 2026-07-24)

AI Comparison keeps its full roster: `PAGE_MODELS.ai_comparisons` is
`includeAll` minus realtime/speech models, a superset of "every model with
structured output and tool access", including Gemini, Kimi, and the
open-weight models. Nothing in this slice modifies it.
