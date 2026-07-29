# Build log: code and notebook files in Ask Anything and My Projects (v6.1.1)

Date: 2026-07-29. Plan: approved in-session (plan mode). Goal: accept
`.R`, `.Rmd`, `.qmd`, `.py`, `.ipynb` (and make `.html` explicit) in Ask
Anything and Job Scout's My Projects polish, converting them client-side
into text the providers can read; notebooks additionally contribute up to
4 plot images as native image parts in Ask Anything (user decision).
No breaking changes.

## Slice 1 — shared notebook converter

- NEW `web/lib/files/notebook-text.ts`: `notebookToText(raw, {maxOutputChars,
  maxImages})` → `{ text, images, cellCount, language } | null`. Markdown
  cells verbatim; code cells fenced with the kernel language (kernelspec →
  language_info → "python"); nbformat `source`/`text` as string or line
  array; stream / text-plain / error outputs capped at 2,000 chars each
  with ANSI codes stripped; `image/png`/`image/jpeg` outputs collected up
  to `maxImages` and replaced in the text by "[plot N from cell M]"
  ("[plot output omitted]" past the cap). Returns null on unparseable
  JSON or a missing cells array so callers fall back to plain text.
  - Gotcha: the ANSI regex needs the ESC character. Writing a literal ESC
    byte into the source survived formatting tools badly, so the pattern
    is built with `String.fromCharCode(27)`; a test pins that plain
    bracketed text ("[1] 0.5", R's vector prefix) survives the strip.
- NEW `web/tests/unit/notebook-text.test.ts` (7 tests): fallback-to-null,
  markdown + R fences, string-array sources + python default, output
  capping, image cap and overflow markers (including `maxImages: 0`),
  image preferred over its text/plain repr, ANSI stripping.

## Slice 2 — Ask Anything

- `web/lib/files/attachments.ts`:
  - `AttachmentKind` and `AttachmentData.kind` gain `"notebook"`
    (additive; stored chats with old kinds render unchanged).
  - NEW `TEXT_EXTENSIONS` set: txt, md, json, py, r, rmd, qmd, html, htm.
    Extension-first matters: Windows browsers report an EMPTY MIME type
    for code files, so the `text/*` fallback never fired for them.
  - `classifyFile`: `ipynb` → notebook; the set drives the text branch.
  - `rejectionReason` copy now names code and notebook files.
  - `attachmentBlockText`: notebooks labeled "[Attached notebook: name]".
- `web/components/ask/AskAnything.tsx`:
  - `ATTACH_ACCEPT` gains `.py,.r,.rmd,.qmd,.ipynb,.html`.
  - New notebook branch in `prepareAttachment`: extract via
    `notebookToText` (cap 4 plots); text rides as a data-attachment part
    (chip detail "N cells, language[, N plots]"); each plot becomes a
    File via base64 → `prepareImage` (downscaling for free) → native
    image part named `<stem>-plot-N.png`, riding the existing image path
    including IndexedDB dehydration. A malformed plot is skipped, never
    fatal; an unparseable notebook falls through to plain text.
- Tests: `ask-attachments.test.ts` +3 (empty-MIME pins for all six
  extensions, mp3 still rejected, notebook block label);
  `ask-anything.spec.ts` +1 e2e (attach .py + .ipynb with
  application/octet-stream MIME, chip shows "1 cell, python", FILE_ACK
  echoes both files' content). The .py fixture is kept short because
  FILE_ACK echoes only 160 chars from the first "[Attached".

## Slice 3 — My Projects polish

- `web/components/scout/ProjectsTab.tsx`:
  - Payload build: `.ipynb` is extracted (`maxImages: 0`) BEFORE the
    30,000-char slice, so the model sees cells, not base64; raw-size
    allowance for notebooks raised 400 KB → 5 MB (`MAX_NOTEBOOK_BYTES`);
    a >400 KB file that fails notebook parsing goes as binary rather
    than shipping megabytes of junk. Zip assembly untouched: originals
    still ship byte-for-byte ("code ships verbatim" holds).
  - Storage fidelity fix: `StoredPolish.textFiles` now stores the FULL
    raw text (was: the 30k-truncated model payload), so re-downloads are
    faithful for text files; binaries still need re-adding.
  - Two silent trapdoors made visible: a note listing text-extension
    files too large to read ("placed in the layout without being read"),
    and a note when the picker truncates at 15 files (new `overflowed`
    state).
- `web/app/api/scout/polish/route.ts`:
  - BUG FIX: a model-supplied layout path containing a space failed the
    `SAFE_PATH` Zod regex and 502'd the whole request; "Final
    Project.ipynb" is the common coursework name. `SAFE_PATH` replaced by
    `SAFE_PATH_LOOSE` (spaces allowed inside segments, still no
    traversal) on `layout.to` and `extraFiles.path`, with a
    deterministic `toSafePath` (spaces → hyphens) applied in the
    placement guard and to extraFiles paths. Responses remain space-free.
  - `INSTRUCTIONS` gain an R/Python ecosystem paragraph: rendered
    artifacts (.html knit from an uploaded .Rmd/.qmd, `_files/`,
    `.Rproj.user`, `.ipynb_checkpoints`, `__pycache__`, `.venv`,
    `renv/library`) → exclude/.gitignore; lockfiles stay; and a note
    that notebook text arrives plot-stripped.
- `web/lib/providers/mock.ts`: polish layout ternary routes `.ipynb` →
  `notebooks/` (before the docs/ rule).
- Tests: `scout-routes.test.ts` +1 ("Final Project.ipynb" + report.qmd:
  every file placed once, notebook at `notebooks/Final-Project.ipynb`, no
  returned path contains a space; runs under a second email because the
  4/60s rate-limit bucket is shared with the earlier polish tests);
  `job-scout.spec.ts` polish scenario extended with a
  "Final Project.ipynb" upload and the hyphenated-path assertion.

## Verification

- `npx tsc --noEmit` clean; `npm run lint` clean.
- `npx vitest run`: 829 tests, 86 files. One unidentified failure on the
  first full run did not reproduce across two immediate re-runs (both
  fully green); noted here for honesty.
- `npx playwright test ask-anything job-scout` (desktop + mobile-320),
  first run: 43 passed, 2 failed — both the NEW attachment scenario, on
  each viewport. Cause: the mock's FILE_ACK echo renders as MARKDOWN, so
  a fixture starting with "# ..." became a separate heading element and
  `getByText(/FILE_ACK/)` resolved only the first paragraph. Fix: plain
  `anchovy_constant = 1` fixture and the assertion scoped to the whole
  assistant article (`getByRole("article", { name: "ChatISA"
  }).filter({ hasText: "FILE_ACK" })`). All pre-existing scenarios,
  including the extended polish one, passed on the first run; the fixed
  scenario then passed on both viewports (2.7s each). Final e2e state:
  45/45 across the two suites.
