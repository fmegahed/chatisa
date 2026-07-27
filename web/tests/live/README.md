# Live hardening runs

Playwright runs that drive **the real models** through the real UI. Opt-in, never
part of `npm run test:e2e`.

## Why this exists separately

`npm run test:e2e` runs with `CHATISA_MOCK_LLM=1`. Nothing there calls a
provider, which is correct: it makes the suite deterministic, free, and safe to
run on every change. It proves our code.

It cannot prove the other half. A canned answer always parses, never fails to
install a package, never loops a hosted sandbox, and never emits a chart that
ignores the palette. The failures students actually hit are in that half. So
these runs exist to **find defects**, and they are expected to be slow, to cost
money, and occasionally to fail for reasons outside the app.

Treat a failure here as a lead to investigate, not as a red build.

## Running

Start a server with real keys and **without** the mock flag:

```bash
AUTH_TEST_MODE=1 AUTH_URL=http://localhost:3200 \
  CHATISA_DATA_DIR=tests/e2e/.data-live \
  CHATISA_PROXY_ALLOW_LOCAL=1 \
  npm run dev -- --port 3200
```

Then:

```bash
npm run test:live                             # everything
npm run test:live -- --grep "networking"      # one area
npm run test:live -- --grep "tutor"
CHATISA_LIVE_BASE_URL=http://localhost:3100 npm run test:live
```

The suite refuses to run against a mock-mode server (`assertLiveServer`, and the
same check in `live.setup.ts`). A live suite quietly pointed at a mock server
produces confident, meaningless passes; that has happened in this project
before, which is why the guard reads the page instead of trusting the operator.

## What lands where

Everything goes under `tests/live/.artifacts/` (git-ignored):

- `<test name>/observations.json` — console errors, page errors, failed
  requests, every HTTP >= 400 with its body, and the timed notes the spec
  emitted. **Read this even when the test passed.** A model can produce a
  plausible answer while a tool call 500s and silently retries.
- `<test name>/*.txt`, `*.md`, `*.png`, `*.pptx` — whatever the spec saved:
  transcripts, generated files, run output.
- `trace/` — Playwright traces, kept for every test, pass or fail.
- `report.json` — the machine-readable run summary.

## Notes on writing a live spec

- **Never assert on a model's wording.** Assert on artifacts and behaviour: a
  file exists and opens, code runs, a plot appears, the palette is the house
  palette. Asserting phrasing measures our guess about the model.
- **Wait on state, not text.** `sendAndSettle` waits for the Stop button to
  disappear, because a real answer's length and timing are unpredictable.
- **Snippets you type must be column-zero.** CodeMirror auto-indents, so a typed
  indented block arrives with compounding whitespace: a syntax error in Python.
- **The first R run in a fresh browser context is slow.** It installs the bundled
  tidyverse set from our own mirror (110 packages) before any code runs. Budget
  minutes, not seconds.
- **Say what you did not check.** A spec that verifies a deck downloads but never
  opens it should say so in a note, so the report is not read as more than it is.

## Files

| File | Purpose |
| --- | --- |
| `../../playwright.live.config.ts` | The opt-in config. No `webServer`: you start it. |
| `live.setup.ts` | Signs in once into `tests/live/.auth/` (separate from e2e's). |
| `support/live.ts` | The fixture, the mock-mode guard, and the shared page helpers. |
| `support/observe.ts` | The recorder that writes `observations.json`. |
| `browser-networking.spec.ts` | Regression suite for the 2026-07-26 cross-origin isolation bug. |
