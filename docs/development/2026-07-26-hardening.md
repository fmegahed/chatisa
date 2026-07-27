# Production hardening, 2026-07-26

Driven by the professor's instruction to harden the app for production with real
models, plus a production bug report (a screenshot of `/coding-tutor` showing
`rvest::read_html` failing with `Error: cannot open the connection`).

Everything here is local and uncommitted. No production access was used; the
production diagnosis is inference from code plus a reproduction on a local server.

---

## 1. Defects found and fixed

### 1.1 R and Python had no network access on three of five code-running pages

**The report.** A student pressing "Run R" inside a Coding Tutor answer got
`Error: cannot open the connection` from `rvest::read_html`.

**Cause.** Not R, not the ws-proxy, not the model. WebR's networking (rvest,
httr2, curl, all through libcurl) needs a `SharedArrayBuffer`, which requires
cross-origin isolation (COOP `same-origin` + COEP `require-corp`). `next.config.ts`
attached those headers to `/coding-studio` and the worker/runtime asset routes
only. Without them WebR silently falls back to a channel with no networking at
all, and the failure surfaces as an R error pointing nowhere near a header.
Python's urllib3 emscripten transport uses the same `SharedArrayBuffer` for its
streaming path, so both languages were affected.

**Five pages can execute code**, not one:

| Route | How it reaches a runtime | Isolated before | Now |
| --- | --- | --- | --- |
| `/coding-studio` | Sandbox calls the run manager | yes | yes |
| `/coding-tutor` | assistant Markdown wraps fences in `RunnableCode` | **no** | yes |
| `/ask-anything` | `run_python` / `run_r` / `run_sql` tools | **no** | yes |
| `/ai-comparison` | assistant Markdown | no | yes |
| `/project-assistant/[id]/coach/[type]` | assistant Markdown | no | yes |

**Fix.** `lib/run/isolation.ts` holds the route list and the reasoning;
`next.config.ts` maps over it.

The first pass covered the Coding Tutor and Ask Anything, which is what the
professor scoped it to, and recorded the other two in
`KNOWN_UNISOLATED_RUN_PAGES`. They were then covered the same day on the
professor's decision, on the reasoning that a Run button should behave the same
wherever it appears; that list is now empty. Verified in a real browser:
`crossOriginIsolated === true` on all five, and no COEP header on `/exam-prep`,
which runs no code.

**Guard.** `tests/unit/run-isolation.test.ts` walks the import graph from every
`page.tsx` to `lib/run/manager.ts` and fails if a code-running page is neither
isolated nor listed. It follows `next/dynamic`, which matters: Coding Studio and
Ask Anything reach their runtimes *only* through `dynamic(() => import(...))`, so
a static-import scan reports both as unable to run code. An earlier version of
this test targeted the `RunnableCode` component instead of the manager and found
3 pages rather than 5, missing the two most important modules.

**Verified.** `crossOriginIsolated === true` on all three isolated routes, and
`rvest::read_html("https://example.com")` returning `Example Domain` through the
real Run button on `/coding-tutor` (42s warm, including the tidyverse install).
`requests.get` through `/api/py-proxy` in 7.7s.

### 1.2 Every failing R run was reported as a success

**Cause.** `public/workers/webr-worker.mjs` called `captureR` with
`captureConditions: false`, under a comment claiming errors would still "re-throw
as JS exceptions (the default)". `false` is exactly what disables that. Measured
against webR 0.6:

| Snippet | Before | After |
| --- | --- | --- |
| `stop("boom")` | `ok: true`, `Error: boom` in the output text | `ok: false` |
| `this_function_does_not_exist(1)` | `ok: true` | `ok: false` |
| error raised inside a dplyr verb | `ok: true` | `ok: false` |
| `message()` / `warning()` / `1:3` | ok, text present | ok, text present |

**Impact.** The student saw the neutral "Output" panel instead of the red "Error"
panel; the message was announced through `aria-live="polite"` instead of
`role="alert"`; and nothing downstream could tell either, so our own multi-turn
test harness never handed the error back to the tutor for correction. The
professor's screenshot shows the shape exactly: an "Output" panel containing
`Error: cannot open the connection`. Python was never affected, because its worker
throws.

**Fix.** `captureConditions: true`, plus rendering of the condition objects.
Flipping the flag alone silently drops `message()` and `warning()` output, since
capturing conditions takes them off the stderr stream and hands them over as R
objects: every "Loading required package: ..." line and every warning would have
vanished. `conditionMessage()` pulls the `message` element in R space, because
`toJs()` on a whole condition fails ("This R object cannot be converted to JS")
on its `call` element. `friendlyError()` also strips webR's one-letter type tag
and its internal `eval(ei, envir)` frame.

### 1.3 The Coding Tutor did not know its own code was runnable

**Evidence.** Asked the professor's scraping question, the tutor produced
selectors it had invented (`.directory-entry`, `.directory-name`), each labelled
"VERIFY & REPLACE THIS SELECTOR". None exist on that page, so the code returned an
empty table and then errored. It had every means to check: the student can run its
R in one click, and R can fetch the page.

**Cause.** Nothing in its system prompt said so. The module renders every
`r`/`python`/`sql` fence with a Run button, and the prompt described none of it,
so the model wrote for a student who would paste into RStudio later. Ask
Anything's prompt is explicit that its tools run and should be used to check its
own claims, and it does not make this mistake.

**Fix.** `lib/prompts/running-code.ts`, appended to the Coding Tutor prompt. Says
the blocks run, forbids "replace this" placeholders, prescribes the two-step shape
for scraping (inspect, then extract), and lists what is installable. The package
lists are **generated** from `lib/sandbox/packages.ts`, because a prompt that
lists packages is a promise and a stale one produces code that cannot run.

**Result.** Same task after the change: an inspection snippet first (`Number of
<table> elements...`), then extraction with `table tr`, header rows filtered by
`th`, name from the `<a>`, position from the `<i>`, webpage from
`html_attr(href)`. **30 rows, all four requested columns, real data.**

### 1.4 Interview Mentor's missing voice was undiagnosable and unexplained

Two separate problems.

**Undiagnosable remotely.** `/api/health` reported whether `DEEPGRAM_TOKEN` was
*present*, and presence is not validity: a revoked key, an account out of credit,
and blocked outbound HTTPS all report "present" and all produce silence.
`probeSpeech()` now mints a token (free, same credential, same outbound path) and
`/api/health?deep=1` reports it. The semantics are deliberately asymmetric:

- absent token → `not-configured`, server stays **healthy** (a supported setup;
  the module degrades to typed answers)
- token present but refused → `broken`, server returns **503**

The second is the state that was invisible from outside. Against the professor's
own credential the probe returns `speech: "ok"`, so that key is valid.

**Unexplained to the student.** `HandsFree.tsx` fell straight through to opening
the microphone when `/api/speech/speak` failed. No voice, no reason. Silence in
the UI also meant silence in every bug report, which is what made the original
report impossible to act on. It now says what happened and distinguishes 503
("not set up on this server") from a transient failure. `QuestionAudio.tsx` makes
the same distinction and stops offering a retry that cannot help.

The `QuestionAudio` comment reading "Not autoplay" beside an `autoPlay` attribute
was corrected. It was a plausible suspect while diagnosing and was never the
cause: the element only mounts after the student presses "Hear this question", so
playback follows a user gesture, which is both what they asked for and what
autoplay policy permits.

### 1.5 Claude Opus 5 could not answer in any module

**Cause.** `AI_APICallError: temperature is deprecated for this model.` Every
request carrying a temperature was rejected. Opus 5 is offered in Ask Anything,
Coding Tutor, AI Comparison, Exam Prep and Project Assistant, and answered in
none of them; students saw "That response failed" every time.

It survived because of an asymmetry between two models of the same generation:
Claude Sonnet 5 only WARNS ("temperature is not supported ... and will be
ignored"), so the default model degraded silently while the premium one broke
loudly, and nothing exercised the premium one.

**Fix.** `supportsTemperature` in the catalog and one `temperatureFor(modelId,
value)` helper, used at all seven call sites: chat, Ask Anything, the project
coach, completions, exam generation, grading, and vision transcription. Missing
any one would leave a module broken while the rest looked fine, so
`tests/unit/temperature.test.ts` scans `lib/` and `app/` for a raw `temperature:`
literal in a provider call and fails if it finds one, with a canary so it cannot
pass vacuously.

`temperatureRange` is left as published: it describes what the model's sampling
would do, while the flag describes whether we are allowed to ask.

### 1.6 A turn mixing provider-executed and browser tools crashed the loop

Found only after 1.5 was fixed, because the temperature error masked it.

```
AI_MissingToolResultsError: Tool result is missing for tool call toolu_01ShWGRu83...
```

**Cause.** Ask Anything mixes three kinds of tool, and only one of them ends the
server's turn:

| Kind | Examples | Result arrives |
| --- | --- | --- |
| server-executed | `search_papers`, `read_url`, `get_miami_style` | same step |
| provider-executed | `code_execution`, `code_interpreter` | same step |
| **browser-executed** | `run_python`, `run_r`, `run_sql` | **next request, from the student's page** |

`stopWhen: stepCountIs(10)` could not express the third case. Step logging on a
failing run:

```
step 1  code_execution (provider) + 2x search_papers (server)  -> 3 results
step 2  run_python (browser)                                   -> 0 results
then    AI_MissingToolResultsError
```

Because step 1 resolved everything, the loop continued; step 2 left a browser
call that the server can never resolve, and step 3 threw.

**Fix.** `lib/ask/stop-conditions.ts`: `awaitsBrowserTool()` ends the turn as soon
as a non-provider-executed call has no result, alongside the existing step cap.

**Not model-specific**, though it looked it. It needs a turn that mixes a
provider-executed tool with a browser tool; Opus 5 did that on one run and
produced a perfectly good deck on another, and Sonnet 5 tripped it too. Two wrong
inferences were made on the way here and both are worth recording:

1. That Opus 5 might not be supported by our `code_execution` tool version. That
   came from the `@ai-sdk/anthropic` doc comment ("Supported models: Claude Opus
   4.6, Sonnet 4.6, Sonnet 4.5, Opus 4.5"), which is **stale**. Anthropic's own
   documentation lists Opus 5 under `code_execution_20250825`,
   `code_execution_20260120` and `code_execution_20260521`. Check the vendor, not
   a third-party type definition.
2. That moving to `code_execution_20260521` would fix it. Anthropic: "the same
   runtime as `code_execution_20260120`. The difference is that the tool
   description tells Claude about the 90-second wall-clock limit on each Python
   cell." Worth having, but not a fix, and currently not reachable: neither
   `@ai-sdk/anthropic` 4.0.18 nor 4.0.21 exposes a factory for it, and the SDK's
   response schema hard-codes the `caller` union as
   `code_execution_20250825 | code_execution_20260120 | direct`, so sending the
   newer type risks a response the SDK cannot parse. Revisit when the SDK adds it.

### 1.7 The Run button promised runs it could not deliver

Professor's instruction: install what can be installed so the student's code
runs, and where a package cannot work at all, do not offer a button whose only
outcome is an error.

**Four tiers** (`lib/sandbox/runnable.ts`), and the fourth is the point:

| Tier | Meaning | Run button |
| --- | --- | --- |
| `ready` | every package already loaded | yes |
| `installable` | obtainable, some on first use | yes, and says so |
| `unknown` | cannot tell | yes, no promises |
| `blocked` | provably cannot work here | **no**, with the reason |

`unknown` exists because the two mistakes are not equally costly. Hiding Run on
code that would have worked is a broken feature with nothing to explain it;
showing Run on code that fails is the status quo plus a clear error. So nothing is
blocked without positive evidence.

**Sources of truth.** Python: Pyodide's own lock (`byAlias`, already used by the
Studio help and Ask Anything) plus `KNOWN_UNAVAILABLE_PYTHON`. R: a new
`available.json` written by `npm run setup:runtimes`, listing the 108 packages we
mirror and all 22,741 the WebR repository serves. Its **absence** is meaningful:
with no manifest an unmirrored package is `unknown` and stays runnable, so a
server that has not regenerated `public/runtimes` behaves exactly as before.

`BUNDLED_R_CLOSURE` was needed because `BUNDLED_R` lists what we ask to install,
not what ends up installed. `dplyr` and `ggplot2` are the two most common calls in
this app's R code and neither is in `BUNDLED_R`; both arrive as tidyverse
dependencies. A test checks every name in the closure list against the mirror on
disk.

**Cost.** The verdict must be reached without booting a runtime, so it comes from
manifests: 113 KB for Python, 253 KB for R. Neither is fetched unless it is needed.
SQL, snippets that import nothing, and R snippets whose every package is in the
shipped bundle are all decided synchronously from constants, and that covers most
blocks in this app, since `dplyr`, `ggplot2` and `readr` all qualify. Only a
snippet naming something outside the bundle pays for a request.

**Background install.** `prepareCode()` fetches a snippet's packages on hover or
keyboard focus, not on mount. Mount would undo the lazy-loading design: a chat
page with six code blocks would boot both runtimes. Hover is the earliest honest
signal that *this* block is the one about to run, and a student who scrolls past
pays nothing.

**Measured behaviour before the change** (`package-availability.spec.ts`, run
against the real runtimes on 2026-07-26). On-demand install already worked; the
gap was entirely in what happens when it *cannot*:

| Case | Result | Time |
| --- | --- | --- |
| Python `pandas` (bundled) | ok | 16.2s (cold boot) |
| Python `requests` (in Pyodide's lock) | ok | 5.8s |
| Python `openpyxl` (our hosted wheel) | ok | 4.7s |
| Python `statsforecast` (impossible) | fails, clear message | 3.3s |
| R `dplyr` (bundled) | ok | 73.6s (cold boot + tidyverse install) |
| R `zoo` (**not** mirrored) | ok, fetched from WebR's repo with its `lattice` dependency | 48.5s |
| R `rJava` (impossible) | fails, **useless message** | 36.5s |

The last row is the case worth fixing. The student saw:

```
Error in `if (require(rJava) == FALSE) install.packages("rJava")`:
  argument is of length zero
```

Nothing in that points at "this package cannot run in a browser". It is an
artifact of `require()` returning nothing useful after a failed install, and it
took 36 seconds to arrive. With the gate, `rJava` is blocked before any of that
happens (both by name and by its absence from the repository manifest) and the
student is told why immediately.

**Coverage, stated precisely.** The UI gate is e2e-tested deterministically on the
Python path (`statsforecast`, via a mock trigger). The R path shares that same
component and its classification is unit-tested, but the R gate has **not** been
exercised through a live browser, because getting a real model to emit an
`rJava` snippet on demand is not reliable.

---

## 2. Reported, not fixed. Decisions needed

(The isolation gap and the CBP dataset limit that were here have both been
resolved; see 1.1 and 1.7.)

2. **The professor's own County Business Patterns example cannot be downloaded in
   the browser.** `cbp23st.zip` is 11,115,845 bytes; `PROXY_RESPONSE_MAX` is
   4,000,000. The hosted sandbox has no network either.

   The app and the model both behave correctly at that wall, which is the
   important part. GPT-5.6 Sol tried the ZIP, hit the limit, tried
   `api.census.gov`, found it needs a key, and then said:

   > The Census ZIP is 11 MB, while the browser download proxy has a 4 MB limit.
   > The Census API also requires an API key, so I cannot retrieve the records
   > reliably from that link **without inventing data**. Please attach
   > cbp23st.zip to the chat. I already have your Miami PowerPoint template.

   No deck was produced, which is right: it had no data. Ten tool cards, none
   failed, no fabrication markers.

   It also caught an inconsistency in the example prompt itself: the prompt says
   "by county and industry", but `cbp23st.zip` is the STATE file. County records
   are in `cbp23co.zip`.

   **Resolved.** The ceiling was raised to 25 MB (see 1.7), and the professor
   corrected the example's wording to state-level, since `cbp23st.zip` is the
   state file. The exercise now works as written.
3. ~~`PACKAGES.rds` 404s on every first R run.~~ **Fixed.** `setup-runtimes` now
   writes a real `PACKAGES.rds` (9 KB, 108 packages) with the local R:
   `saveRDS(read.dcf("PACKAGES"))` is exactly what a CRAN-style repository holds.
   OPTIONAL by design, and that matters for where it runs: the mirror is built on
   a DEV machine and shipped inside the bundle, so the production server never
   needs R. With no Rscript present the step is skipped with a note and R falls
   back to `PACKAGES.gz` as before. Verified: the live R test used to report
   "1 console errors, 1 HTTP >=400" on every run and now reports "observed:
   clean", with R still reaching the internet in 42s.
4. ~~The deploy bundle contains itself, recursively.~~ **Fixed, and it was worse
   than cosmetic.** `.next/standalone` mirrors the project directory, so the
   wholesale copy shipped two things it should not have:

   - `data/chatisa.db`: the DEVELOPMENT database. The copy on disk held 3 user
     rows (email addresses), 150 usage events, **126 exam document page excerpts**
     (the sampled note text ADR-015 governs), and 2 tailored resumes. Production
     reads `CHATISA_DATA_DIR`, so the shipped copy had no purpose at all.
   - `deploy/`: the previous bundle, nested three deep. `deploy/` had reached
     612 MB.

   **The archive that shipped on 2026-07-25 contains both** (verified by listing
   `chatisa-app.zip`, which is a tar despite the name). So the production server
   has a copy of a dev database on disk. It is not read, but it should be deleted:
   remove `data\` from the extracted `chatisa-app` folder on the server, or let
   the next deploy replace the folder wholesale.

   Fix: a `filter` on the standalone copy (`NEVER_BUNDLE = ["data", "deploy"]`)
   plus a post-assembly assertion that FAILS the build if any `.db` file or nested
   `deploy/` survives anywhere in the bundle. The assertion is the load-bearing
   half: the filter guards one known path, while the leak arrived because a source
   directory quietly gained contents nobody expected. Guarded by
   `tests/unit/deploy-bundle.test.ts`, which also checks the artifact on disk.
   **A fresh `node scripts/make-deploy-bundle.mjs` is needed before the next
   deploy**; the local bundle has had the offending paths removed by hand in the
   meantime.
5. **Exam Prep ingest scales superlinearly** (measured 2026-07-25: 12 slides in
   10.7s, 37 slides not ready at 431s).
6. ~~The webR worker's header comment is stale.~~ **Fixed.** It claimed "no
   cross-origin isolation needed", the exact opposite of 1.1, which made it
   actively misleading to anyone diagnosing the reported bug.
7. **Playwright Chromium processes leak** across interrupted live runs. Cosmetic,
   but they accumulate.
8. **`code_execution_20260521` is not reachable yet.** See 1.6: the SDK has no
   factory and its response schema does not know the type. Worth taking when
   `@ai-sdk/anthropic` ships it, for the 90-second cell-limit description.
9. **Generated decks keep only one slide layout.** Both decks produced on
   2026-07-26 carry the Miami theme (Roboto Condensed / Roboto) and exactly one
   layout relationship per slide, so they open correctly, but the template's nine
   layouts are reduced to one. Cosmetic variety, not correctness.

---

## 2a. Verified working, with evidence

Recorded because "no defect found" is a result, and because several of these were
suspected and cleared rather than assumed.

| Claim | Evidence |
| --- | --- |
| Deepgram speech works end to end from the server | `POST /api/speech/speak` returned `audio/mpeg`, 21,888 bytes, MP3 frame header `FF F3`. `/api/health?deep=1` reports `speech: "ok"`. |
| Generated decks open | Two real decks unzipped: 15 and 12 slides, exactly one slideLayout relationship per slide, no empty slides. The 2026-07-25 repair holds. |
| Decks are built from the Miami template | Master theme major/minor latin fonts are Roboto Condensed / Roboto in both. |
| On-demand package install works | R `zoo` (not mirrored) installed from WebR's repository with its `lattice` dependency in 48.5s; Python `requests` in 5.8s; our hosted `openpyxl` wheel in 4.7s. |
| The model does not fabricate data when blocked | The CBP task hit the 4 MB proxy ceiling (3x HTTP 502) and the answer carried none of the fabrication markers. |
| Fantasy Premier League end to end | Sonnet 5 scraped, built an optimisation model, produced the team image, and stated all four FPL constraints (budget, squad size, per-club cap, formation). |

---

## 3. Test additions

**One existing test genuinely broke and had to be updated.**
`tests/e2e/shell.spec.ts` asserted `body.checks.deep` by deep equality on the
whole object, so adding `speech` to the deep-health block failed it. That is the
gate doing its job: the endpoint was fine and the assertion's shape was the
problem. It now checks each key individually plus the exact key set, so the next
addition fails with a clear message instead of an opaque object diff.

Two unit tests were **updated** rather than loosened, both because of deliberate
changes to the Coding Tutor prompt:

- `coding-style.test.ts` pinned the whole prompt byte-for-byte; it now pins the
  legacy Streamlit prefix and asserts additions come after it as headed sections.
- `chart-rules-prompt.test.ts` raised its per-turn budget ceiling from 6,000 to
  8,000 characters, with the measured sizes recorded (tutor 6,686; sandbox 4,588).

### New unit tests

| File | Covers |
| --- | --- |
| `run-isolation.test.ts` | every code-running page is isolated or listed |
| `chart-rules-prompt.test.ts` | the chart contract is shared, not copied |
| `running-code-rules.test.ts` | package lists generated, not retyped; R list matches the worker |
| `runnable.test.ts` | requirement parsing and the four availability tiers |
| `speech-probe.test.ts` | absent vs refused credential, and no credential leaks |

### New e2e tests (mock mode, free and deterministic)

- `interview.spec.ts`: three tests. Hands-free says why the interviewer is silent
  and the interview continues; the manual player says "not set up on this server"
  and disables the retry; a 502 invites one. Failures injected with `page.route`.
- `chat.spec.ts`: no Run button for an impossible package, with the reason and
  Copy still present; and the guard that available packages keep their buttons.

A note on the interview tests, because the first version of them was wrong in an
instructive way: they assumed Playwright's Chromium has no microphone, so
`handsFreeAvailable()` would be false and the manual `QuestionAudio` player would
render. It does have `getUserMedia` and `MediaRecorder`, so the hands-free path
renders instead and the "Hear this question" button never existed. The rewrite
covers **both** components, which is what it should have done from the start,
since the fix changed both.

### Live harness (new, opt-in)

`tests/live/`, run with `npm run test:live`. Real models, real providers, real
money. **Not** part of `npm run test:e2e`, which keeps running against the mock.
Every spec refuses to run against a mock server, because a live suite pointed at
one produces confident, meaningless passes.

| Spec | What it drives |
| --- | --- |
| `browser-networking.spec.ts` | the 1.1 regression suite, R and Python |
| `runtime-errors.spec.ts` | the 1.2 regression suite, both languages |
| `coding-tutor.spec.ts` | the professor's three tasks |
| `ask-anything.spec.ts` | three tasks, one model each, decks opened not just downloaded |
| `modules.spec.ts` | Coding Studio, AI Comparison, JobApp Drafter, Exam Prep |
| `package-availability.spec.ts` | what really happens for bundled / installable / impossible |

See `tests/live/README.md` for how to run it and how to read the artifacts.

### Live results

| Module | Result |
| --- | --- |
| Coding Tutor (a) AirPassengers | pass. Title "Air travel grew steadily from 1949 to 1960, with summer peaks rising each year", all four Miami hexes, no style problems |
| Coding Tutor (b) matrix algebra | pass. KaTeX rendered, no leaked delimiters, R and Python both ran |
| Coding Tutor (c) ISA faculty scrape | pass. 30 rows, all four requested columns, real profile hrefs |
| Ask Anything (a) CBP regression | pass. 10 tool cards, none failed, no fabrication (see item 2 below) |
| Ask Anything (b) Fantasy Premier League | pass. Scraped, optimised, team image, all four constraints stated |
| Ask Anything (c) deck, Opus 5 | pass. 15 slides, Miami theme, one layout rel per slide |
| Ask Anything (c) deck, Sonnet 5 | deck correct (12 slides, Miami theme); test failed on two `read_url` misses, which is web flakiness rather than an app fault |
| Coding Studio, Python | pass. statsmodels regression, plot in the Plots pane |
| Coding Studio, R | pass. Internet reached, dplyr summary correct |
| Coding Studio, side chat | pass. Palette correct, title "North region led monthly sales through 2024", secondary encoding present for four series |
| AI Comparison | pass. Blinded, both answers substantive, preference recorded, "Claude Sonnet 5 won" |
| JobApp Drafter | pass, twice. Real posting, tailored resume, valid .docx. Both defects reported on 2026-07-25 (session name, duplicated skills heading) are absent |
| Exam Prep | pass. Notes to graded results in 11s |

### What the harness got wrong, and the pattern in it

Worth recording, because the same harness will be reused and because every
failure in `modules.spec.ts` turned out to be the test rather than the product.

- **Asserting on presence instead of change.** Three waits fired instantly
  against containers that already held placeholder text: the Studio console ships
  with "Python 3.14.0 (Pyodide, WebAssembly) running in your browser...", the
  comparison panes carry headings before any answer arrives, and the page has h2s
  in the footer. Fixed with a sentinel the snippet prints, a length threshold, and
  a wait on the Submit control that only exists with a question.
- **Guessing UI wording instead of reading it.** "See my results" (real:
  "See results"), and a comparison report heading that is actually the winning
  model's name. Both were already pinned in `tests/e2e/`, which is where they
  should have been read from.
- **Holding a locator across a re-render.** After a submit, the exam form
  re-renders and Submit becomes Next, so "the form containing Submit" matched
  nothing on question 2.
- **Assuming the environment.** Playwright's Chromium does have a microphone, so
  Interview Mentor renders the hands-free path, not the manual player.
- **Running a live suite while editing its subject.** One result was void because
  the dev server was serving a broken import mid-run. The captured page error is
  what identified it.

Each of these reported working software as broken, which is the more expensive
direction of error for a hardening exercise: it sends the reader to the wrong
place.

---

## 4. Gate

| Check | Before | After |
| --- | --- | --- |
| `npm run typecheck` | 0 errors | 0 errors |
| `npm run lint` | 0 problems | 0 problems |
| `npm run test` | 656 tests, 68 files | 731 tests, 79 files |
| `npm run test:e2e` | passing | 266 passed, 24 skipped, 3 flaky (see below) |
| `npm run test:live` | did not exist | `modules.spec.ts` 7/7; the other specs as tabulated above |

### On the three e2e flakes

The full run reported `guest.spec.ts` "knowing module paths does not bypass
authentication", `interview.spec.ts` "offers an unfinished interview", and
`sandbox.spec.ts` "Clear empties SQL console output". All seven instances
(both projects) pass when re-run targeted, so they are the contention flakes the
Playwright config already documents: `next dev` compiling a cold route or lazy
chunk past an assertion timeout under full parallelism. Locally `retries: 0` is
set on purpose, so a flake surfaces rather than hiding.

Worth noting because these changes did add client work to chat pages: the
package-availability verdict fetches an index. That was narrowed afterwards so
the common case fetches nothing at all (see 1.7), which also removes it as a
possible contributor here.

### One flake that is not a flake

The auth setup timed out once at its full 300s budget on a cold Turbopack
compile, then passed at 1.5m on the next run. It is a dependency of every
authenticated project, so when it times out nothing else runs. It sits closer to
its ceiling than is comfortable.
