# Deck template swap and Miami chart style (design, 2026-07-25)

Two changes to Ask Anything, agreed with the professor on 2026-07-25.

**(a)** Replace the deck template with `miami_template_by_fadel_megahed.pptx`.

**(b)** Give every chart the model produces a house style: no pie charts by
default, the Miami palette, descriptive titles carrying the insight, and
non-overlapping annotations when the data is small enough to label.

## Part (a): the template

`webapp/assets/miami_template_by_fadel_megahed.pptx` replaces
`web/assets/brand/miami-template.pptx`. They differ structurally, so the copy
that describes the template has to change with it:

| | old | new |
| --- | --- | --- |
| slides | 12 | 10 |
| layouts | 2 | 1 |
| heading font | Georgia | Roboto Condensed |
| body font | | Roboto |
| code font | | Courier New |

The new deck carries its own **style guide on slide 10**, which is the
authoritative palette for anything shipped alongside it:

| Hex | Name | Role |
| --- | --- | --- |
| `#C3142D` | Miami red | headings, emphasis, primary series |
| `#585E60` | Charcoal | body text, secondary series |
| `#84D6D3` | Teal | hyperlinks ONLY, never a data colour |
| `#EE5863` | Coral | accent |
| `#FF7436` | Orange | accent |
| `#FFDF65` | Highlight | highlight, weakest series colour |

Slide 8 is a content-box palette (blue/gray/green/purple/red/yellow soft
fills with assigned meanings). Slide 10's usage notes: 16:9, titles
left-aligned, red bullets and bold for emphasis, teal for links only, bold on
title and divider slides, the beveled M on white slides.

Note the pptx theme XML itself carries the **stock Office palette**
(`4472C4`, `ED7D31`, ...), not Miami colours, in both the old and new file.
The style-guide slide, not the theme, is the source of truth.

### Why a constant

The filename appears in six places today, which is how the prompt came to say
"12-slide template" while the file had changed underneath it. A single exported
`DECK_TEMPLATE` becomes the only spelling of it.

It lives in its own module, `lib/ask/deck-template.ts`, deliberately WITHOUT
`server-only`. The obvious home was `lib/ask/hosted.ts`, but the system prompt
interpolates the name, and that prompt travels through `lib/chat/config` into
`components/ask/AskAnything.tsx`, which is `"use client"`. Importing it from
`hosted.ts` (which reads the file from disk and holds provider keys) breaks the
client build. `hosted.ts` re-exports it so server callers are unaffected.

Consumers:

- `hosted.ts`: the read path, the upload filename, the container-upload note
- `lib/prompts/ask-anything.ts`: interpolated, plus a rewritten description of
  what the template offers (the layouts, not a slide count)
- `assets/brand/miami-colors.md`: the PowerPoint paragraph, plus the
  style-guide palette recorded above
- `lib/providers/mock.ts` and two tests

`scripts/make-deploy-bundle.mjs` copies `web/assets` wholesale, so
`deploy/chatisa-app` regenerates. It must be rerun before the next deploy.

## Part (b): chart style

### The palette contract

`lib/ask/chart-style.ts`, pure and client-safe, owns this and nothing else.

| Series | Palette | Colour alone enough? |
| --- | --- | --- |
| 1 | `#C3142D` on `#FFFFFF` | yes |
| 2 | `#C3142D`, `#585E60` | yes (red = focus, gray = context) |
| 3 | + `#1D5FAD` | **no**, labels or shapes required |
| 4 | + `#FF7436` | **no** |
| 5 to 8 | ColorBrewer **Dark2** | **no** |
| 9+ | refuse: group the tail into Other, facet, or plot the top few | n/a |

### Measured, not chosen

The first draft of this palette was wrong and a palette validator caught it.
Numbers below are OKLab deltaE times 100, all pairs, against surface `#FFFFFF`.

| Combination | Result |
| --- | --- |
| `#3E5468` slate blue vs `#585E60` charcoal | **normal-vision deltaE 5.5**, floor is 15 |
| `#1D5FAD` agent blue vs `#585E60` charcoal | deltaE 13.4, still under floor at 4 slots |
| `#585E60` charcoal | chroma 0.008, a pure gray |
| `#FFDF65` corn yellow | lightness 0.908 (outside band), contrast 1.31:1 |
| `#C3142D`, `#1D5FAD`, `#1B9E77` | clean pass, zero failures |
| Dark2 at n=4 | 2 hard failures |
| Dark2 at n=8 | `#E7298A` vs `#1B9E77` **deutan deltaE 1.7** |
| Paired at n=12 | `#FF7F00` vs `#33A02C` **protan deltaE 0.6** |

What that changed:

- **The blue is `#1D5FAD`** (Agent blue, already recorded in
  `assets/brand/miami-colors.md`), not slate blue. Slate blue beside charcoal is
  unreadable for everyone, not only colourblind readers.
- **Charcoal is a gray, and that is the point.** It earns the second slot, where
  red means "the series in focus" and gray means "context". It is fragile as a
  peer among other dark hues, which is what caps the brand palette at 4.
- **Direct labels or distinct shapes are mandatory from 3 series up.** No
  categorical palette, brand or brewer, separates reliably on colour alone past
  two or three slots. This is also what makes a 4-colour brand palette
  defensible, and it coincides with the "show the data" requirement.
- **Paired was rejected for 9 to 12.** Its orange and green are effectively one
  colour under protanopia. Past 8, change the question, not the palette.
- **Black is never a series colour.** Text, axes, and annotations only.
- **`#FFDF65` is a fill only**, with a `#585E60` outline and a visible label.
- `RColorBrewer` is already in the WebR mirror via `scales` and matplotlib ships
  Dark2 natively, so the escalation costs no new dependency.

### Two tiers, because hosted sandboxes have no network

The provider sandboxes that build decks cannot install packages. So guidance
splits, and the split is stated in the tool output itself, so the model never
emits an import that will fail inside a container.

**Portable tier, valid everywhere including hosted sandboxes:**
palette and white background; descriptive title plus insight subtitle;
minimal gridlines; legend at bottom or direct labels; sentence case; black for
text and axes. Plain ggplot2 or matplotlib, no extra packages. Annotation
repelling is explicitly NOT required here.

**Rich tier, browser runtimes only:**

| Purpose | R | Python |
| --- | --- | --- |
| coloured-word subtitle replacing a legend | `ggtext` `element_markdown` | `highlight_text` |
| non-overlapping labels | `ggrepel` | `adjustText` |

Fonts are a portable-tier trap: the template's Roboto Condensed almost
certainly is not present in a provider container or in Pyodide, and matplotlib
substitutes silently with a findfont warning rather than failing. Guidance
specifies a fallback list and says to accept the default sans, never a single
hardcoded family.

### The title contract

The title states the finding; the subtitle carries the insight or caveat.
Neither restates the axes. "ISA 401 grades ran higher for three of four
students", not "Grades by course".

### The pie chart policy

In the system prompt, so it lands before any code is written. Do not produce a
pie or donut by default. Say once that pies force angle and area comparisons
that people read poorly, and that a bar chart ranks nominal categories more
accurately while a dot chart handles many categories or two-value comparisons
better. If the student still wants one, build it without further argument.

### Delivery

- System prompt: palette with literal hex values, the escalation, the title
  contract, the pie policy, and the instruction to call `get_miami_style`.
- `get_miami_style` gains kinds `charts-r` and `charts-python`, each returning
  the rules plus a working exemplar for that language. Unlike the other kinds
  these are generated, not files, since they must state the palette and the
  tier split in one place.
- `hosted.ts`'s container-upload note gains the palette and title contract, so
  the rules sit next to the deck-building instruction.

### Bundling

- `WEBR_PACKAGES` gains `ggtext`, `ggrepel`: +9 packages, 4.2 MB on top of the
  existing 99 packages / 72 MB (Rcpp, commonmark, gridtext, jpeg, litedown,
  markdown, png come along). All are prebuilt WebAssembly binaries in the
  r-wasm repo; nothing compiles.
- `PYPI_WHEELS` gains `adjustText`, `highlight_text`. Both are pure
  `py3-none-any`. Spelled with the IMPORT name, not the distribution name
  (`highlight_text`, which PyPI resolves to `highlight-text`), because the
  wheel manifest is keyed by the name the worker matches against imports.
- `HOSTED_WHEELS` in `public/workers/pyodide-worker.mjs` gains both, with
  `pyodideDeps` of numpy/matplotlib/scipy and matplotlib respectively.
- Then `node scripts/setup-runtimes.mjs`. Students take a one-time
  re-download of about 4 MB.

## Testing

- `paletteFor` boundaries: 1, 4, 5, 8, 9, 12, 13, and the exact hex strings.
- Black never appears as a series colour at any n.
- The `charts-python` output for the hosted tier never names `highlight_text`
  or `adjustText`; the browser tier does.
- A drift guard that the prompt and the mock both use `DECK_TEMPLATE`.
- Live render both exemplars in the real WebR and Pyodide runtimes, the way the
  dumbbell starters were verified. Code that only typechecks is not verified.

### Verified 2026-07-25

- `tests/unit/chart-style.test.ts`, 19 tests: palette boundaries, the
  colour-alone rule, black and corn yellow and slate blue absent at every n, the
  tool output per language, and the container note carrying the rules.
- `tests/e2e/chart-examples.spec.ts`, opt-in with `CHATISA_LIVE_NET=1`: all
  three exemplars run. R draws with ggtext and ggrepel from the mirror, Python
  draws with adjustText from the hosted wheel, and the hosted exemplar produces
  a PNG over 10 KB using matplotlib alone with no extra imports.
- Mirror rebuilt: 108 packages, 76 MB (was 99 / 72 MB). Wheel manifest keyed
  `adjustText` and `highlight_text`, matching the worker's import matching.
- Full suite: 644 unit tests, Ask Anything e2e 27 passed / 2 skipped,
  `next build` clean (which is what proves the client boundary above).

## Also in this change

`SEMANTIC_SCHOLAR_API_KEY` reached production, so it is added to the deployed
`chatisa.env.example` generated by `make-deploy-bundle.mjs`, alongside
`OPENALEX_MAILTO`. Without the key the keyless Semantic Scholar pool answers
429 to nearly every request (see the 2026-07-25 paper-search fixes).
