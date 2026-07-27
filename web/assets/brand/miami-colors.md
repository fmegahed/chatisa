# Miami University brand notes (ChatISA assets, distilled 2026-07-24)

Palette (use these exact values):

| Name | Hex | Role |
| --- | --- | --- |
| Miami red | #C41230 | Primary. Emphasis, headings, main flow, human decisions. |
| Agent blue | #1D5FAD | Secondary. Systems, AI, process elements. |
| Eval gold | #EFDB72 | Checks, evaluation, highlights. Darken for borders/text. |
| Neutral tan | #CCC9B8 | Outcomes and de-emphasized elements. |
| Ink | #1A1A1A | Body text on light backgrounds. |
| Warm paper | #EDECE2 | Background panels (matches the PowerPoint template). |

Typography: sans-serif throughout (Helvetica family in LaTeX via `helvet`;
the deck template's own fonts in PowerPoint). Bold for box labels and
headings; sentence case, not title case, for labels.

Composition conventions: rounded-corner boxes with ultra-thick borders in the
role color filled with a light tint of the same color; Stealth arrowheads;
white-backed labels on arrows; a square-swatch legend row when a figure uses
more than two role colors; numbered boxes when order matters.

LaTeX documents (reports, memos): `\usepackage[margin=1in]{geometry}`, helvet
as above, Miami red for `\section` headings via
`\usepackage{sectsty} \allsectionsfont{\color{miamired}}`, hyperlinks colored
Miami red with `\usepackage[colorlinks=true, allcolors=miamired]{hyperref}`.

PowerPoint: build from the 10-slide template
(`miami_template_by_fadel_megahed.pptx`), which carries the branding in its own
slides. Its layouts: title, section divider, two-column comparison, sidebar
plus main content, code and output, activity prompt, content-box palette,
table, and a style-guide slide. Do not restyle from scratch.

The template states its own palette and fonts on that style-guide slide, and
those values govern anything shipped alongside a deck:

| Hex | Name | Role |
| --- | --- | --- |
| #C3142D | Miami red | Headings, emphasis, the primary chart series. |
| #585E60 | Charcoal | Body text, and the "context" chart series. |
| #84D6D3 | Teal | Hyperlinks ONLY. Never a data colour. |
| #EE5863 | Coral | Accent. |
| #FF7436 | Orange | Accent, and the fourth chart series. |
| #FFDF65 | Highlight | Highlight fills. Invisible as a line on white. |

Template fonts: Roboto Condensed for headings, Roboto for body, Courier New
for code. Usage notes from the same slide: 16:9, titles left-aligned, red
bullets and bold for emphasis, bold on title and divider slides, the beveled M
on white slides.

Charts have their own contract, which is checked rather than chosen by eye; ask
for the `charts-r` or `charts-python` style kind rather than deriving a chart
palette from this table.
