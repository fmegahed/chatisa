# ChatISA Design System → Miami Brand Guide Mapping

Source of truth: `brand-standards-2025_508.pdf` (Miami University Brand Guide 2024 v.2).
Tokens live in `app/globals.css` (`@theme` block); guardrail tests in `tests/unit/tokens.test.ts`.

## Color (guide pp. 28–31)

| Token | Hex | Guide role | Use in ChatISA |
|---|---|---|---|
| `--color-miami-red` | #C41230 | Primary (PMS 186) | Primary actions, ribbon labels, focus ring, nav current-state, header rule |
| `--color-paper` | #FFFFFF | Primary | Cards, header surface |
| `--color-warm-white` | #FAF9F7 | Neutral (web only) | Page background |
| `--color-accent-red` | #AD102A | Secondary | Hover/pressed red, inline links |
| `--color-light-tan` | #EDECE2 | Secondary | Footer, status panels, quiet surfaces |
| `--color-medium-tan` | #CCC9B8 | Secondary | Borders, dividers |
| `--color-dark-tan` | #70685C | Secondary | Secondary text on light surfaces |
| `--color-medium-gray` | #666666 | Neutral | De-emphasized text (≥ 14px) |
| `--color-corn-yellow` | #EFDB72 | Tertiary | Sparing emphasis / data viz only |
| `--color-slate-blue` | #3E5468 | Tertiary | Sparing emphasis / data viz only |
| `--color-ink` | #000000 | Neutral | Body text |

Rules enforced: 100% opacity only (no tints/shades, guide p. 28); red+white predominant with
tans in support (p. 30); text/background pairs restricted to the guide's WCAG 2.1 AA matrix
(p. 31) — e.g. white↔Miami Red, Miami Red on white/warm white/light tan, black on all light
surfaces, dark tan on light surfaces. Corn Yellow text is only used on Accent Red/Slate Blue
per the matrix. Legacy off-brand values (#c3142d, rgb(200,16,45)) are test-blocked.

## Typography (guide pp. 32–34)

| Role | Guide typeface | ChatISA stack | Notes |
|---|---|---|---|
| Display/headings | FreightText Pro | Source Serif 4 (self-hosted, OFL) → Georgia | Title case, never all caps (p. 32) |
| Body/UI | Gotham / Gotham Narrow | Arial → Helvetica | Approved common-app substitute (p. 33) |
| Ribbon labels | Gotham uppercase | Arial bold, uppercase, +tracking, extra leading space | Mirrors p. 34 ribbon-text note |
| CTAs/URLs | Gotham Bold | Arial bold | p. 32 |

Licensed faces (FreightText/Display Pro, Gotham, Proxima Nova, Brushability) are **not**
bundled — no license. Fonts are self-hosted via npm (`@fontsource-variable/source-serif-4`);
no external font CDN (CSP-compatible).

## Logos (guide pp. 8–21)

- Official artwork only, from `public/brand/` (sourced from user-provided `assets/`):
  `beveled-m.png` (full-color Beveled-M — header mark, favicon),
  `logo-vertical-stacked.png` (primary vertical stacked — login page, Slice 2).
- Full-color marks appear **only on white/warm-white/light surfaces** (p. 11). No white/
  reverse variant was provided, so logos are never placed on Miami Red. **Missing assets**
  (recorded limitation): one-color white Beveled-M, horizontal lockups.
- Never: recolor, stretch, rotate, add effects, use as text, or recreate the wordmark
  (p. 11, p. 21). "ChatISA" in the header is the app's own name set in the display face —
  it is not, and must never visually imitate, the Miami University wordmark.

## Signature element

The angled **ribbon label** (`.ribbon`): white uppercase sans on Miami Red with the guide's
slanted edge (pp. 34–35 ribbon/typesetting examples). Used for page eyebrows and section
callouts — sparingly, once per view.

## States & signifiers

Error/warning/success/disabled always pair icon or text with color (never color alone).
Focus: 3px Miami Red outline, 2px offset (≥3:1 against white, warm white, light tan).
Reduced motion: all animation/transitions collapse under `prefers-reduced-motion`.
