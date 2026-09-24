# FSB course catalog, skills, and the major-first picker, design

**Date:** 2026-09-24. **Release target:** v6.7.0 (projects 2 and 3 ship
together). **Status:** design agreed in conversation with the professor on
2026-09-23 and 2026-09-24; written spec reviewed by the lead engineer under
the professor's delegation.

Projects 2 and 3 of the four agreed on 2026-09-23. They ship together
because a catalog of about 170 courses without the major-first picker would
turn today's level-tiered chip picker into a wall of chips.

## Goal

Miami students record every course that counts toward their FSB major,
co-major, and the AI for Business minor, quickly and honestly, and that
record makes Job Scout matching and the Portfolio Builder better without
overselling anyone. The data stays current through a scheduled pipeline
that proposes changes and never merges them itself.

## Constraints (professor, 2026-09-23)

- Do not break the running app: saved Job Scout profiles, Portfolio drafts
  and published sites keep working; students are never asked to redo work.
- Slick and usable; axe WCAG A/AA, keyboard, 320px.
- Value-adding: every change earns its place.
- Empower the student rather than model every rule: the picker is a record
  of what was taken, not a degree audit.

## Decisions already made (with the professor)

1. Scope: courses listed in the FSB business core, the 9 BSB majors, the 2
   co-majors, and the AI for Business minor (150 codes on 2026-09-24), plus
   every course already in the catalog. Nothing outside those requirement
   lists (no number theory).
2. Syllabi from the public Simple Syllabus library add detail beyond the
   bulletin; they cover only courses offered in the current terms, so the
   bulletin stays the complete list. The next syllabus pull is 2027-02-01.
3. Skills: add about 25 to 35 business-domain skills at the level analytics
   postings use, designed to split later through `implies` edges. No full
   per-major taxonomy until the job feed expands.
4. Anti-overselling guards: a skill reaches Strong only through an anchor
   link, a student override, or a confirmed extra; stacks of exposure-only
   links stop at Introduced; displays show a focused, grouped set.
5. The job tagger learns the new skills with two rules: tag a domain skill
   only when the role's duties or qualifications ask for it, never from the
   employer's industry; domain skills default to "preferred" unless the
   posting explicitly requires them. Gated on a before/after check of about
   60 stored postings that the professor spot-checks.
6. ISA course links are refreshed from syllabus evidence as well.
7. Picker: "What are you studying?" first; the checklist follows the
   bulletin's groups; each course is Done, Taking now, or Not yet; a
   business-core shortcut; search for anything in scope.
8. Prerequisites of any Done or Taking-now course are added automatically as
   Done, clearly labelled ("Added because you're taking ISA 401") and
   removable. "Or" groups choose the student's own path: an option their
   programs require first, then one from their major's department, then the
   first listed. Instructor-permission routes and groups of four or more
   options are skipped, as are courses outside the scope.
9. "Taking now" counts at reduced weight, capped at Working, until marked
   Done; student overrides always win. Next term the profile asks "You were
   taking ISA 444 in Fall 2026. Did you finish it?"
10. Admit years: groups follow the current bulletin and say so; the catalog
    only grows (courses leaving the bulletin stay searchable with their
    links); ticked courses never disappear; renumberings become equivalent
    codes after the professor confirms them in the pipeline's PR.
11. Renames keep the code as identity; the catalog keeps previous titles,
    search finds either, and the course is re-mapped for review. Each Done
    course records its term where known so a future release could map by
    era if it ever matters.
12. The data refresh runs as a GitHub Action (scheduled 1 Feb, 1 Jun,
    1 Sep, plus manual), including the model mapping, with API keys as
    repository secrets and a per-run spend cap. It opens a pull request with
    a plain-English report; it never commits to `main` and never edits the
    skill list.

## Architecture

### Data files (committed, under `web/catalog/`)

| File | Written by | Read by | Contents |
|---|---|---|---|
| `programs.config.json` | people | collector | One line per program: key, display name, kind (`core`, `major`, `comajor`, `minor`), bulletin URL. Adding a program is one line. |
| `courses.json` | collector | app (client-safe) | Every course ever in scope: code, equivalent codes, title, previous titles, credits, parsed prerequisites, `retired` flag. Never shrinks. |
| `descriptions.json` | collector | mapper, server | Bulletin descriptions and prerequisite text. Not imported by client code. |
| `programs.json` | collector | app | Each program's groups as the bulletin lists them: group title, the bulletin's instruction ("Select two of the following"), and items (one code, or several codes for an "or" row). Plus the bulletin year. |
| `syllabi.json` | collector | mapper | Course-level evidence only: terms seen, number of sections, weekly topics, named tools, stated outcomes. No instructor names, emails, or personal data. |
| `course-skills.json` | mapper proposes, people approve via PR | app | The approved course-to-skill links (replacing the hand-written `lib/scout/course-skills.ts` array, which is migrated to this file unchanged). |
| `mapping-state.json` | mapper | mapper | Per course: fingerprint of description plus syllabus evidence, the last models' proposals, and status. Makes each run incremental. |

Stable output: sorted keys and arrays, no timestamps inside data files, so
a diff shows only real change. A small `manifest.json` records when each
source was last fetched.

### Pipeline (`web/scripts/catalog/`, run by the Action and locally)

1. `collect.ts`: fetch every program in the config and parse its
   requirement table with one generic parser; fetch course pages for every
   prefix referenced; parse title, credits, equivalent codes (including
   "ISA 401/ISA 501." graduate pairs), description, and prerequisite text
   into an AND-of-ORs structure (terms joined by commas or "and", options by
   "or"), marking `uncertain` when the text involves instructor permission
   or class standing; page through the Simple Syllabus library and extract
   evidence for in-scope courses. Anything it cannot parse is listed in the
   report, never silently dropped. Suspected renumberings (same title, new
   code) and title changes are listed.
2. `map.ts`: for each course whose fingerprint changed, ask two models
   (Claude Sonnet 5 and GPT-6 Sol) to link it to the taxonomy with levels
   and evidence phrases, using the bulletin text and the syllabus evidence.
   Links both models agree on are written into `course-skills.json` in the
   PR; disagreements are not written, they go to the review report with
   both proposals. A link a person already approved is never removed or
   re-levelled by the pipeline: agreed new skills are added beside it, an
   agreed level that differs from the approved one goes to the review
   list, and approved links neither model proposed are kept and listed.
   (Refined 2026-09-24 during implementation.) Hard spend cap per run (`CATALOG_MAX_RUN_USD`, default
   10).
3. `report.ts`: writes `catalog/REPORT.md` (the PR body): programs changed,
   courses added, retired, retitled, suspected renumberings, prerequisite
   changes, links added, links needing a decision, rows not understood, and
   the run's cost.

### GitHub Action (`.github/workflows/catalog-refresh.yml`)

Why an Action and not the server's scheduler: the weekly Job Scout harvest
runs inside the ChatISA server (`instrumentation.ts` starts
`lib/scout/scheduler.ts`) because its output is runtime data students
should see at once. The catalog refresh changes committed code and data
that must be reviewed before release, which means a pull request; the
server has no GitHub write access and should not be given it. This is the
repository's first workflow.

- Triggers: `schedule` (1 Feb, 1 Jun, 1 Sep) and `workflow_dispatch`. Never
  `pull_request`, so secrets are never exposed to outside contributors on
  this public repository.
- Steps: checkout, Node 24, `npm ci --ignore-scripts` in `web/`, collect,
  map, report; if anything changed, push a branch `catalog/refresh-<date>`
  and open a pull request with `REPORT.md` as its body using the built-in
  `GITHUB_TOKEN` (`contents: write`, `pull-requests: write`).
- Secrets: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (set by the professor).
  Repository setting required once: allow GitHub Actions to create pull
  requests.

### Skills (`lib/scout/taxonomy.ts`)

- Add a `business` display category and about 25 to 35 domain skills
  (proposed list below), each with a label, lowercase aliases that separate
  it from its neighbours, and `implies` edges only where specific leads to
  general. `TAXONOMY_VERSION` becomes 2 (posting tags already record their
  version; the weekly harvest re-tags every posting it collects, so the
  new vocabulary applies within one week without a migration).
- Proposed skills (the professor reviews the list with the first link
  review): Financial Accounting & Reporting, Managerial & Cost Accounting,
  Auditing, Tax, Corporate Finance, Financial Modeling & Valuation,
  Investments & Portfolio Management, Personal Financial Planning, Real
  Estate Analysis, Economic Analysis, Econometrics, Marketing Strategy,
  Market Research, Consumer Behavior, Digital Marketing, Sales, Brand
  Management, Operations Management, Process Improvement (Lean and Six
  Sigma), Logistics, Human Capital Management, Organizational Behavior,
  Negotiation, Entrepreneurship, Innovation and Design Thinking, Business
  Law, International Business, Sustainability.

### Tagger (`lib/scout/tag.ts`)

- Instructions add the two domain rules (decision 5) and a one-line
  definition for each new skill.
- `scripts/catalog/tag-eval.ts` re-tags a fixed sample of stored postings
  with the old and new vocabularies and writes a side-by-side report: old
  skills that changed (must be near zero) and new domain tags with the text
  that triggered them. Shipping the tagger change requires the professor's
  spot-check of that report.

### Matching guards (`lib/scout/matching.ts`)

- A skill's computed strength is capped below Strong (at 0.79) unless it has
  at least one anchor link from a Done course, a confirmed extra at anchor
  level, or a student override.
- A skill whose only contributions are exposure links is capped at
  Introduced (0.44).
- "Taking now" courses contribute at half weight, and a skill fed only by
  Taking-now courses is capped at Working (0.79).
- Overrides always win, as today.

### Picker (project 3)

Shared by Job Scout's profile and the Portfolio Builder's Classes step;
the showcase's single Miami course uses its search plus "Your courses".

1. **What are you studying?** Multi-select of the programs in
   `programs.config.json` (majors, co-majors, minors). Stored in the
   profile.
2. **The checklist.** Business core first, then each chosen program, each
   as a collapsible section showing the bulletin's own instruction ("Select
   two of the following"). Each course row: code, title, and a
   Done / Taking now / Not yet control (a native radio group per row,
   keyboard operable). "Or" rows render as one row with the options in a
   small select. A "I've finished the business core" button marks every
   core course Done. A note: "Grouped as in the 2026-27 Bulletin. Took
   something different? Search for it below."
3. **Search** across the whole catalog, including retired courses and
   previous titles ("formerly ...").
4. **Automatic prerequisites** (decision 8), shown under the course that
   caused them: "Added because you're taking ISA 401: ISA 345, ISA 235",
   each with Remove. A removed prerequisite is remembered and never re-added
   for that course.
5. **Next term** (decision 9): Taking-now courses record the term; on a
   later term the profile opens with a short, dismissible prompt listing
   them with "Finished" and "Still taking" buttons.

Profile storage (`js-profile-v1` → `js-profile-v2`): programs, and courses
as `{ code, status: "done" | "now", term?, addedBecause? }`. A v1 profile
migrates on read: every course becomes Done with no term. The Portfolio
Builder reads Done and Taking-now courses (Taking now renders as "(in
progress)" on the page).

## What stays the same

- The 45 existing ISA courses, their cross-listings, and their approved
  links carry over unchanged until a reviewed pipeline PR changes them.
- Published portfolio pages are static and never change.
- Guests: no change (they type courses; the picker is for Miami students).

## Accessibility

Native radios and selects, visible labels, sections as headings with
buttons that expose `aria-expanded`, focus kept on the row after an
automatic prerequisite appears (announced through a polite live region),
no colour-only status, axe A/AA at desktop and 320px.

## Testing

- Parser fixture tests: saved copies of three program pages (business core,
  a concentration major, the AI minor), two course pages (including a
  "401/501" pair and an "or permission" prerequisite), and one syllabus
  record; asserts groups, items, prerequisites, uncertainty flags, and that
  unknown rows are reported.
- Prerequisite inference: unit tests for path choice (program-required
  first, then department, then first listed), skipping permission and 4+
  groups, transitive chains, out-of-scope skips, and remembered removals.
- Matching guards: worked examples pinning each cap and override.
- Profile migration: v1 to v2 round trip.
- Pipeline: `map.ts` against a mock model (agreement written, disagreement
  reported, unchanged fingerprint skipped, spend cap stops the run).
- Tag eval: runs locally against stored postings; its report is an input to
  review, not a unit test.
- End to end: a student picks two programs, uses the core shortcut, marks a
  course Taking now and sees its prerequisites added and removable, searches
  a retired or retitled course, and the Portfolio Builder and Job Scout both
  show the result; axe A/AA on the picker at desktop and 320px.

## Release gate

v6.7.0 ships after: the professor's review of the proposed skill list and
the disputed links from the first full mapping run, and the professor's
spot-check of the tag-eval report. Everything else follows the usual gates
(full suites, independent review, browser check, bundle).

## Out of scope

- Degree audit or requirement validation.
- Mapping courses outside the FSB programs and the AI minor.
- Expanding the job feed beyond analytics and IS roles.
- Reading GitHub (project 4).
