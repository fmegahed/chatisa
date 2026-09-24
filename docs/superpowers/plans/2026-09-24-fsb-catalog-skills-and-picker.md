# FSB Catalog, Skills and Picker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A reviewed, self-refreshing FSB course catalog with syllabus evidence, business-domain skills with anti-overselling guards, and a major-first course picker shared by Job Scout and the Portfolio Builder (v6.7.0).

**Architecture:** Committed JSON under `web/catalog/` is the single source of course, program, syllabus and link data; `web/scripts/catalog/` collects, maps and reports; a GitHub Action runs those scripts on a schedule and opens a PR. The app reads the JSON through small typed modules (`lib/scout/courses.ts`, `lib/scout/programs.ts`, `lib/scout/course-skills.ts`), so components never parse raw data. Prerequisite inference and matching guards are pure functions with unit tests.

**Tech Stack:** Next.js 16, React 19, TypeScript, zod 4, AI SDK 7, Vitest, Playwright + axe, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-24-fsb-catalog-skills-and-picker-design.md`

## Global Constraints

- Do not break the running app: v1 Job Scout profiles migrate on read; existing course links unchanged until a reviewed PR changes them; published pages static.
- Parsers never drop silently: every unparsed row or prerequisite is reported.
- Syllabus evidence stores no instructor names, emails, or personal data.
- Stable output: sorted, no timestamps inside data files (timestamps only in `manifest.json`).
- The pipeline never edits `lib/scout/taxonomy.ts` and never commits to `main`.
- Spend cap per mapping run: `CATALOG_MAX_RUN_USD` default 10.
- Accessibility: native radios/selects, visible labels, `aria-expanded` on section toggles, polite live region for added prerequisites, axe A/AA desktop and 320px.
- Copy: no em dashes in user-facing text.
- Release gate: professor reviews the skill list, disputed links, and the tag-eval report before v6.7.0 ships.

## Review Focus

1. A student already has a v1 profile with 12 ISA courses: after upgrade the picker shows them all as Done, nothing is lost, and saving writes v2. (Task 9 migration test.)
2. A student removes an automatically added prerequisite, then marks another course that shares it: it stays removed. (Task 9 inference test.)
3. A bulletin page adds a row type the parser has never seen: the collector reports it and the rest of the program still parses. (Task 2 fixture test with an unknown row.)
4. A course is Taking now and also has an exposure link from a Done course: the skill is Working, not Strong; after marking Done with an anchor it can reach Strong. (Task 7 guard tests.)
5. The mapping run hits the spend cap halfway: it stops cleanly, writes what it has, and the report says how many courses were not mapped. (Task 5 test.)

---

### Task 1: Catalog data layout and link migration
**Files:** Create `web/catalog/programs.config.json`, `web/catalog/course-skills.json`; modify `web/lib/scout/course-skills.ts` to load the JSON; test `web/tests/unit/catalog-links.test.ts`.
- [ ] Test: the JSON-backed `COURSE_SKILLS` deep-equals the current array (snapshot generated from the TS before migration), and the existing integrity tests still pass.
- [ ] Migrate by script (write current `COURSE_SKILLS` to JSON, sorted by course then skill), keep the doc comment, export the same names and types.
- [ ] Config lists the 13 programs found on 2026-09-24 (core, 9 majors, 2 co-majors, AI minor) with URLs and kinds.

### Task 2: Bulletin parsers
**Files:** Create `web/scripts/catalog/bulletin.ts` (pure: `parseProgram(html)`, `parseCoursePage(html)`, `parsePrereq(text)`); fixtures under `web/tests/fixtures/bulletin/`; test `web/tests/unit/catalog-bulletin.test.ts`.
- **Produces:** `ProgramGroup = { title: string; instruction: string | null; items: { codes: string[] }[] }`; `ParsedCourse = { code, altCodes, title, credits, description, prereqText, prereq: string[][] | null, prereqUncertain: boolean }` (prereq is AND of OR-lists of codes); `parseProgram` returns `{ groups, unparsed: string[] }`.
- [ ] Fixtures: save live HTML for business core, ISCM (concentrations), AI minor, the ISA and FIN course pages.
- [ ] Tests: core groups include "Select one of the following" with MTH options; "or" rows merge into one item; concentrations become separate groups; an injected unknown row lands in `unparsed`; "ISA 401/ISA 501." parses with altCode; prerequisite "ACC 221 and ACC 222" gives `[["ACC 221"],["ACC 222"]]`; "ECO 311 or ISA 291 or ISA 391 or STA 463/STA 563" gives one group of four; "...; or permission of the instructor" sets `prereqUncertain`; standing phrases set it too.
- [ ] Implement until green.

### Task 3: Syllabus collector
**Files:** Create `web/scripts/catalog/syllabi.ts` (`listSections(fetch)`, paginated; `extractEvidence(docJson)`); fixture `web/tests/fixtures/syllabus/`; test `web/tests/unit/catalog-syllabi.test.ts`.
- **Produces:** `SyllabusEvidence = { terms: string[]; sections: number; topics: string[]; tools: string[]; outcomes: string[] }` per course code.
- [ ] Tests: pagination follows `pagination.total` across pages of 500; evidence keeps topics and named tools, dedupes across sections, and contains no editor names or emails from the fixture.
- [ ] Implement until green; tools detected from a closed list (Python, R, SQL, Excel, Tableau, Power BI, SAS, SPSS, JMP, Snowflake, Databricks, Spark, AWS, Azure, Git, Bloomberg, QuickBooks, SAP, Salesforce, Minitab).

### Task 4: collect.ts, manifest and report
**Files:** Create `web/scripts/catalog/collect.ts`, `web/scripts/catalog/report.ts`; add npm scripts `catalog:collect`, `catalog:map`, `catalog:report`; outputs in `web/catalog/`.
- [ ] Merge rules: courses never removed (missing ones get `retired: true`); title changes append to `previousTitles`; same-title-new-code pairs listed as suspected renumberings; existing ISA catalog courses kept.
- [ ] Test `web/tests/unit/catalog-merge.test.ts`: retire, retitle, renumber detection, stable sorted output.
- [ ] Run live; commit the generated data; read `REPORT.md` and fix parser gaps it exposes (each fix with a fixture test).

### Task 5: Mapper (incremental, capped)
**Files:** Create `web/scripts/catalog/map.ts` (pure core `planMapping(state, evidence)` and `mergeProposals(a, b)`); test `web/tests/unit/catalog-map.test.ts` with a mock model.
- [ ] Tests: unchanged fingerprint skipped; agreement (same skill, same level) accepted into links; level disagreement and one-sided links reported not written; cap reached stops the loop and reports the remainder; existing approved links for unchanged courses untouched.
- [ ] Implement; models Claude Sonnet 5 and GPT-6 Sol via the app's provider factory; evidence phrases in student voice; no tool claimed unless the evidence names it.

### Task 6: Taxonomy v2
**Files:** Modify `web/lib/scout/taxonomy.ts` (category `business`, the domain skills from the spec with aliases and implies, `TAXONOMY_VERSION = 2`); tests in `web/tests/unit/scout-taxonomy.test.ts`.
- [ ] Tests: new ids resolve by label and alias; no alias collides with an existing id or alias; implies edges resolve; version is 2.

### Task 7: Matching guards
**Files:** Modify `web/lib/scout/matching.ts` (`profileStrengths` accepts per-course status); tests in `web/tests/unit/scout-matching.test.ts`.
- **Produces:** `profileStrengths(courses: { code: string; status: "done" | "now" }[] | string[], extras, overrides)` (string[] still accepted, meaning all done).
- [ ] Tests: exposure-only stack capped at 0.44; no anchor (and no anchor extra, no override) capped at 0.79; Taking-now half weight and capped at 0.79; override wins both ways; existing worked examples unchanged where an anchor exists.

### Task 8: Tagger rules and tag eval
**Files:** Modify `web/lib/scout/tag.ts` instructions; create `web/scripts/catalog/tag-eval.ts`.
- [ ] Rules and one-line definitions added; eval re-tags a fixed sample (up to 60 stored postings) with old and new instructions and writes `docs/development/2026-09-24-tag-eval.md`.

### Task 9: Profile v2 and prerequisite inference
**Files:** Modify `web/lib/scout/profile-store.ts`; create `web/lib/scout/prereqs.ts`; create `web/lib/scout/programs.ts`; tests `web/tests/unit/scout-profile-v2.test.ts`, `web/tests/unit/scout-prereqs.test.ts`.
- **Produces:** `ScoutProfileV2 = { v: 2; programs: string[]; courses: ProfileCourse[]; removedPrereqs: string[]; extras; overrides }`, `ProfileCourse = { code; status: "done" | "now"; term?: string; addedBecause?: string }`; `inferPrereqs(profile, programs, catalog): ProfileCourse[]`; `currentTerm(date): string`.
- [ ] Tests: v1 migrates to v2 (all done); path choice order; permission and 4+ groups skipped; transitive; out-of-scope skipped; removed prerequisites never re-added; already-present courses untouched.

### Task 10: Picker UI
**Files:** Create `web/components/scout/CourseChecklist.tsx` (programs step, sections, rows, core shortcut, search, added-prerequisite lines, live region); use it in `components/scout/ProfileTab.tsx` and `components/portfolio/career/ClassesStep.tsx`; showcase `CoursePicker` single mode gains "Your courses" and full-catalog search.
- [ ] Next-term prompt in ProfileTab; Portfolio passes Taking-now courses so the page marks them "(in progress)".
- [ ] `tsc`, lint clean.

### Task 11: GitHub Action
**Files:** Create `.github/workflows/catalog-refresh.yml`.
- [ ] Schedule `0 12 1 2,6,9 *` and `workflow_dispatch`; permissions `contents: write`, `pull-requests: write`; steps collect, map, report; branch and PR via `gh`; secrets referenced only as env on the map step.
- [ ] Validate with `actionlint` if available, else a YAML parse test.

### Task 12: End to end, review, release
- [ ] E2E for the picker flows in Job Scout and the Portfolio Builder, axe desktop and 320px (foreground runs, per the memory-pressure lesson).
- [ ] Review packet for the professor: `docs/development/2026-09-24-catalog-review.md` (skill list, disputed links, tag-eval summary, parser gaps).
- [ ] After the professor's review: apply decisions, full suites, independent review, browser check, bump 6.7.0, notes, bundle, commit, tag, push.
