# Guests and Project Origin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Guests can build a career portfolio with optional typed courses, and every Showcase declares where the project came from (Miami course, another school, self-study, personal), with pages, READMEs and repository names that read naturally in each case.

**Architecture:** One shared origin helper module (`lib/portfolio/origin.ts`) owns the origin type, its defaults and every origin-dependent string (header label, README line, repo name, prompt line) so the renderer, publish plan, route and UI agree. Typed courses are a separate content field (`otherCourses`) with its own post-validation, never mixed with Miami catalog codes.

**Tech Stack:** Next.js 16 App Router, React 19, zod 4, AI SDK 7 (`generateObject`), Vitest, Playwright + axe.

**Spec:** `docs/superpowers/specs/2026-09-23-guests-and-project-origin-design.md`

## Global Constraints

- Do not break the running app: every new field optional on read with defaults (`origin` absent = `"miami"`, `otherCourses` absent = `[]`).
- Route validation clips, never rejects, lengths (the 6.4.1 lesson); outside course `name`/`school` clipped to 80, max 5 rows.
- Accessibility: native fieldset/legend radios, visible labels, focused `role="alert"` errors, axe A/AA on changed steps, 320px.
- Copy: no em dashes in user-facing text. Labels: "Self-study project", "Personal project".
- Miami showcase header: `CODE - Catalog title` (code alone when the catalog has no title).
- Guests: session email ends with `@guest.chatisa` (`GUEST_EMAIL_DOMAIN`).
- Release convention: one commit `v6.6.0: ...`, annotated tag, notes in `docs/releases/v6.6.0.md`, CHANGELOG entry, deploy bundle.

## Review Focus

1. A guest types a course but leaves School blank, or types the same course twice: the page shows it once, labelled exactly as typed. (Task 3 test.)
2. A student switches origin after typing (other to self-study): the stale typed course must not leak into the header, README or repo name. (Task 1 and Task 5 tests: label/repo derive from origin, and the Course step clears `course` on origin change.)
3. Opening a site saved before this release (no `origin`, no `otherCourses`) and republishing: same repo name, same header except the added catalog title. (Task 4 test.)
4. The model rewords an outside course ("Applied regression analysis" for "Applied Regression"): kept only if it matches the student's text case/space-insensitively; otherwise dropped, never shown with the model's wording. (Task 3 test.)
5. Special characters in typed course text (`<b>`, `&`, quotes) render escaped in header, Coursework and README-safe slug. (Task 2 test.)

---

### Task 1: Origin helpers

**Files:**
- Create: `web/lib/portfolio/origin.ts`
- Test: `web/tests/unit/portfolio-origin.test.ts`

**Interfaces:**
- Produces:
  - `type ProjectOrigin = "miami" | "other" | "self" | "personal"`
  - `ORIGINS: ProjectOrigin[]`
  - `originOf(value: unknown): ProjectOrigin` (anything unknown or absent to `"miami"`)
  - `originLabel(origin, course): string` (header label: miami `CODE - Title` via `getCourse`, other = trimmed course, self `Self-study project`, personal `Personal project`; empty string for miami/other with empty course)
  - `originReadmeLine(origin, course): string` (`Built for X.` / `A self-study project.` / `A personal project.`; `""` when miami/other course is empty)
  - `originPromptLine(origin, course): string` (`Course: X` / `Course (another school): X` / `A self-study project.` / `A personal project.`)
  - `isGuestEmail(email: string | null | undefined): boolean`

- [ ] Write failing tests covering each origin for each function, `originOf(undefined)`, `originOf("bogus")`, an empty course for miami/other, a catalog course with title ("ISA 225" -> "ISA 225 - Principles of Business Analytics"), and `isGuestEmail("guest-3@guest.chatisa")` true / student false / null false.
- [ ] Run `npx vitest run tests/unit/portfolio-origin.test.ts`; expect failures (module missing).
- [ ] Implement `origin.ts` (imports `getCourse` from `@/lib/scout/courses`, `GUEST_EMAIL_DOMAIN` from `@/lib/auth/guest` is server-only because of `node:crypto`; define the domain check inline with a comment pointing at the constant, and a unit test asserting equality with `GUEST_EMAIL_DOMAIN`).
- [ ] Run the test; expect pass.

### Task 2: Content schema and renderer

**Files:**
- Modify: `web/lib/portfolio/content.ts` (add `otherCourseSchema`, optional `otherCourses` on `careerContentSchema`, export `careerGenerationSchema` with it required; `emptyCareer` gets `otherCourses: []`)
- Modify: `web/lib/portfolio/html.ts` (Coursework appends outside courses; `renderShowcase` meta gains `origin`, header built from parts with `originLabel`)
- Test: `web/tests/unit/portfolio-html.test.ts`, `web/tests/unit/portfolio-content.test.ts`

**Interfaces:**
- Produces: `careerContentSchema` (otherCourses optional), `careerGenerationSchema`, `renderShowcase(content, { origin?: ProjectOrigin; course; semester; team; repoUrl; figures; deliverablePaths })` (origin defaults to `"miami"`).

- [ ] Tests: outside courses rendered as `<strong>label</strong>: why`, escaped (`<b>` and `&`); no Coursework section when both lists empty; stored content without `otherCourses` parses; showcase header per origin with and without semester/team shows no leading/duplicated separators; miami header shows catalog title; `renderShowcase` without `origin` behaves as miami.
- [ ] Run; expect failures.
- [ ] Implement. Header: `const parts = [originLabel(origin, course), semester].filter(Boolean).join(", ")`, then team and repository appended with ` · ` only when present; the whole `<p class="meta">` omitted when empty.
- [ ] Run tests; expect pass.

### Task 3: Generate route

**Files:**
- Modify: `web/app/api/portfolio/generate/route.ts`
- Modify: `web/lib/providers/mock.ts` (career: echo `Other courses` lines as `otherCourses` plus one invented; showcase unchanged)
- Test: `web/tests/unit/portfolio-route.test.ts`

**Interfaces:**
- Consumes: `careerGenerationSchema`, `originOf`, `originPromptLine`.
- Produces: career payload field `otherCourses: {name, school}[]` (optional); showcase payload `origin` (optional), `course` may be empty; career response `content.otherCourses` always an array.

- [ ] Tests:
  - career with `otherCourses` [{Applied Regression, Ohio State}, {Data Mining, ""}] returns both with labels `Applied Regression, Ohio State` and `Data Mining`; the mock's invented course is dropped; duplicates submitted twice appear once.
  - career with no courses at all: prompt contains neither `Courses taken` nor `Other courses`; response `courses` and `otherCourses` are `[]`.
  - a 300-char outside course name is clipped, not rejected; more than 5 rows keeps the first 5.
  - older payload without `otherCourses` still 200.
  - showcase with each origin returns 200; prompt first line per origin; payload without `origin` treated as miami; empty `course` with `origin: "self"` accepted.
- [ ] Run; expect failures.
- [ ] Implement payload schema (`otherCourses: z.array(z.object({ name: clipped(80), school: clipped(80).default("") })).default([]).transform(rows => rows.map(trim).filter(name).slice(0,5))` with `.max` removed in favour of slicing, so a sixth row is dropped not rejected), prompt blocks, instructions line, post-validation (`key = s => s.toLowerCase().replace(/\s+/g, " ").trim()`; match model name against submitted `name` or `name, school`; label = student's `school ? name + ", " + school : name`; dedupe by key).
- [ ] Update mock career branch.
- [ ] Run tests; expect pass. Run the whole route file.

### Task 4: Draft, publish plan, repo name, store, restore

**Files:**
- Modify: `web/lib/portfolio/draft.ts` (`origin: ProjectOrigin`, `otherCourses: {name, school}[]` in `Draft` and `initialDraft`)
- Modify: `web/lib/portfolio/files.ts` (`showcaseRepoName(origin, course, title)`)
- Modify: `web/lib/portfolio/publish-plan.ts` (origin-aware repo name, header meta, README fallback)
- Modify: `web/lib/portfolio/store.ts` (`ShowcaseMeta.origin?`, `CareerStudent.otherCourses?`)
- Modify: `web/components/portfolio/Publish.tsx` (repo default and stored meta carry origin / otherCourses)
- Modify: `web/components/portfolio/PortfolioBuilder.tsx` (`openSite` restores `origin` via `originOf` and `otherCourses ?? []`; autosave resume normalises the same)
- Modify: `web/components/portfolio/ReviewStep.tsx` (pass `origin` to `renderShowcase`)
- Test: `web/tests/unit/portfolio-publish.test.ts`, `web/tests/unit/portfolio-files.test.ts`, `web/tests/unit/portfolio-wip.test.ts`

- [ ] Tests: `showcaseRepoName("miami","ISA 444","Sales forecast")` = `isa-444-sales-forecast`; other/self/personal = `sales-forecast`; publish plan for each origin (repo name, README fallback, header); republish keeps `existingRepoName`; a draft object without `origin`/`otherCourses` publishes as miami with no outside courses.
- [ ] Run; expect failures. Implement. Run; expect pass.

### Task 5: UI steps

**Files:**
- Modify: `web/app/(app)/portfolio/page.tsx` (pass `isGuest={isGuestEmail(session.user.email)}`)
- Modify: `web/components/portfolio/PortfolioBuilder.tsx` (accept `isGuest`, pass to Classes and Course steps and ContentEditor via ReviewStep)
- Create: `web/components/portfolio/career/GuestCoursesStep.tsx`
- Modify: `web/components/portfolio/career/ClassesStep.tsx` (render `GuestCoursesStep` when `isGuest`)
- Modify: `web/components/portfolio/showcase/CourseStep.tsx` (origin fieldset; guests default `other`, no miami radio; switching origin clears `course`)
- Modify: `web/components/portfolio/career/DetailsStep.tsx` (send `otherCourses`)
- Modify: `web/components/portfolio/showcase/StoryStep.tsx` (send `origin`, render with origin)
- Modify: `web/components/portfolio/ContentEditor.tsx` (Other courses list when present)

- [ ] Implement per spec copy. Guest rows: `Course 1` / `School 1` labels, "Add a course" up to 5, remove button per row ("Remove course 1"); primary button label "Continue" or "Continue without courses".
- [ ] Course step: `<fieldset><legend>Where did this project come from?</legend>` with four radios; Miami picker shown only under miami; text input `Course and school` (maxLength 80) only under other; hints under self; Continue gating per spec.
- [ ] `npx tsc --noEmit` and `npm run lint` clean.

### Task 6: End to end and accessibility

**Files:**
- Test: `web/tests/e2e/portfolio.spec.ts` (student self-study and personal showcases; header text and repository name in the fake GitHub push)
- Test: `web/tests/e2e/portfolio-guest.spec.ts` (new: guest via `/guest?pass=`; career with two courses then skipping; showcase from another school; axe A/AA on guest Classes and Course steps at desktop and mobile)

- [ ] Write the specs, run `npx playwright test tests/e2e/portfolio.spec.ts tests/e2e/portfolio-guest.spec.ts --workers=2`, fix until green.
- [ ] Run the existing Miami showcase and career tests unchanged; expect green.

### Task 7: Release v6.6.0

- [ ] Full unit suite, typecheck, lint; e2e with `--workers=2`.
- [ ] Independent whole-change review (fresh reviewer), fix findings.
- [ ] Browser check on a dev server as a guest and as a student (both wizards, 320px).
- [ ] Bump to 6.6.0, `docs/releases/v6.6.0.md`, CHANGELOG, `rm -rf .next/dev/types`, `node scripts/make-deploy-bundle.mjs`, verify footer version/date inside the archive (tar), commit, annotated tag, push.
