# Guests and project origin (Portfolio Builder), design

**Date:** 2026-09-23. **Release target:** v6.6.0. **Status:** approved in
conversation (professor, 2026-09-23); written spec reviewed by the lead
engineer per the professor's delegation.

This is project 1 of 4 agreed on 2026-09-23:

1. Guests and project origin (this spec).
2. Course and skill data: FSB majors, co-majors and the AI for Business minor
   from the bulletin; Simple Syllabus evidence (next pull 2027-02-01);
   25 to 35 business-domain skills; anti-overselling guards.
3. Major-first course picker with Done / Taking now / Not yet and automatic,
   labelled prerequisites.
4. Reading GitHub through the student's existing connection.

## Goal

Guests (invited collaborators outside Miami) and students building a showcase
for work that did not come from a Miami course can use the Portfolio Builder
without being forced through Miami-only inputs, and the published page reads
naturally in every case.

## Constraints (professor, 2026-09-23)

- Do not break the running app: existing drafts, autosaves, published sites,
  and Miami students' flows behave exactly as before.
- Slick and usable; passes the existing axe WCAG A/AA checks, works by
  keyboard and at 320px.
- Value-adding: every change must earn its place.

## What exists today

- Guests sign in as `guest-<n>@guest.chatisa` (`lib/auth/guest.ts`,
  `GUEST_EMAIL_DOMAIN`).
- Career wizard: resume, classes, projects, details, review. The Classes step
  (`components/portfolio/career/ClassesStep.tsx`) requires at least one
  Miami course from the catalog.
- The generate route (`app/api/portfolio/generate/route.ts`) accepts an empty
  course list, but clips each course to 20 characters and keeps only codes
  matching `[A-Z]{2,4} \d{3}` that the student listed, so any course from
  another school would be silently dropped.
- The renderer (`lib/portfolio/html.ts`) omits an empty Coursework section.
- Showcase wizard: course, files, story, review. The Course step requires a
  Miami course. The course feeds the model prompt, the page header (code
  only today), the README fallback ("Built for ISA 444"), and the repository
  name (`showcaseRepoName` = slug of code plus title).

## Design

### A. Career portfolio: guest courses

1. `app/(app)/portfolio/page.tsx` passes `isGuest` (the session email ends
   with `@guest.chatisa`) to `PortfolioBuilder`, which passes it to the
   Classes and Course steps. Miami students see no change.
2. For guests the Classes step renders **Relevant courses (optional)**:
   - A short line: "Add 4 to 5 courses that fit the work you want to show.
     They help the page tell your story. You can skip this."
   - Up to 5 rows, each with two labelled inputs: **Course** (required for
     the row to count) and **School (optional)**. Rows start with one empty
     row and an **Add a course** button up to 5.
   - Continue is always enabled; the primary button reads **Continue** when
     at least one course is filled and **Continue without courses** when none
     is.
3. Draft field `otherCourses: { name: string; school: string }[]` (default
   `[]`; absent in older drafts and autosaves means `[]`).
4. Route: the career payload gains `otherCourses` (max 5; `name` and `school`
   clipped to 80, never rejected; rows with an empty name dropped). Missing
   field means `[]`, so an older client still works.
5. Prompt: when there are Miami courses, the existing "Courses taken" block;
   when there are other courses, an "Other courses (outside Miami)" block
   listing `name, school`. When there are neither, no course line at all
   (the model is told nothing about coursework, so it cannot write "no
   courses listed").
6. Content: `careerContentSchema` gains `otherCourses` as an optional array of
   `{ name (1..120), why (1..240) }`, max 5, so every stored site still
   parses. Generation uses a derived schema where the field is required
   (strict structured output handles optional fields poorly).
7. Instructions add: "otherCourses: for each listed outside course that
   supports the story, one sentence on why it matters. Copy the course name
   exactly as given. Leave it empty if none were listed."
8. Post-validation: each returned outside course is kept only if it matches
   a submitted course (case- and whitespace-insensitive, by name, or by
   `name, school`); its displayed label is the student's own text
   (`name` or `name, school`), never the model's wording. Duplicates removed.
   With no outside courses submitted, the field is forced to `[]`.
9. Renderer: Coursework lists the Miami courses as today, then outside
   courses as `<strong>label</strong>: why`, all escaped. No courses of
   either kind means no Coursework section.
10. Editor: the review step's content editor shows an **Other courses** list
    (name, why; max 5) whenever the content has any.
11. Restoring a saved site and the autosave carry `otherCourses`.

### B. Showcase: "Where did this project come from?"

1. Draft field `origin: "miami" | "other" | "self" | "personal"`; absent means
   `"miami"` everywhere (drafts, autosaves, stored sites, route payloads).
2. The Course step becomes a fieldset with legend **Where did this project
   come from?** and four radios:
   - **A Miami course** (hidden for guests): the existing single-course
     picker below it.
   - **A course at another school**: one labelled text input, **Course and
     school**, placeholder "STAT 4520, Ohio State" (max 80, counted).
   - **Self-study**, hint "Working through a book, an online course, or your
     own plan".
   - **A hobby or personal project**.
   Students default to A Miami course; guests default to A course at another
   school. Semester and team stay optional for every origin. Continue needs a
   picked Miami course or a non-empty typed course for those two origins, and
   nothing for the other two.
3. Route: the showcase payload gains `origin` (default `"miami"`); `course` is
   clipped to 80 and may be empty. The prompt's first line depends on origin:
   `Course: <x>`, `Course (another school): <x>`, `A self-study project.`, or
   `A personal project.`.
4. Header line (renderer), before semester, team and repository link:
   - miami: `ISA 444 - Business Forecasting` (code plus catalog title, the
     same convention the career Coursework adopted in 6.4.5; code alone when
     the catalog has no title);
   - other: the typed text;
   - self: `Self-study project`; personal: `Personal project`.
   Separators are only emitted between present parts, so there is never a
   leading comma or dot.
5. Repository name: `showcaseRepoName(origin, course, title)`: miami keeps
   `slug(code + title)`; the other three use `slug(title)`. The existing
   name-clash handling is unchanged. A republish keeps its stored repository
   name, so existing sites never move.
6. README fallback (used only when the model returns none): "Built for
   ISA 444.", "Built for STAT 4520, Ohio State.", "A self-study project.", or
   "A personal project.", followed by the existing credit line.
7. Stored `ShowcaseMeta` gains optional `origin`.

### C. Compatibility

- Every new field is optional on read, with the defaults above, so existing
  drafts, autosaves (`pb-wip`), stored sites and published pages behave as
  before. A republished Miami showcase gains the course title in its header,
  which is the only visible change to existing sites.
- The route accepts payloads without the new fields (a browser with an old
  bundle cached mid-deploy keeps working).
- Job Scout's published-work records carry no course, so nothing changes
  there.

## Accessibility

- The origin choice is a native fieldset, legend and radio group; the
  picker and text input for the chosen origin follow it in DOM order.
- Every input has a visible label; the course rows are labelled
  "Course 1" to "Course 5" and "School 1" to "School 5" for screen readers.
- Errors use the existing focused `role="alert"` pattern.
- New and changed steps are covered by the axe A/AA scan at desktop and
  320px.

## Testing

Unit:
- Renderer: outside courses render escaped with the student's label; no
  Coursework section when both lists are empty; showcase header for each
  origin, with and without semester and team (no stray separators); Miami
  header includes the catalog title.
- `showcaseRepoName` for each origin; README fallback for each origin.
- Route: accepts every origin; defaults a missing origin to miami; clips long
  course text rather than rejecting; drops outside courses the student did
  not submit and relabels kept ones with the student's text; forces `[]`
  when none were submitted; omits all course lines from the prompt when
  there are no courses.
- Contract test: payloads built through the real client code at every cap
  pass the route schema.
- Store: a stored site and an autosave without the new fields load with the
  defaults.

End to end (mock model):
- A guest builds a career portfolio with two outside courses (they appear
  with the guest's own labels) and again skipping courses (no Coursework).
- A guest showcase from another school, and a student showcase for
  self-study and for a personal project (header and repository name).
- A Miami student's career and showcase flows are unchanged.
- axe A/AA on the new steps.

The mock model echoes submitted outside courses and adds one invented course
so the post-validation filter is exercised.

## Out of scope

- Mapping outside courses to skills or Job Scout matching (guests have no
  course-skill mapping; project 2 covers Miami courses only).
- Guest access to Job Scout's profile.
- Any change to Miami students' career course picking (project 3).
