# Project Assistant — design spec

Date: 2026-07-23. Status: approved in brainstorming, pending user review of this
document before an implementation plan is written.

This rebuilds the legacy Streamlit "Project Coach" (`webapp/pages/02_project_coach.py`)
as the `project-coach` module (display name "Project Assistant"), currently a
placeholder route. Standing constraints apply: no git commits, no production
access, no secrets in the client, self-hosted (no third-party runtime
dependencies), WCAG 2.1 AA, Miami brand tokens, and no em dashes in user-facing
text.

## 1. What it is

A team project workspace, organized by course. A student picks a course from the
ISA catalog, creates a named project, and invites teammates. The team shares the
project and works with up to five AI coaches. Each coach runs a guided,
one-question-at-a-time conversation on one side of a split view and fills a live,
editable deliverable on the other. Deliverables save to the shared project and
export to Word.

## 2. Goals and non-goals

Goals:
- A persistent, per-course, team-shared project workspace.
- Five coaches, each producing a structured, editable, exportable deliverable.
- The scoping worksheet codified as a typed constant (ADR-010).
- Reuse of existing infrastructure (auth, model picker, streaming chat with tool
  calling, persistence, Word export).

Non-goals (explicitly deferred or excluded):
- **Real-time collaboration** (simultaneous editing, presence). Deferred to its
  own later cross-cutting slice (self-hosted Yjs over WSS, designed once and
  reused by the Coding Studio editor too). This build is shared-async only.
- **Course enrollment data.** The app has none and looks none up. "Course" is
  only a label the student selects for their own project.
- **Student ID / Banner ID.** Not collected. Teammates are identified by their
  authenticated Google name and `@miamioh.edu` email.
- **Shared runtime execution** (belongs to the deferred real-time slice, and is
  bounded by the browser-only execution model anyway).

## 3. Domain model

- **Course** — a typed constant `ISA_COURSES` (code + title), built from the ISA
  catalog, so the project-creation picker is a searchable list (e.g. ISA 401
  Business Intelligence and Data Visualization, ISA 444 Business Forecasting, ISA
  496 Business Analytics Practicum, ISA 650 Business Analytics Practicum). No
  enrollment lookup.
- **Project** — `{ id, courseCode, name, organization, ownerEmail, coachTypes[],
  createdAt, updatedAt }`. `coachTypes` is the subset of coaches the lead enabled
  for this project (any, some, or all five). Belongs to exactly one course.
- **Project member** — `{ projectId, email, name, role }` where role is `lead` or
  `member`. The creator is the sole `lead`.
- **Deliverable** — `{ id, projectId, coachType, content (JSON, schema per coach),
  transcript (JSON chat messages), lastUpdatedBy, updatedAt }`. One row per
  enabled coach type per project, created lazily on first use.

## 4. The five coaches and their deliverable schemas

Prompts are ported from the legacy (they are pedagogically tuned) and adapted so
the coach both chats one question at a time and calls the deliverable-update
tools as each part is settled (see section 5). Each coach type has a typed
content schema:

- **Scoping** — the 10-section worksheet:
  1. `projectName` (text)
  2. `organizationName` (text)
  3. `contacts` (text: names and titles)
  4. `problem` `{ whatProblem, whoAffected, howMuch, whyPriority }` (text each)
  5. `goals` — up to 3 rows of `{ goal, constraints }`
  6. `data` — `internalSources[]` and `externalSources[]` (up to 3 each) of
     `{ name, contains, granularity, frequency, identifiers, owner, storage,
     comments }`, plus `idealData` (text)
  7. `analysis` — up to 3 rows of `{ type, purpose, validation }`
  8. `ethics` `{ privacy, transparency, discriminationEquity, socialLicense,
     accountability, other }` (text each)
  9. `stakeholders` — rows of `{ orgDept, involvement, counterpart }`
  10. `experiment` `{ successMeasure, howTested, duration }` (text each)
- **Premortem** — `{ projectDescription, rows: [{ failure, howToAvoid }] }`
- **Team Structuring** — `{ rows: [{ name, skills, possibleTask }] }`
- **Devil's Advocate** — `{ decision, alternatives, risks, mitigations }`
- **Reflection** — `{ challenges, insights, growth }`

The scoping schema is the largest and most complex deliverable; it is the first
vertical slice (see section 12). The other four are small tables or field sets
that follow the same pattern.

## 5. Interaction: chat plus a live deliverable

A coach session is a split view: the coach chat on one side, the deliverable on
the other. The deliverable fills as the conversation progresses and is directly
editable by any team member.

Mechanism: the coach model is given a small, typed-per-coach set of tools that
write into the deliverable, for example `setField({ field, value })` and
`addRow` / `setRow({ table, index, row })`. As the coach and student settle a
section, the coach calls the tool; the AI SDK routes the call, the server applies
it to the deliverable content, persists it, and the panel reflects the change.
This is more robust than parsing a transcript at the end and reuses the existing
streaming-chat + tool-calling setup. Direct edits by a student write the same
content through the data layer.

## 6. Sharing and access control (async)

- The creating student is the **lead**. The lead: creates the project, picks and
  later changes which coaches it includes, and invites or removes teammates by
  `@miamioh.edu` email.
- All **members** may open the project and work on any enabled deliverable.
- An invited student sees the project under "Shared with me" once they sign in
  with the invited email. No separate acceptance flow.
- **Async only**: any member edits; saves are shared. Two near-simultaneous edits
  to the same deliverable resolve **last-save-wins**, and the panel shows "last
  updated by <name>, <time>." Real-time is the deferred slice.
- **Access control is enforced on every project/deliverable request**: the
  authenticated user must be the owner or a listed member. This is the privacy
  boundary; it is checked server-side on read and write.

## 7. Screens

- **My Projects** — the student's own projects grouped by course, a "Shared with
  me" section, and "New project" (pick course, name, organization; choose which
  coaches to include).
- **Project workspace** — header (course, name, organization, team), the enabled
  coaches with a done/started indicator, team management (invite by email; lead
  only), coach selection (lead only), and Export.
- **Coach session** — the split view (chat left, live editable deliverable
  right), the model picker, and "back to project."

## 8. Data layer and persistence

Drizzle/SQLite, following the existing persistence patterns:
- `projects` (id, courseCode, name, organization, ownerEmail, coachTypes JSON,
  createdAt, updatedAt)
- `project_members` (projectId, email, name, role) — enables the "shared with me"
  query (members where email = current user)
- `deliverables` (id, projectId, coachType, content JSON, transcript JSON,
  lastUpdatedBy, updatedAt)

## 9. Architecture and reuse

New route `app/(app)/project-coach/` with a small set of focused components and
server routes. It reuses, not reinvents:
- **Auth** (`@miamioh.edu` Google, JWT sessions) and its guards.
- **Model picker** + model catalog (the coaches run on the reasoning-capable
  models, like the legacy did).
- **Streaming chat** engine with cancellation and error recovery, extended with
  the deliverable-update **tools**.
- **Persistence** layer (Drizzle/SQLite).
- **Word export**: the `docx`-based document-builder approach and FSB styling
  from the JobApp export (new deliverable layouts, per section 10, not the
  resume/cover-letter templates).

New, focused pieces: the `ISA_COURSES` and coach constants (prompts + schemas),
the projects/deliverables/members data layer with team access checks, the coach
tool definitions, and the split-view workspace UI.

## 10. Export

Word (`.docx`) built with the same `docx`-library-based approach and FSB styling
as the JobApp export, but with new document layouts for these deliverables (not
the resume/cover-letter templates):
- The scoping deliverable renders to match the worksheet layout (its fields and
  the Goals / Data / Analysis / Ethics / Stakeholders tables).
- Each other deliverable renders its own table or field set.
- A cover carries the course, project name, organization, and the team members'
  names (no student ID).
- Export is available per-project (all started deliverables) and per-deliverable.

## 11. Error handling and accessibility

- Streaming errors, and empty or truncated model answers, are explained to the
  student rather than shown blank (as in the chat modules).
- A failed tool call leaves the deliverable unchanged and surfaces a quiet note;
  the student can still edit directly.
- WCAG 2.1 AA: the split view is keyboard-navigable and labelled; forms and
  tables are properly associated; the axe scans in the e2e suite cover the new
  screens. No em dashes in user-facing text.

## 12. Testing

- **Vitest**: the data layer and access checks (owner/member gating), the
  deliverable schemas and tool-call application, the course constant, and the
  Word export rendering (against a known deliverable).
- **Playwright** (with the mock model): create a project (pick course, choose
  coaches), invite a teammate (second identity sees it under "Shared with me"),
  open a coach, drive a short conversation that fills the deliverable, edit a
  field directly, and export. Axe scans on the new screens.

## 13. Build order

1. Data layer + projects/members/deliverables + access checks, and the My
   Projects and project-workspace screens.
2. The **Scoping Coach** and its 10-section document end to end (the large, most
   complex vertical slice): split view, tool-call fill, direct edit, persist,
   Word export.
3. The other four coaches, which follow the same pattern over much simpler
   schemas.
4. Per-project export and the team-management UI.

Each step is verified (typecheck, lint, unit, e2e incl. axe) and recorded in the
migration log, per the project's practice. Working tree stays uncommitted.

## 14. Open questions

None outstanding. The design decisions (workspace with deliverables; persistent
projects; course + async team sharing; split-view chat plus live deliverable; all
enabled coaches produce a deliverable; lead selects coaches; names only, no
student ID; real-time deferred) were settled in brainstorming.
