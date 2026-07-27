# Project Assistant — Plan 1: Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the data layer, course/coach constants, access control, and the two non-coach screens (My Projects and the project workspace) for the Project Assistant module, so a student can create a course-scoped team project, see it listed, and open its workspace.

**Architecture:** A new `project-coach` route under `app/(app)/`, backed by three new Drizzle/SQLite tables (`projects`, `project_members`, `deliverables`) and a focused data-access module (`lib/db/projects.ts`) that enforces owner-or-member access on every read and write. Course and coach metadata live in typed constants (`lib/project/courses.ts`, `lib/project/coaches.ts`). This plan builds the skeleton only: the coach split-view sessions, deliverable editing, and Word export come in Plans 2 to 4.

**Tech Stack:** Next.js 16 (App Router, server components), TypeScript, Drizzle ORM + better-sqlite3, next-auth (Google `@miamioh.edu`), Tailwind v4, Vitest, Playwright + axe.

## Global Constraints

- **No git commits, no deploys, no production access.** The working tree stays uncommitted; every task ends by running its gate, not by committing.
- **No secrets in the client;** check env var names only, never values.
- **No em dashes in any user-facing text** (copy uses periods, commas, or parentheses instead).
- **WCAG 2.1 AA:** new screens are keyboard-navigable and labelled; the e2e suite runs axe on them.
- **Miami brand tokens** and existing component idioms (`ribbon`, `rounded-card`, `border-medium-tan`, `bg-light-tan`, model-option empty states) are reused, not reinvented.
- **Follow the codebase, not training data:** this is a modified Next.js. Read `node_modules/next/dist/docs/` before writing framework code (per `web/AGENTS.md`).
- **Emails are stored and compared lowercased** (`.toLowerCase()`), matching every existing data-layer function.
- **Timestamps are ISO-8601 UTC strings** (`new Date().toISOString()`), matching the schema.
- **Access control is the privacy boundary:** the authenticated user must be the project owner or a listed member on every project/deliverable request. Enforced server-side.
- **Module slug is `project-coach`; display name is "Project Assistant".** Analytics module key is `project_coach`.

---

## File Structure

**Created:**
- `lib/project/courses.ts` — `Course` type, `ISA_COURSES` constant, `findCourse(code)`, `courseLabel(course)`.
- `lib/project/coaches.ts` — `CoachType` union, `COACHES` metadata registry, `isCoachType(x)`, `coachLabel(type)`.
- `lib/db/projects.ts` — projects/members/deliverables data layer with access checks.
- `app/(app)/project-coach/page.tsx` — My Projects screen (server component).
- `app/(app)/project-coach/new/page.tsx` — New project screen (server component wrapper).
- `app/(app)/project-coach/[projectId]/page.tsx` — Project workspace screen (server component).
- `components/project/NewProjectForm.tsx` — client form (course select, name, organization, coach checkboxes).
- `components/project/ProjectList.tsx` — client-free presentational list used by My Projects.
- `app/api/projects/route.ts` — `POST` create, `GET` list.
- `tests/unit/project-courses.test.ts`, `tests/unit/project-coaches.test.ts`, `tests/unit/project-db.test.ts`.
- `tests/e2e/project-assistant.spec.ts`.

**Modified:**
- `lib/db/schema.ts` — add `projects`, `projectMembers`, `deliverables` tables.
- `drizzle/0005_*.sql` — generated migration (via `npx drizzle-kit generate`).

**Boundary note:** the new domain goes in `lib/db/projects.ts` rather than growing `lib/db/index.ts` (already ~980 lines). It imports `getDb` from `./index` and `* as schema` from `./schema`, following the same idioms (drizzle query builder, `randomUUID`, lowercased emails, ISO timestamps, ownership-checked reads that return `undefined`).

---

## Interfaces produced by this plan (relied on by Plans 2 to 4)

```ts
// lib/project/courses.ts
export interface Course { code: string; title: string }        // code e.g. "401/501"
export const ISA_COURSES: readonly Course[];
export function findCourse(code: string): Course | undefined;
export function courseLabel(course: Course): string;           // "ISA 401/501 — no: uses hyphen? see below"

// lib/project/coaches.ts
export type CoachType =
  | "scoping" | "premortem" | "team_structuring" | "devils_advocate" | "reflection";
export interface CoachMeta { type: CoachType; label: string; blurb: string; order: number }
export const COACHES: readonly CoachMeta[];
export function isCoachType(x: string): x is CoachType;
export function coachLabel(type: CoachType): string;

// lib/db/projects.ts
export interface ProjectRow { id: string; courseCode: string; name: string;
  organization: string; ownerEmail: string; coachTypes: CoachType[];
  createdAt: string; updatedAt: string }
export interface MemberRow { id: string; projectId: string; email: string;
  name: string | null; role: "lead" | "member"; createdAt: string }
export function createProject(params: { ownerEmail: string; ownerName: string | null;
  courseCode: string; name: string; organization: string;
  coachTypes: CoachType[] }): string;                          // returns project id
export function getAccessibleProject(id: string, userEmail: string): ProjectRow | undefined;
export function isProjectMember(projectId: string, userEmail: string): boolean;
export function listOwnedProjects(userEmail: string): ProjectRow[];
export function listSharedProjects(userEmail: string): ProjectRow[];
export function listProjectMembers(projectId: string): MemberRow[];
export function addProjectMember(params: { projectId: string; email: string;
  name: string | null; role: "lead" | "member" }): void;
```

> Copy rule reminder: no em dashes. `courseLabel` renders `"ISA 401/501: Business Intelligence and Data Visualization"` (a colon), never with an em dash. The interface comment above is a note to self, not final copy.

---

### Task 1: Course constant and lookup

**Files:**
- Create: `lib/project/courses.ts`
- Test: `tests/unit/project-courses.test.ts`

**Interfaces:**
- Produces: `Course`, `ISA_COURSES`, `findCourse(code)`, `courseLabel(course)`.

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/project-courses.test.ts
import { describe, expect, it } from "vitest";
import { ISA_COURSES, findCourse, courseLabel } from "@/lib/project/courses";

describe("ISA course catalog", () => {
  it("includes every catalog course with a code and title", () => {
    expect(ISA_COURSES.length).toBe(50);
    for (const c of ISA_COURSES) {
      expect(c.code).toMatch(/^\d{3}(\/\d{3})?$/);
      expect(c.title.length).toBeGreaterThan(0);
    }
  });

  it("keeps codes unique", () => {
    const codes = ISA_COURSES.map((c) => c.code);
    expect(new Set(codes).size).toBe(codes.length);
  });

  it("finds a course by code, including dual-listed codes", () => {
    expect(findCourse("401/501")?.title).toBe(
      "Business Intelligence and Data Visualization",
    );
    expect(findCourse("444/544")?.title).toBe("Business Forecasting");
    expect(findCourse("nope")).toBeUndefined();
  });

  it("labels a course with ISA prefix and a colon, no em dash", () => {
    const label = courseLabel(findCourse("444/544")!);
    expect(label).toBe("ISA 444/544: Business Forecasting");
    expect(label).not.toContain("—");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/unit/project-courses.test.ts`
Expected: FAIL, cannot resolve `@/lib/project/courses`.

- [ ] **Step 3: Write the implementation**

Create `lib/project/courses.ts` with the full ISA catalog (transcribed verbatim from the professor's course list, dual-listed graduate crosswalks kept as combined codes):

```ts
/**
 * The ISA course catalog. "Course" here is only a label a student selects for
 * their own project. The app looks up no enrollment data and stores no roster.
 * Codes are combined for dual-listed courses (e.g. "401/501").
 */
export interface Course {
  code: string;
  title: string;
}

export const ISA_COURSES: readonly Course[] = [
  { code: "125", title: "Introduction to Business Statistics" },
  { code: "177", title: "Independent Studies" },
  { code: "211", title: "Information Technology and Data Driven Decision Making in Business" },
  { code: "225", title: "Principles of Business Analytics" },
  { code: "235", title: "Information Technology and the Intelligent Enterprise" },
  { code: "241", title: "Database for Analytics" },
  { code: "242", title: "Programming for Analytics" },
  { code: "250", title: "Basic Math for Analytics" },
  { code: "277", title: "Independent Studies" },
  { code: "301", title: "Business Data Communications and Security" },
  { code: "303", title: "Enterprise Systems" },
  { code: "305", title: "Information Technology Governance, Risk Management, Security and Audit" },
  { code: "321", title: "Optimization in Business Analytics" },
  { code: "333", title: "Nonparametric Statistics" },
  { code: "335", title: "Blockchain and Business Applications" },
  { code: "336", title: "Generative AI in Business" },
  { code: "340", title: "Internship" },
  { code: "345", title: "Database Systems and Data Warehousing" },
  { code: "365", title: "Statistical Monitoring and Design of Experiments" },
  { code: "377", title: "Independent Studies" },
  { code: "381", title: "Concepts in Business Programming" },
  { code: "387", title: "Designing Business Systems" },
  { code: "391", title: "Applied Regression Analysis in Business" },
  { code: "401/501", title: "Business Intelligence and Data Visualization" },
  { code: "403", title: "Building Web and Mobile Business Applications" },
  { code: "405", title: "Information Security" },
  { code: "406", title: "IT Project Management" },
  { code: "414/514", title: "Managing Big Data" },
  { code: "419", title: "Data Driven Security" },
  { code: "424", title: "Data Infrastructure for the Enterprise" },
  { code: "444/544", title: "Business Forecasting" },
  { code: "477", title: "Independent Studies" },
  { code: "480", title: "Topics in Business Analytics" },
  { code: "481", title: "Topics in Information Systems" },
  { code: "491/591", title: "Introduction to Data Mining in Business" },
  { code: "495", title: "Managing the Intelligent Enterprise" },
  { code: "496", title: "Business Analytics Practicum" },
  { code: "612", title: "Advanced Business Intelligence" },
  { code: "616", title: "Communicating with Data" },
  { code: "621", title: "Enabling Technology Topics I" },
  { code: "628", title: "Information Technology and Analytic's Role in the Enterprise" },
  { code: "629", title: "Leveraging IT and Data Across the Business" },
  { code: "630", title: "Machine Learning Applications in Business" },
  { code: "632", title: "Big Data Analytics and Modern AI" },
  { code: "633", title: "Experimental Design and Causal Methods" },
  { code: "634", title: "Systems Modeling and Optimization" },
  { code: "641", title: "Data Discovery Through Business Analytics for Managers" },
  { code: "645", title: "Business Analytics for the Executive" },
  { code: "650", title: "Business Analytics Practicum" },
  { code: "677", title: "Independent Studies" },
];

const BY_CODE = new Map(ISA_COURSES.map((c) => [c.code, c]));

export function findCourse(code: string): Course | undefined {
  return BY_CODE.get(code);
}

/** "ISA 401/501: Business Intelligence and Data Visualization". No em dash. */
export function courseLabel(course: Course): string {
  return `ISA ${course.code}: ${course.title}`;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run tests/unit/project-courses.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Checkpoint (no commit)**

Run: `npm run typecheck`
Expected: no errors. Leave the working tree uncommitted.

---

### Task 2: Coach registry metadata

Only the coach identity and display metadata are defined here. Each coach's system prompt and deliverable content schema arrive with that coach's own plan (Scoping in Plan 2, the other four in Plan 3). Defining the `CoachType` union now lets `projects.coachTypes` be typed and lets the create form list the choices.

**Files:**
- Create: `lib/project/coaches.ts`
- Test: `tests/unit/project-coaches.test.ts`

**Interfaces:**
- Produces: `CoachType`, `CoachMeta`, `COACHES`, `isCoachType(x)`, `coachLabel(type)`.

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/project-coaches.test.ts
import { describe, expect, it } from "vitest";
import { COACHES, isCoachType, coachLabel } from "@/lib/project/coaches";

describe("coach registry", () => {
  it("defines exactly the five coaches in a stable order", () => {
    expect(COACHES.map((c) => c.type)).toEqual([
      "scoping",
      "premortem",
      "team_structuring",
      "devils_advocate",
      "reflection",
    ]);
  });

  it("gives every coach a label and a blurb with no em dash", () => {
    for (const c of COACHES) {
      expect(c.label.length).toBeGreaterThan(0);
      expect(c.blurb.length).toBeGreaterThan(0);
      expect(c.label).not.toContain("—");
      expect(c.blurb).not.toContain("—");
    }
  });

  it("narrows valid coach types", () => {
    expect(isCoachType("scoping")).toBe(true);
    expect(isCoachType("banana")).toBe(false);
    expect(coachLabel("premortem")).toBe("Premortem");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/unit/project-coaches.test.ts`
Expected: FAIL, cannot resolve `@/lib/project/coaches`.

- [ ] **Step 3: Write the implementation**

```ts
// lib/project/coaches.ts
/**
 * The five project coaches. Each produces one structured, editable deliverable.
 * Prompts and per-coach content schemas are defined in each coach's own slice;
 * this module carries only identity and display metadata so a project can store
 * which coaches its lead enabled.
 */
export type CoachType =
  | "scoping"
  | "premortem"
  | "team_structuring"
  | "devils_advocate"
  | "reflection";

export interface CoachMeta {
  type: CoachType;
  label: string;
  blurb: string;
  order: number;
}

export const COACHES: readonly CoachMeta[] = [
  {
    type: "scoping",
    label: "Project Scoping",
    blurb:
      "Turns a vague idea into a clear brief: the problem, goals, data, analysis, ethics, stakeholders, and how you will measure success.",
    order: 1,
  },
  {
    type: "premortem",
    label: "Premortem",
    blurb:
      "Imagines the project has already failed, then works backward to name the likely failures and how to avoid them.",
    order: 2,
  },
  {
    type: "team_structuring",
    label: "Team Structuring",
    blurb:
      "Maps each teammate's skills to the tasks that suit them, so the work is shared deliberately.",
    order: 3,
  },
  {
    type: "devils_advocate",
    label: "Devil's Advocate",
    blurb:
      "Pressure tests a key decision by arguing the other side, surfacing alternatives, risks, and mitigations.",
    order: 4,
  },
  {
    type: "reflection",
    label: "Reflection",
    blurb:
      "Helps the team look back on challenges, insights, and growth once the work is under way or done.",
    order: 5,
  },
];

const BY_TYPE = new Map(COACHES.map((c) => [c.type, c]));

export function isCoachType(x: string): x is CoachType {
  return BY_TYPE.has(x as CoachType);
}

export function coachLabel(type: CoachType): string {
  return BY_TYPE.get(type)?.label ?? type;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run tests/unit/project-coaches.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Checkpoint (no commit)**

Run: `npm run typecheck` — expected clean. Working tree stays uncommitted.

---

### Task 3: Schema tables and migration

**Files:**
- Modify: `lib/db/schema.ts` (append three tables)
- Create: `drizzle/0005_*.sql` (generated, do not hand-write)

**Interfaces:**
- Produces: `schema.projects`, `schema.projectMembers`, `schema.deliverables`.

- [ ] **Step 1: Append the tables to `lib/db/schema.ts`**

Add at the end of the file (keep the existing import block; it already imports `index`, `integer`, `sqliteTable`, `text`, `uniqueIndex`):

```ts
/**
 * A team project workspace, scoped to a course the student selects (ADR-010).
 * "Course" is a label only; no enrollment data is looked up or stored.
 * `coachTypesJson` is the subset of coaches the lead enabled, as a JSON array.
 */
export const projects = sqliteTable(
  "projects",
  {
    id: text("id").primaryKey(),
    courseCode: text("course_code").notNull(),
    name: text("name").notNull(),
    organization: text("organization").notNull().default(""),
    ownerEmail: text("owner_email")
      .notNull()
      .references(() => users.email, { onDelete: "cascade" }),
    coachTypesJson: text("coach_types_json").notNull().default("[]"),
    createdAt: text("created_at").notNull(),
    updatedAt: text("updated_at").notNull(),
  },
  (t) => [index("projects_owner_updated").on(t.ownerEmail, t.updatedAt)],
);

/**
 * Team membership. Identifies teammates by their authenticated Google name and
 * `@miamioh.edu` email only. No student or Banner ID. The `email` index powers
 * the "shared with me" query for an invited student.
 */
export const projectMembers = sqliteTable(
  "project_members",
  {
    id: text("id").primaryKey(),
    projectId: text("project_id")
      .notNull()
      .references(() => projects.id, { onDelete: "cascade" }),
    email: text("email").notNull(),
    /** Known once the member has signed in at least once; may be null before. */
    name: text("name"),
    /** lead | member. Exactly one lead: the creator. */
    role: text("role").notNull(),
    createdAt: text("created_at").notNull(),
  },
  (t) => [
    uniqueIndex("project_members_project_email").on(t.projectId, t.email),
    index("project_members_email").on(t.email),
  ],
);

/**
 * One deliverable per enabled coach per project, created lazily on first use.
 * `contentJson` is the structured deliverable (schema per coach); `transcriptJson`
 * is the coach chat. `lastUpdatedBy` is a display name (or email) for the
 * "last updated by <name>" note, since edits are shared last-save-wins.
 */
export const deliverables = sqliteTable(
  "deliverables",
  {
    id: text("id").primaryKey(),
    projectId: text("project_id")
      .notNull()
      .references(() => projects.id, { onDelete: "cascade" }),
    coachType: text("coach_type").notNull(),
    contentJson: text("content_json").notNull().default("{}"),
    transcriptJson: text("transcript_json").notNull().default("[]"),
    lastUpdatedBy: text("last_updated_by"),
    updatedAt: text("updated_at").notNull(),
  },
  (t) => [
    uniqueIndex("deliverables_project_coach").on(t.projectId, t.coachType),
  ],
);
```

- [ ] **Step 2: Generate the migration**

Run: `npx drizzle-kit generate`
Expected: a new file `drizzle/0005_<random>.sql` is created containing `CREATE TABLE projects`, `project_members`, `deliverables` and their indexes. Do not edit it by hand.

- [ ] **Step 3: Verify the migration content**

Read the generated `drizzle/0005_*.sql`. Confirm it has three `CREATE TABLE` statements and the four indexes (`projects_owner_updated`, `project_members_project_email`, `project_members_email`, `deliverables_project_coach`), and that no other table is altered (the diff should be additive only).

- [ ] **Step 4: Confirm migration applies**

Migrations run automatically on first `getDb()`. Confirm with a throwaway check:

Run: `CHATISA_DATA_DIR=$(mktemp -d) npx tsx -e "import('@/lib/db').then(m => { m.dbReady(); console.log('migrated ok'); m.closeDb(); })"`
Expected: prints `migrated ok` with no migration error. (On Windows PowerShell, set `$env:CHATISA_DATA_DIR` to a fresh temp dir first, then run the `npx tsx -e` command.)

- [ ] **Step 5: Checkpoint (no commit)**

Run: `npm run typecheck` — expected clean. Working tree stays uncommitted.

---

### Task 4: Project data layer with access checks

**Files:**
- Create: `lib/db/projects.ts`
- Test: `tests/unit/project-db.test.ts`

**Interfaces:**
- Consumes: `getDb` from `@/lib/db`, `schema.projects/projectMembers/deliverables`, `CoachType`.
- Produces: `ProjectRow`, `MemberRow`, `createProject`, `getAccessibleProject`, `isProjectMember`, `listOwnedProjects`, `listSharedProjects`, `listProjectMembers`, `addProjectMember`.

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/project-db.test.ts
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-project-db-"));
process.env.CHATISA_DATA_DIR = dataDir;

const { closeDb, upsertUser } = await import("@/lib/db");
const {
  createProject,
  getAccessibleProject,
  isProjectMember,
  listOwnedProjects,
  listSharedProjects,
  listProjectMembers,
  addProjectMember,
} = await import("@/lib/db/projects");

const LEAD = "lead@miamioh.edu";
const MATE = "mate@miamioh.edu";
const STRANGER = "stranger@miamioh.edu";

beforeAll(() => {
  upsertUser(LEAD, "Team Lead");
  upsertUser(MATE, "Teammate");
  upsertUser(STRANGER, "Nobody");
  // Owner rows are FK-referenced by projects (foreign_keys = ON), so every
  // owner used below must be seeded, including the mixed-case one.
  upsertUser("Mixed.Case@Miamioh.edu", "Mixed");
});

afterAll(() => {
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
});

describe("project data layer", () => {
  it("creates a project and enrolls the owner as the sole lead", () => {
    const id = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "401/501",
      name: "Retail dashboard",
      organization: "Kroger",
      coachTypes: ["scoping", "premortem"],
    });
    const project = getAccessibleProject(id, LEAD);
    expect(project?.name).toBe("Retail dashboard");
    expect(project?.courseCode).toBe("401/501");
    expect(project?.coachTypes).toEqual(["scoping", "premortem"]);

    const members = listProjectMembers(id);
    expect(members).toHaveLength(1);
    expect(members[0].email).toBe(LEAD);
    expect(members[0].role).toBe("lead");
  });

  it("hides a project from a non-member", () => {
    const id = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "444/544",
      name: "Demand forecast",
      organization: "",
      coachTypes: ["scoping"],
    });
    expect(getAccessibleProject(id, STRANGER)).toBeUndefined();
    expect(isProjectMember(id, STRANGER)).toBe(false);
  });

  it("shares a project with an invited member and lists it for them", () => {
    const id = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "496",
      name: "Practicum project",
      organization: "Acme",
      coachTypes: ["scoping", "reflection"],
    });
    addProjectMember({ projectId: id, email: MATE, name: "Teammate", role: "member" });

    expect(isProjectMember(id, MATE)).toBe(true);
    expect(getAccessibleProject(id, MATE)?.name).toBe("Practicum project");

    // Owner sees it under "owned", member sees it under "shared".
    expect(listOwnedProjects(LEAD).some((p) => p.id === id)).toBe(true);
    expect(listOwnedProjects(MATE).some((p) => p.id === id)).toBe(false);
    expect(listSharedProjects(MATE).some((p) => p.id === id)).toBe(true);
    expect(listSharedProjects(LEAD).some((p) => p.id === id)).toBe(false);
  });

  it("treats emails case-insensitively", () => {
    const id = createProject({
      ownerEmail: "Mixed.Case@Miamioh.edu",
      ownerName: "Mixed",
      courseCode: "225",
      name: "Casing",
      organization: "",
      coachTypes: [],
    });
    expect(getAccessibleProject(id, "mixed.case@miamioh.edu")?.name).toBe("Casing");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/unit/project-db.test.ts`
Expected: FAIL, cannot resolve `@/lib/db/projects`.

- [ ] **Step 3: Write the implementation**

```ts
// lib/db/projects.ts
import { randomUUID } from "node:crypto";
import { and, desc, eq, ne } from "drizzle-orm";
import { getDb } from "./index";
import * as schema from "./schema";
import type { CoachType } from "@/lib/project/coaches";

export interface ProjectRow {
  id: string;
  courseCode: string;
  name: string;
  organization: string;
  ownerEmail: string;
  coachTypes: CoachType[];
  createdAt: string;
  updatedAt: string;
}

export interface MemberRow {
  id: string;
  projectId: string;
  email: string;
  name: string | null;
  role: "lead" | "member";
  createdAt: string;
}

function toProjectRow(row: typeof schema.projects.$inferSelect): ProjectRow {
  let coachTypes: CoachType[] = [];
  try {
    const parsed = JSON.parse(row.coachTypesJson);
    if (Array.isArray(parsed)) coachTypes = parsed as CoachType[];
  } catch {
    // A malformed value should not break the whole workspace; treat as none.
  }
  return {
    id: row.id,
    courseCode: row.courseCode,
    name: row.name,
    organization: row.organization,
    ownerEmail: row.ownerEmail,
    coachTypes,
    createdAt: row.createdAt,
    updatedAt: row.updatedAt,
  };
}

function toMemberRow(row: typeof schema.projectMembers.$inferSelect): MemberRow {
  return {
    id: row.id,
    projectId: row.projectId,
    email: row.email,
    name: row.name,
    role: row.role === "lead" ? "lead" : "member",
    createdAt: row.createdAt,
  };
}

/** Creates the project and enrolls the creator as the sole lead, atomically. */
export function createProject(params: {
  ownerEmail: string;
  ownerName: string | null;
  courseCode: string;
  name: string;
  organization: string;
  coachTypes: CoachType[];
}): string {
  const db = getDb();
  const id = randomUUID();
  const now = new Date().toISOString();
  const ownerEmail = params.ownerEmail.toLowerCase();
  db.transaction((tx) => {
    tx.insert(schema.projects)
      .values({
        id,
        courseCode: params.courseCode,
        name: params.name,
        organization: params.organization,
        ownerEmail,
        coachTypesJson: JSON.stringify(params.coachTypes),
        createdAt: now,
        updatedAt: now,
      })
      .run();
    tx.insert(schema.projectMembers)
      .values({
        id: randomUUID(),
        projectId: id,
        email: ownerEmail,
        name: params.ownerName,
        role: "lead",
        createdAt: now,
      })
      .run();
  });
  return id;
}

/** True when the user owns or is a listed member of the project. */
export function isProjectMember(projectId: string, userEmail: string): boolean {
  const member = getDb()
    .select({ id: schema.projectMembers.id })
    .from(schema.projectMembers)
    .where(
      and(
        eq(schema.projectMembers.projectId, projectId),
        eq(schema.projectMembers.email, userEmail.toLowerCase()),
      ),
    )
    .get();
  return member !== undefined;
}

/**
 * Returns the project only when the user is the owner or a member. This is the
 * privacy boundary: every workspace read goes through it, and a non-member gets
 * the same undefined a bad id would give, so no project ids leak.
 */
export function getAccessibleProject(
  id: string,
  userEmail: string,
): ProjectRow | undefined {
  if (!isProjectMember(id, userEmail)) return undefined;
  const row = getDb()
    .select()
    .from(schema.projects)
    .where(eq(schema.projects.id, id))
    .get();
  return row ? toProjectRow(row) : undefined;
}

/** Projects the user created, newest activity first. */
export function listOwnedProjects(userEmail: string): ProjectRow[] {
  return getDb()
    .select()
    .from(schema.projects)
    .where(eq(schema.projects.ownerEmail, userEmail.toLowerCase()))
    .orderBy(desc(schema.projects.updatedAt))
    .all()
    .map(toProjectRow);
}

/** Projects the user was invited to but does not own ("shared with me"). */
export function listSharedProjects(userEmail: string): ProjectRow[] {
  const email = userEmail.toLowerCase();
  const rows = getDb()
    .select({ project: schema.projects })
    .from(schema.projectMembers)
    .innerJoin(
      schema.projects,
      eq(schema.projectMembers.projectId, schema.projects.id),
    )
    .where(
      and(
        eq(schema.projectMembers.email, email),
        ne(schema.projects.ownerEmail, email),
      ),
    )
    .orderBy(desc(schema.projects.updatedAt))
    .all();
  return rows.map((r) => toProjectRow(r.project));
}

export function listProjectMembers(projectId: string): MemberRow[] {
  return getDb()
    .select()
    .from(schema.projectMembers)
    .where(eq(schema.projectMembers.projectId, projectId))
    .orderBy(schema.projectMembers.createdAt)
    .all()
    .map(toMemberRow);
}

/**
 * Adds (or updates the name of) a member. Idempotent on (projectId, email):
 * re-inviting an existing member does not duplicate them.
 */
export function addProjectMember(params: {
  projectId: string;
  email: string;
  name: string | null;
  role: "lead" | "member";
}): void {
  const email = params.email.toLowerCase();
  getDb()
    .insert(schema.projectMembers)
    .values({
      id: randomUUID(),
      projectId: params.projectId,
      email,
      name: params.name,
      role: params.role,
      createdAt: new Date().toISOString(),
    })
    .onConflictDoUpdate({
      target: [schema.projectMembers.projectId, schema.projectMembers.email],
      set: { ...(params.name ? { name: params.name } : {}) },
    })
    .run();
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run tests/unit/project-db.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Checkpoint (no commit)**

Run: `npm run typecheck && npm run lint`
Expected: clean. Working tree stays uncommitted.

---

### Task 5: Projects API route (create and list)

**Files:**
- Create: `app/api/projects/route.ts`

**Interfaces:**
- Consumes: `auth`, `upsertUser`, `createProject`, `listOwnedProjects`, `listSharedProjects`, `recordUsageEvent`, `isCoachType`, `findCourse`.
- Produces: `POST /api/projects` (create, returns `{ id }`), `GET /api/projects` (returns `{ owned, shared }`).

- [ ] **Step 1: Write the implementation**

The list screen (Task 6) reads through the data layer directly in its server component, so `GET` here exists for the client create-flow redirect and future use; `POST` is the write path used by the New Project form.

```ts
// app/api/projects/route.ts
import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import {
  createProject,
  listOwnedProjects,
  listSharedProjects,
} from "@/lib/db/projects";
import { recordUsageEvent } from "@/lib/db";
import { findCourse } from "@/lib/project/courses";
import { isCoachType } from "@/lib/project/coaches";

export const runtime = "nodejs";

const MODULE = "project_coach";

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}

const createSchema = z.object({
  courseCode: z.string().min(1).max(20),
  name: z.string().trim().min(1).max(160),
  organization: z.string().trim().max(160).default(""),
  coachTypes: z.array(z.string()).max(5),
});

export async function GET() {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return errorResponse(401, "Sign in to continue.");
  return NextResponse.json({
    owned: listOwnedProjects(email),
    shared: listSharedProjects(email),
  });
}

export async function POST(request: Request) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return errorResponse(401, "Sign in to continue.");

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return errorResponse(400, "Send a valid request.");
  }

  const parsed = createSchema.safeParse(body);
  if (!parsed.success) return errorResponse(400, "Check the project details.");

  if (!findCourse(parsed.data.courseCode)) {
    return errorResponse(400, "Pick a course from the list.");
  }
  const coachTypes = parsed.data.coachTypes.filter(isCoachType);

  const id = createProject({
    ownerEmail: email,
    ownerName: session.user?.name ?? null,
    courseCode: parsed.data.courseCode,
    name: parsed.data.name,
    organization: parsed.data.organization,
    coachTypes,
  });

  recordUsageEvent({
    userEmail: email,
    module: MODULE,
    eventType: "project_created",
  });

  return NextResponse.json({ id }, { status: 201 });
}
```

- [ ] **Step 2: Verify types and lint**

Run: `npm run typecheck && npm run lint`
Expected: clean. (This route is covered end to end by the Playwright spec in Task 9.)

- [ ] **Step 3: Checkpoint (no commit)** — working tree stays uncommitted.

---

### Task 6: My Projects screen

**Files:**
- Create: `app/(app)/project-coach/page.tsx`
- Create: `components/project/ProjectList.tsx`

**Interfaces:**
- Consumes: `auth`, `listOwnedProjects`, `listSharedProjects`, `recordUsageEvent`, `courseLabel`, `findCourse`, `ProjectRow`.

**Routing note:** creating `app/(app)/project-coach/page.tsx` makes this concrete segment take precedence over the existing `app/(app)/[module]` placeholder for the `project-coach` slug. The other placeholder slugs (`general-chat`, `ai-comparisons`) are unaffected. Confirm after building that `/project-coach` renders this page, not the placeholder.

- [ ] **Step 1: Write the presentational list component**

```tsx
// components/project/ProjectList.tsx
import Link from "next/link";
import type { ProjectRow } from "@/lib/db/projects";
import { courseLabel, findCourse } from "@/lib/project/courses";

function courseName(code: string): string {
  const course = findCourse(code);
  return course ? courseLabel(course) : `ISA ${code}`;
}

export function ProjectList({ projects }: { projects: ProjectRow[] }) {
  return (
    <ul className="mt-4 grid gap-4 sm:grid-cols-2">
      {projects.map((p) => (
        <li key={p.id}>
          <Link
            href={`/project-coach/${p.id}`}
            className="block rounded-card border border-medium-tan bg-light-tan p-4 hover:border-miami-red focus-visible:outline focus-visible:outline-2"
          >
            <p className="text-sm text-neutral-700">{courseName(p.courseCode)}</p>
            <p className="mt-1 text-lg font-bold">{p.name}</p>
            {p.organization ? (
              <p className="mt-1 text-sm text-neutral-700">{p.organization}</p>
            ) : null}
          </Link>
        </li>
      ))}
    </ul>
  );
}
```

> Confirm the brand class names against an existing card (`app/(app)/jobapp-assistant/page.tsx` uses `rounded-card border border-medium-tan bg-light-tan`). If `miami-red` is not a configured token, use the same hover treatment the other cards use. Match, do not invent.

- [ ] **Step 2: Write the My Projects page**

```tsx
// app/(app)/project-coach/page.tsx
import type { Metadata } from "next";
import Link from "next/link";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { recordUsageEvent } from "@/lib/db";
import { listOwnedProjects, listSharedProjects } from "@/lib/db/projects";
import { ProjectList } from "@/components/project/ProjectList";

export const metadata: Metadata = { title: "Project Assistant" };

export default async function ProjectAssistantPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");
  const email = session.user.email;

  const owned = listOwnedProjects(email);
  const shared = listSharedProjects(email);

  recordUsageEvent({
    userEmail: email,
    module: "project_coach",
    eventType: "module_open",
  });

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">Project Assistant</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        Set up a team project for your course, invite your teammates, and work
        with AI coaches that help you scope, plan, and reflect. Each coach fills
        a deliverable you can edit together and export to Word.
      </p>

      <div className="mt-6">
        <Link
          href="/project-coach/new"
          className="inline-block rounded-card bg-miami-red px-5 py-2.5 font-bold text-white"
        >
          New project
        </Link>
      </div>

      <section className="mt-10" aria-labelledby="my-projects-heading">
        <h2 id="my-projects-heading" className="text-2xl">
          My projects
        </h2>
        {owned.length === 0 ? (
          <p className="mt-3 text-neutral-700">
            You have not created a project yet. Start one with New project above.
          </p>
        ) : (
          <ProjectList projects={owned} />
        )}
      </section>

      {shared.length > 0 ? (
        <section className="mt-10" aria-labelledby="shared-projects-heading">
          <h2 id="shared-projects-heading" className="text-2xl">
            Shared with me
          </h2>
          <ProjectList projects={shared} />
        </section>
      ) : null}
    </div>
  );
}
```

> Confirm the primary-button class against an existing call to action (look at how `New project`-style buttons are styled elsewhere, e.g. the login or module pages). Reuse the exact token, whether that is `bg-miami-red` or another brand class.

- [ ] **Step 3: Verify it renders**

Run: `npm run dev`, sign in with the test login (`AUTH_TEST_MODE=1`), visit `/project-coach`. Confirm the header, "New project" button, and the empty-state copy render, and that this is the new page (not the `[module]` placeholder). Stop the dev server.

- [ ] **Step 4: Checkpoint (no commit)**

Run: `npm run typecheck && npm run lint` — expected clean. Working tree stays uncommitted.

---

### Task 7: New project screen and form

**Files:**
- Create: `app/(app)/project-coach/new/page.tsx`
- Create: `components/project/NewProjectForm.tsx`

**Interfaces:**
- Consumes: `auth`, `ISA_COURSES`, `courseLabel`, `COACHES`, `POST /api/projects`.

- [ ] **Step 1: Write the client form**

```tsx
// components/project/NewProjectForm.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ISA_COURSES, courseLabel } from "@/lib/project/courses";
import { COACHES, type CoachType } from "@/lib/project/coaches";

export function NewProjectForm() {
  const router = useRouter();
  const [courseCode, setCourseCode] = useState("");
  const [name, setName] = useState("");
  const [organization, setOrganization] = useState("");
  const [coaches, setCoaches] = useState<CoachType[]>(["scoping"]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggleCoach(type: CoachType) {
    setCoaches((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type],
    );
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!courseCode) {
      setError("Pick a course.");
      return;
    }
    if (!name.trim()) {
      setError("Give the project a name.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ courseCode, name, organization, coachTypes: coaches }),
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        setError(data.error ?? "Could not create the project. Try again.");
        setSubmitting(false);
        return;
      }
      const { id } = (await res.json()) as { id: string };
      router.push(`/project-coach/${id}`);
    } catch {
      setError("Could not reach the server. Check your connection and try again.");
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="mt-6 max-w-2xl">
      <div className="mb-5">
        <label htmlFor="course" className="block font-bold">
          Course
        </label>
        <select
          id="course"
          value={courseCode}
          onChange={(e) => setCourseCode(e.target.value)}
          className="mt-1 w-full rounded border border-medium-tan p-2"
          required
        >
          <option value="">Select a course</option>
          {ISA_COURSES.map((c) => (
            <option key={c.code} value={c.code}>
              {courseLabel(c)}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-5">
        <label htmlFor="name" className="block font-bold">
          Project name
        </label>
        <input
          id="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="mt-1 w-full rounded border border-medium-tan p-2"
          maxLength={160}
          required
        />
      </div>

      <div className="mb-5">
        <label htmlFor="organization" className="block font-bold">
          Organization (optional)
        </label>
        <input
          id="organization"
          value={organization}
          onChange={(e) => setOrganization(e.target.value)}
          className="mt-1 w-full rounded border border-medium-tan p-2"
          maxLength={160}
        />
      </div>

      <fieldset className="mb-6">
        <legend className="font-bold">Coaches to include</legend>
        <p className="text-sm text-neutral-700">
          Pick the coaches this project will use. You can change this later.
        </p>
        <div className="mt-2 grid gap-2">
          {COACHES.map((c) => (
            <label key={c.type} className="flex items-start gap-2">
              <input
                type="checkbox"
                checked={coaches.includes(c.type)}
                onChange={() => toggleCoach(c.type)}
                className="mt-1"
              />
              <span>
                <span className="font-bold">{c.label}.</span> {c.blurb}
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      {error ? (
        <p role="alert" className="mb-4 text-miami-red">
          {error}
        </p>
      ) : null}

      <button
        type="submit"
        disabled={submitting}
        className="rounded-card bg-miami-red px-5 py-2.5 font-bold text-white disabled:opacity-60"
      >
        {submitting ? "Creating..." : "Create project"}
      </button>
    </form>
  );
}
```

- [ ] **Step 2: Write the page wrapper**

```tsx
// app/(app)/project-coach/new/page.tsx
import type { Metadata } from "next";
import Link from "next/link";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { NewProjectForm } from "@/components/project/NewProjectForm";

export const metadata: Metadata = { title: "New project" };

export default async function NewProjectPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <Link href="/project-coach" className="text-sm underline">
        Back to my projects
      </Link>
      <h1 className="mt-4 text-3xl">New project</h1>
      <p className="mt-2 max-w-2xl text-neutral-700">
        Choose the course, name the project, and pick the coaches. You will be
        the team lead and can invite teammates from the project page.
      </p>
      <NewProjectForm />
    </div>
  );
}
```

- [ ] **Step 3: Verify types and lint**

Run: `npm run typecheck && npm run lint`
Expected: clean. Replace any brand class that does not resolve (`bg-miami-red`, `text-miami-red`) with the exact tokens the rest of the app uses; verify against a rendered button.

- [ ] **Step 4: Checkpoint (no commit)** — working tree stays uncommitted.

---

### Task 8: Project workspace screen

This is the skeleton workspace: it proves access control end to end and lists the enabled coaches. The coach session links point at `/project-coach/[projectId]/coach/[coachType]`, which Plan 2 (Scoping) builds first; until then those links resolve to a not-yet-built route, which is expected at this stage.

**Files:**
- Create: `app/(app)/project-coach/[projectId]/page.tsx`

**Interfaces:**
- Consumes: `auth`, `getAccessibleProject`, `listProjectMembers`, `courseLabel`, `findCourse`, `COACHES`, `coachLabel`.

- [ ] **Step 1: Write the workspace page**

```tsx
// app/(app)/project-coach/[projectId]/page.tsx
import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { getAccessibleProject, listProjectMembers } from "@/lib/db/projects";
import { courseLabel, findCourse } from "@/lib/project/courses";
import { COACHES } from "@/lib/project/coaches";

export default async function ProjectWorkspacePage({
  params,
}: {
  params: Promise<{ projectId: string }>;
}) {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");
  const { projectId } = await params;

  const project = getAccessibleProject(projectId, session.user.email);
  // Access control: a non-member gets the same not-found a bad id gives.
  if (!project) notFound();

  const members = listProjectMembers(projectId);
  const isLead =
    members.find((m) => m.email === session.user!.email!.toLowerCase())?.role ===
    "lead";
  const course = findCourse(project.courseCode);
  const enabled = COACHES.filter((c) => project.coachTypes.includes(c.type));

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <Link href="/project-coach" className="text-sm underline">
        Back to my projects
      </Link>

      <p className="ribbon mt-4">
        {course ? courseLabel(course) : `ISA ${project.courseCode}`}
      </p>
      <h1 className="mt-3 text-4xl">{project.name}</h1>
      {project.organization ? (
        <p className="mt-2 text-lg text-neutral-700">{project.organization}</p>
      ) : null}

      <section className="mt-8" aria-labelledby="team-heading">
        <h2 id="team-heading" className="text-2xl">
          Team
        </h2>
        <ul className="mt-3 flex flex-wrap gap-2">
          {members.map((m) => (
            <li
              key={m.id}
              className="rounded-card border border-medium-tan bg-light-tan px-3 py-1 text-sm"
            >
              {m.name ?? m.email}
              {m.role === "lead" ? " (lead)" : ""}
            </li>
          ))}
        </ul>
        {/* Invite UI (lead only) arrives with the team-management slice (Plan 4). */}
      </section>

      <section className="mt-10" aria-labelledby="coaches-heading">
        <h2 id="coaches-heading" className="text-2xl">
          Coaches
        </h2>
        {enabled.length === 0 ? (
          <p className="mt-3 text-neutral-700">
            No coaches are enabled for this project yet.
            {isLead ? " As the lead you can add them when coach selection ships." : ""}
          </p>
        ) : (
          <ul className="mt-4 grid gap-4 sm:grid-cols-2">
            {enabled.map((c) => (
              <li key={c.type}>
                <Link
                  href={`/project-coach/${project.id}/coach/${c.type}`}
                  className="block rounded-card border border-medium-tan bg-light-tan p-4 hover:border-miami-red focus-visible:outline focus-visible:outline-2"
                >
                  <p className="text-lg font-bold">{c.label}</p>
                  <p className="mt-1 text-sm text-neutral-700">{c.blurb}</p>
                </Link>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
```

- [ ] **Step 2: Verify types and lint**

Run: `npm run typecheck && npm run lint`
Expected: clean.

- [ ] **Step 3: Manual access check**

With `npm run dev` and the test login: create a project, open it, confirm the workspace renders the course, name, team chip (you, marked lead), and the enabled coaches. Then confirm that visiting a made-up id (`/project-coach/does-not-exist`) returns the not-found page, not a crash. Stop the dev server.

- [ ] **Step 4: Checkpoint (no commit)** — working tree stays uncommitted.

---

### Task 9: End-to-end coverage with axe

**Files:**
- Create: `tests/e2e/project-assistant.spec.ts`

**Interfaces:**
- Consumes: the test login (`AUTH_TEST_MODE=1`) and mock model flag conventions used by the existing e2e specs.

- [ ] **Step 1: Read an existing spec for the login and axe helpers**

Open a passing spec such as `tests/e2e/*jobapp*` or `tests/e2e/*exam*` and copy its exact patterns for: signing in via the test login, the base URL, and running the axe scan (the helper import and call). Reuse those helpers verbatim; do not invent a new login flow.

- [ ] **Step 2: Write the spec**

```ts
// tests/e2e/project-assistant.spec.ts
import { test, expect } from "@playwright/test";
// Reuse the project's existing e2e helpers for login + axe. Import them the
// same way the jobapp/exam specs do (adjust the path to match those specs).
import { signInTestUser, runAxe } from "./helpers"; // <- match the real helper module

test.describe("Project Assistant foundations", () => {
  test("create a project, see it listed, open its workspace", async ({ page }) => {
    await signInTestUser(page);

    await page.goto("/project-coach");
    await expect(
      page.getByRole("heading", { name: "Project Assistant" }),
    ).toBeVisible();
    await runAxe(page);

    await page.getByRole("link", { name: "New project" }).click();
    await expect(
      page.getByRole("heading", { name: "New project" }),
    ).toBeVisible();
    await runAxe(page);

    await page.getByLabel("Course").selectOption("401/501");
    await page.getByLabel("Project name").fill("Playwright project");
    await page.getByLabel("Organization (optional)").fill("Test Org");
    await page.getByRole("button", { name: "Create project" }).click();

    // Lands on the workspace for the new project.
    await expect(
      page.getByRole("heading", { name: "Playwright project" }),
    ).toBeVisible();
    await expect(page.getByText("Test Org")).toBeVisible();
    await expect(page.getByText("(lead)")).toBeVisible();
    await runAxe(page);

    // The project now appears under My projects.
    await page.getByRole("link", { name: "Back to my projects" }).click();
    await expect(
      page.getByRole("link", { name: /Playwright project/ }),
    ).toBeVisible();
  });
});
```

> If the existing helpers are named differently (e.g. a `test.use` fixture or an inline login), match that. The assertions above are stable regardless of helper shape.

- [ ] **Step 3: Run the spec**

Run: `npm run test:e2e -- project-assistant`
Expected: PASS, including the axe scans on the three screens.

- [ ] **Step 4: Full gate**

Run: `npm run typecheck && npm run lint && npm test && npm run test:e2e`
Expected: all green. Record the results honestly (counts, any skips). Working tree stays uncommitted.

- [ ] **Step 5: Record in the migration log**

Append a dated entry to `webapp/docs/development/migration-log.md` (using the existing `### YYYY-MM-DD —` header style) summarizing: the three new tables and migration `0005`, the course and coach constants, the `lib/db/projects.ts` access-checked data layer, and the My Projects / New project / workspace screens. Note that coach sessions, deliverables, and export are Plans 2 to 4.

---

## Self-Review

**1. Spec coverage (against `2026-07-23-project-assistant-design.md`):**
- Section 3 domain model: `Course` constant (Task 1), `Project` with `coachTypes` (Task 3/4), `Project member` with role (Task 3/4), `Deliverable` table created (Task 3; its CRUD is Plan 2 where the coach uses it). Covered.
- Section 6 access control (owner or member on every request): `getAccessibleProject`/`isProjectMember` (Task 4), enforced in the workspace page and API (Tasks 5, 8). Covered.
- Section 7 screens: My Projects (Task 6), New project (Task 7), Project workspace (Task 8). Coach session is Plan 2. Covered for this slice.
- Section 8 data layer (three tables): Task 3. Covered.
- Section 11 accessibility (labelled, keyboard, axe): forms use `<label htmlFor>`, sections use `aria-labelledby`, errors use `role="alert"`, axe scans in Task 9. Covered.
- Section 13 build order step 1: this whole plan. Covered.
- Deferred correctly to later plans: coach split-view + tool-call fill + deliverable editing (Plan 2/3), Word export (Plan 4), team invite UI and coach re-selection (Plan 4), real-time (its own future slice).

**2. Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". The two soft references (brand-token confirmation, e2e helper names) are explicit "match the real thing" instructions with the concrete fallback stated, not logic placeholders. Acceptable because the exact token/helper names are environment facts the executor verifies in seconds, and inventing them would be worse than instructing verification.

**3. Type consistency:** `CoachType` is defined once (Task 2) and imported by `lib/db/projects.ts` (Task 4), the API (Task 5), and the form (Task 7). `ProjectRow`/`MemberRow` are defined in Task 4 and consumed by Tasks 6 and 8. `createProject` params match between the data layer (Task 4), its test (Task 4), and the API caller (Task 5). `coachTypes` is a `CoachType[]` everywhere and serialized as `coachTypesJson`. `findCourse`/`courseLabel` signatures match across Tasks 1, 5, 6, 8. Consistent.

---

## Execution Handoff

This is Plan 1 of a four-plan sequence for the Project Assistant module. The remaining plans (to be written next, each shipping working software on its own):

- **Plan 2 — Scoping Coach end to end:** the split-view coach session (chat left, live deliverable right), the 10-section scoping content schema, the deliverable data layer (get-or-create, save content, save transcript, "last updated by"), the coach system prompt and `setField`/`addRow`/`setRow` tools, direct editing, and its Word export.
- **Plan 3 — The other four coaches:** Premortem, Team Structuring, Devil's Advocate, Reflection, over the same split-view pattern with their smaller schemas.
- **Plan 4 — Team management and per-project export:** invite/remove members (lead only), change enabled coaches (lead only), and per-project Word export of all started deliverables.

**Plan 1 is saved to `webapp/docs/development/2026-07-23-project-assistant-plan-1-foundations.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, two-stage review between tasks, fast iteration.

**2. Inline Execution** — execute the tasks in this session with checkpoints for review.

Which approach would you like? (And should I write Plans 2 to 4 now, or after Plan 1 is executed and reviewed?)
