# Project Assistant — Plan 2A: Scoping Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the fully unit-testable backend for the Scoping coach: the typed 10-section scoping content schema, the pure deliverable-mutation reducer the coach's tools will call, and the deliverable data layer (get-or-create, save content, save transcript, list) on the `deliverables` table from Plan 1.

**Architecture:** No UI and no model in this plan. The scoping content shape and its mutation operations (`setField`, `addRow`, `setRow`) are pure functions so they can be tested exhaustively without a language model. The deliverable data layer stores one row per (project, coach) in the existing `deliverables` table. Plan 2B wires these to the split-view coach session and the tool-calling route; Plan 2C adds the Word export.

**Tech Stack:** TypeScript, Zod, Drizzle ORM + better-sqlite3, Vitest.

## Global Constraints

- **No git commits, no deploys, no production access.** Working tree stays uncommitted; each task ends by running its gate.
- **No secrets in the client;** env var names only.
- **No em dashes in any user-facing text.** (The field labels here are data keys, not user copy; the coach prompt and UI copy come in Plan 2B.)
- **This is a customized Next.js;** follow existing patterns.
- **Emails lowercased; timestamps ISO-8601 UTC strings**, matching every existing data-layer function.
- **Access control lives at the project layer** (`getAccessibleProject` from Plan 1). The deliverable functions here operate by `projectId` and assume the caller has already checked project access; Plan 2B enforces that in the route.
- **Sequencing:** run after the chat-retention execution has settled (shared gates), consistent with the note in the retention plan.

---

## File Structure

**Created:**
- `lib/project/scoping.ts` — `ScopingContent` type + Zod schema, `emptyScopingContent()`, the `ScopingOp` union, and `applyScopingOp(content, op)` reducer.
- `tests/unit/project-scoping.test.ts` — schema + reducer tests.

**Modified:**
- `lib/db/projects.ts` — add the deliverable data layer.
- `tests/unit/project-db.test.ts` — add deliverable-layer tests (same file as Plan 1's project-db tests).

## Interfaces produced (relied on by Plans 2B and 2C)

```ts
// lib/project/scoping.ts
export interface ScopingContent { /* the 10 sections, see Task 1 */ }
export const scopingContentSchema: z.ZodType<ScopingContent>;
export function emptyScopingContent(): ScopingContent;

export type ScopingTable =
  | "goals" | "data.internalSources" | "data.externalSources"
  | "analysis" | "stakeholders";
export type ScopingOp =
  | { kind: "setField"; path: string; value: string }
  | { kind: "addRow"; table: ScopingTable }
  | { kind: "setRow"; table: ScopingTable; index: number; row: Record<string, string> };
/** Pure. Returns content unchanged on an unknown path/table or out-of-range index. */
export function applyScopingOp(content: ScopingContent, op: ScopingOp): ScopingContent;

// lib/db/projects.ts
export interface DeliverableRow {
  id: string; projectId: string; coachType: string;
  contentJson: string; transcriptJson: string;
  lastUpdatedBy: string | null; updatedAt: string;
}
export function getDeliverable(projectId: string, coachType: string): DeliverableRow | undefined;
export function getOrCreateDeliverable(projectId: string, coachType: string): DeliverableRow;
export function saveDeliverableContent(params: { projectId: string; coachType: string;
  contentJson: string; updatedBy: string | null }): void;
export function saveDeliverableTranscript(params: { projectId: string; coachType: string;
  transcriptJson: string; updatedBy: string | null }): void;
export function listDeliverables(projectId: string): DeliverableRow[];
```

---

### Task 1: Scoping content schema and reducer

**Files:**
- Create: `lib/project/scoping.ts`
- Test: `tests/unit/project-scoping.test.ts`

**Interfaces:**
- Produces: `ScopingContent`, `scopingContentSchema`, `emptyScopingContent`, `ScopingTable`, `ScopingOp`, `applyScopingOp`.

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/project-scoping.test.ts
import { describe, expect, it } from "vitest";
import {
  scopingContentSchema,
  emptyScopingContent,
  applyScopingOp,
} from "@/lib/project/scoping";

describe("scoping content schema", () => {
  it("an empty deliverable validates", () => {
    expect(scopingContentSchema.safeParse(emptyScopingContent()).success).toBe(true);
  });

  it("caps the bounded tables at three rows", () => {
    const c = emptyScopingContent();
    c.goals = [
      { goal: "a", constraints: "" },
      { goal: "b", constraints: "" },
      { goal: "c", constraints: "" },
      { goal: "d", constraints: "" },
    ];
    expect(scopingContentSchema.safeParse(c).success).toBe(false);
  });
});

describe("applyScopingOp", () => {
  it("sets a top-level field", () => {
    const c = applyScopingOp(emptyScopingContent(), {
      kind: "setField",
      path: "organizationName",
      value: "Kroger",
    });
    expect(c.organizationName).toBe("Kroger");
  });

  it("sets a nested field", () => {
    const c = applyScopingOp(emptyScopingContent(), {
      kind: "setField",
      path: "problem.whatProblem",
      value: "Stockouts",
    });
    expect(c.problem.whatProblem).toBe("Stockouts");
  });

  it("ignores an unknown path and returns content unchanged", () => {
    const before = emptyScopingContent();
    const after = applyScopingOp(before, {
      kind: "setField",
      path: "problem.nonsense",
      value: "x",
    });
    expect(after).toEqual(before);
  });

  it("adds a row to a table and caps at three", () => {
    let c = emptyScopingContent();
    c = applyScopingOp(c, { kind: "addRow", table: "goals" });
    expect(c.goals).toHaveLength(1);
    c = applyScopingOp(c, { kind: "addRow", table: "goals" });
    c = applyScopingOp(c, { kind: "addRow", table: "goals" });
    c = applyScopingOp(c, { kind: "addRow", table: "goals" }); // 4th ignored
    expect(c.goals).toHaveLength(3);
  });

  it("sets a row's known keys and ignores unknown keys", () => {
    let c = emptyScopingContent();
    c = applyScopingOp(c, { kind: "addRow", table: "stakeholders" });
    c = applyScopingOp(c, {
      kind: "setRow",
      table: "stakeholders",
      index: 0,
      row: { orgDept: "Ops", involvement: "Owner", bogus: "drop me" },
    });
    expect(c.stakeholders[0]).toEqual({
      orgDept: "Ops",
      involvement: "Owner",
      counterpart: "",
    });
    expect("bogus" in c.stakeholders[0]).toBe(false);
  });

  it("leaves content unchanged for an out-of-range setRow", () => {
    const before = emptyScopingContent();
    const after = applyScopingOp(before, {
      kind: "setRow",
      table: "goals",
      index: 5,
      row: { goal: "x" },
    });
    expect(after).toEqual(before);
  });

  it("does not mutate the input", () => {
    const before = emptyScopingContent();
    const snapshot = JSON.stringify(before);
    applyScopingOp(before, { kind: "setField", path: "contacts", value: "Jo" });
    expect(JSON.stringify(before)).toBe(snapshot);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/unit/project-scoping.test.ts`
Expected: FAIL, cannot resolve `@/lib/project/scoping`.

- [ ] **Step 3: Write the implementation**

```ts
// lib/project/scoping.ts
import { z } from "zod";

/**
 * The project scoping worksheet as a typed deliverable (ADR-010). Ten sections,
 * ported from the legacy scoping worksheet. Every text field defaults to an
 * empty string and every table to an empty array, so a fresh deliverable is
 * fully shaped and directly editable before the coach has filled anything.
 *
 * Bounded tables (goals, data sources, analysis) cap at three rows to match the
 * worksheet. Stakeholders is unbounded.
 */

const str = () => z.string().default("");

const goalRow = z.object({ goal: str(), constraints: str() });
const dataSource = z.object({
  name: str(),
  contains: str(),
  granularity: str(),
  frequency: str(),
  identifiers: str(),
  owner: str(),
  storage: str(),
  comments: str(),
});
const analysisRow = z.object({ type: str(), purpose: str(), validation: str() });
const stakeholderRow = z.object({
  orgDept: str(),
  involvement: str(),
  counterpart: str(),
});

export const scopingContentSchema = z.object({
  projectName: str(),
  organizationName: str(),
  contacts: str(),
  problem: z.object({
    whatProblem: str(),
    whoAffected: str(),
    howMuch: str(),
    whyPriority: str(),
  }),
  goals: z.array(goalRow).max(3).default([]),
  data: z.object({
    internalSources: z.array(dataSource).max(3).default([]),
    externalSources: z.array(dataSource).max(3).default([]),
    idealData: str(),
  }),
  analysis: z.array(analysisRow).max(3).default([]),
  ethics: z.object({
    privacy: str(),
    transparency: str(),
    discriminationEquity: str(),
    socialLicense: str(),
    accountability: str(),
    other: str(),
  }),
  stakeholders: z.array(stakeholderRow).default([]),
  experiment: z.object({
    successMeasure: str(),
    howTested: str(),
    duration: str(),
  }),
});

export type ScopingContent = z.infer<typeof scopingContentSchema>;

export function emptyScopingContent(): ScopingContent {
  // Parsing an empty object applies every default, so the shape stays in one
  // place (the schema) rather than being duplicated here.
  return scopingContentSchema.parse({
    problem: {},
    data: {},
    ethics: {},
    experiment: {},
  });
}

// ---- deliverable mutation operations (called by the coach's tools in 2B) ----

export type ScopingTable =
  | "goals"
  | "data.internalSources"
  | "data.externalSources"
  | "analysis"
  | "stakeholders";

export type ScopingOp =
  | { kind: "setField"; path: string; value: string }
  | { kind: "addRow"; table: ScopingTable }
  | { kind: "setRow"; table: ScopingTable; index: number; row: Record<string, string> };

/** Every scalar field path a setField op may target. */
const FIELD_PATHS = new Set<string>([
  "projectName",
  "organizationName",
  "contacts",
  "problem.whatProblem",
  "problem.whoAffected",
  "problem.howMuch",
  "problem.whyPriority",
  "data.idealData",
  "ethics.privacy",
  "ethics.transparency",
  "ethics.discriminationEquity",
  "ethics.socialLicense",
  "ethics.accountability",
  "ethics.other",
  "experiment.successMeasure",
  "experiment.howTested",
  "experiment.duration",
]);

/** The empty row and its known keys, per table. */
const ROW_SHAPES: Record<ScopingTable, () => Record<string, string>> = {
  goals: () => ({ goal: "", constraints: "" }),
  "data.internalSources": () => ({
    name: "", contains: "", granularity: "", frequency: "",
    identifiers: "", owner: "", storage: "", comments: "",
  }),
  "data.externalSources": () => ({
    name: "", contains: "", granularity: "", frequency: "",
    identifiers: "", owner: "", storage: "", comments: "",
  }),
  analysis: () => ({ type: "", purpose: "", validation: "" }),
  stakeholders: () => ({ orgDept: "", involvement: "", counterpart: "" }),
};

const TABLE_CAPS: Record<ScopingTable, number> = {
  goals: 3,
  "data.internalSources": 3,
  "data.externalSources": 3,
  analysis: 3,
  stakeholders: Number.POSITIVE_INFINITY,
};

function getTable(content: ScopingContent, table: ScopingTable): Record<string, string>[] {
  switch (table) {
    case "goals":
      return content.goals;
    case "data.internalSources":
      return content.data.internalSources;
    case "data.externalSources":
      return content.data.externalSources;
    case "analysis":
      return content.analysis;
    case "stakeholders":
      return content.stakeholders;
  }
}

/**
 * Applies one operation and returns a NEW content object. On anything malformed
 * (unknown path, unknown table, out-of-range index) it returns the content
 * unchanged, so a bad tool call from the model is a no-op rather than a crash.
 */
export function applyScopingOp(content: ScopingContent, op: ScopingOp): ScopingContent {
  const next: ScopingContent = structuredClone(content);

  if (op.kind === "setField") {
    if (!FIELD_PATHS.has(op.path)) return content;
    const parts = op.path.split(".");
    // Two levels at most, matching FIELD_PATHS.
    if (parts.length === 1) {
      (next as unknown as Record<string, unknown>)[parts[0]] = op.value;
    } else {
      const parent = (next as unknown as Record<string, Record<string, unknown>>)[parts[0]];
      parent[parts[1]] = op.value;
    }
    return next;
  }

  if (op.kind === "addRow") {
    if (!(op.table in ROW_SHAPES)) return content;
    const rows = getTable(next, op.table);
    if (rows.length >= TABLE_CAPS[op.table]) return content;
    rows.push(ROW_SHAPES[op.table]());
    return next;
  }

  // setRow
  if (!(op.table in ROW_SHAPES)) return content;
  const rows = getTable(next, op.table);
  if (op.index < 0 || op.index >= rows.length) return content;
  const known = ROW_SHAPES[op.table]();
  const merged = { ...known, ...rows[op.index] };
  for (const key of Object.keys(known)) {
    if (key in op.row) merged[key] = op.row[key];
  }
  rows[op.index] = merged;
  return next;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run tests/unit/project-scoping.test.ts`
Expected: PASS (all cases).

- [ ] **Step 5: Checkpoint (no commit)**

Run: `npm run typecheck && npm run lint` — expected clean. Working tree stays uncommitted.

---

### Task 2: Deliverable data layer

**Files:**
- Modify: `lib/db/projects.ts`
- Test: `tests/unit/project-db.test.ts` (append to the Plan 1 file)

**Interfaces:**
- Consumes: `getDb`, `schema.deliverables`, `randomUUID`, and the project helpers already in the file.
- Produces: `DeliverableRow`, `getDeliverable`, `getOrCreateDeliverable`, `saveDeliverableContent`, `saveDeliverableTranscript`, `listDeliverables`.

- [ ] **Step 1: Write the failing test (append to `tests/unit/project-db.test.ts`)**

Add these imports to the existing destructured import from `@/lib/db/projects` in that file:
`getDeliverable`, `getOrCreateDeliverable`, `saveDeliverableContent`, `saveDeliverableTranscript`, `listDeliverables`. Then append:

```ts
describe("deliverable data layer", () => {
  it("creates a deliverable once and returns the same row thereafter", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "401/501",
      name: "Deliverable project",
      organization: "",
      coachTypes: ["scoping"],
    });

    const first = getOrCreateDeliverable(projectId, "scoping");
    const second = getOrCreateDeliverable(projectId, "scoping");
    expect(second.id).toBe(first.id);
    expect(first.contentJson).toBe("{}");
    expect(first.transcriptJson).toBe("[]");
  });

  it("saves content and transcript with a last-updated-by name", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "444/544",
      name: "Save project",
      organization: "",
      coachTypes: ["scoping"],
    });
    getOrCreateDeliverable(projectId, "scoping");

    saveDeliverableContent({
      projectId,
      coachType: "scoping",
      contentJson: JSON.stringify({ organizationName: "Kroger" }),
      updatedBy: "Team Lead",
    });
    saveDeliverableTranscript({
      projectId,
      coachType: "scoping",
      transcriptJson: JSON.stringify([{ role: "user", text: "hi" }]),
      updatedBy: "Team Lead",
    });

    const row = getDeliverable(projectId, "scoping");
    expect(row?.contentJson).toContain("Kroger");
    expect(row?.transcriptJson).toContain("hi");
    expect(row?.lastUpdatedBy).toBe("Team Lead");
  });

  it("saving before a get-or-create still creates the row", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "225",
      name: "Lazy project",
      organization: "",
      coachTypes: ["scoping"],
    });
    saveDeliverableContent({
      projectId,
      coachType: "scoping",
      contentJson: JSON.stringify({ contacts: "Jo" }),
      updatedBy: null,
    });
    expect(getDeliverable(projectId, "scoping")?.contentJson).toContain("Jo");
  });

  it("lists deliverables for a project", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Team Lead",
      courseCode: "496",
      name: "List project",
      organization: "",
      coachTypes: ["scoping", "reflection"],
    });
    getOrCreateDeliverable(projectId, "scoping");
    getOrCreateDeliverable(projectId, "reflection");
    const types = listDeliverables(projectId).map((d) => d.coachType).sort();
    expect(types).toEqual(["reflection", "scoping"]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/unit/project-db.test.ts`
Expected: FAIL, the deliverable functions are not exported.

- [ ] **Step 3: Write the implementation (append to `lib/db/projects.ts`)**

```ts
export interface DeliverableRow {
  id: string;
  projectId: string;
  coachType: string;
  contentJson: string;
  transcriptJson: string;
  lastUpdatedBy: string | null;
  updatedAt: string;
}

function toDeliverableRow(
  row: typeof schema.deliverables.$inferSelect,
): DeliverableRow {
  return {
    id: row.id,
    projectId: row.projectId,
    coachType: row.coachType,
    contentJson: row.contentJson,
    transcriptJson: row.transcriptJson,
    lastUpdatedBy: row.lastUpdatedBy,
    updatedAt: row.updatedAt,
  };
}

export function getDeliverable(
  projectId: string,
  coachType: string,
): DeliverableRow | undefined {
  const row = getDb()
    .select()
    .from(schema.deliverables)
    .where(
      and(
        eq(schema.deliverables.projectId, projectId),
        eq(schema.deliverables.coachType, coachType),
      ),
    )
    .get();
  return row ? toDeliverableRow(row) : undefined;
}

/** Returns the deliverable, creating an empty one on first use. */
export function getOrCreateDeliverable(
  projectId: string,
  coachType: string,
): DeliverableRow {
  const existing = getDeliverable(projectId, coachType);
  if (existing) return existing;
  getDb()
    .insert(schema.deliverables)
    .values({
      id: randomUUID(),
      projectId,
      coachType,
      contentJson: "{}",
      transcriptJson: "[]",
      lastUpdatedBy: null,
      updatedAt: new Date().toISOString(),
    })
    // A concurrent first-use race resolves to the row that won.
    .onConflictDoNothing({
      target: [schema.deliverables.projectId, schema.deliverables.coachType],
    })
    .run();
  // Non-null: it exists now, whether we or the racer inserted it.
  return getDeliverable(projectId, coachType)!;
}

export function saveDeliverableContent(params: {
  projectId: string;
  coachType: string;
  contentJson: string;
  updatedBy: string | null;
}): void {
  const now = new Date().toISOString();
  getDb()
    .insert(schema.deliverables)
    .values({
      id: randomUUID(),
      projectId: params.projectId,
      coachType: params.coachType,
      contentJson: params.contentJson,
      transcriptJson: "[]",
      lastUpdatedBy: params.updatedBy,
      updatedAt: now,
    })
    .onConflictDoUpdate({
      target: [schema.deliverables.projectId, schema.deliverables.coachType],
      set: {
        contentJson: params.contentJson,
        lastUpdatedBy: params.updatedBy,
        updatedAt: now,
      },
    })
    .run();
}

export function saveDeliverableTranscript(params: {
  projectId: string;
  coachType: string;
  transcriptJson: string;
  updatedBy: string | null;
}): void {
  const now = new Date().toISOString();
  getDb()
    .insert(schema.deliverables)
    .values({
      id: randomUUID(),
      projectId: params.projectId,
      coachType: params.coachType,
      contentJson: "{}",
      transcriptJson: params.transcriptJson,
      lastUpdatedBy: params.updatedBy,
      updatedAt: now,
    })
    .onConflictDoUpdate({
      target: [schema.deliverables.projectId, schema.deliverables.coachType],
      set: {
        transcriptJson: params.transcriptJson,
        lastUpdatedBy: params.updatedBy,
        updatedAt: now,
      },
    })
    .run();
}

export function listDeliverables(projectId: string): DeliverableRow[] {
  return getDb()
    .select()
    .from(schema.deliverables)
    .where(eq(schema.deliverables.projectId, projectId))
    .all()
    .map(toDeliverableRow);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run tests/unit/project-db.test.ts`
Expected: PASS (Plan 1 cases plus the four new deliverable cases).

- [ ] **Step 5: Full gate (no commit)**

Run: `npm run typecheck && npm run lint && npx vitest run tests/unit/project-scoping.test.ts tests/unit/project-db.test.ts`
Expected: clean and green. Working tree stays uncommitted.

- [ ] **Step 6: Record in the migration log**

Append a dated `### YYYY-MM-DD —` entry to `webapp/docs/development/migration-log.md`: the scoping content schema and reducer, and the deliverable data layer. Note that the coach session UI, tool-calling route, and Word export are Plans 2B and 2C.

---

## Self-Review

**1. Spec coverage (against the design spec, section 4 and 8):**
- The 10-section scoping schema, exactly as specified (problem 4 fields; goals up to 3; internal/external data sources up to 3 with the 8 attributes; analysis up to 3; ethics 6 fields; stakeholders unbounded; experiment 3 fields): Task 1. Covered.
- The `setField`/`addRow`/`setRow` mutation mechanism the coach tools use (section 5): `applyScopingOp`, pure and tested. Covered.
- Deliverable persistence (one row per coach per project, lazy create, content + transcript, last-updated-by): Task 2. Covered.
- Deferred to 2B/2C as intended: the coach prompt and tool definitions, the split-view UI, direct editing wired to `saveDeliverableContent`, and the Word export.

**2. Placeholder scan:** none. Every field, path, and row shape is spelled out.

**3. Type consistency:** `ScopingTable` values match `getTable`, `ROW_SHAPES`, and `TABLE_CAPS` exactly (goals, data.internalSources, data.externalSources, analysis, stakeholders). `FIELD_PATHS` matches the schema's scalar fields. `DeliverableRow` fields match the `deliverables` columns from Plan 1 (`contentJson`, `transcriptJson`, `lastUpdatedBy`, `updatedAt`). The deliverable functions' `onConflictDoUpdate` targets the `deliverables_project_coach` unique index defined in Plan 1.

---

## Execution Handoff

Plan 2A is the backend for the Scoping coach. After it, **Plan 2B** builds the split-view coach session: the coach system prompt (ported from the legacy `project_scoping_prompt` and adapted for one-question-at-a-time plus tool calls), the tool-calling route (`streamText` with `tool({ inputSchema, execute })` mapping `setField`/`addRow`/`setRow` onto `applyScopingOp` then `saveDeliverableContent`, with `stopWhen: stepCountIs(...)`), the live editable deliverable panel wired to `saveDeliverableContent`, transcript persistence, the model picker (a new `project_coach` page-models key), and access enforcement via `getAccessibleProject`. **Plan 2C** adds `renderScopingDocx` and the per-deliverable export route, following the JobApp docx pattern.

**Plan saved to `webapp/docs/development/2026-07-23-project-assistant-plan-2a-scoping-backend.md`. Run it after the chat-retention execution settles. Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks.

**2. Inline Execution** — execute here with checkpoints.

Which approach, and shall I write Plan 2B next while this runs?
