# Project Assistant — Plan 3A: Coaches Backend (generic framework) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the backend for the other four coaches (Premortem, Team Structuring, Devil's Advocate, Reflection) with a small generic "fields plus tables" framework, and unify the coach and deliverable routes behind one coach engine so scoping and the four generic coaches share the same route code.

**Architecture:** The four new coaches all have the shape "some scalar fields and at most one table," unlike scoping's deep nesting. A generic content model (`{ fields, tables }`), a generic op reducer, and a per-coach `CoachSpec` (fields, tables, prompt) express all four. A `CoachEngine` interface unifies scoping (which keeps its bespoke schema and reducer, only wrapped) and the generic coaches, so the coach route and deliverable route become coach-agnostic. No UI in this plan: the session page still renders only scoping until Plan 3B, so the generic routes are exercised by tests, not yet reachable by a student.

**Tech Stack:** TypeScript, Zod, AI SDK v7, Drizzle, Vitest.

## Global Constraints

- **No git commits, no deploys, no production access.** Working tree stays uncommitted; each task ends by running its gate. (Git repo exists at `webapp/`; `web/` and `docs/` are untracked; never run git write commands.)
- **No secrets in the client;** env var names only.
- **No em dashes in any user-facing text**, including coach prompts.
- **This is a customized Next.js;** follow existing patterns.
- **Scoping must keep working unchanged.** Its schema (`scopingContentSchema`), reducer (`applyScopingOp`), and prompt are not modified, only wrapped behind the engine. The Plan 2B route tool-wiring test and session e2e must still pass.
- **Access control unchanged:** the routes still resolve the project through `getAccessibleProject`.
- **Sequencing:** run after the Scoping slice (2A/2B/2C) and the e2e hardening pass.

---

## File Structure

**Created:**
- `lib/project/coach-framework.ts` — generic `GenericContent`, `CoachSpec`, `buildEmptyContent`, `coachContentSchema`, `applyGenericOp`.
- `lib/project/coach-specs.ts` — `COACH_SPECS` for the four coaches (fields, tables, prompts), `getCoachSpec`.
- `lib/project/coach-engine.ts` — `CoachEngine`, `GenericOp`, `getCoachEngine` (scoping wrapper + generic factory).
- `tests/unit/coach-framework.test.ts`, `tests/unit/coach-specs.test.ts`, `tests/unit/coach-engine.test.ts`.

**Modified:**
- `app/api/project-coach/[projectId]/coach/[coachType]/route.ts` — drive tools and prompt from `getCoachEngine`, keyed by `coachType`.
- `app/api/project-coach/[projectId]/coach/[coachType]/deliverable/route.ts` — drive read/validate/save from `getCoachEngine`.

## Interfaces produced (relied on by Plan 3B)

```ts
// lib/project/coach-framework.ts
export interface GenericContent { fields: Record<string, string>; tables: Record<string, Record<string, string>[]> }
export interface CoachFieldDef { key: string; label: string; multiline?: boolean }
export interface CoachTableDef { key: string; label: string; columns: { key: string; label: string }[] }
export interface CoachSpec { type: string; title: string; fields: CoachFieldDef[]; tables: CoachTableDef[]; systemPrompt: string }
export function buildEmptyContent(spec: CoachSpec): GenericContent;
export function coachContentSchema(spec: CoachSpec): z.ZodType<GenericContent>;
export function applyGenericOp(spec: CoachSpec, content: GenericContent, op: GenericOp): GenericContent;

// lib/project/coach-specs.ts
export const COACH_SPECS: Record<string, CoachSpec>;   // premortem, team_structuring, devils_advocate, reflection
export function getCoachSpec(type: string): CoachSpec | undefined;

// lib/project/coach-engine.ts
export type GenericOp =
  | { kind: "setField"; path: string; value: string }
  | { kind: "addRow"; table: string }
  | { kind: "setRow"; table: string; index: number; row: Record<string, string> };
export interface CoachEngine {
  emptyContent(): unknown;
  parseContent(contentJson: string): unknown;
  parseUnknown(value: unknown): unknown | null;
  applyOp(content: unknown, op: GenericOp): unknown;
  serializeForPrompt(content: unknown): string;
  systemPrompt: string;
}
export function getCoachEngine(coachType: string): CoachEngine | null;
```

---

### Task 1: Generic content framework

**Files:**
- Create: `lib/project/coach-framework.ts`
- Test: `tests/unit/coach-framework.test.ts`

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/coach-framework.test.ts
import { describe, expect, it } from "vitest";
import {
  buildEmptyContent,
  coachContentSchema,
  applyGenericOp,
  type CoachSpec,
} from "@/lib/project/coach-framework";

const SPEC: CoachSpec = {
  type: "demo",
  title: "Demo",
  fields: [{ key: "decision", label: "Decision" }],
  tables: [
    { key: "rows", label: "Rows", columns: [{ key: "a", label: "A" }, { key: "b", label: "B" }] },
  ],
  systemPrompt: "x",
};

describe("coach framework", () => {
  it("builds an empty content with every field and table present", () => {
    const c = buildEmptyContent(SPEC);
    expect(c).toEqual({ fields: { decision: "" }, tables: { rows: [] } });
    expect(coachContentSchema(SPEC).safeParse(c).success).toBe(true);
  });

  it("sets a known field and ignores an unknown one", () => {
    let c = buildEmptyContent(SPEC);
    c = applyGenericOp(SPEC, c, { kind: "setField", path: "decision", value: "Go" });
    expect(c.fields.decision).toBe("Go");
    const before = c;
    const after = applyGenericOp(SPEC, c, { kind: "setField", path: "nope", value: "x" });
    expect(after).toEqual(before);
  });

  it("adds and sets table rows, ignoring unknown tables and columns", () => {
    let c = buildEmptyContent(SPEC);
    c = applyGenericOp(SPEC, c, { kind: "addRow", table: "rows" });
    expect(c.tables.rows).toHaveLength(1);
    c = applyGenericOp(SPEC, c, { kind: "setRow", table: "rows", index: 0, row: { a: "1", z: "drop" } });
    expect(c.tables.rows[0]).toEqual({ a: "1", b: "" });
    const before = c;
    expect(applyGenericOp(SPEC, c, { kind: "addRow", table: "ghost" })).toEqual(before);
    expect(applyGenericOp(SPEC, c, { kind: "setRow", table: "rows", index: 9, row: {} })).toEqual(before);
  });

  it("does not mutate its input", () => {
    const c = buildEmptyContent(SPEC);
    const snap = JSON.stringify(c);
    applyGenericOp(SPEC, c, { kind: "setField", path: "decision", value: "x" });
    expect(JSON.stringify(c)).toBe(snap);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run tests/unit/coach-framework.test.ts`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

```ts
// lib/project/coach-framework.ts
import { z } from "zod";
import type { GenericOp } from "@/lib/project/coach-engine";

export interface CoachFieldDef {
  key: string;
  label: string;
  multiline?: boolean;
}
export interface CoachTableDef {
  key: string;
  label: string;
  columns: { key: string; label: string }[];
}
export interface CoachSpec {
  type: string;
  title: string;
  fields: CoachFieldDef[];
  tables: CoachTableDef[];
  systemPrompt: string;
}

export interface GenericContent {
  fields: Record<string, string>;
  tables: Record<string, Record<string, string>[]>;
}

export function buildEmptyContent(spec: CoachSpec): GenericContent {
  return {
    fields: Object.fromEntries(spec.fields.map((f) => [f.key, ""])),
    tables: Object.fromEntries(spec.tables.map((t) => [t.key, []])),
  };
}

export function coachContentSchema(spec: CoachSpec): z.ZodType<GenericContent> {
  const fieldShape = Object.fromEntries(
    spec.fields.map((f) => [f.key, z.string().default("")]),
  );
  const tableShape = Object.fromEntries(
    spec.tables.map((t) => {
      const rowShape = Object.fromEntries(
        t.columns.map((c) => [c.key, z.string().default("")]),
      );
      return [t.key, z.array(z.object(rowShape)).default([])];
    }),
  );
  return z.object({
    fields: z.object(fieldShape),
    tables: z.object(tableShape),
  }) as unknown as z.ZodType<GenericContent>;
}

function emptyRow(table: CoachTableDef): Record<string, string> {
  return Object.fromEntries(table.columns.map((c) => [c.key, ""]));
}

/** Pure. Returns content unchanged on an unknown field, table, or bad index. */
export function applyGenericOp(
  spec: CoachSpec,
  content: GenericContent,
  op: GenericOp,
): GenericContent {
  const next: GenericContent = structuredClone(content);

  if (op.kind === "setField") {
    if (!spec.fields.some((f) => f.key === op.path)) return content;
    next.fields[op.path] = op.value;
    return next;
  }

  const table = spec.tables.find((t) => t.key === op.table);
  if (!table) return content;
  const rows = next.tables[table.key] ?? (next.tables[table.key] = []);

  if (op.kind === "addRow") {
    rows.push(emptyRow(table));
    return next;
  }

  // setRow
  if (op.index < 0 || op.index >= rows.length) return content;
  const merged = { ...emptyRow(table), ...rows[op.index] };
  for (const col of table.columns) {
    if (col.key in op.row) merged[col.key] = op.row[col.key];
  }
  rows[op.index] = merged;
  return next;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run tests/unit/coach-framework.test.ts`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 2: The four coach specs and prompts

**Files:**
- Create: `lib/project/coach-specs.ts`
- Test: `tests/unit/coach-specs.test.ts`

The prompts are adapted from the legacy `devils_advocate_prompt`, `structuring_prompt`, `premortem_prompt`, and `reflective_prompt`, keeping the one-question-at-a-time coaching and adding the instruction to record settled answers via tools. Each prompt gets a tool guide generated from its own spec, so the valid field and table names never drift from the schema.

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/coach-specs.test.ts
import { describe, expect, it } from "vitest";
import { COACH_SPECS, getCoachSpec } from "@/lib/project/coach-specs";
import { buildEmptyContent, coachContentSchema } from "@/lib/project/coach-framework";

describe("coach specs", () => {
  it("defines the four generic coaches (not scoping)", () => {
    expect(Object.keys(COACH_SPECS).sort()).toEqual([
      "devils_advocate",
      "premortem",
      "reflection",
      "team_structuring",
    ]);
    expect(getCoachSpec("scoping")).toBeUndefined();
  });

  it("matches the design schema for each coach", () => {
    expect(COACH_SPECS.premortem.fields.map((f) => f.key)).toEqual(["projectDescription"]);
    expect(COACH_SPECS.premortem.tables[0].columns.map((c) => c.key)).toEqual(["failure", "howToAvoid"]);
    expect(COACH_SPECS.team_structuring.fields).toHaveLength(0);
    expect(COACH_SPECS.team_structuring.tables[0].columns.map((c) => c.key)).toEqual(["name", "skills", "possibleTask"]);
    expect(COACH_SPECS.devils_advocate.fields.map((f) => f.key)).toEqual(["decision", "alternatives", "risks", "mitigations"]);
    expect(COACH_SPECS.reflection.fields.map((f) => f.key)).toEqual(["challenges", "insights", "growth"]);
  });

  it("every spec has a prompt with no em dash and a valid empty content", () => {
    for (const spec of Object.values(COACH_SPECS)) {
      expect(spec.systemPrompt).not.toContain("—");
      expect(spec.systemPrompt.length).toBeGreaterThan(50);
      expect(coachContentSchema(spec).safeParse(buildEmptyContent(spec)).success).toBe(true);
    }
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run tests/unit/coach-specs.test.ts`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

```ts
// lib/project/coach-specs.ts
import type { CoachFieldDef, CoachSpec, CoachTableDef } from "@/lib/project/coach-framework";

interface SpecBase {
  type: string;
  title: string;
  fields: CoachFieldDef[];
  tables: CoachTableDef[];
  basePrompt: string;
}

/** Generates the tool guide from the spec so field/table names never drift. */
function toolGuide(fields: CoachFieldDef[], tables: CoachTableDef[]): string {
  const lines: string[] = ["Record settled answers into the worksheet by calling tools. Do not paste the worksheet into the chat; it updates from your tool calls."];
  if (fields.length > 0) {
    lines.push(`- setField: set one field. Paths: ${fields.map((f) => f.key).join(", ")}.`);
  }
  if (tables.length > 0) {
    const tableNames = tables.map((t) => t.key).join(", ");
    lines.push(`- addRow: add an empty row before filling it. Tables: ${tableNames}.`);
    const rowKeys = tables
      .map((t) => `${t.key} {${t.columns.map((c) => c.key).join(", ")}}`)
      .join("; ");
    lines.push(`- setRow: set an existing row by zero-based index. Row keys per table: ${rowKeys}. Always addRow before setRow.`);
  }
  return lines.join("\n");
}

const BASES: SpecBase[] = [
  {
    type: "premortem",
    title: "Premortem",
    fields: [{ key: "projectDescription", label: "Project description", multiline: true }],
    tables: [
      {
        key: "failures",
        label: "Anticipated failures",
        columns: [
          { key: "failure", label: "Possible failure" },
          { key: "howToAvoid", label: "How to avoid it" },
        ],
      },
    ],
    basePrompt:
      "You are a friendly team coach guiding a student team through a project premortem, one question at a time. A premortem makes it safe to voice concerns during planning: the team imagines the project has already failed and works backward to name the reasons. Introduce yourself and briefly explain why premortems help. Then ask the student to describe their project briefly, and record it with setField (projectDescription). Wait for each answer before moving on. Then ask them to imagine the project has failed and to name every reason they can think of; record each reason as a row (addRow to failures, then setRow with the failure). Do not describe the failure yourself or judge the project. Then, for each failure, ask how they could strengthen the plan to avoid it, and record it in the same row (setRow, howToAvoid). Keep your chat replies short.",
  },
  {
    type: "team_structuring",
    title: "Team Structuring",
    fields: [],
    tables: [
      {
        key: "members",
        label: "Team members",
        columns: [
          { key: "name", label: "Name" },
          { key: "skills", label: "Skills and expertise" },
          { key: "possibleTask", label: "Possible task" },
        ],
      },
    ],
    basePrompt:
      "You are a friendly AI teammate helping a team recognize and use the skills on the team, one question at a time. Introduce yourself and ask the team to tell you about their project. Then explain that effective teams understand and use each member's skills. Ask them to list their team members and each person's skills; record each member as a row (addRow to members, then setRow with name and skills). Then ask how they might organize the tasks given those skills, and record a possible task per member (setRow, possibleTask). Keep talking until they have a sense of who will do what. Keep your chat replies short.",
  },
  {
    type: "devils_advocate",
    title: "Devil's Advocate",
    fields: [
      { key: "decision", label: "The decision", multiline: true },
      { key: "alternatives", label: "Alternative points of view", multiline: true },
      { key: "risks", label: "Risks and drawbacks", multiline: true },
      { key: "mitigations", label: "Mitigations", multiline: true },
    ],
    tables: [],
    basePrompt:
      "You are a friendly AI teammate who helps a team pressure test a decision by playing devil's advocate, one question at a time. Introduce yourself as a teammate who wants to help the team reconsider a decision from another point of view. Ask what recent team decision they have made or are considering, and record it with setField (decision). Explain that groups can fall into a consensus trap, and that questioning a decision does not mean it is wrong. Ask them to name alternative points of view, and record them (setField, alternatives). Ask what the risks or drawbacks are if they proceed, and record them (setField, risks). Then draw out what would reduce those risks and record it (setField, mitigations). You may ask what data supports the decision and what assumptions they are making. Keep your chat replies short.",
  },
  {
    type: "reflection",
    title: "Reflection",
    fields: [
      { key: "challenges", label: "Challenges", multiline: true },
      { key: "insights", label: "Insights", multiline: true },
      { key: "growth", label: "Growth", multiline: true },
    ],
    tables: [],
    basePrompt:
      "You are a helpful coach guiding a student to reflect on a recent team experience, one question at a time. Introduce yourself and explain you are here to help them reflect. Ask them to name one challenge they overcame and one they or their team did not, and record it with setField (challenges). Wait for a response before continuing. Then ask how their understanding of themselves as a team member has changed and what new insights they gained, and record it (setField, insights). Push for specific examples: if they name an insight, ask about their old and new understanding and what led to the change, and record how they have grown (setField, growth). Ask open-ended questions, one at a time. Keep your chat replies short.",
  },
];

export const COACH_SPECS: Record<string, CoachSpec> = Object.fromEntries(
  BASES.map((b) => [
    b.type,
    {
      type: b.type,
      title: b.title,
      fields: b.fields,
      tables: b.tables,
      systemPrompt: `${b.basePrompt}\n\n${toolGuide(b.fields, b.tables)}`,
    },
  ]),
);

export function getCoachSpec(type: string): CoachSpec | undefined {
  return COACH_SPECS[type];
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run tests/unit/coach-specs.test.ts`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 3: The coach engine (unifies scoping and generic)

**Files:**
- Create: `lib/project/coach-engine.ts`
- Test: `tests/unit/coach-engine.test.ts`

**Interfaces:**
- Produces: `GenericOp`, `CoachEngine`, `getCoachEngine`.

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/coach-engine.test.ts
import { describe, expect, it } from "vitest";
import { getCoachEngine } from "@/lib/project/coach-engine";

describe("coach engine", () => {
  it("returns null for an unknown coach", () => {
    expect(getCoachEngine("banana")).toBeNull();
  });

  it("drives a generic coach (devils_advocate) through setField", () => {
    const engine = getCoachEngine("devils_advocate")!;
    const empty = engine.emptyContent();
    const next = engine.applyOp(empty, { kind: "setField", path: "decision", value: "Ship Friday" });
    const json = JSON.stringify(next);
    expect(json).toContain("Ship Friday");
    // Round-trips through parseContent.
    const reread = engine.parseContent(json) as { fields: { decision: string } };
    expect(reread.fields.decision).toBe("Ship Friday");
    expect(engine.parseUnknown({ nonsense: true })).toBeNull();
  });

  it("wraps scoping without changing its behavior", () => {
    const engine = getCoachEngine("scoping")!;
    const next = engine.applyOp(engine.emptyContent(), {
      kind: "setField",
      path: "organizationName",
      value: "Kroger",
    });
    expect(JSON.stringify(next)).toContain("Kroger");
    expect(engine.systemPrompt.length).toBeGreaterThan(50);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run tests/unit/coach-engine.test.ts`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

```ts
// lib/project/coach-engine.ts
import {
  applyGenericOp,
  buildEmptyContent,
  coachContentSchema,
  type CoachSpec,
  type GenericContent,
} from "@/lib/project/coach-framework";
import { getCoachSpec } from "@/lib/project/coach-specs";
import {
  applyScopingOp,
  emptyScopingContent,
  scopingContentSchema,
  type ScopingContent,
  type ScopingOp,
} from "@/lib/project/scoping";
import {
  SCOPING_COACH_PROMPT,
  serializeScopingForPrompt,
} from "@/lib/prompts/project-scoping";

export type GenericOp =
  | { kind: "setField"; path: string; value: string }
  | { kind: "addRow"; table: string }
  | { kind: "setRow"; table: string; index: number; row: Record<string, string> };

export interface CoachEngine {
  emptyContent(): unknown;
  /** Parse a stored contentJson, falling back to an empty deliverable. */
  parseContent(contentJson: string): unknown;
  /** Validate an untrusted value (a direct edit), or null if it is not valid. */
  parseUnknown(value: unknown): unknown | null;
  applyOp(content: unknown, op: GenericOp): unknown;
  serializeForPrompt(content: unknown): string;
  systemPrompt: string;
}

function safeJson(contentJson: string): unknown {
  try {
    return JSON.parse(contentJson || "{}");
  } catch {
    return {};
  }
}

function scopingEngine(): CoachEngine {
  return {
    emptyContent: () => emptyScopingContent(),
    parseContent: (json) => {
      const parsed = scopingContentSchema.safeParse(safeJson(json));
      return parsed.success ? parsed.data : emptyScopingContent();
    },
    parseUnknown: (value) => {
      const parsed = scopingContentSchema.safeParse(value);
      return parsed.success ? parsed.data : null;
    },
    applyOp: (content, op) => applyScopingOp(content as ScopingContent, op as ScopingOp),
    serializeForPrompt: (content) => serializeScopingForPrompt(content as ScopingContent),
    systemPrompt: SCOPING_COACH_PROMPT,
  };
}

function genericEngine(spec: CoachSpec): CoachEngine {
  const schema = coachContentSchema(spec);
  return {
    emptyContent: () => buildEmptyContent(spec),
    parseContent: (json) => {
      const parsed = schema.safeParse(safeJson(json));
      return parsed.success ? parsed.data : buildEmptyContent(spec);
    },
    parseUnknown: (value) => {
      const parsed = schema.safeParse(value);
      return parsed.success ? parsed.data : null;
    },
    applyOp: (content, op) => applyGenericOp(spec, content as GenericContent, op),
    serializeForPrompt: (content) => JSON.stringify(content),
    systemPrompt: spec.systemPrompt,
  };
}

export function getCoachEngine(coachType: string): CoachEngine | null {
  if (coachType === "scoping") return scopingEngine();
  const spec = getCoachSpec(coachType);
  return spec ? genericEngine(spec) : null;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run tests/unit/coach-engine.test.ts`
Expected: PASS.

- [ ] **Step 5: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 4: Drive the coach route from the engine

**Files:**
- Modify: `app/api/project-coach/[projectId]/coach/[coachType]/route.ts`

Replace the scoping-specific pieces with engine-based ones, so all coaches share the route. Scoping behaves identically because its engine wraps the same functions.

- [ ] **Step 1: Rewrite the route body**

Change the imports and the handler so it: resolves `getCoachEngine(coachType)` (null returns 400); computes `readContent`/`applyAndSave` through the engine; and defines tools whose `table` argument is a plain `z.string()` (the engine's `applyOp` validates and no-ops on unknown tables, so no per-coach enum is needed). Specifically:

- Remove the `coachType !== "scoping"` guard and the `TABLES` constant.
- Remove the imports of `applyScopingOp`, `emptyScopingContent`, `scopingContentSchema`, `ScopingOp`, `ScopingTable`, `SCOPING_COACH_PROMPT`, `serializeScopingForPrompt`. Add `import { getCoachEngine } from "@/lib/project/coach-engine";`.
- Replace the module-level `readContent`/`applyAndSave` helpers and their in-handler use with engine-driven versions defined inside `POST` (they need `projectId`, `engine`, and `updatedBy`):

```ts
  const engine = getCoachEngine(coachType);
  if (!engine) return jsonError(400, "That coach is not available.");

  const project = getAccessibleProject(projectId, email);
  if (!project) return jsonError(404, "That project could not be found.");

  // ... body parse, model checks, rate limit, as before ...

  getOrCreateDeliverable(projectId, coachType);
  const updatedBy = session.user?.name ?? email;

  const readContent = () => {
    const row = getDeliverable(projectId, coachType);
    return row ? engine.parseContent(row.contentJson) : engine.emptyContent();
  };
  const applyAndSave = (op: GenericOp) => {
    const next = engine.applyOp(readContent(), op);
    saveDeliverableContent({
      projectId,
      coachType,
      contentJson: JSON.stringify(next),
      updatedBy,
    });
  };

  const tools = {
    setField: tool({
      description: "Record a single settled field in the worksheet.",
      inputSchema: z.object({ path: z.string(), value: z.string() }),
      execute: async ({ path, value }) => {
        applyAndSave({ kind: "setField", path, value });
        return { ok: true, path };
      },
    }),
    addRow: tool({
      description: "Add an empty row to a worksheet table before filling it.",
      inputSchema: z.object({ table: z.string() }),
      execute: async ({ table }) => {
        applyAndSave({ kind: "addRow", table });
        return { ok: true, table };
      },
    }),
    setRow: tool({
      description: "Set the fields of an existing worksheet table row by index.",
      inputSchema: z.object({
        table: z.string(),
        index: z.number().int().min(0),
        row: z.record(z.string(), z.string()),
      }),
      execute: async ({ table, index, row }) => {
        applyAndSave({ kind: "setRow", table, index, row });
        return { ok: true, table, index };
      },
    }),
  };

  const instructions = `${engine.systemPrompt}\n\n--- Worksheet so far (JSON) ---\n${engine.serializeForPrompt(readContent())}`;
```

Import `GenericOp` from `@/lib/project/coach-engine` for the `applyAndSave` parameter type. Everything else in the handler (auth, `bodySchema`, model validation, rate limit, `streamText` with `tools`, `stopWhen: stepCountIs(8)`, `onFinish`/`onError` usage events, `toUIMessageStreamResponse`) stays as it is.

- [ ] **Step 2: Checkpoint**

Run: `npm run typecheck && npm run lint`
Expected: clean. Then confirm scoping still works at the route level:

Run: `npx vitest run tests/unit/project-scoping-tools.test.ts`
Expected: PASS (that test drives the same effect the route's `applyAndSave` runs; unchanged for scoping).

---

### Task 5: Drive the deliverable route from the engine

**Files:**
- Modify: `app/api/project-coach/[projectId]/coach/[coachType]/deliverable/route.ts`

- [ ] **Step 1: Rewrite the coach guard and content handling**

- In `resolve(...)`, replace the `coachType !== "scoping"` check with an engine lookup, and return the engine plus `coachType`:

```ts
import { getCoachEngine } from "@/lib/project/coach-engine";
// ... in resolve():
  const { projectId, coachType } = await params;
  const engine = getCoachEngine(coachType);
  if (!engine) return { error: jsonError(404, "That coach could not be found.") } as const;
  const project = getAccessibleProject(projectId, email);
  if (!project) return { error: jsonError(404, "That project could not be found.") } as const;
  return { email, projectId, coachType, engine, name: session.user?.name ?? email } as const;
```

- In `GET`, use `getOrCreateDeliverable(r.projectId, r.coachType)` instead of the hardcoded `"scoping"`.
- In `PATCH`, validate content with the engine and save under the resolved coach type:

```ts
  if (parsed.data.content !== undefined) {
    const content = r.engine.parseUnknown(parsed.data.content);
    if (content === null) return jsonError(400, "The worksheet content was not valid.");
    saveDeliverableContent({
      projectId: r.projectId,
      coachType: r.coachType,
      contentJson: JSON.stringify(content),
      updatedBy: r.name,
    });
  }
  if (parsed.data.transcript !== undefined) {
    saveDeliverableTranscript({
      projectId: r.projectId,
      coachType: r.coachType,
      transcriptJson: JSON.stringify(parsed.data.transcript),
      updatedBy: r.name,
    });
  }
  const row = getDeliverable(r.projectId, r.coachType)!;
  return Response.json({ lastUpdatedBy: row.lastUpdatedBy, updatedAt: row.updatedAt });
```

Remove the now-unused `scopingContentSchema` import if nothing else uses it.

- [ ] **Step 2: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 6: Generic route-wiring test and full gate

**Files:**
- Create: `tests/unit/coach-generic-tools.test.ts`

- [ ] **Step 1: Route-level wiring test for a generic coach**

Mirrors the route's `applyAndSave` for a generic coach (the mock model streams no tool calls, so the effect is exercised directly, as in the scoping wiring test).

```ts
// tests/unit/coach-generic-tools.test.ts
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-generic-coach-"));
process.env.CHATISA_DATA_DIR = dataDir;

const { closeDb, upsertUser } = await import("@/lib/db");
const { createProject, getDeliverable, getOrCreateDeliverable, saveDeliverableContent } =
  await import("@/lib/db/projects");
const { getCoachEngine } = await import("@/lib/project/coach-engine");

const LEAD = "lead@miamioh.edu";
beforeAll(() => upsertUser(LEAD, "Lead"));
afterAll(() => {
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
});

function applyAndSave(projectId: string, coachType: string, op: Parameters<ReturnType<typeof getCoachEngine>["applyOp"]>[1]) {
  const engine = getCoachEngine(coachType)!;
  const row = getDeliverable(projectId, coachType);
  const content = row ? engine.parseContent(row.contentJson) : engine.emptyContent();
  const next = engine.applyOp(content, op);
  saveDeliverableContent({ projectId, coachType, contentJson: JSON.stringify(next), updatedBy: "Lead" });
}

describe("generic coach route wiring", () => {
  it("premortem: description plus a failure row accumulate", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Lead",
      courseCode: "496",
      name: "Premortem",
      organization: "",
      coachTypes: ["premortem"],
    });
    getOrCreateDeliverable(projectId, "premortem");
    applyAndSave(projectId, "premortem", { kind: "setField", path: "projectDescription", value: "A forecasting tool" });
    applyAndSave(projectId, "premortem", { kind: "addRow", table: "failures" });
    applyAndSave(projectId, "premortem", { kind: "setRow", table: "failures", index: 0, row: { failure: "No data access", howToAvoid: "Confirm access early" } });

    const saved = JSON.parse(getDeliverable(projectId, "premortem")!.contentJson);
    expect(saved.fields.projectDescription).toBe("A forecasting tool");
    expect(saved.tables.failures[0]).toEqual({ failure: "No data access", howToAvoid: "Confirm access early" });
  });
});
```

- [ ] **Step 2: Run it**

Run: `npx vitest run tests/unit/coach-generic-tools.test.ts`
Expected: PASS.

- [ ] **Step 3: Full gate**

Run: `npm run typecheck && npm run lint && npm test && npm run test:e2e`
Expected: green, quoting real counts. Scoping's e2e (`project-coach-session.spec.ts`) and route-wiring test must still pass, proving the engine refactor did not change scoping. Known parallel-load flakes: none should occur now (the e2e hardening pass fixed them); if one does, re-run in isolation and report.

- [ ] **Step 4: Migration log**

Append a dated entry: the generic coach framework, the four specs and prompts, the coach engine unifying scoping and generic, and the two routes now driven by the engine. Note that the session UI still renders only scoping until Plan 3B, so the generic routes are covered by tests, not yet reachable by a student.

---

## Self-Review

**1. Spec coverage (design section 4):**
- Premortem `{ projectDescription, rows: [{ failure, howToAvoid }] }`, Team Structuring `{ rows: [{ name, skills, possibleTask }] }`, Devil's Advocate `{ decision, alternatives, risks, mitigations }`, Reflection `{ challenges, insights, growth }`: all four as `COACH_SPECS` (Task 2), verified field-for-field in the test. Covered.
- The tool mechanism applied server-side and persisted (section 5): the engine drives the shared route (Task 4). Covered.
- Direct edits validated and saved (section 5, 6): the deliverable route uses `engine.parseUnknown` (Task 5). Covered.
- Scoping unchanged: its engine wraps the existing schema/reducer/prompt; the scoping wiring test and e2e still pass (Task 4/6). Covered.
- Deferred to 3B: the generic editable panel, the session/page dispatch that makes the four reachable, and their Word export.

**2. Placeholder scan:** none. The route edits specify exact removals and the exact replacement blocks. The "remove unused import if nothing else uses it" note states the condition.

**3. Type/name consistency:** `GenericOp` is defined in `coach-engine.ts` and imported by `coach-framework.ts` (for `applyGenericOp`) and the route. `CoachSpec`/`GenericContent`/`buildEmptyContent`/`coachContentSchema`/`applyGenericOp` are used consistently across framework, specs, and engine. The four coach-type strings match `CoachType` from `lib/project/coaches.ts` (Plan 1) and the `COACHES` registry order. `getCoachEngine` returns the same `CoachEngine` shape for both scoping and generic, so the route and deliverable route are coach-agnostic.

---

## Execution Handoff

After 3A, **Plan 3B** adds the generic editable panel (spec-driven, reusing `applyGenericOp`), makes the session page and `CoachSession` dispatch scoping vs generic, adds the generic Word export and export-route dispatch, and an e2e that opens a generic coach (chat, edit, persist, export, axe) plus a smoke check that all five coaches open. That makes the four coaches reachable and completes design build-order step 3.

**Plan saved to `webapp/docs/development/2026-07-23-project-assistant-plan-3a-coaches-backend.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks.

**2. Inline Execution** — execute here with checkpoints.

I will proceed to execute it via a subagent now unless you say otherwise.
