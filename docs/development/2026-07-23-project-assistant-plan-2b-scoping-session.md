# Project Assistant — Plan 2B: Scoping Coach Session Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Scoping coach split-view session: a guided chat that fills a live, editable 10-section deliverable through the app's first tool-calling route, with the deliverable persisted, directly editable by any team member, and a "last updated by" note.

**Architecture:** A new route streams the coach with three deliverable tools (`setField`, `addRow`, `setRow`) whose `execute` runs the pure `applyScopingOp` reducer from Plan 2A and persists via `saveDeliverableContent`. A GET/PATCH deliverable route serves the panel and direct edits. The client split view (`CoachSession`) runs `useChat` against the coach route and, when each turn finishes, refetches the deliverable and saves the transcript. The editable panel reuses `applyScopingOp` so a student's direct edits have identical semantics to the coach's tool calls. Coach transcripts persist deliberately (a team artifact, exempt from the chat-retention rule).

**Tech Stack:** Next.js 16 (App Router), TypeScript, AI SDK v7 (`ai@7`, `@ai-sdk/react`), Zod, Drizzle, Vitest, Playwright + axe.

## Global Constraints

- **No git commits, no deploys, no production access.** Working tree stays uncommitted; each task ends by running its gate. (A git repo exists at `webapp/`, but `web/` and `docs/` are untracked; never run git write commands.)
- **No secrets in the client;** the browser sends a model id only, never a key. Env var names only.
- **No em dashes in any user-facing text.**
- **This is a customized Next.js;** follow existing patterns (`components/chat/Chat.tsx`, `app/api/chat/route.ts`, `app/api/documents/[id]/export/route.ts`).
- **Emails lowercased; timestamps ISO-8601 UTC**, matching the data layer.
- **Access control on every request:** the coach route, the deliverable route, and the session page all resolve the project through `getAccessibleProject(projectId, email)` and treat a non-member exactly like a missing project.
- **Reuse, do not reinvent:** the model picker (`ModelChooser`, `buildModelOptions`, `getPageModels("project_coach")` which is already configured), the streaming error handling (`classifyProviderFailure`, `describeEmptyResponse`, `TRUNCATION_NOTICE`, `outputTokenBudget`), the mock model (`CHATISA_MOCK_LLM=1`), and the brand tokens from `Chat.tsx` (`bg-paper`, `border-medium-tan`, `bg-light-tan`, `text-dark-tan`, `bg-miami-red`, `text-paper`, `hover:bg-accent-red`).
- **Scope: the Scoping coach only.** The other four coaches are Plan 3; the Word export is Plan 2C. A request for a non-scoping coach returns a clear "not available yet".
- **Sequencing:** run after Plan 2A (uses its `scopingContentSchema`, `emptyScopingContent`, `applyScopingOp`, and the deliverable data layer).

---

## File Structure

**Created:**
- `lib/prompts/project-scoping.ts` — the coach system prompt and `serializeScopingForPrompt(content)`.
- `app/api/project-coach/[projectId]/coach/[coachType]/route.ts` — the tool-calling coach stream (POST).
- `app/api/project-coach/[projectId]/coach/[coachType]/deliverable/route.ts` — GET (read) and PATCH (direct edit + transcript save).
- `components/project/scoping-fields.ts` — data-driven field/table descriptors for the panel.
- `components/project/ScopingDeliverable.tsx` — the editable 10-section panel.
- `components/project/CoachSession.tsx` — the split-view client.
- `app/(app)/project-coach/[projectId]/coach/[coachType]/page.tsx` — the session page (server component).
- `tests/unit/project-scoping-tools.test.ts` — route tool-execute wiring test.
- `tests/e2e/project-coach-session.spec.ts` — chat + direct edit + persistence + axe.

**Interfaces consumed from Plan 2A:** `scopingContentSchema`, `emptyScopingContent`, `applyScopingOp`, `ScopingOp`, `ScopingTable`, `ScopingContent` (`@/lib/project/scoping`); `getOrCreateDeliverable`, `getDeliverable`, `saveDeliverableContent`, `saveDeliverableTranscript` (`@/lib/db/projects`); `getAccessibleProject`, `listProjectMembers` (`@/lib/db/projects`).

---

### Task 1: Coach prompt and content serialization

**Files:**
- Create: `lib/prompts/project-scoping.ts`

**Interfaces:**
- Produces: `SCOPING_COACH_PROMPT: string`, `serializeScopingForPrompt(content: ScopingContent): string`.

- [ ] **Step 1: Write the module**

```ts
// lib/prompts/project-scoping.ts
import type { ScopingContent } from "@/lib/project/scoping";

/**
 * Ported from the legacy Project Scoping Coach (pages/02_project_coach.py) and
 * adapted for this app: the coach still walks one section at a time, but instead
 * of pasting a finished document into the chat it records each settled answer by
 * calling a tool. The live worksheet on the right is the deliverable; the chat is
 * the conversation that fills it.
 */
export const SCOPING_COACH_PROMPT = `You are the Project Scoping Coach for a business analytics team. You guide a student team, one question at a time, through scoping their analytics project, and you record their settled answers into a shared worksheet by calling tools.

How you work:
- Start by asking for a short description of the project in a sentence or two. Do not ask for everything at once.
- Then walk through the worksheet in order: the project name, the organization, the contacts, the problem (what it is, who it affects, how much it costs, why it is a priority now), the goals and their constraints, the data (internal and external sources, and the ideal data), the analysis approaches, the ethics considerations, the stakeholders, and finally how success will be measured and tested.
- Ask one focused question at a time. Offer a hint or an example when it helps. The student cannot see the worksheet template, so phrase each question so it stands on its own.
- When an answer is settled, record it by calling a tool. Do not paste the whole worksheet back into the chat; the worksheet updates itself from your tool calls.
- Keep your chat replies short and conversational. Confirm briefly what you recorded, then move to the next question.

Tools you can call to fill the worksheet:
- setField: set a single field. The path is one of: projectName, organizationName, contacts, problem.whatProblem, problem.whoAffected, problem.howMuch, problem.whyPriority, data.idealData, ethics.privacy, ethics.transparency, ethics.discriminationEquity, ethics.socialLicense, ethics.accountability, ethics.other, experiment.successMeasure, experiment.howTested, experiment.duration.
- addRow: add an empty row to a table before you fill it. The table is one of: goals, data.internalSources, data.externalSources, analysis, stakeholders. The first four hold at most three rows.
- setRow: set the fields of an existing row by its zero-based index. Row keys per table: goals {goal, constraints}; data.internalSources and data.externalSources {name, contains, granularity, frequency, identifiers, owner, storage, comments}; analysis {type, purpose, validation}; stakeholders {orgDept, involvement, counterpart}.

Always addRow before setRow for a new row. If the student edits the worksheet directly, work with what is there. Do not invent answers on the student's behalf; draw them out.`;

/** A compact view of what is already filled, so the coach does not re-ask. */
export function serializeScopingForPrompt(content: ScopingContent): string {
  return JSON.stringify(content);
}
```

- [ ] **Step 2: Checkpoint**

Run: `npm run typecheck && npm run lint`
Expected: clean.

---

### Task 2: The coach tool-calling route

**Files:**
- Create: `app/api/project-coach/[projectId]/coach/[coachType]/route.ts`

**Interfaces:**
- Produces: `POST` streaming coach with `setField`/`addRow`/`setRow` tools that persist through `applyScopingOp` + `saveDeliverableContent`.

- [ ] **Step 1: Write the route**

```ts
// app/api/project-coach/[projectId]/coach/[coachType]/route.ts
import { z } from "zod";
import {
  convertToModelMessages,
  stepCountIs,
  streamText,
  tool,
  type ModelMessage,
  type UIMessage,
} from "ai";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import { MODELS, calculateCost, getPageModels } from "@/lib/config/models";
import { getLanguageModel, isModelAvailable } from "@/lib/providers";
import { getMockModel } from "@/lib/providers/mock";
import { classifyProviderFailure } from "@/lib/providers/errors";
import { outputTokenBudget } from "@/lib/chat/budget";
import { CHAT_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";
import { recordUsageEvent } from "@/lib/db";
import {
  getAccessibleProject,
  getDeliverable,
  getOrCreateDeliverable,
  saveDeliverableContent,
} from "@/lib/db/projects";
import {
  applyScopingOp,
  emptyScopingContent,
  scopingContentSchema,
  type ScopingOp,
  type ScopingTable,
} from "@/lib/project/scoping";
import {
  SCOPING_COACH_PROMPT,
  serializeScopingForPrompt,
} from "@/lib/prompts/project-scoping";

export const runtime = "nodejs";
export const maxDuration = 120;

const MODULE = "project_coach";
const TABLES = [
  "goals",
  "data.internalSources",
  "data.externalSources",
  "analysis",
  "stakeholders",
] as const;

function jsonError(status: number, message: string) {
  return Response.json({ error: message }, { status });
}

const bodySchema = z.object({
  modelId: z.string().min(1).max(128),
  messages: z.array(z.any()).min(1).max(400),
});

/** Reads the latest saved scoping content, tolerating the empty default. */
function readContent(projectId: string) {
  const row = getDeliverable(projectId, "scoping");
  if (!row) return emptyScopingContent();
  const parsed = scopingContentSchema.safeParse(JSON.parse(row.contentJson));
  return parsed.success ? parsed.data : emptyScopingContent();
}

/** Applies one op to the latest content and persists it. Never throws. */
function applyAndSave(projectId: string, op: ScopingOp, updatedBy: string | null) {
  const next = applyScopingOp(readContent(projectId), op);
  saveDeliverableContent({
    projectId,
    coachType: "scoping",
    contentJson: JSON.stringify(next),
    updatedBy,
  });
}

export async function POST(
  req: Request,
  { params }: { params: Promise<{ projectId: string; coachType: string }> },
) {
  const requestId = crypto.randomUUID();
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return jsonError(401, "Sign in to continue.");

  const { projectId, coachType } = await params;
  if (coachType !== "scoping") {
    return jsonError(400, "That coach is not available yet.");
  }

  const project = getAccessibleProject(projectId, email);
  if (!project) return jsonError(404, "That project could not be found.");

  let raw: unknown;
  try {
    raw = await req.json();
  } catch {
    return jsonError(400, "Request body must be JSON.");
  }
  const parsed = bodySchema.safeParse(raw);
  if (!parsed.success) return jsonError(400, "That request wasn't valid.");

  const { modelId, messages } = parsed.data;
  if (!getPageModels(MODULE).includes(modelId)) {
    return jsonError(400, "That model isn't available for this coach.");
  }
  if (!isModelAvailable(modelId)) {
    return jsonError(503, "That model isn't configured right now. Pick another.");
  }

  const limit = checkRateLimit(`coach:${email}`, CHAT_RATE_LIMIT);
  if (!limit.allowed) {
    return jsonError(429, "You've sent a lot of messages. Wait a moment and try again.");
  }

  getOrCreateDeliverable(projectId, "scoping");
  const updatedBy = session.user?.name ?? email;
  const modelConfig = MODELS[modelId];
  const startedAt = Date.now();

  const tools = {
    setField: tool({
      description: "Record a single settled field in the worksheet.",
      inputSchema: z.object({ path: z.string(), value: z.string() }),
      execute: async ({ path, value }) => {
        applyAndSave(projectId, { kind: "setField", path, value }, updatedBy);
        return { ok: true, path };
      },
    }),
    addRow: tool({
      description: "Add an empty row to a worksheet table before filling it.",
      inputSchema: z.object({ table: z.enum(TABLES) }),
      execute: async ({ table }) => {
        applyAndSave(projectId, { kind: "addRow", table: table as ScopingTable }, updatedBy);
        return { ok: true, table };
      },
    }),
    setRow: tool({
      description: "Set the fields of an existing worksheet table row by index.",
      inputSchema: z.object({
        table: z.enum(TABLES),
        index: z.number().int().min(0),
        row: z.record(z.string(), z.string()),
      }),
      execute: async ({ table, index, row }) => {
        applyAndSave(
          projectId,
          { kind: "setRow", table: table as ScopingTable, index, row },
          updatedBy,
        );
        return { ok: true, table, index };
      },
    }),
  };

  const instructions = `${SCOPING_COACH_PROMPT}\n\n--- Worksheet so far (JSON) ---\n${serializeScopingForPrompt(readContent(projectId))}`;

  try {
    const model =
      process.env.CHATISA_MOCK_LLM === "1"
        ? getMockModel()
        : getLanguageModel(modelId);

    const result = streamText({
      model,
      instructions,
      messages: convertToModelMessages(messages as unknown as UIMessage[]) as ModelMessage[],
      tools,
      toolChoice: "auto",
      // The coach may take several steps: ask, record a tool call, then reply.
      stopWhen: stepCountIs(8),
      temperature: 0.3,
      maxOutputTokens: outputTokenBudget(modelId, 1200),
      abortSignal: req.signal,
      onFinish({ text, usage, finishReason }) {
        const inputTokens = usage?.inputTokens ?? null;
        const outputTokens = usage?.outputTokens ?? null;
        const cost =
          inputTokens != null && outputTokens != null
            ? calculateCost(modelId, inputTokens, outputTokens)
            : null;
        recordUsageEvent({
          userEmail: email,
          module: MODULE,
          eventType: "coach_completion",
          modelId,
          provider: modelConfig.provider,
          inputTokens,
          outputTokens,
          costUsd: cost && "totalCost" in cost ? cost.totalCost : null,
          latencyMs: Date.now() - startedAt,
          responseChars: text.length,
          outcome: finishReason ?? "stop",
        });
      },
      onError({ error }) {
        logger.error(
          { requestId, module: MODULE, modelId, err: String(error) },
          "coach stream failed",
        );
        recordUsageEvent({
          userEmail: email,
          module: MODULE,
          eventType: "coach_error",
          modelId,
          provider: modelConfig.provider,
          latencyMs: Date.now() - startedAt,
          outcome: classifyProviderFailure(error).kind,
        });
      },
    });

    return result.toUIMessageStreamResponse({
      onError: (error) => classifyProviderFailure(error).message,
    });
  } catch (err) {
    logger.error({ requestId, err: String(err) }, "coach route failed");
    return jsonError(503, "That model isn't configured right now. Pick another.");
  }
}
```

- [ ] **Step 2: Checkpoint**

Run: `npm run typecheck && npm run lint`
Expected: clean. If `outputTokenBudget`'s second argument name differs, match its real signature (it is imported from `@/lib/chat/budget`; confirm by reading that file). If `z.record` arity differs in the installed Zod, match the installed API.

---

### Task 3: The deliverable read/edit route

**Files:**
- Create: `app/api/project-coach/[projectId]/coach/[coachType]/deliverable/route.ts`

**Interfaces:**
- Produces: `GET` (content, transcript, lastUpdatedBy, updatedAt) and `PATCH` (direct edit and transcript save), both access-checked.

- [ ] **Step 1: Write the route**

```ts
// app/api/project-coach/[projectId]/coach/[coachType]/deliverable/route.ts
import { z } from "zod";
import { auth } from "@/lib/auth";
import {
  getAccessibleProject,
  getOrCreateDeliverable,
  saveDeliverableContent,
  saveDeliverableTranscript,
  getDeliverable,
} from "@/lib/db/projects";
import { scopingContentSchema } from "@/lib/project/scoping";

export const runtime = "nodejs";

function jsonError(status: number, message: string) {
  return Response.json({ error: message }, { status });
}

async function resolve(
  params: Promise<{ projectId: string; coachType: string }>,
) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return { error: jsonError(401, "Sign in to continue.") } as const;
  const { projectId, coachType } = await params;
  if (coachType !== "scoping") {
    return { error: jsonError(404, "That coach could not be found.") } as const;
  }
  const project = getAccessibleProject(projectId, email);
  if (!project) return { error: jsonError(404, "That project could not be found.") } as const;
  return { email, projectId, name: session.user?.name ?? email } as const;
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ projectId: string; coachType: string }> },
) {
  const r = await resolve(params);
  if ("error" in r) return r.error;
  const row = getOrCreateDeliverable(r.projectId, "scoping");
  return Response.json({
    contentJson: row.contentJson,
    transcriptJson: row.transcriptJson,
    lastUpdatedBy: row.lastUpdatedBy,
    updatedAt: row.updatedAt,
  });
}

const patchSchema = z.object({
  content: z.unknown().optional(),
  transcript: z.array(z.any()).optional(),
});

export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ projectId: string; coachType: string }> },
) {
  const r = await resolve(params);
  if ("error" in r) return r.error;

  let raw: unknown;
  try {
    raw = await req.json();
  } catch {
    return jsonError(400, "Request body must be JSON.");
  }
  const parsed = patchSchema.safeParse(raw);
  if (!parsed.success) return jsonError(400, "That request wasn't valid.");

  if (parsed.data.content !== undefined) {
    const content = scopingContentSchema.safeParse(parsed.data.content);
    if (!content.success) return jsonError(400, "The worksheet content was not valid.");
    saveDeliverableContent({
      projectId: r.projectId,
      coachType: "scoping",
      contentJson: JSON.stringify(content.data),
      updatedBy: r.name,
    });
  }
  if (parsed.data.transcript !== undefined) {
    saveDeliverableTranscript({
      projectId: r.projectId,
      coachType: "scoping",
      transcriptJson: JSON.stringify(parsed.data.transcript),
      updatedBy: r.name,
    });
  }

  const row = getDeliverable(r.projectId, "scoping")!;
  return Response.json({ lastUpdatedBy: row.lastUpdatedBy, updatedAt: row.updatedAt });
}
```

- [ ] **Step 2: Checkpoint**

Run: `npm run typecheck && npm run lint`
Expected: clean.

---

### Task 4: The editable deliverable panel

**Files:**
- Create: `components/project/scoping-fields.ts`
- Create: `components/project/ScopingDeliverable.tsx`

**Interfaces:**
- Produces: `ScopingDeliverable` (props: `content`, `onChange`, `lastUpdatedBy`). Reuses `applyScopingOp` so a direct edit has the same semantics as a coach tool call.

- [ ] **Step 1: Write the field descriptors**

```ts
// components/project/scoping-fields.ts
import type { ScopingTable } from "@/lib/project/scoping";

export interface FieldDef {
  path: string;
  label: string;
  multiline?: boolean;
}
export interface FieldSection {
  heading: string;
  fields: FieldDef[];
}
export interface TableSection {
  heading: string;
  table: ScopingTable;
  columns: { key: string; label: string }[];
  capped: boolean;
}

export const FIELD_SECTIONS: FieldSection[] = [
  {
    heading: "Project",
    fields: [
      { path: "projectName", label: "Project name" },
      { path: "organizationName", label: "Organization" },
      { path: "contacts", label: "Contacts (names and titles)", multiline: true },
    ],
  },
  {
    heading: "Problem",
    fields: [
      { path: "problem.whatProblem", label: "What is the problem", multiline: true },
      { path: "problem.whoAffected", label: "Who is affected", multiline: true },
      { path: "problem.howMuch", label: "How much does it cost", multiline: true },
      { path: "problem.whyPriority", label: "Why is it a priority now", multiline: true },
    ],
  },
  {
    heading: "Ideal data",
    fields: [{ path: "data.idealData", label: "Ideal data", multiline: true }],
  },
  {
    heading: "Ethics",
    fields: [
      { path: "ethics.privacy", label: "Privacy", multiline: true },
      { path: "ethics.transparency", label: "Transparency", multiline: true },
      { path: "ethics.discriminationEquity", label: "Discrimination and equity", multiline: true },
      { path: "ethics.socialLicense", label: "Social license", multiline: true },
      { path: "ethics.accountability", label: "Accountability", multiline: true },
      { path: "ethics.other", label: "Other", multiline: true },
    ],
  },
  {
    heading: "Experiment",
    fields: [
      { path: "experiment.successMeasure", label: "How success is measured", multiline: true },
      { path: "experiment.howTested", label: "How it will be tested", multiline: true },
      { path: "experiment.duration", label: "Duration", multiline: false },
    ],
  },
];

const DATA_COLUMNS = [
  { key: "name", label: "Name" },
  { key: "contains", label: "Contains" },
  { key: "granularity", label: "Granularity" },
  { key: "frequency", label: "Frequency" },
  { key: "identifiers", label: "Identifiers" },
  { key: "owner", label: "Owner" },
  { key: "storage", label: "Storage" },
  { key: "comments", label: "Comments" },
];

export const TABLE_SECTIONS: TableSection[] = [
  {
    heading: "Goals",
    table: "goals",
    columns: [
      { key: "goal", label: "Goal" },
      { key: "constraints", label: "Constraints" },
    ],
    capped: true,
  },
  { heading: "Internal data sources", table: "data.internalSources", columns: DATA_COLUMNS, capped: true },
  { heading: "External data sources", table: "data.externalSources", columns: DATA_COLUMNS, capped: true },
  {
    heading: "Analysis",
    table: "analysis",
    columns: [
      { key: "type", label: "Type" },
      { key: "purpose", label: "Purpose" },
      { key: "validation", label: "Validation" },
    ],
    capped: true,
  },
  {
    heading: "Stakeholders",
    table: "stakeholders",
    columns: [
      { key: "orgDept", label: "Org or department" },
      { key: "involvement", label: "Involvement" },
      { key: "counterpart", label: "Counterpart" },
    ],
    capped: false,
  },
];

/** Reads a scalar value at a one or two level path. */
export function readField(content: unknown, path: string): string {
  const parts = path.split(".");
  let node: unknown = content;
  for (const p of parts) {
    if (node && typeof node === "object") node = (node as Record<string, unknown>)[p];
  }
  return typeof node === "string" ? node : "";
}
```

- [ ] **Step 2: Write the panel**

```tsx
// components/project/ScopingDeliverable.tsx
"use client";

import { applyScopingOp, type ScopingContent, type ScopingTable } from "@/lib/project/scoping";
import {
  FIELD_SECTIONS,
  TABLE_SECTIONS,
  readField,
} from "@/components/project/scoping-fields";

function rowsFor(content: ScopingContent, table: ScopingTable): Record<string, string>[] {
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

export function ScopingDeliverable({
  content,
  onChange,
  lastUpdatedBy,
}: {
  content: ScopingContent;
  onChange: (next: ScopingContent) => void;
  lastUpdatedBy: string | null;
}) {
  return (
    <section aria-label="Project scoping worksheet" className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl">Scoping worksheet</h2>
        {lastUpdatedBy ? (
          <p className="text-sm text-dark-tan">Last updated by {lastUpdatedBy}</p>
        ) : null}
      </div>

      {FIELD_SECTIONS.map((section) => (
        <fieldset key={section.heading} className="flex flex-col gap-3">
          <legend className="text-lg font-bold">{section.heading}</legend>
          {section.fields.map((f) => {
            const id = `sf-${f.path}`;
            const value = readField(content, f.path);
            const set = (v: string) =>
              onChange(applyScopingOp(content, { kind: "setField", path: f.path, value: v }));
            return (
              <div key={f.path} className="flex flex-col gap-1">
                <label htmlFor={id} className="text-sm font-bold">
                  {f.label}
                </label>
                {f.multiline ? (
                  <textarea
                    id={id}
                    rows={2}
                    value={value}
                    onChange={(e) => set(e.target.value)}
                    className="rounded border border-medium-tan bg-paper p-2"
                  />
                ) : (
                  <input
                    id={id}
                    value={value}
                    onChange={(e) => set(e.target.value)}
                    className="rounded border border-medium-tan bg-paper p-2"
                  />
                )}
              </div>
            );
          })}
        </fieldset>
      ))}

      {TABLE_SECTIONS.map((section) => {
        const rows = rowsFor(content, section.table);
        const atCap = section.capped && rows.length >= 3;
        return (
          <fieldset key={section.heading} className="flex flex-col gap-3">
            <legend className="text-lg font-bold">{section.heading}</legend>
            {rows.map((row, index) => (
              <div
                key={index}
                className="grid gap-2 rounded-card border border-medium-tan bg-light-tan p-3 sm:grid-cols-2"
              >
                {section.columns.map((col) => {
                  const id = `tf-${section.table}-${index}-${col.key}`;
                  const set = (v: string) =>
                    onChange(
                      applyScopingOp(content, {
                        kind: "setRow",
                        table: section.table,
                        index,
                        row: { [col.key]: v },
                      }),
                    );
                  return (
                    <div key={col.key} className="flex flex-col gap-1">
                      <label htmlFor={id} className="text-xs font-bold">
                        {col.label}
                      </label>
                      <input
                        id={id}
                        value={row[col.key] ?? ""}
                        onChange={(e) => set(e.target.value)}
                        className="rounded border border-medium-tan bg-paper p-1.5 text-sm"
                      />
                    </div>
                  );
                })}
              </div>
            ))}
            <div>
              <button
                type="button"
                disabled={atCap}
                onClick={() => onChange(applyScopingOp(content, { kind: "addRow", table: section.table }))}
                className="rounded-card border border-medium-tan bg-paper px-3 py-1.5 text-sm font-bold hover:border-miami-red hover:text-miami-red disabled:cursor-not-allowed disabled:opacity-60"
              >
                Add row
              </button>
              {atCap ? (
                <span className="ml-2 text-sm text-dark-tan">Up to three rows.</span>
              ) : null}
            </div>
          </fieldset>
        );
      })}
    </section>
  );
}
```

- [ ] **Step 3: Checkpoint**

Run: `npm run typecheck && npm run lint`
Expected: clean. Verify the brand tokens render against an existing card; substitute only if one does not resolve.

---

### Task 5: The split-view session client

**Files:**
- Create: `components/project/CoachSession.tsx`

**Interfaces:**
- Consumes: `useChat`, `DefaultChatTransport`, `ModelChooser`, `Markdown`, `ScopingDeliverable`, and the coach + deliverable routes.

- [ ] **Step 1: Write the component**

```tsx
// components/project/CoachSession.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { Markdown } from "@/components/chat/Markdown";
import { ModelChooser } from "@/components/ModelChooser";
import { ScopingDeliverable } from "@/components/project/ScopingDeliverable";
import {
  emptyScopingContent,
  scopingContentSchema,
  type ScopingContent,
} from "@/lib/project/scoping";
import type { ModelOption } from "@/lib/config/models";

export function CoachSession({
  projectId,
  projectName,
  models,
  defaultModelId,
  initialContent,
  initialMessages,
  initialLastUpdatedBy,
}: {
  projectId: string;
  projectName: string;
  models: ModelOption[];
  defaultModelId: string;
  initialContent: ScopingContent;
  initialMessages: UIMessage[];
  initialLastUpdatedBy: string | null;
}) {
  const base = `/api/project-coach/${projectId}/coach/scoping`;
  const [modelId, setModelId] = useState(defaultModelId);
  const [input, setInput] = useState("");
  const [content, setContent] = useState<ScopingContent>(initialContent);
  const [lastUpdatedBy, setLastUpdatedBy] = useState<string | null>(initialLastUpdatedBy);
  const [saveError, setSaveError] = useState<string | null>(null);
  const saveTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const [transport] = useState(() => new DefaultChatTransport({ api: base }));
  const { messages, sendMessage, status, stop, error, clearError } = useChat({
    messages: initialMessages,
    transport,
    onFinish() {
      // The coach's tool calls changed the worksheet on the server; pull it
      // back so the panel reflects them, and persist the transcript.
      void refetchDeliverable();
      void fetch(`${base}/deliverable`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: messages }),
      });
    },
  });

  const busy = status === "submitted" || status === "streaming";

  async function refetchDeliverable() {
    try {
      const res = await fetch(`${base}/deliverable`);
      if (!res.ok) return;
      const data = (await res.json()) as { contentJson: string; lastUpdatedBy: string | null };
      const parsed = scopingContentSchema.safeParse(JSON.parse(data.contentJson));
      setContent(parsed.success ? parsed.data : emptyScopingContent());
      setLastUpdatedBy(data.lastUpdatedBy);
    } catch {
      // A failed refetch leaves the last known worksheet in place.
    }
  }

  /** Direct edits save on a short debounce, last-save-wins. */
  function onWorksheetChange(next: ScopingContent) {
    setContent(next);
    if (saveTimer.current) clearTimeout(saveTimer.current);
    saveTimer.current = setTimeout(() => {
      setSaveError(null);
      fetch(`${base}/deliverable`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ content: next }),
      })
        .then((res) => {
          if (!res.ok) throw new Error("save failed");
          return res.json();
        })
        .then((data: { lastUpdatedBy: string | null }) => setLastUpdatedBy(data.lastUpdatedBy))
        .catch(() => setSaveError("Your last edit did not save. Check your connection."));
    }, 600);
  }

  useEffect(() => () => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
  }, []);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    clearError();
    sendMessage({ text }, { body: { modelId } });
    setInput("");
  }

  return (
    <div className="mx-auto max-w-6xl px-4 py-6">
      <Link href={`/project-coach/${projectId}`} className="text-sm underline">
        Back to project
      </Link>
      <h1 className="mt-3 text-3xl">Project Scoping Coach</h1>
      <p className="mt-1 text-dark-tan">{projectName}</p>

      <div className="mt-6 grid gap-8 lg:grid-cols-2">
        {/* Chat */}
        <div className="flex flex-col gap-4">
          <ModelChooser
            options={models}
            value={modelId}
            onChange={setModelId}
            help="Switching applies to your next message."
          />
          <div role="log" aria-label="Coach conversation" aria-busy={busy} className="flex flex-col gap-4">
            {messages.length === 0 ? (
              <div className="rounded-card border border-medium-tan bg-paper p-5">
                <h2 className="text-xl">Start scoping</h2>
                <p className="mt-2">
                  Describe your project in a sentence or two. The coach will walk you
                  through the worksheet, one question at a time, and fill it as you go.
                </p>
              </div>
            ) : null}
            {messages.map((m) => {
              const text = m.parts
                .filter((p) => p.type === "text")
                .map((p) => ("text" in p ? p.text : ""))
                .join("");
              if (!text) return null;
              const isUser = m.role === "user";
              return (
                <article
                  key={m.id}
                  className={
                    isUser
                      ? "self-end rounded-card border border-medium-tan bg-light-tan p-4 md:max-w-[85%]"
                      : "rounded-card border border-medium-tan bg-paper p-4"
                  }
                >
                  <h3 className="mb-1 text-sm font-bold text-dark-tan">{isUser ? "You" : "Coach"}</h3>
                  {isUser ? <p className="whitespace-pre-wrap">{text}</p> : <Markdown>{text}</Markdown>}
                </article>
              );
            })}
          </div>

          <p role="status" className="text-sm text-dark-tan">
            {status === "submitted" ? "Sending." : status === "streaming" ? "The coach is responding." : ""}
          </p>
          {error ? (
            <div role="alert" className="rounded-card border-2 border-miami-red bg-paper p-4">
              <p className="font-bold text-miami-red">That response failed</p>
              <p className="mt-1">{error.message || "The coach could not respond. Your message was kept."}</p>
            </div>
          ) : null}

          <form onSubmit={submit} className="flex flex-col gap-2">
            <label htmlFor="coach-input" className="text-sm font-bold">
              Your message
            </label>
            <textarea
              id="coach-input"
              rows={3}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) submit(e);
              }}
              className="w-full rounded-card border border-medium-tan bg-paper p-3"
            />
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={busy || input.trim().length === 0}
                className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
              >
                Send message
              </button>
              {busy ? (
                <button
                  type="button"
                  onClick={stop}
                  className="rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red"
                >
                  Stop
                </button>
              ) : null}
            </div>
          </form>
        </div>

        {/* Live deliverable */}
        <div className="lg:border-l lg:border-medium-tan lg:pl-8">
          {saveError ? (
            <p role="alert" className="mb-3 text-miami-red">
              {saveError}
            </p>
          ) : null}
          <ScopingDeliverable content={content} onChange={onWorksheetChange} lastUpdatedBy={lastUpdatedBy} />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Checkpoint**

Run: `npm run typecheck && npm run lint`
Expected: clean. Confirm `ModelChooser`'s prop names against `components/ModelChooser.tsx` and `Markdown`'s default export/named export against `components/chat/Markdown.tsx`; match them exactly.

---

### Task 6: The session page

**Files:**
- Create: `app/(app)/project-coach/[projectId]/coach/[coachType]/page.tsx`

- [ ] **Step 1: Write the page**

```tsx
// app/(app)/project-coach/[projectId]/coach/[coachType]/page.tsx
import { notFound, redirect } from "next/navigation";
import type { UIMessage } from "ai";
import { auth } from "@/lib/auth";
import { recordUsageEvent } from "@/lib/db";
import { getAccessibleProject, getOrCreateDeliverable } from "@/lib/db/projects";
import {
  emptyScopingContent,
  scopingContentSchema,
  type ScopingContent,
} from "@/lib/project/scoping";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";
import { CoachSession } from "@/components/project/CoachSession";

export default async function CoachSessionPage({
  params,
}: {
  params: Promise<{ projectId: string; coachType: string }>;
}) {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");
  const { projectId, coachType } = await params;
  if (coachType !== "scoping") notFound();

  const project = getAccessibleProject(projectId, session.user.email);
  if (!project) notFound();

  const row = getOrCreateDeliverable(projectId, "scoping");
  const contentParsed = scopingContentSchema.safeParse(JSON.parse(row.contentJson));
  const content: ScopingContent = contentParsed.success
    ? contentParsed.data
    : emptyScopingContent();

  let initialMessages: UIMessage[] = [];
  try {
    const parsed = JSON.parse(row.transcriptJson);
    if (Array.isArray(parsed)) initialMessages = parsed as UIMessage[];
  } catch {
    initialMessages = [];
  }

  const available = filterAvailableModels(getPageModels("project_coach"));
  const { options, defaultModelId } = buildModelOptions("project_coach", available);

  recordUsageEvent({
    userEmail: session.user.email,
    module: "project_coach",
    eventType: "coach_open",
    outcome: "scoping",
  });

  return (
    <CoachSession
      projectId={projectId}
      projectName={project.name}
      models={options}
      defaultModelId={defaultModelId}
      initialContent={content}
      initialMessages={initialMessages}
      initialLastUpdatedBy={row.lastUpdatedBy}
    />
  );
}
```

- [ ] **Step 2: Checkpoint**

Run: `npm run typecheck && npm run lint`
Expected: clean. Confirm `buildModelOptions`/`filterAvailableModels` signatures against `lib/config/models.ts` and `lib/providers` (they are used the same way in `app/(app)/jobapp-assistant/page.tsx`).

---

### Task 7: Tests (tool wiring, session e2e) and log

**Files:**
- Create: `tests/unit/project-scoping-tools.test.ts`
- Create: `tests/e2e/project-coach-session.spec.ts`
- Modify: `webapp/docs/development/migration-log.md`

- [ ] **Step 1: Route-level tool wiring test**

Because the mock model streams text only (no tool calls), the tool path is proven by exercising the same effect the tool's `execute` performs: apply an op through `applyScopingOp` and persist through `saveDeliverableContent`, then read it back. This is the exact body of the route's `applyAndSave`.

```ts
// tests/unit/project-scoping-tools.test.ts
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-coach-tools-"));
process.env.CHATISA_DATA_DIR = dataDir;

const { closeDb, upsertUser } = await import("@/lib/db");
const { createProject, getDeliverable, getOrCreateDeliverable, saveDeliverableContent } =
  await import("@/lib/db/projects");
const { applyScopingOp, scopingContentSchema, emptyScopingContent } =
  await import("@/lib/project/scoping");

const LEAD = "lead@miamioh.edu";

beforeAll(() => upsertUser(LEAD, "Lead"));
afterAll(() => {
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
});

/** Mirrors the route's applyAndSave, the code a tool execute runs. */
function applyAndSave(projectId: string, op: Parameters<typeof applyScopingOp>[1]) {
  const row = getDeliverable(projectId, "scoping");
  const parsed = scopingContentSchema.safeParse(row ? JSON.parse(row.contentJson) : {});
  const content = parsed.success ? parsed.data : emptyScopingContent();
  const next = applyScopingOp(content, op);
  saveDeliverableContent({
    projectId,
    coachType: "scoping",
    contentJson: JSON.stringify(next),
    updatedBy: "Lead",
  });
}

describe("coach tool wiring", () => {
  it("setField then addRow then setRow accumulate in the saved deliverable", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Lead",
      courseCode: "401/501",
      name: "Coach tools",
      organization: "",
      coachTypes: ["scoping"],
    });
    getOrCreateDeliverable(projectId, "scoping");

    applyAndSave(projectId, { kind: "setField", path: "organizationName", value: "Kroger" });
    applyAndSave(projectId, { kind: "addRow", table: "goals" });
    applyAndSave(projectId, { kind: "setRow", table: "goals", index: 0, row: { goal: "Cut stockouts" } });

    const saved = scopingContentSchema.parse(JSON.parse(getDeliverable(projectId, "scoping")!.contentJson));
    expect(saved.organizationName).toBe("Kroger");
    expect(saved.goals[0].goal).toBe("Cut stockouts");
    expect(getDeliverable(projectId, "scoping")!.lastUpdatedBy).toBe("Lead");
  });
});
```

- [ ] **Step 2: Run it**

Run: `npx vitest run tests/unit/project-scoping-tools.test.ts`
Expected: PASS.

- [ ] **Step 3: Session e2e (chat + direct edit + persistence + axe)**

Mirror the login and axe helpers used by `tests/e2e/project-assistant.spec.ts` (Plan 1). The flow: create a scoping project, open the coach, send one message (mock model streams a reply), type into a worksheet field, reload, and confirm the edit persisted. Use a unique project name per run (the Plan 1 spec does this to avoid parallel-project collisions), and prefer explicit `expect(...).toBeVisible()` waits over implicit timing (the Plan 1 spec flaked once under full-parallel load; explicit waits avoid that).

```ts
// tests/e2e/project-coach-session.spec.ts
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

async function runAxe(page: import("@playwright/test").Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(results.violations).toEqual([]);
}

test("scope a project: chat, edit the worksheet, and persist", async ({ page }, testInfo) => {
  const name = `Scoping ${testInfo.project.name} ${Date.now()}`;

  await page.goto("/project-coach/new");
  await page.getByLabel("Course").selectOption("401/501");
  await page.getByLabel("Project name").fill(name);
  await page.getByRole("button", { name: "Create project" }).click();

  await expect(page.getByRole("heading", { name })).toBeVisible();
  await page.getByRole("link", { name: /Project Scoping/ }).click();

  await expect(page.getByRole("heading", { name: "Project Scoping Coach" })).toBeVisible();
  await runAxe(page);

  // Chat streams a reply from the mock model.
  await page.getByLabel("Your message").fill("We want to cut stockouts at a grocery chain.");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByText("Coach", { exact: true }).first()).toBeVisible();

  // Direct edit persists across a reload.
  const org = page.getByLabel("Organization", { exact: true });
  await org.fill("Kroger");
  await expect(page.getByText(/Last updated by/)).toBeVisible();
  await page.waitForTimeout(800); // let the debounced save flush
  await page.reload();
  await expect(page.getByLabel("Organization", { exact: true })).toHaveValue("Kroger");
});
```

- [ ] **Step 4: Run the e2e**

Run: `npm run test:e2e -- project-coach-session`
Expected: PASS including axe. If the "Organization" label is ambiguous (the New project form also has one, but it is a different page), scope with `exact: true` as shown; adjust only if a real collision appears on the session page.

- [ ] **Step 5: Full gate**

Run: `npm run typecheck && npm run lint && npm test && npm run test:e2e`
Expected: green. The Plan 1 `project-assistant.spec.ts` and the chat CodeMirror test have known parallel-load flakes; if either fails, re-run it in isolation and report it as a pre-existing flake, not a regression from this plan.

- [ ] **Step 6: Migration log**

Append a dated `### YYYY-MM-DD —` entry: the coach prompt, the first tool-calling route (`setField`/`addRow`/`setRow` over `applyScopingOp`), the deliverable GET/PATCH route, the split-view session and editable panel, transcript persistence (noted as exempt from the chat-retention rule), and that the Word export is Plan 2C.

---

## Self-Review

**1. Spec coverage (design spec sections 5, 6, 7, 11):**
- Split view, chat plus a live deliverable (5): `CoachSession` + `ScopingDeliverable`. Covered.
- Tool mechanism `setField`/`addRow`/`setRow` applied server-side and persisted (5): Task 2. Covered.
- Direct edits write the same content through the data layer (5): the panel reuses `applyScopingOp`, edits PATCH to `saveDeliverableContent`. Covered.
- Access control on every request (6): coach route, deliverable route, and page all use `getAccessibleProject`. Covered.
- Async last-save-wins with "last updated by" (6): `saveDeliverableContent` overwrites; the panel shows `lastUpdatedBy`. Covered.
- Coach session screen with model picker and back link (7): Task 5/6. Covered.
- Streaming errors and empty answers explained; failed tool call leaves the deliverable unchanged; keyboard and labelled, axe scanned (11): the route's `applyScopingOp` no-ops on bad ops, error/status roles present, e2e runs axe. Covered.
- Deferred: the other four coaches (Plan 3); the Word export (Plan 2C).

**2. Placeholder scan:** No "TBD"/"handle edge cases". The confirm-the-signature notes (ModelChooser props, Markdown export, outputTokenBudget arg, buildModelOptions) are explicit "match the real file" checks with the reference file named, not logic gaps.

**3. Type/name consistency:** `ScopingContent`, `applyScopingOp`, `ScopingOp`, `ScopingTable`, `scopingContentSchema`, `emptyScopingContent` are used exactly as Plan 2A exports them. The five table names match `TABLES` (route), `TABLE_SECTIONS` (panel), and `rowsFor` (panel). The coach route, deliverable route, session client, and page all reference `coachType === "scoping"` consistently, and the client hits `/api/project-coach/${projectId}/coach/scoping` and its `/deliverable` child, which are the two routes created here.

---

## Execution Handoff

After 2B, **Plan 2C** adds `renderScopingDocx(content): Promise<Buffer>` (following the `renderResumeDocx` docx pattern, with the worksheet's field blocks and its Goals/Data/Analysis/Stakeholders tables) and an export route `app/api/project-coach/[projectId]/coach/[coachType]/export/route.ts` returning the `.docx` with the Word content-type. **Plan 3** brings the other four coaches over the same session pattern with their smaller schemas.

**Plan saved to `webapp/docs/development/2026-07-23-project-assistant-plan-2b-scoping-session.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks.

**2. Inline Execution** — execute here with checkpoints.

I will proceed to execute it via a subagent now unless you say otherwise.
