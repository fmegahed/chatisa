# Project Assistant — Plan 3B: Coaches Frontend and Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the four generic coaches reachable: a spec-driven editable panel, a session page and `CoachSession` that dispatch scoping vs generic, and a generic Word export. This completes design build-order step 3 (all five coaches produce an editable, exportable deliverable).

**Architecture:** A `GenericDeliverable` panel renders any `CoachSpec` (fields plus tables) and reuses `applyGenericOp`, exactly as `ScopingDeliverable` reuses `applyScopingOp`. `CoachSession` becomes coach-agnostic through a `kind` discriminator ("scoping" renders `ScopingDeliverable`, "generic" renders `GenericDeliverable` with its spec), keyed on `coachType` for its route URLs. The session page resolves any coach through `getCoachEngine`/`getCoachSpec`. A generic docx renderer plus an export-route dispatch handle the four coaches' downloads.

**Tech Stack:** Next.js 16, TypeScript, `docx`, Vitest, Playwright + axe.

## Global Constraints

- **No git commits, no deploys, no production access.** Working tree stays uncommitted; each task ends by running its gate. (Git repo at `webapp/`; `web/` and `docs/` untracked; never run git write commands.)
- **No secrets in the client;** env var names only.
- **No em dashes in any user-facing text.**
- **This is a customized Next.js;** follow existing patterns (`ScopingDeliverable.tsx`, `CoachSession.tsx`, `scoping-docx.ts`).
- **Scoping must keep working unchanged**, including its e2e (`project-coach-session.spec.ts`). The dispatch keeps scoping on its existing `ScopingDeliverable` and `renderScopingDocx` paths.
- **Access control unchanged:** the session page and export route resolve the project through `getAccessibleProject`.
- **Sequencing:** run after Plan 3A (uses `CoachSpec`, `COACH_SPECS`/`getCoachSpec`, `getCoachEngine`, `buildEmptyContent`, `applyGenericOp`, `GenericContent`). The generic coach and deliverable routes are already engine-driven from 3A.

---

## File Structure

**Created:**
- `components/project/GenericDeliverable.tsx` — spec-driven editable panel.
- `lib/documents/generic-coach-docx.ts` — `renderGenericCoachDocx(spec, content, header)`.
- `tests/unit/generic-coach-docx.test.ts`.

**Modified:**
- `components/project/CoachSession.tsx` — coach-agnostic via a `kind` discriminator and `coachType`/`coachTitle`.
- `app/(app)/project-coach/[projectId]/coach/[coachType]/page.tsx` — resolve any coach, pass the dispatch props.
- `app/api/project-coach/[projectId]/coach/[coachType]/export/route.ts` — dispatch scoping vs generic renderer.
- `tests/e2e/project-coach-session.spec.ts` — add a generic-coach flow and an all-coaches smoke check.

**Interfaces consumed from 3A:** `CoachSpec`, `GenericContent`, `applyGenericOp`, `buildEmptyContent` (`@/lib/project/coach-framework`); `getCoachSpec` (`@/lib/project/coach-specs`); `getCoachEngine` (`@/lib/project/coach-engine`). From earlier: `coachLabel` (`@/lib/project/coaches`), `renderScopingDocx` (`@/lib/documents/scoping-docx`), `courseLabel`/`findCourse` (`@/lib/project/courses`).

---

### Task 1: The generic editable panel

**Files:**
- Create: `components/project/GenericDeliverable.tsx`

Mirrors `ScopingDeliverable`, but the sections come from the spec and edits go through `applyGenericOp`. Fields read from `content.fields[key]`, table rows from `content.tables[table.key]`.

- [ ] **Step 1: Write the panel**

```tsx
// components/project/GenericDeliverable.tsx
"use client";

import {
  applyGenericOp,
  type CoachSpec,
  type GenericContent,
} from "@/lib/project/coach-framework";

export function GenericDeliverable({
  spec,
  content,
  onChange,
  lastUpdatedBy,
}: {
  spec: CoachSpec;
  content: GenericContent;
  onChange: (next: GenericContent) => void;
  lastUpdatedBy: string | null;
}) {
  return (
    <section aria-label={`${spec.title} worksheet`} className="flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl">{spec.title} worksheet</h2>
        {lastUpdatedBy ? (
          <p className="text-sm text-dark-tan">Last updated by {lastUpdatedBy}</p>
        ) : null}
      </div>

      {spec.fields.length > 0 ? (
        <fieldset className="flex flex-col gap-3">
          <legend className="text-lg font-bold">Details</legend>
          {spec.fields.map((f) => {
            const id = `gf-${f.key}`;
            const value = content.fields[f.key] ?? "";
            const set = (v: string) =>
              onChange(applyGenericOp(spec, content, { kind: "setField", path: f.key, value: v }));
            return (
              <div key={f.key} className="flex flex-col gap-1">
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
      ) : null}

      {spec.tables.map((table) => {
        const rows = content.tables[table.key] ?? [];
        return (
          <fieldset key={table.key} className="flex flex-col gap-3">
            <legend className="text-lg font-bold">{table.label}</legend>
            {rows.map((row, index) => (
              <div
                key={index}
                className="grid gap-2 rounded-card border border-medium-tan bg-light-tan p-3 sm:grid-cols-2"
              >
                {table.columns.map((col) => {
                  const id = `gt-${table.key}-${index}-${col.key}`;
                  const set = (v: string) =>
                    onChange(
                      applyGenericOp(spec, content, {
                        kind: "setRow",
                        table: table.key,
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
                onClick={() => onChange(applyGenericOp(spec, content, { kind: "addRow", table: table.key }))}
                className="rounded-card border border-medium-tan bg-paper px-3 py-1.5 text-sm font-bold hover:border-miami-red hover:text-miami-red"
              >
                Add row
              </button>
            </div>
          </fieldset>
        );
      })}
    </section>
  );
}
```

- [ ] **Step 2: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 2: Make `CoachSession` coach-agnostic

**Files:**
- Modify: `components/project/CoachSession.tsx`

- [ ] **Step 1: Generalize the props and rendering**

Change the component so it works for any coach:
- Replace the props type with a discriminated union and add `coachType`, `coachTitle`:

```tsx
import { ScopingDeliverable } from "@/components/project/ScopingDeliverable";
import { GenericDeliverable } from "@/components/project/GenericDeliverable";
import type { ScopingContent } from "@/lib/project/scoping";
import type { CoachSpec, GenericContent } from "@/lib/project/coach-framework";
import type { ModelOption } from "@/lib/config/models";
import type { UIMessage } from "ai";

type CoachSessionProps = {
  projectId: string;
  projectName: string;
  coachType: string;
  coachTitle: string;
  models: ModelOption[];
  defaultModelId: string;
  initialContent: unknown;
  initialMessages: UIMessage[];
  initialLastUpdatedBy: string | null;
} & ({ kind: "scoping" } | { kind: "generic"; spec: CoachSpec });

export function CoachSession(props: CoachSessionProps) {
  const { projectId, projectName, coachType, coachTitle, models, defaultModelId } = props;
```

- `base` becomes `/api/project-coach/${projectId}/coach/${coachType}`.
- `content` state is `unknown`, seeded from `props.initialContent`. `onWorksheetChange(next: unknown)` and `refetchDeliverable` set it from `JSON.parse(data.contentJson)` (the server already validated on save), removing the scoping-schema import and the `scopingContentSchema`/`emptyScopingContent` usage:

```tsx
  const [content, setContent] = useState<unknown>(props.initialContent);
  // ...
  async function refetchDeliverable() {
    try {
      const res = await fetch(`${base}/deliverable`);
      if (!res.ok) return;
      const data = (await res.json()) as { contentJson: string; lastUpdatedBy: string | null };
      setContent(JSON.parse(data.contentJson));
      setLastUpdatedBy(data.lastUpdatedBy);
    } catch {
      // A failed refetch leaves the last known worksheet in place.
    }
  }
  function onWorksheetChange(next: unknown) {
    setContent(next);
    // ...unchanged debounced PATCH of { content: next } ...
  }
```

- The `<h1>` uses `coachTitle` instead of the literal "Project Scoping Coach".
- Where `<ScopingDeliverable ... />` is rendered, dispatch on `kind`:

```tsx
          {props.kind === "scoping" ? (
            <ScopingDeliverable
              content={content as ScopingContent}
              onChange={onWorksheetChange}
              lastUpdatedBy={lastUpdatedBy}
            />
          ) : (
            <GenericDeliverable
              spec={props.spec}
              content={content as GenericContent}
              onChange={onWorksheetChange}
              lastUpdatedBy={lastUpdatedBy}
            />
          )}
```

`onWorksheetChange` has type `(next: unknown) => void`, which is assignable to both panels' `onChange` (a handler accepting `unknown` accepts a narrower type). Keep the existing chat column, model chooser, error/status handling, the debounced-save cleanup effect, and the "Download Word" link (which already uses `${base}/export`, now the correct per-coach route) unchanged.

- [ ] **Step 2: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 3: Generalize the session page

**Files:**
- Modify: `app/(app)/project-coach/[projectId]/coach/[coachType]/page.tsx`

- [ ] **Step 1: Resolve any coach and pass dispatch props**

```tsx
import { notFound, redirect } from "next/navigation";
import type { UIMessage } from "ai";
import { auth } from "@/lib/auth";
import { recordUsageEvent } from "@/lib/db";
import { getAccessibleProject, getOrCreateDeliverable } from "@/lib/db/projects";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";
import { CoachSession } from "@/components/project/CoachSession";
import { getCoachEngine } from "@/lib/project/coach-engine";
import { getCoachSpec } from "@/lib/project/coach-specs";
import { coachLabel, isCoachType } from "@/lib/project/coaches";

export default async function CoachSessionPage({
  params,
}: {
  params: Promise<{ projectId: string; coachType: string }>;
}) {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");
  const { projectId, coachType } = await params;

  const engine = getCoachEngine(coachType);
  if (!engine || !isCoachType(coachType)) notFound();

  const project = getAccessibleProject(projectId, session.user.email);
  if (!project) notFound();

  const row = getOrCreateDeliverable(projectId, coachType);
  const content = engine.parseContent(row.contentJson);

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
    outcome: coachType,
  });

  const common = {
    projectId,
    projectName: project.name,
    coachType,
    coachTitle: `${coachLabel(coachType)} Coach`,
    models: options,
    defaultModelId,
    initialContent: content,
    initialMessages,
    initialLastUpdatedBy: row.lastUpdatedBy,
  };
  const spec = getCoachSpec(coachType);
  return spec ? (
    <CoachSession {...common} kind="generic" spec={spec} />
  ) : (
    <CoachSession {...common} kind="scoping" />
  );
}
```

`coachLabel("scoping")` is "Project Scoping", so the heading stays "Project Scoping Coach" and the scoping e2e assertion is unchanged.

- [ ] **Step 2: Checkpoint** — `npm run typecheck && npm run lint`. Confirm `isCoachType` and `coachLabel` exist in `@/lib/project/coaches` (Plan 1). The workspace page already links every enabled coach to `/project-coach/[projectId]/coach/[coachType]`, so all five are now reachable.

---

### Task 4: Generic Word export and route dispatch

**Files:**
- Create: `lib/documents/generic-coach-docx.ts`
- Test: `tests/unit/generic-coach-docx.test.ts`
- Modify: `app/api/project-coach/[projectId]/coach/[coachType]/export/route.ts`

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/generic-coach-docx.test.ts
import { describe, expect, it } from "vitest";
import { renderGenericCoachDocx } from "@/lib/documents/generic-coach-docx";
import { COACH_SPECS } from "@/lib/project/coach-specs";
import { buildEmptyContent } from "@/lib/project/coach-framework";

const header = {
  projectName: "Retail dashboard",
  courseLabel: "ISA 496: Business Analytics Practicum",
  organization: "Kroger",
  members: ["Lead", "Mate"],
};

describe("renderGenericCoachDocx", () => {
  it("renders a valid .docx for a premortem deliverable", async () => {
    const spec = COACH_SPECS.premortem;
    const content = buildEmptyContent(spec);
    content.fields.projectDescription = "A forecasting tool";
    content.tables.failures = [{ failure: "No data", howToAvoid: "Confirm early" }];
    const buf = await renderGenericCoachDocx(spec, content, header);
    expect(buf.byteLength).toBeGreaterThan(0);
    expect(buf.subarray(0, 2).toString("latin1")).toBe("PK");
  });

  it("renders a valid .docx for a fields-only coach (reflection)", async () => {
    const spec = COACH_SPECS.reflection;
    const buf = await renderGenericCoachDocx(spec, buildEmptyContent(spec), header);
    expect(buf.subarray(0, 2).toString("latin1")).toBe("PK");
  });
});
```

- [ ] **Step 2: Run to verify it fails** — `npx vitest run tests/unit/generic-coach-docx.test.ts` (module missing).

- [ ] **Step 3: Write the renderer**

Mirror `lib/documents/scoping-docx.ts` (same helpers: `labelledValue`, `heading`, `cell`, `tableFor`, the US Letter page constants, `Packer.toBuffer`). Fields render as labelled paragraphs; tables render as Word tables.

```ts
// lib/documents/generic-coach-docx.ts
import "server-only";
import {
  AlignmentType,
  Document,
  HeadingLevel,
  Packer,
  Paragraph,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
} from "docx";
import type { CoachSpec, GenericContent } from "@/lib/project/coach-framework";
import type { ScopingDocHeader } from "@/lib/documents/scoping-docx";

const PAGE = { width: 12240, height: 15840, margin: 720 };
const FONT = "Arial";

function labelledValue(label: string, value: string): Paragraph {
  return new Paragraph({
    spacing: { after: 80 },
    children: [
      new TextRun({ text: `${label}: `, bold: true, font: FONT, size: 22 }),
      new TextRun({ text: value || "Not recorded", font: FONT, size: 22 }),
    ],
  });
}

function heading(text: string): Paragraph {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, font: FONT, bold: true, size: 26 })],
  });
}

function cell(text: string, bold?: boolean): TableCell {
  return new TableCell({
    children: [new Paragraph({ children: [new TextRun({ text, bold, font: FONT, size: 20 })] })],
  });
}

function tableFor(
  columns: { key: string; label: string }[],
  rows: Record<string, string>[],
): (Table | Paragraph)[] {
  if (rows.length === 0) {
    return [
      new Paragraph({
        children: [new TextRun({ text: "None recorded.", italics: true, font: FONT, size: 22 })],
      }),
    ];
  }
  return [
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      rows: [
        new TableRow({ tableHeader: true, children: columns.map((c) => cell(c.label, true)) }),
        ...rows.map((r) => new TableRow({ children: columns.map((c) => cell(r[c.key] ?? "")) })),
      ],
    }),
  ];
}

export async function renderGenericCoachDocx(
  spec: CoachSpec,
  content: GenericContent,
  header: ScopingDocHeader,
): Promise<Buffer> {
  const children: (Paragraph | Table)[] = [];

  children.push(
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [
        new TextRun({ text: `${spec.title}: ${header.projectName || "Project"}`, font: FONT, bold: true, size: 32 }),
      ],
    }),
  );
  children.push(labelledValue("Course", header.courseLabel));
  if (header.organization) children.push(labelledValue("Organization", header.organization));
  if (header.members.length > 0) children.push(labelledValue("Team", header.members.join(", ")));

  if (spec.fields.length > 0) {
    children.push(heading("Details"));
    for (const f of spec.fields) {
      children.push(labelledValue(f.label, content.fields[f.key] ?? ""));
    }
  }
  for (const table of spec.tables) {
    children.push(heading(table.label));
    for (const node of tableFor(table.columns, content.tables[table.key] ?? [])) {
      children.push(node);
    }
  }

  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            size: { width: PAGE.width, height: PAGE.height },
            margin: { top: PAGE.margin, bottom: PAGE.margin, left: PAGE.margin, right: PAGE.margin },
          },
        },
        children,
      },
    ],
  });
  return Buffer.from(await Packer.toBuffer(doc));
}
```

(Confirm `ScopingDocHeader` is exported from `lib/documents/scoping-docx.ts`; it is, per Plan 2C. If it is not, define the same `{ projectName, courseLabel, organization, members }` interface here and export it.)

- [ ] **Step 4: Run to verify it passes** — `npx vitest run tests/unit/generic-coach-docx.test.ts`.

- [ ] **Step 5: Dispatch the export route**

In `app/api/project-coach/[projectId]/coach/[coachType]/export/route.ts`, replace the scoping-only guard and rendering with a dispatch:
- Replace `if (coachType !== "scoping") return jsonError(404, ...)` with an engine lookup: `const engine = getCoachEngine(coachType); if (!engine || !isCoachType(coachType)) return jsonError(404, "That coach could not be found.");`
- Parse content with `engine.parseContent(row.contentJson)`.
- Choose the renderer by spec:

```ts
import { getCoachEngine } from "@/lib/project/coach-engine";
import { getCoachSpec } from "@/lib/project/coach-specs";
import { isCoachType } from "@/lib/project/coaches";
import { renderGenericCoachDocx } from "@/lib/documents/generic-coach-docx";
import type { GenericContent } from "@/lib/project/coach-framework";
import type { ScopingContent } from "@/lib/project/scoping";
// ... inside GET, after resolving project and reading `content` via engine.parseContent(row.contentJson):
    const spec = getCoachSpec(coachType);
    const buffer = spec
      ? await renderGenericCoachDocx(spec, content as GenericContent, headerData)
      : await renderScopingDocx(content as ScopingContent, headerData);
```

where `headerData` is the existing `{ projectName, courseLabel, organization, members }` object. The filename helper can stay; optionally include the coach type in the filename (for example, `${slug}-${coachType}.docx`) so different coaches do not overwrite each other on disk. Keep `getOrCreateDeliverable(projectId, coachType)` (not the hardcoded "scoping"), the access checks, the usage event, and the Word content-type response unchanged.

- [ ] **Step 6: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 5: e2e, all-coaches smoke, gate, and log

**Files:**
- Modify: `tests/e2e/project-coach-session.spec.ts`

- [ ] **Step 1: Add a generic-coach flow and an all-coaches smoke test**

Add tests that (a) create a project with all coaches enabled, (b) open a generic coach (Reflection: fields only, simplest) and exercise chat + a direct field edit + persistence + the Word download, and (c) confirm every enabled coach page opens with its heading.

```ts
test("a generic coach fills, edits, persists, and exports", async ({ page }, testInfo) => {
  const name = `Reflection ${testInfo.project.name} ${Date.now()}`;

  await page.goto("/project-coach/new");
  await page.getByLabel("Course").selectOption("496");
  await page.getByLabel("Project name").fill(name);
  // Enable Reflection (the New project form lists all five coaches as checkboxes).
  await page.getByRole("checkbox", { name: /Reflection/ }).check();
  await page.getByRole("button", { name: "Create project" }).click();

  await expect(page.getByRole("heading", { name })).toBeVisible();
  await page.getByRole("link", { name: /Reflection/ }).click();

  await expect(page.getByRole("heading", { name: "Reflection Coach" })).toBeVisible();

  await page.getByLabel("Your message").fill("We struggled with scheduling but shipped on time.");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByText("Coach", { exact: true }).first()).toBeVisible();

  const challenges = page.getByLabel("Challenges", { exact: true });
  await challenges.fill("Scheduling across time zones");
  await expect(page.getByText(/Last updated by/)).toBeVisible({ timeout: 20000 });
  await page.waitForTimeout(800);
  await page.reload();
  await expect(page.getByLabel("Challenges", { exact: true })).toHaveValue("Scheduling across time zones");

  const downloadPromise = page.waitForEvent("download", { timeout: 20000 });
  await page.getByRole("link", { name: "Download Word" }).click();
  expect((await downloadPromise).suggestedFilename()).toMatch(/\.docx$/);
});

test("every enabled coach opens", async ({ page }, testInfo) => {
  const name = `All coaches ${testInfo.project.name} ${Date.now()}`;
  await page.goto("/project-coach/new");
  await page.getByLabel("Course").selectOption("496");
  await page.getByLabel("Project name").fill(name);
  for (const c of ["Premortem", "Team Structuring", "Devil's Advocate", "Reflection"]) {
    await page.getByRole("checkbox", { name: new RegExp(c) }).check();
  }
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible();

  for (const [label, heading] of [
    ["Project Scoping", "Project Scoping Coach"],
    ["Premortem", "Premortem Coach"],
    ["Team Structuring", "Team Structuring Coach"],
    ["Devil's Advocate", "Devil's Advocate Coach"],
    ["Reflection", "Reflection Coach"],
  ] as const) {
    await page.getByRole("link", { name: new RegExp(label) }).click();
    await expect(page.getByRole("heading", { name: heading })).toBeVisible();
    await page.getByRole("link", { name: "Back to project" }).click();
    await expect(page.getByRole("heading", { name })).toBeVisible();
  }
});
```

(Scoping is enabled by default in the New project form, per Plan 1's `NewProjectForm` initial state `["scoping"]`; the other four are checked above. If the default changed, check Scoping too.)

- [ ] **Step 2: Run the new e2e** — `npm run test:e2e -- project-coach-session`. Expected: all pass, including axe from the earlier tests in the file.

- [ ] **Step 3: Full gate**

Run: `npm run typecheck && npm run lint && npm test && npm run test:e2e`
Expected: green, real counts quoted. The scoping session flow must still pass unchanged.

- [ ] **Step 4: Migration log**

Append a dated entry: the generic panel, the coach-agnostic `CoachSession` and session page, the generic Word export and export dispatch, and the all-coaches e2e. Note this completes design build-order step 3 (all five coaches produce an editable, exportable deliverable). Plan 4 remains: team invite and coach selection (lead only) and per-project export of all deliverables.

---

## Self-Review

**1. Spec coverage (design sections 5, 7, 10, 13 step 3):**
- All five coaches reachable with an editable deliverable: the session page dispatch and `GenericDeliverable` (Tasks 1 to 3). Covered.
- The four generic coaches export to Word: `renderGenericCoachDocx` and the export dispatch (Task 4). Covered.
- Direct edits persist for generic coaches: `GenericDeliverable` uses `applyGenericOp` and `CoachSession` PATCHes content (the deliverable route validates via the engine from 3A). Covered.
- Scoping unchanged: dispatch keeps it on `ScopingDeliverable`/`renderScopingDocx`; its e2e still passes. Covered.
- Deferred to Plan 4: team invite, coach re-selection, per-project export of all deliverables.

**2. Placeholder scan:** none. The `CoachSession`/export edits name exact replacements; the `ScopingDocHeader` reuse states the fallback if the export is missing.

**3. Type/name consistency:** `CoachSpec`/`GenericContent`/`applyGenericOp`/`buildEmptyContent` are used as 3A exports them. `getCoachEngine`/`getCoachSpec` drive the page and export dispatch identically to how 3A drives the API routes. `coachLabel`/`isCoachType` are Plan 1 exports. The heading `${coachLabel(coachType)} Coach` yields "Project Scoping Coach" for scoping (unchanged assertion) and "Premortem Coach" etc. for the rest, matching the e2e. `renderGenericCoachDocx` and `renderScopingDocx` share the `ScopingDocHeader` shape.

---

## Execution Handoff

This completes design build-order step 3. **Plan 4** remains: team management (invite and remove members, and change which coaches a project includes, both lead only) and per-project export of all started deliverables (a zip or a combined document).

**Plan saved to `webapp/docs/development/2026-07-23-project-assistant-plan-3b-coaches-frontend.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks.

**2. Inline Execution** — execute here with checkpoints.

Run it only after Plan 3A has executed and verified (it depends on 3A's framework, specs, and engine). I will execute it after 3A reports green, unless you say otherwise.
