# Project Assistant — Plan 4B: Per-Project Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a team download one combined Word document with every started deliverable in the project (each coach's worksheet as its own section), from a "Download all deliverables" button on the workspace.

**Architecture:** Extract the docx layout helpers and per-coach block builders into a shared module (`lib/documents/coach-docx.ts`), so the existing single-deliverable renderers and a new combined renderer all share one layout (and the Miami Red styling). The single renderers (`renderScopingDocx`, `renderGenericCoachDocx`) become thin wrappers over the shared blocks, keeping their public API and tests unchanged. A new access-checked project export route assembles the started deliverables into one document.

**Tech Stack:** TypeScript, `docx`, Next.js route handlers, Vitest, Playwright.

## Global Constraints

- **No git commits, no deploys, no production access.** Working tree stays uncommitted; each task ends by running its gate. (Git repo at `webapp/`; `web/` and `docs/` untracked; never run git write commands.)
- **No secrets in the client;** env var names only.
- **No em dashes in any user-facing text**, including document headings.
- **This is a customized Next.js;** follow existing patterns.
- **Access control:** the project export route resolves the project through `getAccessibleProject` (any member may export); `cache-control: private, no-store`.
- **Preserve the Miami Red title and headings** the professor asked for. The refactor must keep `color: "C41230"` on the title and section headings; a regression test locks this.
- **Sequencing:** run after Plan 4A. It refactors `scoping-docx.ts` and `generic-coach-docx.ts` (which 4A does not touch) and modifies the workspace page (which 4A also modifies, so run 4B after 4A to avoid overlapping edits).

---

## File Structure

**Created:**
- `lib/documents/coach-docx.ts` — shared helpers, block builders, `renderProjectDeliverablesDocx`.
- `app/api/project-coach/[projectId]/export/route.ts` — GET combined `.docx`.
- `tests/unit/project-docx.test.ts` — combined renderer plus a Miami Red regression check.

**Modified:**
- `lib/documents/scoping-docx.ts` — thin wrapper over the shared module.
- `lib/documents/generic-coach-docx.ts` — thin wrapper over the shared module.
- `app/(app)/project-coach/[projectId]/page.tsx` — add the "Download all deliverables" link.
- `tests/e2e/project-coach-session.spec.ts` (or `project-team.spec.ts`) — add a project-export download assertion.

**Interfaces consumed:** `ScopingContent`/`ScopingTable` (`@/lib/project/scoping`), `FIELD_SECTIONS`/`TABLE_SECTIONS`/`readField` (`@/components/project/scoping-fields`), `CoachSpec`/`GenericContent` (`@/lib/project/coach-framework`), `getCoachEngine` (`@/lib/project/coach-engine`), `getCoachSpec` (`@/lib/project/coach-specs`), `getAccessibleProject`/`listDeliverables`/`listProjectMembers` (`@/lib/db/projects`), `coachLabel`/`isCoachType` (`@/lib/project/coaches`), `courseLabel`/`findCourse` (`@/lib/project/courses`).

---

### Task 1: Shared docx module

**Files:**
- Create: `lib/documents/coach-docx.ts`

- [ ] **Step 1: Write the shared module** (helpers extracted verbatim from `scoping-docx.ts`, plus block builders and the combined renderer)

```ts
// lib/documents/coach-docx.ts
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
import type { ScopingContent, ScopingTable } from "@/lib/project/scoping";
import { FIELD_SECTIONS, TABLE_SECTIONS, readField } from "@/components/project/scoping-fields";
import type { CoachSpec, GenericContent } from "@/lib/project/coach-framework";

export interface ScopingDocHeader {
  projectName: string;
  courseLabel: string;
  organization: string;
  members: string[];
}

export const FONT = "Arial";
export const MIAMI_RED = "C41230"; // app --color-miami-red; docx wants hex, no #
const PAGE = { width: 12240, height: 15840, margin: 720 };

export function labelledValue(label: string, value: string): Paragraph {
  return new Paragraph({
    spacing: { after: 80 },
    children: [
      new TextRun({ text: `${label}: `, bold: true, font: FONT, size: 22 }),
      new TextRun({ text: value || "Not recorded", font: FONT, size: 22 }),
    ],
  });
}

export function heading(text: string): Paragraph {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text, font: FONT, bold: true, size: 26, color: MIAMI_RED })],
  });
}

function cell(text: string, bold?: boolean): TableCell {
  return new TableCell({
    children: [new Paragraph({ children: [new TextRun({ text, bold, font: FONT, size: 20 })] })],
  });
}

export function tableFor(
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

/** Centered H1 title (Miami Red) plus the course, organization, and team. */
export function coverBlocks(header: ScopingDocHeader, title: string, size = 36): Paragraph[] {
  const blocks: Paragraph[] = [
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [new TextRun({ text: title, font: FONT, bold: true, size, color: MIAMI_RED })],
    }),
    labelledValue("Course", header.courseLabel),
  ];
  if (header.organization) blocks.push(labelledValue("Organization", header.organization));
  if (header.members.length > 0) blocks.push(labelledValue("Team", header.members.join(", ")));
  return blocks;
}

function scopingRows(content: ScopingContent, table: ScopingTable): Record<string, string>[] {
  switch (table) {
    case "goals": return content.goals;
    case "data.internalSources": return content.data.internalSources;
    case "data.externalSources": return content.data.externalSources;
    case "analysis": return content.analysis;
    case "stakeholders": return content.stakeholders;
  }
}

/** The scoping worksheet body (no cover). */
export function scopingBlocks(content: ScopingContent): (Paragraph | Table)[] {
  const blocks: (Paragraph | Table)[] = [];
  for (const section of FIELD_SECTIONS) {
    blocks.push(heading(section.heading));
    for (const f of section.fields) blocks.push(labelledValue(f.label, readField(content, f.path)));
  }
  for (const section of TABLE_SECTIONS) {
    blocks.push(heading(section.heading));
    for (const node of tableFor(section.columns, scopingRows(content, section.table))) blocks.push(node);
  }
  return blocks;
}

/** A generic coach's worksheet body (no cover). */
export function genericBlocks(spec: CoachSpec, content: GenericContent): (Paragraph | Table)[] {
  const blocks: (Paragraph | Table)[] = [];
  if (spec.fields.length > 0) {
    blocks.push(heading("Details"));
    for (const f of spec.fields) blocks.push(labelledValue(f.label, content.fields[f.key] ?? ""));
  }
  for (const table of spec.tables) {
    blocks.push(heading(table.label));
    for (const node of tableFor(table.columns, content.tables[table.key] ?? [])) blocks.push(node);
  }
  return blocks;
}

export async function docFromChildren(children: (Paragraph | Table)[]): Promise<Buffer> {
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

/** One document, every started deliverable as its own section on a new page. */
export async function renderProjectDeliverablesDocx(
  header: ScopingDocHeader,
  sections: { title: string; blocks: (Paragraph | Table)[] }[],
): Promise<Buffer> {
  const children: (Paragraph | Table)[] = coverBlocks(
    header,
    `${header.projectName || "Project"}: all deliverables`,
  );
  sections.forEach((s, i) => {
    children.push(
      new Paragraph({
        heading: HeadingLevel.HEADING_1,
        pageBreakBefore: i > 0,
        spacing: { before: 240, after: 120 },
        children: [new TextRun({ text: s.title, font: FONT, bold: true, size: 32, color: MIAMI_RED })],
      }),
    );
    for (const b of s.blocks) children.push(b);
  });
  if (sections.length === 0) {
    children.push(
      new Paragraph({
        children: [new TextRun({ text: "No deliverables have been started yet.", font: FONT, size: 22 })],
      }),
    );
  }
  return docFromChildren(children);
}
```

- [ ] **Step 2: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 2: Reduce the single renderers to wrappers

**Files:**
- Modify: `lib/documents/scoping-docx.ts`
- Modify: `lib/documents/generic-coach-docx.ts`

- [ ] **Step 1: `scoping-docx.ts`**

Replace the whole file with a thin wrapper. It keeps the same exports (`renderScopingDocx`, `ScopingDocHeader`) so the coach export route (Plan 2C) and the docx tests are unchanged.

```ts
// lib/documents/scoping-docx.ts
import "server-only";
import type { ScopingContent } from "@/lib/project/scoping";
import {
  coverBlocks,
  docFromChildren,
  scopingBlocks,
  type ScopingDocHeader,
} from "@/lib/documents/coach-docx";

export type { ScopingDocHeader };

export async function renderScopingDocx(
  content: ScopingContent,
  header: ScopingDocHeader,
): Promise<Buffer> {
  return docFromChildren([
    ...coverBlocks(header, header.projectName || "Project scope"),
    ...scopingBlocks(content),
  ]);
}
```

- [ ] **Step 2: `generic-coach-docx.ts`**

```ts
// lib/documents/generic-coach-docx.ts
import "server-only";
import type { CoachSpec, GenericContent } from "@/lib/project/coach-framework";
import {
  coverBlocks,
  docFromChildren,
  genericBlocks,
  type ScopingDocHeader,
} from "@/lib/documents/coach-docx";

export async function renderGenericCoachDocx(
  spec: CoachSpec,
  content: GenericContent,
  header: ScopingDocHeader,
): Promise<Buffer> {
  return docFromChildren([
    ...coverBlocks(header, `${spec.title}: ${header.projectName || "Project"}`, 32),
    ...genericBlocks(spec, content),
  ]);
}
```

(If anything imported `ScopingDocHeader` from `generic-coach-docx.ts`, it still resolves through `scoping-docx.ts` or directly from `coach-docx.ts`; the export route imports it from `scoping-docx.ts` per Plan 3B, which still re-exports it.)

- [ ] **Step 3: Checkpoint and existing docx tests**

Run: `npm run typecheck && npm run lint`
Run: `npx vitest run tests/unit/scoping-docx.test.ts tests/unit/generic-coach-docx.test.ts`
Expected: clean and 4/4. The single renderers produce the same documents through the shared blocks.

---

### Task 3: Combined renderer test with Miami Red regression

**Files:**
- Create: `tests/unit/project-docx.test.ts`

- [ ] **Step 1: Write the test**

Uses JSZip (a `docx` dependency) to read `word/document.xml` and assert the brand color survives, and that a combined document renders.

```ts
// tests/unit/project-docx.test.ts
import { describe, expect, it } from "vitest";
import JSZip from "jszip";
import { renderProjectDeliverablesDocx, scopingBlocks, genericBlocks } from "@/lib/documents/coach-docx";
import { renderScopingDocx } from "@/lib/documents/scoping-docx";
import { emptyScopingContent } from "@/lib/project/scoping";
import { COACH_SPECS } from "@/lib/project/coach-specs";
import { buildEmptyContent } from "@/lib/project/coach-framework";

const header = {
  projectName: "Retail dashboard",
  courseLabel: "ISA 496: Business Analytics Practicum",
  organization: "Kroger",
  members: ["Lead", "Mate"],
};

async function documentXml(buf: Buffer): Promise<string> {
  const zip = await JSZip.loadAsync(buf);
  return zip.file("word/document.xml")!.async("string");
}

describe("project export", () => {
  it("combines multiple deliverables into one document", async () => {
    const scoping = emptyScopingContent();
    scoping.projectName = "Retail dashboard";
    const premortem = COACH_SPECS.premortem;
    const pmContent = buildEmptyContent(premortem);
    pmContent.fields.projectDescription = "A forecasting tool";

    const buf = await renderProjectDeliverablesDocx(header, [
      { title: "Project Scoping", blocks: scopingBlocks(scoping) },
      { title: "Premortem", blocks: genericBlocks(premortem, pmContent) },
    ]);
    const xml = await documentXml(buf);
    expect(buf.subarray(0, 2).toString("latin1")).toBe("PK");
    expect(xml).toContain("A forecasting tool");
    // Both section titles present.
    expect(xml).toContain("Project Scoping");
    expect(xml).toContain("Premortem");
  });

  it("keeps the title and headings Miami Red (regression)", async () => {
    const xml = await documentXml(await renderScopingDocx(emptyScopingContent(), header));
    expect((xml.match(/C41230/g) ?? []).length).toBeGreaterThanOrEqual(2);
  });
});
```

- [ ] **Step 2: Run it** — `npx vitest run tests/unit/project-docx.test.ts`. Expected: PASS. If `jszip` cannot be imported (it is a `docx` dependency, normally present), report it and fall back to asserting only the `PK` signature and buffer growth for the combined doc, dropping the color regression assertion.

- [ ] **Step 3: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 4: The project export route

**Files:**
- Create: `app/api/project-coach/[projectId]/export/route.ts`

- [ ] **Step 1: Write the route**

```ts
// app/api/project-coach/[projectId]/export/route.ts
import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import { recordUsageEvent } from "@/lib/db";
import {
  getAccessibleProject,
  listDeliverables,
  listProjectMembers,
} from "@/lib/db/projects";
import { getCoachEngine } from "@/lib/project/coach-engine";
import { getCoachSpec } from "@/lib/project/coach-specs";
import { coachLabel, isCoachType } from "@/lib/project/coaches";
import { courseLabel, findCourse } from "@/lib/project/courses";
import {
  genericBlocks,
  renderProjectDeliverablesDocx,
  scopingBlocks,
} from "@/lib/documents/coach-docx";
import type { GenericContent } from "@/lib/project/coach-framework";
import type { ScopingContent } from "@/lib/project/scoping";
import { COACHES } from "@/lib/project/coaches";

export const runtime = "nodejs";

function jsonError(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}

function exportFilename(projectName: string): string {
  const slug =
    projectName.replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) ||
    "project";
  return `${slug}-Deliverables.docx`;
}

/** A deliverable counts as started once it has content or a transcript. */
function isStarted(contentJson: string, transcriptJson: string): boolean {
  return contentJson.trim() !== "{}" || transcriptJson.trim() !== "[]";
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ projectId: string }> },
) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return jsonError(401, "Sign in to continue.");
  const { projectId } = await params;

  const project = getAccessibleProject(projectId, email);
  if (!project) return jsonError(404, "That project could not be found.");

  try {
    const course = findCourse(project.courseCode);
    const header = {
      projectName: project.name,
      courseLabel: course ? courseLabel(course) : `ISA ${project.courseCode}`,
      organization: project.organization,
      members: listProjectMembers(projectId).map((m) => m.name ?? m.email),
    };

    // Order sections by the canonical coach order, including only started ones.
    const byType = new Map(listDeliverables(projectId).map((d) => [d.coachType, d]));
    const sections = [];
    for (const coach of COACHES) {
      const row = byType.get(coach.type);
      if (!row || !isCoachType(coach.type)) continue;
      if (!isStarted(row.contentJson, row.transcriptJson)) continue;
      const engine = getCoachEngine(coach.type);
      if (!engine) continue;
      const content = engine.parseContent(row.contentJson);
      const spec = getCoachSpec(coach.type);
      const blocks = spec
        ? genericBlocks(spec, content as GenericContent)
        : scopingBlocks(content as ScopingContent);
      sections.push({ title: coachLabel(coach.type), blocks });
    }

    const buffer = await renderProjectDeliverablesDocx(header, sections);

    recordUsageEvent({
      userEmail: email,
      module: "project_coach",
      eventType: "project_exported",
      outcome: String(sections.length),
    });

    return new NextResponse(new Uint8Array(buffer), {
      headers: {
        "content-type":
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "content-disposition": `attachment; filename="${exportFilename(project.name)}"`,
        "content-length": String(buffer.byteLength),
        "cache-control": "private, no-store",
      },
    });
  } catch (err) {
    logger.error({ err: String(err) }, "project export failed");
    return jsonError(500, "That project could not be exported. Try again.");
  }
}
```

- [ ] **Step 2: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 5: Workspace button, e2e, gate, log

**Files:**
- Modify: `app/(app)/project-coach/[projectId]/page.tsx`
- Modify: `tests/e2e/project-coach-session.spec.ts`

- [ ] **Step 1: Add the "Download all deliverables" link**

In the workspace page (after Plan 4A's edits), add a link near the header, below the organization line, pointing at the project export route. It is a plain download anchor:

```tsx
      <div className="mt-4">
        <a
          href={`/api/project-coach/${project.id}/export`}
          className="inline-block rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red"
        >
          Download all deliverables
        </a>
      </div>
```

- [ ] **Step 2: Extend an e2e with a project-export download**

Add to `tests/e2e/project-coach-session.spec.ts` a test that creates a project, starts a coach (so there is a started deliverable), returns to the workspace, and downloads the project export:

```ts
test("project export downloads a combined .docx", async ({ page }, testInfo) => {
  const name = `Export ${testInfo.project.name} ${Date.now()}`;
  await page.goto("/project-coach/new");
  await page.getByLabel("Course").selectOption("496");
  await page.getByLabel("Project name").fill(name);
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible();

  // Start the scoping deliverable so there is something to export.
  await page.getByRole("link", { name: /Project Scoping/ }).click();
  await expect(page.getByRole("heading", { name: "Project Scoping Coach" })).toBeVisible();
  await page.getByLabel("Organization", { exact: true }).fill("Kroger");
  await expect(page.getByText(/Last updated by/)).toBeVisible({ timeout: 20000 });
  await page.waitForTimeout(800);

  await page.getByRole("link", { name: "Back to project" }).click();
  const downloadPromise = page.waitForEvent("download", { timeout: 20000 });
  await page.getByRole("link", { name: "Download all deliverables" }).click();
  expect((await downloadPromise).suggestedFilename()).toMatch(/\.docx$/);
});
```

- [ ] **Step 3: Full gate** — `npm run typecheck && npm run lint && npm test && npm run test:e2e`. Expected: green, real counts quoted, existing docx and coach specs still pass.

- [ ] **Step 4: Migration log** — append a dated entry: the shared docx module, the single renderers reduced to wrappers, the combined `renderProjectDeliverablesDocx`, the project export route (started-deliverables only, canonical coach order), the workspace button, the Miami Red regression test, and the e2e. Note this completes the Project Assistant module (design build-order step 4).

---

## Self-Review

**1. Spec coverage (design section 10, build-order step 4):**
- Per-project export of all started deliverables, one document, each deliverable rendered: `renderProjectDeliverablesDocx` + the route (Tasks 1, 4). Covered.
- The cover carries course, project name, organization, and team names: `coverBlocks` (Task 1). Covered.
- Available per project from the workspace: the "Download all deliverables" link (Task 5). Covered.
- Per-deliverable export still works (Plan 2C/3B), now through the shared blocks: the single renderers are thin wrappers and their tests still pass (Task 2). Covered.
- Miami Red preserved: `coverBlocks`/`heading` keep `color: MIAMI_RED`, locked by the regression test (Tasks 1, 3). Covered.

**2. Placeholder scan:** none. The workspace edit specifies the exact link; the JSZip note gives a concrete fallback.

**3. Type/name consistency:** `ScopingDocHeader` now lives in `coach-docx.ts` and is re-exported by `scoping-docx.ts`, so existing importers still resolve. `scopingBlocks`/`genericBlocks`/`coverBlocks`/`docFromChildren`/`renderProjectDeliverablesDocx` are used consistently by the wrappers, the route, and the test. The route dispatches scoping vs generic via `getCoachSpec`, matching the engine dispatch used everywhere else. `isStarted` uses the same `"{}"`/`"[]"` defaults that `getOrCreateDeliverable` writes.

---

## Execution Handoff

This completes the Project Assistant module (all four design build-order steps). The only remaining design item is the deferred real-time collaboration slice, which is its own future cross-cutting work (self-hosted Yjs over WSS, reused by the Coding Studio editor too).

**Plan saved to `webapp/docs/development/2026-07-23-project-assistant-plan-4b-project-export.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks.

**2. Inline Execution** — execute here with checkpoints.

Run it after Plan 4A has executed and verified. I will execute it after 4A reports green, unless you say otherwise.
