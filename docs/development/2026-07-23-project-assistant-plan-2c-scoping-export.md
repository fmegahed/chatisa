# Project Assistant — Plan 2C: Scoping Word Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export the Scoping deliverable to Word (.docx): the worksheet's field sections as labelled paragraphs and its Goals, Data, Analysis, and Stakeholders sections as Word tables, with a cover carrying the course, project name, organization, and team member names. This completes the Scoping vertical slice (design build-order step 2).

**Architecture:** A server-only renderer (`renderScopingDocx`) builds the document with the `docx` library, mirroring the JobApp export approach (`renderResumeDocx`). An access-checked GET route streams the `.docx` with the Word content type, exactly as `app/api/documents/[id]/export/route.ts` does. A download link is added to the coach session. Layout reuses the field and table descriptors from Plan 2B so labels stay consistent between the on-screen panel and the document.

**Tech Stack:** TypeScript, `docx`, Next.js route handlers, Vitest, Playwright.

## Global Constraints

- **No git commits, no deploys, no production access.** Working tree stays uncommitted; each task ends by running its gate. (A git repo exists at `webapp/`; `web/` and `docs/` are untracked; never run git write commands.)
- **No secrets in the client;** env var names only.
- **No em dashes in any user-facing text**, including document headings and labels.
- **This is a customized Next.js;** follow the existing export pattern (`app/api/documents/[id]/export/route.ts`) and docx pattern (`lib/documents/docx.ts`).
- **Access control on every request:** the export route resolves the project through `getAccessibleProject` and treats a non-member like a missing project. The document is one team's material: `cache-control: private, no-store`.
- **Scope: the Scoping deliverable only.** Per-project export of all deliverables and the other coaches' exports are Plan 4 / Plan 3. A non-scoping coach export returns 404.
- **Sequencing:** run after Plan 2B.

---

## File Structure

**Created:**
- `lib/documents/scoping-docx.ts` — `renderScopingDocx(content, header): Promise<Buffer>`.
- `app/api/project-coach/[projectId]/coach/[coachType]/export/route.ts` — GET returning the `.docx`.
- `tests/unit/scoping-docx.test.ts` — renderer produces a valid docx buffer.

**Modified:**
- `components/project/CoachSession.tsx` — add a "Download Word" link to the export route.
- `tests/e2e/project-coach-session.spec.ts` — assert the export downloads a `.docx`.
- `webapp/docs/development/migration-log.md`.

**Interfaces consumed:** `ScopingContent`, `scopingContentSchema`, `emptyScopingContent` (`@/lib/project/scoping`); `FIELD_SECTIONS`, `TABLE_SECTIONS`, `readField` (`@/components/project/scoping-fields`, plain data + a pure helper, safe to import server-side); `getAccessibleProject`, `getDeliverable`, `listProjectMembers` (`@/lib/db/projects`); `findCourse`, `courseLabel` (`@/lib/project/courses`).

---

### Task 1: The docx renderer

**Files:**
- Create: `lib/documents/scoping-docx.ts`
- Test: `tests/unit/scoping-docx.test.ts`

**Interfaces:**
- Produces: `ScopingDocHeader`, `renderScopingDocx(content: ScopingContent, header: ScopingDocHeader): Promise<Buffer>`.

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/scoping-docx.test.ts
import { describe, expect, it } from "vitest";
import { renderScopingDocx } from "@/lib/documents/scoping-docx";
import { emptyScopingContent } from "@/lib/project/scoping";

const header = {
  projectName: "Retail dashboard",
  courseLabel: "ISA 401/501: Business Intelligence and Data Visualization",
  organization: "Kroger",
  members: ["Team Lead", "Teammate"],
};

describe("renderScopingDocx", () => {
  it("produces a non-empty .docx buffer for an empty worksheet", async () => {
    const buf = await renderScopingDocx(emptyScopingContent(), header);
    expect(buf.byteLength).toBeGreaterThan(0);
    // .docx is a zip: it starts with the "PK" local-file signature.
    expect(buf.subarray(0, 2).toString("latin1")).toBe("PK");
  });

  it("produces a larger buffer once the worksheet has content", async () => {
    const empty = await renderScopingDocx(emptyScopingContent(), header);
    const filled = emptyScopingContent();
    filled.organizationName = "Kroger";
    filled.goals = [{ goal: "Cut stockouts", constraints: "One quarter" }];
    filled.stakeholders = [
      { orgDept: "Operations", involvement: "Owner", counterpart: "Analyst" },
    ];
    const withContent = await renderScopingDocx(filled, header);
    expect(withContent.byteLength).toBeGreaterThan(empty.byteLength);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run tests/unit/scoping-docx.test.ts`
Expected: FAIL, cannot resolve `@/lib/documents/scoping-docx`.

- [ ] **Step 3: Write the renderer**

```ts
// lib/documents/scoping-docx.ts
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
import {
  FIELD_SECTIONS,
  TABLE_SECTIONS,
  readField,
} from "@/components/project/scoping-fields";

export interface ScopingDocHeader {
  projectName: string;
  courseLabel: string;
  organization: string;
  members: string[];
}

// US Letter in twips (1 inch = 1440), half-inch margins.
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

function cell(text: string, opts?: { bold?: boolean }): TableCell {
  return new TableCell({
    children: [
      new Paragraph({
        children: [new TextRun({ text, bold: opts?.bold, font: FONT, size: 20 })],
      }),
    ],
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
  const headerRow = new TableRow({
    tableHeader: true,
    children: columns.map((c) => cell(c.label, { bold: true })),
  });
  const bodyRows = rows.map(
    (r) => new TableRow({ children: columns.map((c) => cell(r[c.key] ?? "")) }),
  );
  return [
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      rows: [headerRow, ...bodyRows],
    }),
  ];
}

export async function renderScopingDocx(
  content: ScopingContent,
  header: ScopingDocHeader,
): Promise<Buffer> {
  const children: (Paragraph | Table)[] = [];

  // Cover.
  children.push(
    new Paragraph({
      heading: HeadingLevel.HEADING_1,
      alignment: AlignmentType.CENTER,
      spacing: { after: 120 },
      children: [
        new TextRun({
          text: header.projectName || "Project scope",
          font: FONT,
          bold: true,
          size: 36,
        }),
      ],
    }),
  );
  children.push(labelledValue("Course", header.courseLabel));
  if (header.organization) children.push(labelledValue("Organization", header.organization));
  if (header.members.length > 0) {
    children.push(labelledValue("Team", header.members.join(", ")));
  }

  // Field sections.
  for (const section of FIELD_SECTIONS) {
    children.push(heading(section.heading));
    for (const f of section.fields) {
      children.push(labelledValue(f.label, readField(content, f.path)));
    }
  }

  // Table sections.
  for (const section of TABLE_SECTIONS) {
    children.push(heading(section.heading));
    for (const node of tableFor(section.columns, rowsFor(content, section.table))) {
      children.push(node);
    }
  }

  const doc = new Document({
    sections: [
      {
        properties: {
          page: {
            size: { width: PAGE.width, height: PAGE.height },
            margin: {
              top: PAGE.margin,
              bottom: PAGE.margin,
              left: PAGE.margin,
              right: PAGE.margin,
            },
          },
        },
        children,
      },
    ],
  });
  return Buffer.from(await Packer.toBuffer(doc));
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run tests/unit/scoping-docx.test.ts`
Expected: PASS (2 cases). If `TableRow`'s `tableHeader` option name or `WidthType.PERCENTAGE` differs in the installed `docx`, match the installed types (confirmed present: `Table`, `TableRow`, `TableCell`, `WidthType`, `BorderStyle`, `HeadingLevel`).

- [ ] **Step 5: Checkpoint** — `npm run typecheck && npm run lint`, working tree uncommitted.

---

### Task 2: The export route

**Files:**
- Create: `app/api/project-coach/[projectId]/coach/[coachType]/export/route.ts`

- [ ] **Step 1: Write the route**

```ts
// app/api/project-coach/[projectId]/coach/[coachType]/export/route.ts
import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import { recordUsageEvent } from "@/lib/db";
import {
  getAccessibleProject,
  getOrCreateDeliverable,
  listProjectMembers,
} from "@/lib/db/projects";
import { emptyScopingContent, scopingContentSchema } from "@/lib/project/scoping";
import { renderScopingDocx } from "@/lib/documents/scoping-docx";
import { courseLabel, findCourse } from "@/lib/project/courses";

export const runtime = "nodejs";

function jsonError(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}

/** Safe for a Content-Disposition header and a filesystem. */
function exportFilename(projectName: string): string {
  const slug =
    projectName.replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) ||
    "project";
  return `${slug}-Scoping.docx`;
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ projectId: string; coachType: string }> },
) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return jsonError(401, "Sign in to continue.");

  const { projectId, coachType } = await params;
  if (coachType !== "scoping") return jsonError(404, "That coach could not be found.");

  const project = getAccessibleProject(projectId, email);
  if (!project) return jsonError(404, "That project could not be found.");

  try {
    const row = getOrCreateDeliverable(projectId, "scoping");
    const parsed = scopingContentSchema.safeParse(JSON.parse(row.contentJson));
    const content = parsed.success ? parsed.data : emptyScopingContent();

    const course = findCourse(project.courseCode);
    const members = listProjectMembers(projectId).map((m) => m.name ?? m.email);
    const buffer = await renderScopingDocx(content, {
      projectName: project.name,
      courseLabel: course ? courseLabel(course) : `ISA ${project.courseCode}`,
      organization: project.organization,
      members,
    });

    recordUsageEvent({
      userEmail: email,
      module: "project_coach",
      eventType: "deliverable_exported",
      outcome: "scoping",
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
    logger.error({ err: String(err) }, "scoping export failed");
    return jsonError(500, "That worksheet could not be exported. Try again.");
  }
}
```

- [ ] **Step 2: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 3: Export link in the session, and the e2e

**Files:**
- Modify: `components/project/CoachSession.tsx`
- Modify: `tests/e2e/project-coach-session.spec.ts`

- [ ] **Step 1: Add the download link**

In `components/project/CoachSession.tsx`, in the deliverable column (the `<div className="lg:border-l ...">`), directly above `<ScopingDeliverable ... />`, add a download link to the export route. It is a plain anchor (a GET download), not a fetch:

```tsx
          <div className="mb-4">
            <a
              href={`${base}/export`}
              className="inline-block rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red"
            >
              Download Word
            </a>
          </div>
```

(`base` is already `/api/project-coach/${projectId}/coach/scoping` in the component, so `${base}/export` is the route from Task 2.)

- [ ] **Step 2: Extend the session e2e with a download assertion**

Append to the existing test in `tests/e2e/project-coach-session.spec.ts`, after the persistence checks (the page is on the coach session after the reload):

```ts
  // The worksheet exports to a .docx.
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download Word" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/\.docx$/);
```

- [ ] **Step 3: Run the e2e**

Run: `npm run test:e2e -- project-coach-session`
Expected: PASS. Keep the explicit-wait style already in this spec (the routes compile on first hit under load). If the download event is slow to fire, allow `page.waitForEvent("download", { timeout: 20000 })`.

- [ ] **Step 4: Checkpoint** — working tree uncommitted.

---

### Task 4: Full gate and log

- [ ] **Step 1: Full gate**

Run: `npm run typecheck && npm run lint && npm test && npm run test:e2e`
Expected: green. Quote real counts. The Plan 1 `project-assistant.spec.ts` and the chat CodeMirror test have known parallel-load flakes; if either fails, re-run in isolation and report it as pre-existing, not a regression.

- [ ] **Step 2: Migration log**

Append a dated `### YYYY-MM-DD —` entry: the `renderScopingDocx` renderer (field sections as labelled paragraphs, table sections as Word tables, a cover with course/project/organization/team), the access-checked export route, and the session download link. Note that per-project export of all deliverables is Plan 4 and the other coaches are Plan 3. This completes the Scoping vertical slice (design build-order step 2).

---

## Self-Review

**1. Spec coverage (design spec section 10):**
- Scoping deliverable renders to match the worksheet layout, with its fields and its Goals / Data / Analysis / Ethics / Stakeholders content: `renderScopingDocx` renders every `FIELD_SECTIONS` field and every `TABLE_SECTIONS` table (Ethics is a field section here, matching the panel). Covered.
- A cover carries course, project name, organization, and team member names (no student id): the cover block, members from `listProjectMembers` mapped to `name ?? email`. Covered.
- Export available per-deliverable: the session download link and route. Covered. (Per-project export of all started deliverables is deferred to Plan 4, as the design allows.)

**2. Placeholder scan:** none. The one conditional (docx option-name confirmation) names the installed exports verified present.

**3. Type/name consistency:** `ScopingContent`, `scopingContentSchema`, `emptyScopingContent` used as Plan 2A exports them. `FIELD_SECTIONS`, `TABLE_SECTIONS`, `readField` used as Plan 2B defines them, so the document labels match the panel. `rowsFor` covers the same five `ScopingTable` values. The export route path `/api/project-coach/${projectId}/coach/scoping/export` matches the link `${base}/export` in the session.

---

## Execution Handoff

This completes the Scoping vertical slice (design build-order step 2: split view, tool-call fill, direct edit, persist, Word export). Next: **Plan 3** ports the other four coaches (Premortem, Team Structuring, Devil's Advocate, Reflection) over the same session pattern with their smaller schemas, and **Plan 4** adds team management (invite and coach selection, lead only) and per-project export of all started deliverables.

**Plan saved to `webapp/docs/development/2026-07-23-project-assistant-plan-2c-scoping-export.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks.

**2. Inline Execution** — execute here with checkpoints.

I will proceed to execute it via a subagent now unless you say otherwise.
