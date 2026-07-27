# Project Assistant — Plan 4A: Team Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the project lead manage the team (invite teammates by `@miamioh.edu` email, remove members) and change which coaches the project includes, all enforced lead-only, with invited members' names filled in on their first visit.

**Architecture:** Two access-checked routes (members, coaches) that require the caller to be the lead. Two small client components (`TeamManager`, `CoachSelector`) render in the workspace only for the lead and call the routes, then `router.refresh()` to re-render the server component with fresh data. The lead is the project owner (`ownerEmail`); the owner can never be removed.

**Tech Stack:** Next.js 16, TypeScript, Drizzle, Zod, Vitest, Playwright + axe.

## Global Constraints

- **No git commits, no deploys, no production access.** Working tree stays uncommitted; each task ends by running its gate. (Git repo at `webapp/`; `web/` and `docs/` untracked; never run git write commands.)
- **No secrets in the client;** env var names only.
- **No em dashes in any user-facing text.**
- **This is a customized Next.js;** follow existing patterns.
- **Emails lowercased; timestamps ISO-8601 UTC.**
- **Access control is the privacy boundary:** every member/coach mutation resolves the project through `getAccessibleProject` and additionally requires `isProjectLead`. A non-lead member gets a 403; a non-member gets the same 404 a bad id gives.
- **Names only, no student id** (design): invites carry an email; the name is filled from the invited student's authenticated Google name on their first visit.
- **Sequencing:** run after Plan 3B. Independent of Plan 4B (per-project export).

---

## File Structure

**Created:**
- `app/api/project-coach/[projectId]/members/route.ts` — POST invite, DELETE remove (lead only).
- `app/api/project-coach/[projectId]/coaches/route.ts` — PUT set enabled coaches (lead only).
- `components/project/TeamManager.tsx` — member list with remove, plus invite form (lead view).
- `components/project/CoachSelector.tsx` — coach checkboxes with save (lead view).
- `tests/e2e/project-team.spec.ts` — lead invites, toggles a coach, removes.

**Modified:**
- `lib/db/projects.ts` — add `isProjectLead`, `removeProjectMember`, `updateProjectCoaches`, `fillMemberName`.
- `app/(app)/project-coach/[projectId]/page.tsx` — name-fill on visit; render lead controls.
- `tests/unit/project-db.test.ts` — tests for the new data-layer functions.

## Interfaces produced

```ts
// lib/db/projects.ts
export function isProjectLead(projectId: string, userEmail: string): boolean;
export function removeProjectMember(projectId: string, email: string): boolean; // false if it is the owner or absent
export function updateProjectCoaches(projectId: string, coachTypes: CoachType[]): void;
export function fillMemberName(projectId: string, email: string, name: string): void; // sets name only when currently null
```

---

### Task 1: Data-layer additions

**Files:**
- Modify: `lib/db/projects.ts`
- Test: `tests/unit/project-db.test.ts`

- [ ] **Step 1: Write the failing test (append to `tests/unit/project-db.test.ts`)**

Add `isProjectLead`, `removeProjectMember`, `updateProjectCoaches`, `fillMemberName` to the existing destructured import from `@/lib/db/projects`, then append:

```ts
describe("team management", () => {
  it("identifies the lead as the owner only", () => {
    const projectId = createProject({
      ownerEmail: LEAD, ownerName: "Team Lead", courseCode: "401/501",
      name: "Lead check", organization: "", coachTypes: [],
    });
    addProjectMember({ projectId, email: MATE, name: "Teammate", role: "member" });
    expect(isProjectLead(projectId, LEAD)).toBe(true);
    expect(isProjectLead(projectId, MATE)).toBe(false);
    expect(isProjectLead(projectId, STRANGER)).toBe(false);
  });

  it("removes a member but never the owner", () => {
    const projectId = createProject({
      ownerEmail: LEAD, ownerName: "Team Lead", courseCode: "444/544",
      name: "Remove", organization: "", coachTypes: [],
    });
    addProjectMember({ projectId, email: MATE, name: "Teammate", role: "member" });

    expect(removeProjectMember(projectId, MATE)).toBe(true);
    expect(isProjectMember(projectId, MATE)).toBe(false);
    // The owner cannot be removed.
    expect(removeProjectMember(projectId, LEAD)).toBe(false);
    expect(isProjectMember(projectId, LEAD)).toBe(true);
  });

  it("updates which coaches a project includes", () => {
    const projectId = createProject({
      ownerEmail: LEAD, ownerName: "Team Lead", courseCode: "225",
      name: "Coaches", organization: "", coachTypes: ["scoping"],
    });
    updateProjectCoaches(projectId, ["premortem", "reflection"]);
    expect(getAccessibleProject(projectId, LEAD)?.coachTypes).toEqual(["premortem", "reflection"]);
  });

  it("fills a member name only when it was empty", () => {
    const projectId = createProject({
      ownerEmail: LEAD, ownerName: "Team Lead", courseCode: "496",
      name: "Names", organization: "", coachTypes: [],
    });
    addProjectMember({ projectId, email: MATE, name: null, role: "member" });
    fillMemberName(projectId, MATE, "Real Name");
    expect(listProjectMembers(projectId).find((m) => m.email === MATE)?.name).toBe("Real Name");
    // Does not overwrite an existing name.
    fillMemberName(projectId, MATE, "Different");
    expect(listProjectMembers(projectId).find((m) => m.email === MATE)?.name).toBe("Real Name");
  });
});
```

- [ ] **Step 2: Run to verify it fails** — `npx vitest run tests/unit/project-db.test.ts` (functions not exported).

- [ ] **Step 3: Write the implementation (append to `lib/db/projects.ts`)**

Add `isNull` to the existing `drizzle-orm` import (it currently imports `and`, `desc`, `eq`, `ne`). Then:

```ts
/** The lead is the project owner. */
export function isProjectLead(projectId: string, userEmail: string): boolean {
  const row = getDb()
    .select({ ownerEmail: schema.projects.ownerEmail })
    .from(schema.projects)
    .where(eq(schema.projects.id, projectId))
    .get();
  return row?.ownerEmail === userEmail.toLowerCase();
}

/** Removes a member. Refuses to remove the owner, and reports whether it did. */
export function removeProjectMember(projectId: string, email: string): boolean {
  const target = email.toLowerCase();
  const project = getDb()
    .select({ ownerEmail: schema.projects.ownerEmail })
    .from(schema.projects)
    .where(eq(schema.projects.id, projectId))
    .get();
  if (!project || project.ownerEmail === target) return false;
  const result = getDb()
    .delete(schema.projectMembers)
    .where(
      and(
        eq(schema.projectMembers.projectId, projectId),
        eq(schema.projectMembers.email, target),
      ),
    )
    .run();
  return result.changes > 0;
}

export function updateProjectCoaches(projectId: string, coachTypes: CoachType[]): void {
  getDb()
    .update(schema.projects)
    .set({
      coachTypesJson: JSON.stringify(coachTypes),
      updatedAt: new Date().toISOString(),
    })
    .where(eq(schema.projects.id, projectId))
    .run();
}

/** Fills an invited member's name on first visit, without overwriting one. */
export function fillMemberName(projectId: string, email: string, name: string): void {
  getDb()
    .update(schema.projectMembers)
    .set({ name })
    .where(
      and(
        eq(schema.projectMembers.projectId, projectId),
        eq(schema.projectMembers.email, email.toLowerCase()),
        isNull(schema.projectMembers.name),
      ),
    )
    .run();
}
```

- [ ] **Step 4: Run to verify it passes** — `npx vitest run tests/unit/project-db.test.ts`.

- [ ] **Step 5: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 2: Members and coaches routes

**Files:**
- Create: `app/api/project-coach/[projectId]/members/route.ts`
- Create: `app/api/project-coach/[projectId]/coaches/route.ts`

- [ ] **Step 1: Members route**

```ts
// app/api/project-coach/[projectId]/members/route.ts
import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { ALLOWED_EMAIL_DOMAIN } from "@/lib/auth/domain";
import {
  addProjectMember,
  isProjectLead,
  listProjectMembers,
  removeProjectMember,
} from "@/lib/db/projects";

export const runtime = "nodejs";

function jsonError(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}

async function requireLead(
  params: Promise<{ projectId: string }>,
): Promise<{ error: NextResponse } | { projectId: string }> {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return { error: jsonError(401, "Sign in to continue.") };
  const { projectId } = await params;
  // A non-lead (member or non-member) is refused. Members can read the roster
  // from the workspace; only the lead changes it.
  if (!isProjectLead(projectId, email)) {
    return { error: jsonError(403, "Only the team lead can change the team.") };
  }
  return { projectId };
}

const inviteSchema = z.object({
  email: z.string().trim().toLowerCase().email().max(200),
  name: z.string().trim().max(200).optional(),
});

export async function POST(
  req: Request,
  { params }: { params: Promise<{ projectId: string }> },
) {
  const r = await requireLead(params);
  if ("error" in r) return r.error;

  let raw: unknown;
  try {
    raw = await req.json();
  } catch {
    return jsonError(400, "Request body must be JSON.");
  }
  const parsed = inviteSchema.safeParse(raw);
  if (!parsed.success) return jsonError(400, "Enter a valid email address.");
  if (!parsed.data.email.endsWith(`@${ALLOWED_EMAIL_DOMAIN}`)) {
    return jsonError(400, `Invite a ${ALLOWED_EMAIL_DOMAIN} email address.`);
  }

  addProjectMember({
    projectId: r.projectId,
    email: parsed.data.email,
    name: parsed.data.name ?? null,
    role: "member",
  });
  return NextResponse.json({ members: listProjectMembers(r.projectId) });
}

const removeSchema = z.object({ email: z.string().trim().toLowerCase().email() });

export async function DELETE(
  req: Request,
  { params }: { params: Promise<{ projectId: string }> },
) {
  const r = await requireLead(params);
  if ("error" in r) return r.error;

  let raw: unknown;
  try {
    raw = await req.json();
  } catch {
    return jsonError(400, "Request body must be JSON.");
  }
  const parsed = removeSchema.safeParse(raw);
  if (!parsed.success) return jsonError(400, "That request wasn't valid.");

  const removed = removeProjectMember(r.projectId, parsed.data.email);
  if (!removed) return jsonError(400, "That member could not be removed.");
  return NextResponse.json({ members: listProjectMembers(r.projectId) });
}
```

- [ ] **Step 2: Coaches route**

```ts
// app/api/project-coach/[projectId]/coaches/route.ts
import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { getAccessibleProject, isProjectLead, updateProjectCoaches } from "@/lib/db/projects";
import { isCoachType } from "@/lib/project/coaches";

export const runtime = "nodejs";

function jsonError(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}

const schema = z.object({ coachTypes: z.array(z.string()).max(5) });

export async function PUT(
  req: Request,
  { params }: { params: Promise<{ projectId: string }> },
) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return jsonError(401, "Sign in to continue.");
  const { projectId } = await params;
  if (!isProjectLead(projectId, email)) {
    return jsonError(403, "Only the team lead can change the coaches.");
  }

  let raw: unknown;
  try {
    raw = await req.json();
  } catch {
    return jsonError(400, "Request body must be JSON.");
  }
  const parsed = schema.safeParse(raw);
  if (!parsed.success) return jsonError(400, "That request wasn't valid.");

  updateProjectCoaches(projectId, parsed.data.coachTypes.filter(isCoachType));
  return NextResponse.json({ project: getAccessibleProject(projectId, email) });
}
```

- [ ] **Step 3: Checkpoint** — `npm run typecheck && npm run lint`.

---

### Task 3: Lead-only client controls

**Files:**
- Create: `components/project/TeamManager.tsx`
- Create: `components/project/CoachSelector.tsx`

- [ ] **Step 1: TeamManager**

```tsx
// components/project/TeamManager.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { MemberRow } from "@/lib/db/projects";

export function TeamManager({
  projectId,
  members,
  ownerEmail,
}: {
  projectId: string;
  members: MemberRow[];
  ownerEmail: string;
}) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const base = `/api/project-coach/${projectId}/members`;

  async function invite(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const value = email.trim();
    if (!value) return;
    setBusy(true);
    try {
      const res = await fetch(base, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: value }),
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        setError(data.error ?? "Could not invite that teammate.");
        return;
      }
      setEmail("");
      router.refresh();
    } catch {
      setError("Could not reach the server. Try again.");
    } finally {
      setBusy(false);
    }
  }

  async function remove(memberEmail: string) {
    setError(null);
    const res = await fetch(base, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: memberEmail }),
    });
    if (res.ok) router.refresh();
    else setError("Could not remove that teammate.");
  }

  return (
    <div className="mt-3">
      <ul className="flex flex-wrap gap-2">
        {members.map((m) => (
          <li
            key={m.id}
            className="flex items-center gap-2 rounded-card border border-medium-tan bg-light-tan px-3 py-1 text-sm"
          >
            <span>
              {m.name ?? m.email}
              {m.email === ownerEmail.toLowerCase() ? " (lead)" : ""}
            </span>
            {m.email !== ownerEmail.toLowerCase() ? (
              <button
                type="button"
                onClick={() => remove(m.email)}
                aria-label={`Remove ${m.name ?? m.email}`}
                className="font-bold text-miami-red hover:underline"
              >
                Remove
              </button>
            ) : null}
          </li>
        ))}
      </ul>

      <form onSubmit={invite} className="mt-3 flex flex-wrap items-end gap-2">
        <div className="flex flex-col gap-1">
          <label htmlFor="invite-email" className="text-sm font-bold">
            Invite a teammate by email
          </label>
          <input
            id="invite-email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="name@miamioh.edu"
            className="rounded border border-medium-tan bg-paper p-2"
          />
        </div>
        <button
          type="submit"
          disabled={busy || email.trim().length === 0}
          className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
        >
          Invite
        </button>
      </form>
      {error ? (
        <p role="alert" className="mt-2 text-miami-red">
          {error}
        </p>
      ) : null}
    </div>
  );
}
```

- [ ] **Step 2: CoachSelector**

```tsx
// components/project/CoachSelector.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { COACHES, type CoachType } from "@/lib/project/coaches";

export function CoachSelector({
  projectId,
  enabled,
}: {
  projectId: string;
  enabled: CoachType[];
}) {
  const router = useRouter();
  const [selected, setSelected] = useState<CoachType[]>(enabled);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggle(type: CoachType) {
    setSelected((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type],
    );
  }

  async function save() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/project-coach/${projectId}/coaches`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ coachTypes: selected }),
      });
      if (!res.ok) {
        setError("Could not save the coaches. Try again.");
        return;
      }
      router.refresh();
    } catch {
      setError("Could not reach the server. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mt-3 rounded-card border border-medium-tan p-4">
      <p className="font-bold">Choose coaches (lead only)</p>
      <div className="mt-2 grid gap-2">
        {COACHES.map((c) => (
          <label key={c.type} className="flex items-start gap-2">
            <input
              type="checkbox"
              checked={selected.includes(c.type)}
              onChange={() => toggle(c.type)}
              className="mt-1"
            />
            <span>
              <span className="font-bold">{c.label}.</span> {c.blurb}
            </span>
          </label>
        ))}
      </div>
      <button
        type="button"
        onClick={save}
        disabled={busy}
        className="mt-3 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
      >
        {busy ? "Saving..." : "Save coaches"}
      </button>
      {error ? (
        <p role="alert" className="mt-2 text-miami-red">
          {error}
        </p>
      ) : null}
    </div>
  );
}
```

- [ ] **Step 3: Checkpoint** — `npm run typecheck && npm run lint`. Confirm the brand tokens (`text-miami-red`, `bg-miami-red`, `text-paper`, `hover:bg-accent-red`, `bg-medium-gray`) resolve against existing components; substitute exact tokens if any does not.

---

### Task 4: Wire the workspace page

**Files:**
- Modify: `app/(app)/project-coach/[projectId]/page.tsx`

- [ ] **Step 1: Add name-fill and lead controls**

- Add imports:

```tsx
import { getAccessibleProject, listProjectMembers, fillMemberName } from "@/lib/db/projects";
import { TeamManager } from "@/components/project/TeamManager";
import { CoachSelector } from "@/components/project/CoachSelector";
```

- After confirming `project` (access), fill the current member's name on first visit, then list members:

```tsx
  if (session.user.name) {
    fillMemberName(projectId, session.user.email, session.user.name);
  }
  const members = listProjectMembers(projectId);
```

- In the Team section, replace the static `<ul>` and the Plan 4 comment with a lead/member split:

```tsx
        {isLead ? (
          <TeamManager
            projectId={project.id}
            members={members}
            ownerEmail={project.ownerEmail}
          />
        ) : (
          <ul className="mt-3 flex flex-wrap gap-2">
            {members.map((m) => (
              <li
                key={m.id}
                className="rounded-card border border-medium-tan bg-light-tan px-3 py-1 text-sm"
              >
                {m.name ?? m.email}
                {m.email === project.ownerEmail.toLowerCase() ? " (lead)" : ""}
              </li>
            ))}
          </ul>
        )}
```

- In the Coaches section, when `isLead`, render `<CoachSelector projectId={project.id} enabled={project.coachTypes} />` below the enabled-coaches list (and drop the "when coach selection ships" text). The enabled-coaches list stays as it is, so the lead sees both the live coach links and the selector.

- [ ] **Step 2: Checkpoint** — `npm run typecheck && npm run lint`.

- [ ] **Step 3: Manual sanity (optional)** — with `npm run dev`, open a project you own: the invite form, remove buttons, and coach selector appear; toggling a coach and saving updates the coach list after refresh.

---

### Task 5: e2e, gate, and log

**Files:**
- Create: `tests/e2e/project-team.spec.ts`

- [ ] **Step 1: e2e (lead actions)**

The suite runs as one authenticated student, who is the lead of the projects they create. Invite adds a member row by email (the invitee is a different address, so it shows as an email chip); removal takes it away; the coach selector changes the enabled list. Reuse the axe helper from the other project specs.

```ts
// tests/e2e/project-team.spec.ts
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("lead invites and removes a teammate and changes coaches", async ({ page }, testInfo) => {
  const name = `Team ${testInfo.project.name} ${Date.now()}`;
  const mate = `teammate.${Date.now()}@miamioh.edu`;

  await page.goto("/project-coach/new");
  await page.getByLabel("Course").selectOption("496");
  await page.getByLabel("Project name").fill(name);
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible();

  // Invite a teammate.
  await page.getByLabel("Invite a teammate by email").fill(mate);
  await page.getByRole("button", { name: "Invite" }).click();
  await expect(page.getByText(mate)).toBeVisible();

  // Axe on the lead workspace with the controls present.
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(results.violations).toEqual([]);

  // Enable a coach that was not chosen at creation (Premortem), then confirm the link.
  await page.getByRole("checkbox", { name: /Premortem/ }).check();
  await page.getByRole("button", { name: "Save coaches" }).click();
  await expect(page.getByRole("link", { name: /Premortem/ })).toBeVisible();

  // Remove the teammate.
  await page.getByRole("button", { name: `Remove ${mate}` }).click();
  await expect(page.getByText(mate)).toHaveCount(0);
});
```

- [ ] **Step 2: Run the e2e** — `npm run test:e2e -- project-team`. Expected: pass, including axe.

- [ ] **Step 3: Full gate** — `npm run typecheck && npm run lint && npm test && npm run test:e2e`. Expected: green, real counts quoted. The suite is deterministic; a failure is likely real.

- [ ] **Step 4: Migration log** — append a dated entry: the four data-layer functions, the members and coaches routes (lead-only, domain-checked), the `TeamManager`/`CoachSelector` controls, name-fill on visit, and the e2e. Note per-project export is Plan 4B.

---

## Self-Review

**1. Spec coverage (design section 6):**
- The lead invites and removes teammates by `@miamioh.edu` email: members route + `TeamManager` (Tasks 2, 3), domain-checked. Covered.
- The lead picks and later changes which coaches the project includes: coaches route + `CoachSelector` (Tasks 2, 3). Covered.
- Members are identified by name and email, name filled from the authenticated Google name: `fillMemberName` on visit (Task 4). Covered.
- Access control enforced server-side on every change; only the owner or a member reads, only the lead writes: `getAccessibleProject` (read) plus `isProjectLead` (write) on each route. Covered.
- The owner cannot be removed: `removeProjectMember` refuses the owner (Task 1). Covered.
- Deferred to Plan 4B: per-project export of all started deliverables.

**2. Placeholder scan:** none. The workspace edits name the exact insertions; the brand-token note states the fallback.

**3. Type/name consistency:** `isProjectLead`, `removeProjectMember`, `updateProjectCoaches`, `fillMemberName` are used consistently across the data layer, routes, and page. `MemberRow` (Plan 1) is the prop type for `TeamManager`. `CoachType`/`COACHES`/`isCoachType`/`coachLabel` are Plan 1 exports. `ALLOWED_EMAIL_DOMAIN` is the existing auth constant. The routes live under the same `/api/project-coach/[projectId]/` path the session already uses.

---

## Execution Handoff

**Plan 4B** (the last piece) adds per-project export: a single combined `.docx` of every started deliverable (a shared docx-helpers refactor so the per-coach renderers and the combined renderer share layout), an access-checked project export route, and a "Download all deliverables" button on the workspace.

**Plan saved to `webapp/docs/development/2026-07-23-project-assistant-plan-4a-team-management.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks.

**2. Inline Execution** — execute here with checkpoints.

I will proceed to execute it via a subagent now unless you say otherwise.
