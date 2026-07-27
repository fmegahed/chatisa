// lib/db/projects.ts
import { randomUUID } from "node:crypto";
import { and, desc, eq, isNull, ne } from "drizzle-orm";
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
/**
 * Deletes a project. Only the owner may delete it, and doing so cascades to the
 * team and every deliverable (foreign keys with ON DELETE CASCADE). Returns
 * whether a row was removed, so a non-owner gets the same nothing a bad id does.
 */
export function deleteProject(projectId: string, userEmail: string): boolean {
  const result = getDb()
    .delete(schema.projects)
    .where(
      and(
        eq(schema.projects.id, projectId),
        eq(schema.projects.ownerEmail, userEmail.toLowerCase()),
      ),
    )
    .run();
  return result.changes > 0;
}

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
  const insert = getDb()
    .insert(schema.projectMembers)
    .values({
      id: randomUUID(),
      projectId: params.projectId,
      email,
      name: params.name,
      role: params.role,
      createdAt: new Date().toISOString(),
    });
  const target = [
    schema.projectMembers.projectId,
    schema.projectMembers.email,
  ] as const;
  // A name-carrying re-invite refreshes the stored name; a nameless invite
  // (the common case, name filled on the invitee's first visit) leaves any
  // existing row untouched. Drizzle rejects an empty update set, so a nameless
  // conflict must do nothing rather than "set {}".
  if (params.name) {
    insert
      .onConflictDoUpdate({ target: [...target], set: { name: params.name } })
      .run();
  } else {
    insert.onConflictDoNothing({ target: [...target] }).run();
  }
}

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

export function updateProjectCoaches(
  projectId: string,
  coachTypes: CoachType[],
): void {
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
export function fillMemberName(
  projectId: string,
  email: string,
  name: string,
): void {
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
