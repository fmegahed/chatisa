// app/api/project-assistant/[projectId]/members/route.ts
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
