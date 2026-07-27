import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { recordUsageEvent } from "@/lib/db";
import { deleteProject } from "@/lib/db/projects";

export const runtime = "nodejs";

/**
 * Deletes a project. Owner only: deleteProject checks ownership and cascades to
 * the team and every deliverable. A non-owner gets the same 404 a bad id gives,
 * so project ids do not leak.
 */
export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ projectId: string }> },
) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) {
    return NextResponse.json({ error: "Sign in to continue." }, { status: 401 });
  }
  const { projectId } = await params;
  if (!deleteProject(projectId, email)) {
    return NextResponse.json(
      { error: "That project could not be deleted." },
      { status: 404 },
    );
  }
  recordUsageEvent({
    userEmail: email,
    module: "project_coach",
    eventType: "project_deleted",
  });
  return NextResponse.json({ ok: true });
}
