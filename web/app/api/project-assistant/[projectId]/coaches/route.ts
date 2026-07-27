// app/api/project-assistant/[projectId]/coaches/route.ts
import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import {
  getAccessibleProject,
  isProjectLead,
  updateProjectCoaches,
} from "@/lib/db/projects";
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
