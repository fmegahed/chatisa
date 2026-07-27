// app/api/project-assistant/[projectId]/coach/[coachType]/deliverable/route.ts
import { z } from "zod";
import { auth } from "@/lib/auth";
import {
  getAccessibleProject,
  getOrCreateDeliverable,
  saveDeliverableContent,
  saveDeliverableTranscript,
  getDeliverable,
} from "@/lib/db/projects";
import { getCoachEngine } from "@/lib/project/coach-engine";

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
  const engine = getCoachEngine(coachType);
  if (!engine) return { error: jsonError(404, "That coach could not be found.") } as const;
  const project = getAccessibleProject(projectId, email);
  if (!project) return { error: jsonError(404, "That project could not be found.") } as const;
  return { email, projectId, coachType, engine, name: session.user?.name ?? email } as const;
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ projectId: string; coachType: string }> },
) {
  const r = await resolve(params);
  if ("error" in r) return r.error;
  const row = getOrCreateDeliverable(r.projectId, r.coachType);
  // Normalize to the coach's full content shape. A brand-new deliverable is
  // stored as "{}", so returning it raw would leave the client panel without
  // the arrays and field maps it renders. parseContent fills the empty shape.
  return Response.json({
    contentJson: JSON.stringify(r.engine.parseContent(row.contentJson)),
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
}
