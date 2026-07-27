// app/api/project-assistant/[projectId]/export/route.ts
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
