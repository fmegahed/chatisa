import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import { recordUsageEvent } from "@/lib/db";
import {
  getAccessibleProject,
  getOrCreateDeliverable,
  listProjectMembers,
} from "@/lib/db/projects";
import type { ScopingContent } from "@/lib/project/scoping";
import { renderScopingDocx } from "@/lib/documents/scoping-docx";
import { renderGenericCoachDocx } from "@/lib/documents/generic-coach-docx";
import { getCoachEngine } from "@/lib/project/coach-engine";
import { getCoachSpec } from "@/lib/project/coach-specs";
import { isCoachType } from "@/lib/project/coaches";
import type { GenericContent } from "@/lib/project/coach-framework";
import { courseLabel, findCourse } from "@/lib/project/courses";

export const runtime = "nodejs";

function jsonError(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}

/** Safe for a Content-Disposition header and a filesystem. */
function exportFilename(projectName: string, coachType: string): string {
  const slug =
    projectName.replace(/[^a-zA-Z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) ||
    "project";
  return `${slug}-${coachType}.docx`;
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ projectId: string; coachType: string }> },
) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return jsonError(401, "Sign in to continue.");

  const { projectId, coachType } = await params;
  const engine = getCoachEngine(coachType);
  if (!engine || !isCoachType(coachType)) return jsonError(404, "That coach could not be found.");

  const project = getAccessibleProject(projectId, email);
  if (!project) return jsonError(404, "That project could not be found.");

  try {
    const row = getOrCreateDeliverable(projectId, coachType);
    const content = engine.parseContent(row.contentJson);

    const course = findCourse(project.courseCode);
    const members = listProjectMembers(projectId).map((m) => m.name ?? m.email);
    const headerData = {
      projectName: project.name,
      courseLabel: course ? courseLabel(course) : `ISA ${project.courseCode}`,
      organization: project.organization,
      members,
    };
    const spec = getCoachSpec(coachType);
    const buffer = spec
      ? await renderGenericCoachDocx(spec, content as GenericContent, headerData)
      : await renderScopingDocx(content as ScopingContent, headerData);

    recordUsageEvent({
      userEmail: email,
      module: "project_coach",
      eventType: "deliverable_exported",
      outcome: coachType,
    });

    return new NextResponse(new Uint8Array(buffer), {
      headers: {
        "content-type":
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "content-disposition": `attachment; filename="${exportFilename(project.name, coachType)}"`,
        "content-length": String(buffer.byteLength),
        "cache-control": "private, no-store",
      },
    });
  } catch (err) {
    logger.error({ err: String(err) }, "scoping export failed");
    return jsonError(500, "That worksheet could not be exported. Try again.");
  }
}
