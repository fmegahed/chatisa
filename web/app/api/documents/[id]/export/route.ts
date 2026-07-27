import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import { getOwnedDocument, recordUsageEvent } from "@/lib/db";
import { renderCoverLetterDocx, renderResumeDocx } from "@/lib/documents/docx";
import {
  coverLetterContentSchema,
  resumeContentSchema,
} from "@/lib/documents/schema";
import type { TemplateId } from "@/lib/prompts/fsb-standards";

export const runtime = "nodejs";

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}

/** Safe for a Content-Disposition header and for a filesystem. */
function exportFilename(company: string, kind: string): string {
  const slug = company
    .replace(/[^a-zA-Z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 40) || "application";
  const label = kind === "resume" ? "Resume" : "Cover-Letter";
  return `${slug}-${label}.docx`;
}

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const { id } = await params;
  const document = getOwnedDocument(id, userEmail);
  if (!document) return errorResponse(404, "That document could not be found.");

  try {
    const parsed = JSON.parse(document.contentJson);
    let buffer: Buffer;
    let company = "application";

    if (document.kind === "resume") {
      const content = resumeContentSchema.parse(parsed);
      buffer = await renderResumeDocx(content, document.template as TemplateId);
    } else {
      const content = coverLetterContentSchema.parse(parsed);
      company = content.recipient.company;
      buffer = await renderCoverLetterDocx(content);
    }

    recordUsageEvent({
      userEmail,
      module: "jobapp_assistant",
      eventType: "document_exported",
      outcome: document.kind,
    });

    return new NextResponse(new Uint8Array(buffer), {
      headers: {
        "content-type":
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "content-disposition": `attachment; filename="${exportFilename(company, document.kind)}"`,
        "content-length": String(buffer.byteLength),
        // One student's application materials: never a shared cache.
        "cache-control": "private, no-store",
      },
    });
  } catch (err) {
    logger.error({ err: String(err) }, "document export failed");
    return errorResponse(500, "That document could not be exported. Try saving it again.");
  }
}
