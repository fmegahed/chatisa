import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { getOwnedDocument, saveDocumentContent } from "@/lib/db";
import {
  coverLetterContentSchema,
  resumeContentSchema,
} from "@/lib/documents/schema";
import { checkClaims } from "@/lib/documents/grounding";
import { getOwnedApplication } from "@/lib/db";

export const runtime = "nodejs";

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}

const saveSchema = z.object({
  content: z.unknown(),
  markReviewed: z.boolean().default(false),
});

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

  return NextResponse.json({
    id: document.id,
    kind: document.kind,
    template: document.template,
    content: JSON.parse(document.contentJson),
    flagged: document.ungroundedJson ? JSON.parse(document.ungroundedJson) : [],
    reviewed: document.reviewedAt !== null,
  });
}

/**
 * Saves the student's edits.
 *
 * Grounding is re-checked against their resume after every edit, so a warning
 * clears as soon as they fix the line rather than lingering and training them
 * to ignore it. Editing counts as reviewing.
 */
export async function PATCH(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const { id } = await params;
  const document = getOwnedDocument(id, userEmail);
  if (!document) return errorResponse(404, "That document could not be found.");

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Request body must be JSON.");
  }
  const parsed = saveSchema.safeParse(body);
  if (!parsed.success) return errorResponse(400, "That request wasn't valid.");

  const shape =
    document.kind === "resume" ? resumeContentSchema : coverLetterContentSchema;
  const content = shape.safeParse(parsed.data.content);
  if (!content.success) {
    return errorResponse(400, "That document could not be saved. Reload and try again.");
  }

  const application = getOwnedApplication(document.applicationId, userEmail);
  let flagged: unknown[] = [];
  if (application?.resumeText) {
    const claims =
      document.kind === "resume"
        ? (content.data as { sections: { entries: { bullets: { text: string; sourceLine: string | null }[] }[] }[] })
            .sections.flatMap((s) => s.entries)
            .flatMap((e) => e.bullets)
            .map((b) => ({ text: b.text, sourceLine: b.sourceLine }))
        : (content.data as { paragraphs: { text: string; sourceLine: string | null }[] })
            .paragraphs.filter((p) => p.sourceLine !== null)
            .map((p) => ({ text: p.text, sourceLine: p.sourceLine }));
    flagged = checkClaims(claims, application.resumeText).flagged;
  }

  saveDocumentContent({
    id,
    userEmail,
    contentJson: JSON.stringify(content.data),
    markReviewed: parsed.data.markReviewed,
  });

  return NextResponse.json({ saved: true, flagged });
}
