import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import { MODELS, getPageModels } from "@/lib/config/models";
import { isModelAvailable } from "@/lib/providers";
import { classifyProviderFailure } from "@/lib/providers/errors";
import {
  getOwnedApplication,
  listDocumentsForApplication,
  recordUsageEvent,
  upsertTailoredDocument,
} from "@/lib/db";
import {
  generateCoverLetter,
  generateTailoredResume,
} from "@/lib/documents/generate";
import { describeGrounding } from "@/lib/documents/grounding";
import { EXAM_GENERATE_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";
import type { TemplateId } from "@/lib/prompts/fsb-standards";

export const runtime = "nodejs";
export const maxDuration = 300;

const MODULE = "jobapp_assistant";

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

const generateSchema = z.object({
  kind: z.enum(["resume", "cover_letter"]),
  modelId: z.string().min(1),
  template: z.union([z.literal(1), z.literal(2), z.literal(3)]).default(1),
  studentName: z.string().min(1).max(120),
  email: z.string().max(200).nullable().optional(),
  phone: z.string().max(60).nullable().optional(),
  linkedin: z.string().max(200).nullable().optional(),
  recipientName: z.string().max(120).nullable().optional(),
  companyAddress: z.string().max(400).nullable().optional(),
});

export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const { id } = await params;
  const application = getOwnedApplication(id, userEmail);
  if (!application) return errorResponse(404, "That application could not be found.");

  return NextResponse.json({
    documents: listDocumentsForApplication(id).map((d) => ({
      id: d.id,
      kind: d.kind,
      template: d.template,
      content: JSON.parse(d.contentJson),
      flagged: d.ungroundedJson ? JSON.parse(d.ungroundedJson) : [],
      reviewed: d.reviewedAt !== null,
      updatedAt: d.updatedAt,
    })),
  });
}

/** Generates or regenerates one document for this application. */
export async function POST(
  req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const { id } = await params;
  const application = getOwnedApplication(id, userEmail);
  if (!application) return errorResponse(404, "That application could not be found.");

  if (!application.resumeText) {
    // Without the student's own resume there is nothing to ground against, and
    // generating anyway would mean inventing a career for them.
    return errorResponse(
      400,
      "Upload your current resume first. Everything here is built from what your resume already says, so there is nothing to work from without it.",
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Request body must be JSON.");
  }
  const parsed = generateSchema.safeParse(body);
  if (!parsed.success) return errorResponse(400, "That request wasn't valid.");
  const input = parsed.data;

  if (!getPageModels("jobapp_assistant").includes(input.modelId)) {
    return errorResponse(400, "That model isn't available for this module.");
  }
  if (!isModelAvailable(input.modelId)) {
    return errorResponse(503, "That model isn't configured on this server right now.");
  }

  const limit = checkRateLimit(`documents:${userEmail}`, EXAM_GENERATE_RATE_LIMIT);
  if (!limit.allowed) {
    return errorResponse(429, "Too many documents in a short time. Wait a moment.", {
      retryAfterSeconds: limit.retryAfterSeconds,
    });
  }

  const contact = {
    email: input.email ?? null,
    phone: input.phone ?? null,
    linkedin: input.linkedin ?? null,
  };
  const startedAt = Date.now();

  try {
    const result =
      input.kind === "resume"
        ? await generateTailoredResume({
            modelId: input.modelId,
            template: input.template as TemplateId,
            studentName: input.studentName,
            contact,
            resumeText: application.resumeText,
            postingText: application.postingText,
            company: application.company,
            positionTitle: application.positionTitle,
          })
        : await generateCoverLetter({
            modelId: input.modelId,
            studentName: input.studentName,
            contact,
            resumeText: application.resumeText,
            postingText: application.postingText,
            company: application.company,
            positionTitle: application.positionTitle,
            recipientName: input.recipientName ?? null,
            companyAddress: input.companyAddress ?? null,
            todayLabel: new Date().toLocaleDateString("en-US", {
              year: "numeric",
              month: "long",
              day: "numeric",
            }),
          });

    const documentId = upsertTailoredDocument({
      applicationId: id,
      userEmail,
      kind: input.kind,
      template: input.template,
      modelId: input.modelId,
      contentJson: JSON.stringify(result.content),
      ungroundedJson: JSON.stringify(result.grounding.flagged),
    });

    recordUsageEvent({
      userEmail,
      module: MODULE,
      eventType: "document_generated",
      modelId: input.modelId,
      provider: MODELS[input.modelId]?.provider ?? null,
      latencyMs: Date.now() - startedAt,
      outcome: input.kind,
    });

    return NextResponse.json({
      documentId,
      kind: input.kind,
      content: result.content,
      flagged: result.grounding.flagged,
      groundingMessage: describeGrounding(result.grounding),
      checked: result.grounding.checked,
    });
  } catch (err) {
    const failure = classifyProviderFailure(err);
    logger.error(
      { err: String(err), failureKind: failure.kind },
      "document generation failed",
    );
    return errorResponse(502, failure.message);
  }
}
