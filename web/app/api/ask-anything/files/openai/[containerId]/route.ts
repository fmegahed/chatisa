import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { checkRateLimit } from "@/lib/ratelimit";
import {
  OPENAI_CONTAINER_ID,
  openaiContainerFiles,
} from "@/lib/ask/hosted-files";

/**
 * Lists the files an OpenAI code-interpreter run CREATED in its container
 * (slice E), so the tool card can offer downloads. Uploads (the Miami
 * template) are excluded by the helper. Metadata only; content streams from
 * the sibling [fileId] route.
 */
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ containerId: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) {
    return NextResponse.json({ error: "Sign in to continue." }, { status: 401 });
  }
  const limit = checkRateLimit(`hosted-files:${userEmail}`, {
    limit: 60,
    windowMs: 60_000,
  });
  if (!limit.allowed) {
    return NextResponse.json(
      { error: "Too many requests at once. Wait a moment." },
      { status: 429 },
    );
  }

  const { containerId } = await params;
  if (!OPENAI_CONTAINER_ID.test(containerId)) {
    return NextResponse.json({ error: "Unknown container." }, { status: 400 });
  }

  if (process.env.CHATISA_MOCK_LLM === "1") {
    return NextResponse.json({
      files: [{ id: "cfile_mock1", filename: "mock-deck.pptx", sizeBytes: 22 }],
    });
  }

  const files = await openaiContainerFiles(containerId);
  if (files === null) {
    return NextResponse.json(
      { error: "That container is no longer available." },
      { status: 404 },
    );
  }
  return NextResponse.json({ files });
}
