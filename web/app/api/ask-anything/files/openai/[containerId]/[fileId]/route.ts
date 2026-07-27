import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { checkRateLimit } from "@/lib/ratelimit";
import {
  OPENAI_CONTAINER_ID,
  OPENAI_FILE_ID,
  mediaTypeForName,
  mockZipBytes,
  openaiContainerFileContent,
  safeDownloadName,
} from "@/lib/ask/hosted-files";

/**
 * Streams one file created by an OpenAI code-interpreter container to the
 * student (slice E). Pass-through only (ADR-022). The display filename rides
 * as a query parameter from the card's file listing; it is sanitized and used
 * for the download name only, never as a path.
 */
export async function GET(
  req: Request,
  { params }: { params: Promise<{ containerId: string; fileId: string }> },
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
      { error: "Too many downloads at once. Wait a moment." },
      { status: 429 },
    );
  }

  const { containerId, fileId } = await params;
  if (!OPENAI_CONTAINER_ID.test(containerId) || !OPENAI_FILE_ID.test(fileId)) {
    return NextResponse.json({ error: "Unknown file." }, { status: 400 });
  }
  const filename = safeDownloadName(
    new URL(req.url).searchParams.get("name") ?? "",
    fileId,
  );

  if (process.env.CHATISA_MOCK_LLM === "1") {
    return new Response(new Uint8Array(mockZipBytes()), {
      headers: {
        "content-type": mediaTypeForName(filename),
        "content-disposition": `attachment; filename="${filename}"`,
        "cache-control": "private, no-store",
      },
    });
  }

  const upstream = await openaiContainerFileContent(containerId, fileId);
  if (!upstream.ok || !upstream.body) {
    return NextResponse.json(
      { error: "That file could not be retrieved." },
      { status: 502 },
    );
  }
  return new Response(upstream.body, {
    headers: {
      "content-type": mediaTypeForName(filename),
      "content-disposition": `attachment; filename="${filename}"`,
      "cache-control": "private, no-store",
    },
  });
}
