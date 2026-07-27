import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import { repairPptx } from "@/lib/ask/pptx-repair";
import { checkRateLimit } from "@/lib/ratelimit";
import {
  ANTHROPIC_FILE_ID,
  anthropicFileContent,
  anthropicFileMeta,
  mediaTypeForName,
  mockZipBytes,
  safeDownloadName,
} from "@/lib/ask/hosted-files";

/**
 * Streams one file created by Anthropic's code execution container to the
 * student (slice E). Pass-through only: nothing is written to disk or the
 * database (ADR-022); the provider holds the file, we relay the bytes.
 *
 * With ?meta=1 it returns just the filename as JSON. The stream carries file
 * ids but no names, so without this the download button had to be labelled
 * "the file" and a student could not tell a deck from a spreadsheet.
 */
export async function GET(
  req: Request,
  { params }: { params: Promise<{ fileId: string }> },
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

  const { fileId } = await params;
  if (!ANTHROPIC_FILE_ID.test(fileId)) {
    return NextResponse.json({ error: "Unknown file." }, { status: 400 });
  }
  const metaOnly =
    new URL(req.url).searchParams.get("meta") === "1";

  // Mock mode serves a fixture so the e2e download flow runs offline.
  if (process.env.CHATISA_MOCK_LLM === "1") {
    if (metaOnly) {
      return NextResponse.json({ filename: "mock-deck.pptx" });
    }
    return new Response(new Uint8Array(mockZipBytes()), {
      headers: {
        "content-type": mediaTypeForName("mock-deck.pptx"),
        "content-disposition": 'attachment; filename="mock-deck.pptx"',
        "cache-control": "private, no-store",
      },
    });
  }

  const meta = await anthropicFileMeta(fileId);
  if (!meta) {
    return NextResponse.json(
      { error: "That file is no longer available from the provider." },
      { status: 404 },
    );
  }
  if (metaOnly) {
    return NextResponse.json(
      { filename: safeDownloadName(meta.filename, fileId) },
      { headers: { "cache-control": "private, no-store" } },
    );
  }
  const upstream = await anthropicFileContent(fileId);
  if (!upstream.ok || !upstream.body) {
    return NextResponse.json(
      { error: "That file could not be retrieved." },
      { status: 502 },
    );
  }
  const filename = safeDownloadName(meta.filename, fileId);
  const headers = {
    "content-type": meta.mimeType ?? mediaTypeForName(filename),
    "content-disposition": `attachment; filename="${filename}"`,
    "cache-control": "private, no-store",
  };

  // A generated deck is buffered rather than streamed so it can be repaired
  // before it reaches the student: the model's python-pptx code leaves a
  // duplicate slideLayout relationship on every slide, which makes PowerPoint
  // refuse to open the file (see lib/ask/pptx-repair). Decks are tens of
  // kilobytes, so buffering costs nothing. Everything else still streams.
  if (filename.toLowerCase().endsWith(".pptx")) {
    const raw = new Uint8Array(await upstream.arrayBuffer());
    const { bytes, removed } = await repairPptx(raw);
    if (removed > 0) {
      logger.info({ filename, removed }, "repaired generated pptx");
    }
    // Copied into a Uint8Array that is definitely backed by an ArrayBuffer:
    // jszip's output is typed over ArrayBufferLike, which could be a
    // SharedArrayBuffer, and Response's BodyInit will not accept that. A deck
    // is tens of kilobytes, so the copy is free.
    const body = new Uint8Array(bytes.byteLength);
    body.set(bytes);
    return new Response(body, {
      headers: { ...headers, "content-length": String(body.byteLength) },
    });
  }

  return new Response(upstream.body, { headers });
}
