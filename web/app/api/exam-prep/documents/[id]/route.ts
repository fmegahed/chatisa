import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { deleteExamDocument, getOwnedExamDocument } from "@/lib/db";

/**
 * A document belonging to the signed-in student. A document owned by someone
 * else answers exactly as an unknown id does, so ids cannot be probed.
 */
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) {
    return NextResponse.json({ error: "Sign in to continue." }, { status: 401 });
  }

  const doc = getOwnedExamDocument((await params).id, userEmail);
  if (!doc) {
    return NextResponse.json(
      { error: "That document could not be found." },
      { status: 404 },
    );
  }

  return NextResponse.json({
    id: doc.id,
    filename: doc.filename,
    pageCount: doc.pageCount,
    textPageCount: doc.textPageCount,
    visionPageCount: doc.visionPageCount,
    charCount: doc.charCount,
    classification: doc.classification,
    createdAt: doc.createdAt,
    textAvailable: doc.textPurgedAt === null,
  });
}

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) {
    return NextResponse.json({ error: "Sign in to continue." }, { status: 401 });
  }

  const removed = deleteExamDocument((await params).id, userEmail);
  if (!removed) {
    return NextResponse.json(
      { error: "That document could not be found." },
      { status: 404 },
    );
  }
  return new NextResponse(null, { status: 204 });
}
