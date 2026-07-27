import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import {
  deleteInterview,
  getInterviewTurns,
  getOwnedInterview,
} from "@/lib/db";
import { projectInterview } from "@/lib/interview/projection";

export const runtime = "nodejs";

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}

/** Current state, used to resume an interrupted interview. */
export async function GET(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const { id } = await params;
  const interview = getOwnedInterview(id, userEmail);
  // Same response for another student's interview as for one that does not
  // exist, so ids cannot be probed for existence.
  if (!interview) return errorResponse(404, "That interview could not be found.");

  return NextResponse.json({
    interview: projectInterview(interview, getInterviewTurns(id)),
  });
}

export async function DELETE(
  _req: Request,
  { params }: { params: Promise<{ id: string }> },
) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const { id } = await params;
  if (!deleteInterview(id, userEmail)) {
    return errorResponse(404, "That interview could not be found.");
  }
  return NextResponse.json({ deleted: true });
}
