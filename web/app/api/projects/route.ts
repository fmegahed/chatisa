// app/api/projects/route.ts
import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import {
  createProject,
  listOwnedProjects,
  listSharedProjects,
} from "@/lib/db/projects";
import { recordUsageEvent } from "@/lib/db";
import { findCourse } from "@/lib/project/courses";
import { isCoachType } from "@/lib/project/coaches";

export const runtime = "nodejs";

const MODULE = "project_coach";

function errorResponse(status: number, message: string) {
  return NextResponse.json({ error: message }, { status });
}

const createSchema = z.object({
  courseCode: z.string().min(1).max(20),
  name: z.string().trim().min(1).max(160),
  organization: z.string().trim().max(160).default(""),
  coachTypes: z.array(z.string()).max(5),
});

export async function GET() {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return errorResponse(401, "Sign in to continue.");
  return NextResponse.json({
    owned: listOwnedProjects(email),
    shared: listSharedProjects(email),
  });
}

export async function POST(request: Request) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return errorResponse(401, "Sign in to continue.");

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return errorResponse(400, "Send a valid request.");
  }

  const parsed = createSchema.safeParse(body);
  if (!parsed.success) return errorResponse(400, "Check the project details.");

  if (!findCourse(parsed.data.courseCode)) {
    return errorResponse(400, "Pick a course from the list.");
  }
  const coachTypes = parsed.data.coachTypes.filter(isCoachType);

  const id = createProject({
    ownerEmail: email,
    ownerName: session.user?.name ?? null,
    courseCode: parsed.data.courseCode,
    name: parsed.data.name,
    organization: parsed.data.organization,
    coachTypes,
  });

  recordUsageEvent({
    userEmail: email,
    module: MODULE,
    eventType: "project_created",
  });

  return NextResponse.json({ id }, { status: 201 });
}
