import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { scoutRunInProgress } from "@/lib/db";
import { runHarvest } from "@/lib/scout/harvest";
import { logger } from "@/lib/log";

/**
 * Manual harvest trigger, for the operator after a config change or a failed
 * Sunday run. Gated twice: a signed-in session AND membership in
 * CHATISA_SCOUT_ADMINS (comma-separated emails), because a harvest spends
 * real API quota and model dollars.
 */
export async function POST() {
  const session = await auth();
  const email = session?.user?.email?.toLowerCase();
  if (!email) {
    return NextResponse.json({ error: "Sign in required." }, { status: 401 });
  }
  const admins = (process.env.CHATISA_SCOUT_ADMINS ?? "")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
  if (!admins.includes(email)) {
    return NextResponse.json(
      { error: "Only Job Scout administrators can trigger a harvest." },
      { status: 403 },
    );
  }
  if (scoutRunInProgress()) {
    return NextResponse.json(
      { error: "A harvest is already running." },
      { status: 409 },
    );
  }
  // Fire and forget: a full harvest takes minutes and the admin should not
  // hold a request open for it. Progress lands in scout_runs.
  void runHarvest({ trigger: "manual" }).catch((err) =>
    logger.error(
      { err: err instanceof Error ? err.message : String(err) },
      "manual job scout harvest failed",
    ),
  );
  return NextResponse.json(
    { status: "started", note: "Progress is recorded in scout_runs." },
    { status: 202 },
  );
}
