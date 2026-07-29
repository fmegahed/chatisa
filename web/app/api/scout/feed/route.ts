import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import {
  countScoutPostings,
  latestSuccessfulScoutRun,
  listAllScoutPostings,
  listScoutPostings,
  recordUsageEvent,
} from "@/lib/db";

/**
 * The weekly feed. Serves public employer content plus freshness metadata;
 * deliberately takes no profile input and stores nothing per student — the
 * browser computes matches locally (local-first decision, 2026-07-28).
 */
export async function GET(req: Request) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) {
    return NextResponse.json({ error: "Sign in required." }, { status: 401 });
  }

  const url = new URL(req.url);

  // The whole active feed in one compact response: the client scores and
  // filters locally (privacy), so a pagination loop only added round trips
  // (user scale question, 2026-07-29).
  if (url.searchParams.get("shape") === "index") {
    const lastRun = latestSuccessfulScoutRun();
    recordUsageEvent({
      userEmail: email,
      module: "job_scout",
      eventType: "feed_view",
    });
    return NextResponse.json({
      postings: listAllScoutPostings(),
      freshness: {
        updatedAt: lastRun?.startedAt ?? null,
        totalActive: countScoutPostings(),
        sourceErrors: lastRun ? JSON.parse(lastRun.sourceErrorsJson) : {},
      },
    });
  }

  const categoryRaw = url.searchParams.get("category");
  const category = ["fulltime", "internship", "federal"].includes(
    categoryRaw ?? "",
  )
    ? (categoryRaw as "fulltime" | "internship" | "federal")
    : undefined;
  const stateRaw = url.searchParams.get("state");
  const state =
    stateRaw && /^[A-Za-z]{2}$/.test(stateRaw)
      ? stateRaw.toUpperCase()
      : undefined;
  const remoteRaw = url.searchParams.get("remote");
  const remote = remoteRaw === "1" ? true : undefined;
  const limit = Math.min(
    Math.max(Number(url.searchParams.get("limit")) || 50, 1),
    100,
  );
  const offset = Math.max(Number(url.searchParams.get("offset")) || 0, 0);

  const postings = listScoutPostings({ category, state, remote, limit, offset });
  const lastRun = latestSuccessfulScoutRun();

  recordUsageEvent({
    userEmail: email,
    module: "job_scout",
    eventType: "feed_view",
  });

  return NextResponse.json({
    postings,
    freshness: {
      updatedAt: lastRun?.startedAt ?? null,
      totalActive: countScoutPostings(),
      sourceErrors: lastRun ? JSON.parse(lastRun.sourceErrorsJson) : {},
    },
  });
}
