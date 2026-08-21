import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { checkRateLimit } from "@/lib/ratelimit";
import { recordUsageEvent } from "@/lib/db";

/**
 * Publish beacon (2026-08-20). The push to GitHub happens in the browser
 * with the student's own token, so the server never sees it; this route is
 * how a successful publish becomes a usage count. Counts only: the body
 * carries the site kind and nothing else.
 */

const schema = z.object({ kind: z.enum(["career", "showcase"]) });

export async function POST(req: Request) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) return NextResponse.json({ error: "Sign in required." }, { status: 401 });
  if (!checkRateLimit(`portfolio-event:${email}`, { limit: 20, windowMs: 60_000 }).allowed) {
    return NextResponse.json({ ok: false }, { status: 429 });
  }
  const parsed = schema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return NextResponse.json({ ok: false }, { status: 400 });
  recordUsageEvent({
    userEmail: email,
    module: "portfolio",
    eventType: "portfolio_published",
    outcome: parsed.data.kind,
  });
  return NextResponse.json({ ok: true });
}
