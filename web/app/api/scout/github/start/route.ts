import { randomUUID } from "node:crypto";
import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { checkRateLimit } from "@/lib/ratelimit";
import {
  githubAuthorizeUrl,
  githubOauthConfigured,
} from "@/lib/scout/github-oauth";
import { encodeOauthState, safeReturnPath } from "@/lib/scout/github-state";

/**
 * Begins the GitHub connect flow (v6.3.0). Opened in a popup (or as a
 * full-page navigation when popups are blocked). Sets the anti-CSRF state
 * cookie on the SAME response that redirects to GitHub, per the Next 16
 * rule that cookies are only settable in route handlers.
 *
 * CHATISA_MOCK_GITHUB=1 short-circuits straight to the callback with a mock
 * code so e2e tests exercise the real state validation without github.com.
 */
export async function GET(req: Request) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) {
    return NextResponse.json({ error: "Sign in required." }, { status: 401 });
  }
  const limit = checkRateLimit(`scout-gh-oauth:${email}`, {
    limit: 10,
    windowMs: 60_000,
  });
  if (!limit.allowed) {
    return NextResponse.json(
      { error: `Give it a moment. Try again in ${limit.retryAfterSeconds} seconds.` },
      { status: 429 },
    );
  }
  if (!githubOauthConfigured()) {
    return NextResponse.json(
      { error: "GitHub connection is not configured on this server." },
      { status: 503 },
    );
  }

  const url = new URL(req.url);
  const returnPath = safeReturnPath(url.searchParams.get("return"));
  const state = randomUUID();

  const callbackUrl = new URL("/api/scout/github/callback", url.origin);
  const destination =
    process.env.CHATISA_MOCK_GITHUB === "1"
      ? `${callbackUrl.toString()}?code=mock-code&state=${state}`
      : githubAuthorizeUrl(state, callbackUrl.toString());

  const res = NextResponse.redirect(destination, 302);
  res.cookies.set("scout_gh_state", encodeOauthState({ state, returnPath }), {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 600,
  });
  return res;
}
