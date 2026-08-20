import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { recordUsageEvent } from "@/lib/db";
import { logger } from "@/lib/log";
import { exchangeCodeForToken } from "@/lib/scout/github-oauth";
import { decodeOauthState } from "@/lib/scout/github-state";

/**
 * GitHub OAuth callback (v6.3.0). Validates the state cookie (the
 * CareerBridge prototype skipped this, leaving a login-CSRF hole where an
 * attacker could plant their own token in a victim's browser), exchanges the
 * code server-side, and hands the token to the browser in the URL FRAGMENT
 * of a redirect: fragments never appear in server or proxy logs, and the
 * receiving page clears the hash immediately. The token is not stored,
 * logged, or recorded anywhere server-side.
 */
export async function GET(req: Request) {
  const session = await auth();
  const email = session?.user?.email;
  if (!email) {
    return NextResponse.json({ error: "Sign in required." }, { status: 401 });
  }

  const url = new URL(req.url);
  const connectedPage = (hash: string, returnPath: string) => {
    const dest = new URL("/job-scout/github-connected", url.origin);
    dest.searchParams.set("return", returnPath);
    const res = NextResponse.redirect(`${dest.toString()}${hash}`, 303);
    res.headers.set("cache-control", "no-store");
    // One-shot cookie either way: a used or stale state must never validate twice.
    res.cookies.set("scout_gh_state", "", { path: "/", maxAge: 0 });
    return res;
  };

  const cookieHeader = req.headers.get("cookie") ?? "";
  const stateCookie = /(?:^|;\s*)scout_gh_state=([^;]+)/.exec(cookieHeader)?.[1];
  const stored = decodeOauthState(
    stateCookie ? decodeURIComponent(stateCookie) : undefined,
  );
  const returnPath = stored?.returnPath ?? "/job-scout";

  if (url.searchParams.get("error") === "access_denied") {
    return connectedPage("#gh-error=denied", returnPath);
  }
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  if (!stored || !code || !state || state !== stored.state) {
    return connectedPage("#gh-error=state", returnPath);
  }

  try {
    const { token, login } = await exchangeCodeForToken(code);
    recordUsageEvent({
      userEmail: email,
      module: "job_scout",
      eventType: "github_connected",
    });
    const payload = Buffer.from(
      JSON.stringify({ token, login }),
      "utf8",
    ).toString("base64url");
    return connectedPage(`#gh=${payload}`, returnPath);
  } catch (err) {
    logger.error({ err: String(err) }, "github oauth exchange failed");
    return connectedPage("#gh-error=exchange", returnPath);
  }
}
