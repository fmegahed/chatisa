import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";

/**
 * First authentication layer: unauthenticated requests to app pages are
 * redirected to /login before rendering. The authenticated layout and each
 * privileged API route re-verify the session (defense in depth).
 */
export const proxy = auth((req) => {
  if (req.auth?.user) return;

  // API callers get a JSON 401. Redirecting them to an HTML sign-in page
  // would turn an authentication failure into a confusing parse error, and a
  // redirected POST would arrive at a page route carrying its original body.
  if (req.nextUrl.pathname.startsWith("/api/")) {
    return NextResponse.json({ error: "Sign in to continue." }, { status: 401 });
  }

  return NextResponse.redirect(new URL("/login", req.nextUrl.origin));
});

export const config = {
  matcher: [
    // Everything except: login page, the guest invite landing page (its whole
    // point is to admit the not-yet-signed-in), auth endpoints, health, Next
    // internals, static assets, and the self-hosted code-runner assets
    // (workers and WASM runtimes), which are public and must load without an
    // auth round-trip. login and guest are EXACT segments ((?:$|/)): a future
    // route that merely starts with those words must not silently skip this
    // layer.
    "/((?!login(?:$|/)|guest(?:$|/)|api/auth|api/health|_next/static|_next/image|brand/|workers/|runtimes/|favicon.ico|.*\\.(?:png|svg|ico|woff2?|wasm|mjs)$).*)",
  ],
};
