/**
 * Pure helpers for the GitHub OAuth state cookie. Kept free of server-only
 * imports so the encoding, tolerant decoding, and the open-redirect guard
 * are unit-testable in isolation.
 */

export interface GithubOauthState {
  state: string;
  returnPath: string;
}

/**
 * Only a same-origin absolute path may be a post-connect destination.
 * "//evil.example" is a protocol-relative URL and "https://..." is absolute;
 * both would turn the callback into an open redirect, so anything that is
 * not a plain "/path" collapses to the Portfolio Builder, which owns the
 * GitHub connection flow (2026-08-20).
 */
export function safeReturnPath(raw: string | null | undefined): string {
  if (raw && raw.startsWith("/") && !raw.startsWith("//") && !raw.includes("\\")) {
    return raw;
  }
  return "/portfolio";
}

export function encodeOauthState(value: GithubOauthState): string {
  return Buffer.from(JSON.stringify(value), "utf8").toString("base64url");
}

/** Tolerant: a missing, truncated, or tampered cookie decodes to null. */
export function decodeOauthState(raw: string | undefined): GithubOauthState | null {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(
      Buffer.from(raw, "base64url").toString("utf8"),
    ) as Partial<GithubOauthState>;
    if (typeof parsed.state !== "string" || parsed.state.length < 16) return null;
    return { state: parsed.state, returnPath: safeReturnPath(parsed.returnPath) };
  } catch {
    return null;
  }
}

/**
 * The origin the OUTSIDE world uses to reach this server. In production the
 * Next process listens on 127.0.0.1:3000 behind the TLS relay in
 * chatisa-server.mjs, so `new URL(req.url).origin` is the internal address;
 * building the OAuth redirect_uri from it made GitHub reject every connect
 * with "redirect_uri is not associated with this application" (2026-08-20).
 * AUTH_URL is what the OAuth app was registered against, so it wins; the
 * relay's x-forwarded-* headers are the fallback; the raw request origin is
 * only right in plain local development.
 */
export function publicOrigin(
  req: Request,
  env: { AUTH_URL?: string } = { AUTH_URL: process.env.AUTH_URL },
): string {
  if (env.AUTH_URL) {
    try {
      return new URL(env.AUTH_URL).origin;
    } catch {
      /* malformed; fall through */
    }
  }
  const host = req.headers.get("x-forwarded-host")?.split(",")[0]?.trim();
  if (host) {
    const proto =
      req.headers.get("x-forwarded-proto")?.split(",")[0]?.trim() || "https";
    try {
      return new URL(`${proto}://${host}`).origin;
    } catch {
      /* malformed; fall through */
    }
  }
  return new URL(req.url).origin;
}
