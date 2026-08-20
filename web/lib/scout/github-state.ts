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
 * not a plain "/path" collapses to the module home.
 */
export function safeReturnPath(raw: string | null | undefined): string {
  if (raw && raw.startsWith("/") && !raw.startsWith("//") && !raw.includes("\\")) {
    return raw;
  }
  return "/job-scout";
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
