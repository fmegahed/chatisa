import { createHash } from "node:crypto";

/**
 * Guest magic-pass policy (2026-07-24): a small set of invite links so
 * collaborators outside Miami can try ChatISA before deployment. Pure and
 * unit-testable, mirroring lib/auth/domain.ts.
 *
 * Design:
 * - Tokens are 128-bit random strings minted by scripts/make-guest-passes.mjs.
 *   The server stores only their SHA-256 hashes (CHATISA_GUEST_PASS_HASHES,
 *   comma separated), so a leaked env file does not leak working links.
 * - An expiry date is REQUIRED (CHATISA_GUEST_EXPIRES, ISO date). No expiry
 *   means guest access is off: a trial that cannot end silently becomes a
 *   permanent unaudited door.
 * - Each token maps by position to a stable identity, guest-<n>@guest.chatisa,
 *   so usage events attribute per collaborator and revoking one token (remove
 *   its hash; keep positions with an empty slot or reissue) removes one guest.
 * - Unlike the test login, this provider is MEANT for production; its scope is
 *   bounded by the hash list and the expiry, not by NODE_ENV.
 */

/** Structurally matches both process.env and plain test objects (an interface
 * with only optional props trips the weak-type check against ProcessEnv). */
export type GuestEnv = Record<string, string | undefined>;

export const GUEST_EMAIL_DOMAIN = "guest.chatisa";

export function hashGuestPass(pass: string): string {
  return createHash("sha256").update(pass, "utf8").digest("hex");
}

export function parseGuestConfig(env: GuestEnv): {
  hashes: string[];
  expiresAt: number | null;
} {
  const hashes = (env.CHATISA_GUEST_PASS_HASHES ?? "")
    .split(",")
    .map((h) => h.trim().toLowerCase())
    .filter((h) => /^[0-9a-f]{64}$/.test(h));
  const raw = env.CHATISA_GUEST_EXPIRES?.trim();
  const expiresAt = raw ? Date.parse(raw) : NaN;
  return {
    hashes,
    expiresAt: Number.isFinite(expiresAt) ? expiresAt : null,
  };
}

/** True when guest links exist and have not expired. */
export function guestPassesEnabled(env: GuestEnv, now: number): boolean {
  const { hashes, expiresAt } = parseGuestConfig(env);
  return hashes.length > 0 && expiresAt !== null && now < expiresAt;
}

export type GuestDecision =
  | { allowed: true; guestNumber: number; email: string; name: string }
  | {
      allowed: false;
      reason: "disabled" | "expired" | "malformed" | "unknown-pass";
    };

export function evaluateGuestPass(
  rawPass: string,
  env: GuestEnv,
  now: number,
): GuestDecision {
  const { hashes, expiresAt } = parseGuestConfig(env);
  if (hashes.length === 0 || expiresAt === null) {
    return { allowed: false, reason: "disabled" };
  }
  if (now >= expiresAt) return { allowed: false, reason: "expired" };
  const pass = rawPass.trim();
  // Minted tokens are 32 hex chars; the bounds reject junk before hashing.
  if (pass.length < 16 || pass.length > 128) {
    return { allowed: false, reason: "malformed" };
  }
  const index = hashes.indexOf(hashGuestPass(pass));
  if (index === -1) return { allowed: false, reason: "unknown-pass" };
  const guestNumber = index + 1;
  return {
    allowed: true,
    guestNumber,
    email: `guest-${guestNumber}@${GUEST_EMAIL_DOMAIN}`,
    name: `Guest ${guestNumber}`,
  };
}
