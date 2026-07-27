/**
 * Pure sign-in policy: only verified Google accounts on the miamioh.edu
 * domain may use ChatISA (ADR-004). Kept free of framework imports so the
 * rule is directly unit-testable.
 */
export const ALLOWED_EMAIL_DOMAIN = "miamioh.edu";

export interface SignInProfile {
  email?: string | null;
  /** Google's `email_verified` claim. */
  emailVerified?: boolean | null;
  /** Google Workspace `hd` claim, when present. */
  hostedDomain?: string | null;
}

export interface SignInDecision {
  allowed: boolean;
  reason?:
    | "missing-email"
    | "email-not-verified"
    | "wrong-domain"
    | "wrong-hosted-domain";
}

export function evaluateSignIn(profile: SignInProfile): SignInDecision {
  const email = profile.email?.trim().toLowerCase();
  if (!email) return { allowed: false, reason: "missing-email" };
  if (profile.emailVerified !== true) {
    return { allowed: false, reason: "email-not-verified" };
  }
  // Exact domain match: subdomains and look-alikes are rejected.
  if (!email.endsWith(`@${ALLOWED_EMAIL_DOMAIN}`)) {
    return { allowed: false, reason: "wrong-domain" };
  }
  const local = email.slice(0, email.length - ALLOWED_EMAIL_DOMAIN.length - 1);
  if (local.length === 0 || local.includes("@")) {
    return { allowed: false, reason: "wrong-domain" };
  }
  if (
    profile.hostedDomain != null &&
    profile.hostedDomain.toLowerCase() !== ALLOWED_EMAIL_DOMAIN
  ) {
    return { allowed: false, reason: "wrong-hosted-domain" };
  }
  return { allowed: true };
}

/**
 * The Credentials-based test provider may exist ONLY outside production
 * and ONLY when explicitly switched on. Pure so tests can pin the rule.
 */
export function isTestModeEnabled(env: {
  AUTH_TEST_MODE?: string;
  NODE_ENV?: string;
}): boolean {
  return env.AUTH_TEST_MODE === "1" && env.NODE_ENV !== "production";
}
