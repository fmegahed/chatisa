import { describe, expect, it } from "vitest";
import {
  evaluateSignIn,
  isTestModeEnabled,
  ALLOWED_EMAIL_DOMAIN,
} from "@/lib/auth/domain";

describe("evaluateSignIn (miamioh.edu policy)", () => {
  it("allows a verified miamioh.edu account", () => {
    expect(
      evaluateSignIn({ email: "student@miamioh.edu", emailVerified: true }),
    ).toEqual({ allowed: true });
  });

  it("is case-insensitive and trims whitespace", () => {
    expect(
      evaluateSignIn({ email: "  Student@MiamiOH.EDU ", emailVerified: true })
        .allowed,
    ).toBe(true);
  });

  it("rejects missing email", () => {
    expect(evaluateSignIn({ email: null, emailVerified: true }).reason).toBe(
      "missing-email",
    );
  });

  it("rejects unverified email", () => {
    expect(
      evaluateSignIn({ email: "student@miamioh.edu", emailVerified: false })
        .reason,
    ).toBe("email-not-verified");
    expect(
      evaluateSignIn({ email: "student@miamioh.edu", emailVerified: null })
        .reason,
    ).toBe("email-not-verified");
  });

  it.each([
    "student@gmail.com",
    "student@evil-miamioh.edu",
    "student@miamioh.edu.attacker.com",
    "student@sub.miamioh.edu",
    "@miamioh.edu",
  ])("rejects %s", (email) => {
    expect(evaluateSignIn({ email, emailVerified: true }).allowed).toBe(false);
  });

  it("rejects mismatched Google hosted domain even with a matching email", () => {
    expect(
      evaluateSignIn({
        email: "student@miamioh.edu",
        emailVerified: true,
        hostedDomain: "othercollege.edu",
      }).reason,
    ).toBe("wrong-hosted-domain");
  });

  it("accepts matching hosted domain in any case", () => {
    expect(
      evaluateSignIn({
        email: "student@miamioh.edu",
        emailVerified: true,
        hostedDomain: "MiamiOH.edu",
      }).allowed,
    ).toBe(true);
  });

  it("exports the expected domain constant", () => {
    expect(ALLOWED_EMAIL_DOMAIN).toBe("miamioh.edu");
  });
});

describe("isTestModeEnabled", () => {
  it("is on only with the explicit flag outside production", () => {
    expect(
      isTestModeEnabled({ AUTH_TEST_MODE: "1", NODE_ENV: "development" }),
    ).toBe(true);
    expect(
      isTestModeEnabled({ AUTH_TEST_MODE: "1", NODE_ENV: "test" }),
    ).toBe(true);
  });

  it("is NEVER on in production, even with the flag", () => {
    expect(
      isTestModeEnabled({ AUTH_TEST_MODE: "1", NODE_ENV: "production" }),
    ).toBe(false);
  });

  it("is off without the flag", () => {
    expect(isTestModeEnabled({ NODE_ENV: "development" })).toBe(false);
    expect(
      isTestModeEnabled({ AUTH_TEST_MODE: "0", NODE_ENV: "development" }),
    ).toBe(false);
  });
});
