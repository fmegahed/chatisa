import { describe, expect, it } from "vitest";
import { validateEnv } from "@/lib/config/env";

describe("validateEnv", () => {
  it("accepts an empty environment (all provider keys optional at boot)", () => {
    const { env, report } = validateEnv({} as NodeJS.ProcessEnv);
    expect(env).not.toBeNull();
    expect(report.ok).toBe(true);
    expect(report.missingProviders).toContain("OPENAI_API_KEY");
    // Five provider credentials: four model providers after the 2026-07-21
    // refresh (Cohere and Groq direct are gone), plus Deepgram for speech.
    expect(report.missingProviders).toHaveLength(5);
  });

  it("reports present provider keys as not missing", () => {
    const { report } = validateEnv({
      OPENAI_API_KEY: "test-key-value",
      GROQ_API_KEY: "test-key-value",
    } as unknown as NodeJS.ProcessEnv);
    expect(report.ok).toBe(true);
    expect(report.missingProviders).not.toContain("OPENAI_API_KEY");
    expect(report.missingProviders).not.toContain("GROQ_API_KEY");
    expect(report.missingProviders).toContain("ANTHROPIC_API_KEY");
  });

  it("treats blank values as unset, so a half-filled .env still boots", () => {
    // .env.example ships every key blank; copying it must not stop the server.
    const { env, report } = validateEnv({
      OPENAI_API_KEY: "real-value",
      ANTHROPIC_API_KEY: "",
      GOOGLE_API_KEY: "   ",
      COHERE_API_KEY: "",
      GROQ_API_KEY: "",
      HF_TOKEN: "",
      AUTH_SECRET: "",
      AUTH_URL: "",
    } as unknown as NodeJS.ProcessEnv);
    expect(env).not.toBeNull();
    expect(report.ok).toBe(true);
    expect(report.missingProviders).toContain("HF_TOKEN");
    expect(report.missingProviders).toContain("ANTHROPIC_API_KEY");
    expect(report.missingProviders).not.toContain("OPENAI_API_KEY");
    expect(report.authConfigured).toBe(false);
  });

  it("still rejects a blank required auth variable in production", () => {
    const { report } = validateEnv({
      NODE_ENV: "production",
      AUTH_SECRET: "",
      AUTH_GOOGLE_ID: "id",
      AUTH_GOOGLE_SECRET: "secret",
      AUTH_URL: "https://chatisa.fsb.miamioh.edu",
    } as unknown as NodeJS.ProcessEnv);
    expect(report.ok).toBe(false);
    expect(report.invalid).toContain("AUTH_SECRET");
  });

  it("requires all auth variables in production", () => {
    const { env, report } = validateEnv({
      NODE_ENV: "production",
    } as unknown as NodeJS.ProcessEnv);
    expect(env).toBeNull();
    expect(report.ok).toBe(false);
    for (const key of [
      "AUTH_SECRET",
      "AUTH_GOOGLE_ID",
      "AUTH_GOOGLE_SECRET",
      "AUTH_URL",
    ]) {
      expect(report.invalid).toContain(key);
    }
  });

  it("refuses AUTH_TEST_MODE=1 in production", () => {
    const { report } = validateEnv({
      NODE_ENV: "production",
      AUTH_SECRET: "0123456789abcdef0123456789abcdef",
      AUTH_GOOGLE_ID: "id",
      AUTH_GOOGLE_SECRET: "secret",
      AUTH_URL: "https://chatisa.fsb.miamioh.edu",
      AUTH_TEST_MODE: "1",
    } as unknown as NodeJS.ProcessEnv);
    expect(report.ok).toBe(false);
    expect(report.invalid).toContain("AUTH_TEST_MODE");
  });

  it("accepts a complete production configuration and reports authConfigured", () => {
    const { report } = validateEnv({
      NODE_ENV: "production",
      AUTH_SECRET: "0123456789abcdef0123456789abcdef",
      AUTH_GOOGLE_ID: "id",
      AUTH_GOOGLE_SECRET: "secret",
      AUTH_URL: "https://chatisa.fsb.miamioh.edu",
    } as unknown as NodeJS.ProcessEnv);
    expect(report.ok).toBe(true);
    expect(report.authConfigured).toBe(true);
  });

  it("rejects malformed values and names only the variable", () => {
    const { env, report } = validateEnv({
      NODE_ENV: "not-a-real-env",
      AUTH_SECRET: "too-short",
    } as unknown as NodeJS.ProcessEnv);
    expect(env).toBeNull();
    expect(report.ok).toBe(false);
    expect(report.invalid).toContain("NODE_ENV");
    expect(report.invalid).toContain("AUTH_SECRET");
    // Never leaks values:
    expect(JSON.stringify(report)).not.toContain("too-short");
  });
});
