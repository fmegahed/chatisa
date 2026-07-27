import { describe, expect, it } from "vitest";
import { buildProviders } from "@/lib/auth/providers";

function ids(providers: ReturnType<typeof buildProviders>): string[] {
  // A Credentials provider reports its TYPE id ("credentials") at the top
  // level; the custom id ("guest-pass", "test-login") lives in its options
  // until Auth.js initializes it.
  return providers.map((p) => {
    const options = (p as { options?: { id?: string } }).options;
    return options?.id ?? ("id" in p ? String(p.id) : "unknown");
  });
}

describe("buildProviders", () => {
  it("exposes Google and the guest-pass provider when test mode is off", () => {
    // guest-pass is present in production BY DESIGN: its gate is the hashed
    // invite list plus a required expiry (lib/auth/guest.ts), not NODE_ENV.
    // With no passes configured, every guest attempt fails closed.
    expect(ids(buildProviders({ testMode: false }))).toEqual([
      "google",
      "guest-pass",
    ]);
  });

  it("adds the test-login credentials provider only in test mode", () => {
    expect(ids(buildProviders({ testMode: true }))).toEqual([
      "google",
      "guest-pass",
      "test-login",
    ]);
    expect(ids(buildProviders({ testMode: false }))).not.toContain(
      "test-login",
    );
  });
});
