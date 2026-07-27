import { describe, expect, it } from "vitest";
import {
  evaluateGuestPass,
  guestPassesEnabled,
  hashGuestPass,
  parseGuestConfig,
} from "@/lib/auth/guest";

const PASS_1 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const PASS_2 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const ENV = {
  CHATISA_GUEST_PASS_HASHES: `${hashGuestPass(PASS_1)},${hashGuestPass(PASS_2)}`,
  CHATISA_GUEST_EXPIRES: "2026-09-30",
};
const BEFORE = Date.parse("2026-08-01");
const AFTER = Date.parse("2026-10-01");

describe("guest pass policy", () => {
  it("admits a known pass with a stable positional identity", () => {
    const d1 = evaluateGuestPass(PASS_1, ENV, BEFORE);
    const d2 = evaluateGuestPass(` ${PASS_2} `, ENV, BEFORE); // trimmed
    expect(d1).toMatchObject({
      allowed: true,
      guestNumber: 1,
      email: "guest-1@guest.chatisa",
      name: "Guest 1",
    });
    expect(d2).toMatchObject({ allowed: true, guestNumber: 2 });
  });

  it("rejects unknown, malformed, expired, and disabled cases", () => {
    expect(evaluateGuestPass("c".repeat(32), ENV, BEFORE)).toMatchObject({
      allowed: false,
      reason: "unknown-pass",
    });
    expect(evaluateGuestPass("short", ENV, BEFORE)).toMatchObject({
      reason: "malformed",
    });
    expect(evaluateGuestPass(PASS_1, ENV, AFTER)).toMatchObject({
      reason: "expired",
    });
    expect(evaluateGuestPass(PASS_1, {}, BEFORE)).toMatchObject({
      reason: "disabled",
    });
    // An expiry date is REQUIRED: hashes alone must not open the door.
    expect(
      evaluateGuestPass(
        PASS_1,
        { CHATISA_GUEST_PASS_HASHES: ENV.CHATISA_GUEST_PASS_HASHES },
        BEFORE,
      ),
    ).toMatchObject({ reason: "disabled" });
  });

  it("revoking one hash by zeroing keeps other positions stable", () => {
    const env = {
      ...ENV,
      CHATISA_GUEST_PASS_HASHES: `${"0".repeat(64)},${hashGuestPass(PASS_2)}`,
    };
    expect(evaluateGuestPass(PASS_1, env, BEFORE)).toMatchObject({
      reason: "unknown-pass",
    });
    expect(evaluateGuestPass(PASS_2, env, BEFORE)).toMatchObject({
      allowed: true,
      guestNumber: 2,
    });
  });

  it("parses config defensively and reports enablement honestly", () => {
    expect(
      parseGuestConfig({ CHATISA_GUEST_PASS_HASHES: "junk,,ZZZ" }).hashes,
    ).toEqual([]);
    expect(guestPassesEnabled(ENV, BEFORE)).toBe(true);
    expect(guestPassesEnabled(ENV, AFTER)).toBe(false);
    expect(guestPassesEnabled({}, BEFORE)).toBe(false);
  });
});
