import { beforeEach, describe, expect, it } from "vitest";
import { checkRateLimit, resetRateLimits } from "@/lib/ratelimit";

describe("checkRateLimit", () => {
  beforeEach(() => resetRateLimits());

  it("allows requests up to the limit, then blocks", () => {
    const opts = { limit: 3, windowMs: 60_000 };
    const now = 1_000_000;
    expect(checkRateLimit("u1", opts, now).allowed).toBe(true);
    expect(checkRateLimit("u1", opts, now).allowed).toBe(true);
    const third = checkRateLimit("u1", opts, now);
    expect(third.allowed).toBe(true);
    expect(third.remaining).toBe(0);

    const blocked = checkRateLimit("u1", opts, now);
    expect(blocked.allowed).toBe(false);
    expect(blocked.retryAfterSeconds).toBeGreaterThan(0);
  });

  it("keeps separate counters per key", () => {
    const opts = { limit: 1, windowMs: 60_000 };
    const now = 2_000_000;
    expect(checkRateLimit("a@miamioh.edu", opts, now).allowed).toBe(true);
    expect(checkRateLimit("a@miamioh.edu", opts, now).allowed).toBe(false);
    expect(checkRateLimit("b@miamioh.edu", opts, now).allowed).toBe(true);
  });

  it("resets after the window elapses", () => {
    const opts = { limit: 1, windowMs: 1_000 };
    const start = 3_000_000;
    expect(checkRateLimit("u2", opts, start).allowed).toBe(true);
    expect(checkRateLimit("u2", opts, start + 500).allowed).toBe(false);
    expect(checkRateLimit("u2", opts, start + 1_001).allowed).toBe(true);
  });
});
