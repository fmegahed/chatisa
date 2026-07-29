import { describe, expect, it } from "vitest";
import { isHarvestDue, lastDueLabel } from "@/lib/scout/scheduler";

/**
 * DST is the whole reason this logic exists, so the pins are the 2026
 * transition weekends: spring forward 2026-03-08 (2 AM Eastern does not
 * exist) and fall back 2026-11-01 (1 AM happens twice).
 */

// 2026-07-26 was a Sunday; EDT is UTC-4, so 06:05Z = 2:05 AM Eastern.
const SUN_JUL26_0205_ET = "2026-07-26T06:05:00.000Z";
const SUN_JUL19_0205_ET = "2026-07-19T06:05:00.000Z";

describe("lastDueLabel", () => {
  it("labels the current Sunday once 2 AM Eastern has passed", () => {
    expect(lastDueLabel(new Date(SUN_JUL26_0205_ET))).toBe("2026-07-26");
  });

  it("labels the previous Sunday before 2 AM Eastern", () => {
    // Sunday 1:00 AM EDT.
    expect(lastDueLabel(new Date("2026-07-26T05:00:00.000Z"))).toBe(
      "2026-07-19",
    );
  });

  it("still produces a label on the spring-forward Sunday with no 2 AM", () => {
    // 2026-03-08 12:00Z = 8:00 AM EDT (EDT began at 2 AM that morning).
    expect(lastDueLabel(new Date("2026-03-08T12:00:00.000Z"))).toBe(
      "2026-03-08",
    );
  });

  it("handles the fall-back Sunday", () => {
    // 2026-11-01 12:00Z = 7:00 AM EST (EST resumed at 2 AM that morning).
    expect(lastDueLabel(new Date("2026-11-01T12:00:00.000Z"))).toBe(
      "2026-11-01",
    );
  });
});

describe("isHarvestDue", () => {
  const midweek = new Date("2026-07-29T16:00:00.000Z");

  it("is due when no harvest has ever succeeded", () => {
    expect(isHarvestDue(midweek, null)).toBe(true);
  });

  it("is not due when this week's boundary was already served", () => {
    expect(isHarvestDue(midweek, SUN_JUL26_0205_ET)).toBe(false);
  });

  it("catches up when the server slept through Sunday 2 AM", () => {
    expect(isHarvestDue(midweek, SUN_JUL19_0205_ET)).toBe(true);
  });

  it("is not due on Sunday 1 AM even with a week-old success", () => {
    expect(
      isHarvestDue(new Date("2026-07-26T05:00:00.000Z"), SUN_JUL19_0205_ET),
    ).toBe(false);
  });

  it("becomes due right after 2 AM Sunday", () => {
    expect(isHarvestDue(new Date(SUN_JUL26_0205_ET), SUN_JUL19_0205_ET)).toBe(
      true,
    );
  });

  it("runs on the spring-forward Sunday despite the missing hour", () => {
    expect(
      isHarvestDue(
        new Date("2026-03-08T12:00:00.000Z"),
        "2026-03-01T07:05:00.000Z",
      ),
    ).toBe(true);
  });

  it("treats a future success timestamp as clock skew, not work to do", () => {
    expect(isHarvestDue(midweek, "2026-08-30T06:05:00.000Z")).toBe(false);
  });
});
