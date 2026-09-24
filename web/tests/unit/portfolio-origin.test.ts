import { describe, expect, it } from "vitest";
import {
  ORIGINS, guestOriginPatch, isGuestEmail, originLabel, originOf, originPromptLine, originReadmeLine,
} from "@/lib/portfolio/origin";
import { GUEST_EMAIL_DOMAIN } from "@/lib/auth/guest";

describe("project origin", () => {
  it("treats anything missing or unknown as a Miami course, so older drafts keep their meaning", () => {
    expect(originOf(undefined)).toBe("miami");
    expect(originOf(null)).toBe("miami");
    expect(originOf("bogus")).toBe("miami");
    for (const o of ORIGINS) expect(originOf(o)).toBe(o);
  });

  it("labels the page header for each origin", () => {
    expect(originLabel("miami", "ISA 225")).toBe("ISA 225 - Principles of Business Analytics");
    expect(originLabel("miami", "ZZZ 999")).toBe("ZZZ 999");
    expect(originLabel("other", "  STAT 4520, Ohio State ")).toBe("STAT 4520, Ohio State");
    expect(originLabel("self", "leftover text")).toBe("Self-study project");
    expect(originLabel("personal", "leftover text")).toBe("Personal project");
  });

  it("has no label for a course origin with no course typed", () => {
    expect(originLabel("miami", "")).toBe("");
    expect(originLabel("other", "   ")).toBe("");
  });

  it("writes the README line for each origin", () => {
    expect(originReadmeLine("miami", "ISA 444")).toBe("Built for ISA 444.");
    expect(originReadmeLine("other", "STAT 4520, Ohio State")).toBe("Built for STAT 4520, Ohio State.");
    expect(originReadmeLine("self", "x")).toBe("A self-study project.");
    expect(originReadmeLine("personal", "x")).toBe("A personal project.");
    expect(originReadmeLine("other", "")).toBe("");
  });

  it("tells the model where the project came from", () => {
    expect(originPromptLine("miami", "ISA 444")).toBe("Course: ISA 444");
    expect(originPromptLine("other", "STAT 4520, Ohio State")).toBe("Course (another school): STAT 4520, Ohio State");
    expect(originPromptLine("self", "")).toBe("A self-study project.");
    expect(originPromptLine("personal", "")).toBe("A personal project.");
  });
});

describe("guestOriginPatch (v6.6.1)", () => {
  it("moves a guest off the Miami option without losing a course they already chose", () => {
    // A guest's autosave from before v6.6.0 has no origin (so Miami) and may
    // hold a course picked back then; it becomes their "another school" text.
    expect(guestOriginPatch("miami", "ISA 444")).toEqual({ origin: "other", course: "ISA 444" });
    expect(guestOriginPatch("miami", "")).toEqual({ origin: "other", course: "" });
  });

  it("leaves every other origin alone", () => {
    for (const o of ["other", "self", "personal"] as const) expect(guestOriginPatch(o, "x")).toBeNull();
  });
});

describe("isGuestEmail", () => {
  it("recognises guest-pass identities only", () => {
    expect(isGuestEmail("guest-3@guest.chatisa")).toBe(true);
    expect(isGuestEmail("GUEST-3@Guest.Chatisa")).toBe(true);
    expect(isGuestEmail("student@miamioh.edu")).toBe(false);
    expect(isGuestEmail("someone@guest.chatisa.evil.com")).toBe(false);
    expect(isGuestEmail(null)).toBe(false);
    expect(isGuestEmail(undefined)).toBe(false);
  });

  it("uses the same domain the guest provider mints", () => {
    expect(isGuestEmail(`guest-1@${GUEST_EMAIL_DOMAIN}`)).toBe(true);
  });
});
