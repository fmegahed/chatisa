// tests/unit/project-coaches.test.ts
import { describe, expect, it } from "vitest";
import { COACHES, isCoachType, coachLabel } from "@/lib/project/coaches";

describe("coach registry", () => {
  it("defines exactly the five coaches in a stable order", () => {
    expect(COACHES.map((c) => c.type)).toEqual([
      "scoping",
      "premortem",
      "team_structuring",
      "devils_advocate",
      "reflection",
    ]);
  });

  it("gives every coach a label and a blurb with no em dash", () => {
    for (const c of COACHES) {
      expect(c.label.length).toBeGreaterThan(0);
      expect(c.blurb.length).toBeGreaterThan(0);
      expect(c.label).not.toContain("—");
      expect(c.blurb).not.toContain("—");
    }
  });

  it("narrows valid coach types", () => {
    expect(isCoachType("scoping")).toBe(true);
    expect(isCoachType("banana")).toBe(false);
    expect(coachLabel("premortem")).toBe("Premortem");
  });
});
