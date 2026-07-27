// tests/unit/coach-specs.test.ts
import { describe, expect, it } from "vitest";
import { COACH_SPECS, getCoachSpec } from "@/lib/project/coach-specs";
import { buildEmptyContent, coachContentSchema } from "@/lib/project/coach-framework";

describe("coach specs", () => {
  it("defines the four generic coaches (not scoping)", () => {
    expect(Object.keys(COACH_SPECS).sort()).toEqual([
      "devils_advocate",
      "premortem",
      "reflection",
      "team_structuring",
    ]);
    expect(getCoachSpec("scoping")).toBeUndefined();
  });

  it("matches the design schema for each coach", () => {
    expect(COACH_SPECS.premortem.fields.map((f) => f.key)).toEqual(["projectDescription"]);
    expect(COACH_SPECS.premortem.tables[0].columns.map((c) => c.key)).toEqual(["failure", "howToAvoid"]);
    expect(COACH_SPECS.team_structuring.fields).toHaveLength(0);
    expect(COACH_SPECS.team_structuring.tables[0].columns.map((c) => c.key)).toEqual(["name", "skills", "possibleTask"]);
    expect(COACH_SPECS.devils_advocate.fields.map((f) => f.key)).toEqual(["decision", "alternatives", "risks", "mitigations"]);
    expect(COACH_SPECS.reflection.fields.map((f) => f.key)).toEqual(["challenges", "insights", "growth"]);
  });

  it("every spec has a prompt with no em dash and a valid empty content", () => {
    for (const spec of Object.values(COACH_SPECS)) {
      expect(spec.systemPrompt).not.toContain("—");
      expect(spec.systemPrompt.length).toBeGreaterThan(50);
      expect(coachContentSchema(spec).safeParse(buildEmptyContent(spec)).success).toBe(true);
    }
  });
});
