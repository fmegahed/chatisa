/**
 * Worked-example pins for the matching math (design §2.3/§2.4). The numbers
 * here were computed by hand; if a weight changes in matching.ts these fail
 * on purpose so the change is a decision, not an accident.
 */
import { describe, expect, it } from "vitest";
import { profileStrengths, scoreJob } from "@/lib/scout/matching";

describe("profileStrengths (noisy-OR, credit-scaled)", () => {
  it("a 1.5-credit anchor contributes half depth: ISA 241 sql = 0.5", () => {
    const s = profileStrengths(["ISA 241"], []);
    expect(s.get("sql")).toBeCloseTo(0.5, 10);
    expect(s.get("database_design")).toBeCloseTo(0.5, 10);
    // applied 0.6 x 1.5/3
    expect(s.get("data_wrangling")).toBeCloseTo(0.3, 10);
  });

  it("a 3-credit anchor saturates to 1.0", () => {
    const s = profileStrengths(["ISA 391"], []);
    expect(s.get("regression")).toBe(1);
  });

  it("two applied 3-credit courses compose with diminishing returns: 0.84", () => {
    // data_wrangling is applied (0.6) in both ISA 345 and ISA 381.
    const s = profileStrengths(["ISA 345", "ISA 381"], []);
    expect(s.get("data_wrangling")).toBeCloseTo(1 - 0.4 * 0.4, 10);
  });

  it("resolves cross-listed alt codes: ISA 501 counts as ISA 401", () => {
    const s = profileStrengths(["ISA 501"], []);
    expect(s.get("business_intelligence")).toBe(1);
  });

  it("extras weigh like a 3-credit course at their level", () => {
    const s = profileStrengths([], [{ skillId: "tableau", level: "applied" }]);
    expect(s.get("tableau")).toBeCloseTo(0.6, 10);
  });

  it("student overrides beat the computation in both directions", () => {
    // ISA 391 makes regression 1.0; the student says Working.
    const down = profileStrengths(["ISA 391"], [], [
      { skillId: "regression", level: "working" },
    ]);
    expect(down.get("regression")).toBeCloseTo(0.6, 10);
    // ISA 241's half-credit sql (0.5); the student says Strong.
    const up = profileStrengths(["ISA 241"], [], [
      { skillId: "sql", level: "strong" },
    ]);
    expect(up.get("sql")).toBe(1);
    // An override on a skill nothing else contributes still creates it.
    const created = profileStrengths([], [], [
      { skillId: "tableau", level: "introduced" },
    ]);
    expect(created.get("tableau")).toBeCloseTo(0.25, 10);
    // Unknown skill ids in overrides are ignored.
    expect(
      profileStrengths([], [], [{ skillId: "nope", level: "strong" }]).size,
    ).toBe(0);
  });

  it("ignores unknown courses and unknown extra skills", () => {
    const s = profileStrengths(
      ["ISA 999"],
      [{ skillId: "not_a_skill", level: "anchor" }],
    );
    expect(s.size).toBe(0);
  });
});

describe("ranking shrinks thin tags", () => {
  it("a one-skill 1/1 never outranks a broad near-complete match", () => {
    const strengths = new Map([
      ["sql", 1.0], ["data_visualization", 1.0], ["python", 1.0],
      ["excel", 1.0], ["statistics", 1.0], ["power_bi", 1.0],
    ]);
    const thin = scoreJob(strengths, [
      { skillId: "sql", importance: "required" },
    ]);
    const broad = scoreJob(strengths, [
      { skillId: "sql", importance: "required" },
      { skillId: "python", importance: "required" },
      { skillId: "excel", importance: "required" },
      { skillId: "data_visualization", importance: "required" },
      { skillId: "statistics", importance: "preferred" },
      { skillId: "power_bi", importance: "preferred" },
    ]);
    // The thin posting still DISPLAYS a perfect score and 1/1 coverage...
    expect(thin.score).toBe(1);
    expect(thin.coveredRequired).toBe(1);
    // ...but ranks below the posting with real evidence (the GIS
    // professorship problem, video review 2026-07-29).
    expect(broad.rank).toBeGreaterThan(thin.rank);
  });
});

describe("scoreJob (requirement coverage)", () => {
  it("computes the hand-worked example", () => {
    const strengths = new Map([
      ["sql", 1.0],
      ["tableau", 0.8],
    ]);
    const match = scoreJob(strengths, [
      { skillId: "sql", importance: "required" },
      { skillId: "python", importance: "required" },
      { skillId: "data_visualization", importance: "preferred" },
    ]);
    // (1x1.0 + 1x0 + 0.5x(0.8x0.6)) / 2.5 = 1.24/2.5
    expect(match.score).toBeCloseTo(0.496, 10);
    expect(match.band).toBe("good");
    expect(match.coveredRequired).toBe(1);
    expect(match.totalRequired).toBe(2);
    // tableau implies data_visualization but 0.48 < 0.5 stays a gap,
    // and required gaps sort before preferred ones.
    expect(match.gaps.map((g) => g.skillId)).toEqual([
      "python",
      "data_visualization",
    ]);
    const viz = match.gaps.find((g) => g.skillId === "data_visualization");
    expect(viz?.via).toBe("tableau");
  });

  it("band edges are inclusive: 0.70 strong, 0.45 good", () => {
    const exactly = (target: number) =>
      scoreJob(new Map([["sql", target]]), [
        { skillId: "sql", importance: "required" },
      ]).band;
    expect(exactly(0.7)).toBe("strong");
    expect(exactly(0.699)).toBe("good");
    expect(exactly(0.45)).toBe("good");
    expect(exactly(0.449)).toBe("stretch");
  });

  it("implies credit flows both directions", () => {
    // Student has the general skill, job wants the specific tool.
    const general = scoreJob(new Map([["data_visualization", 1.0]]), [
      { skillId: "tableau", importance: "required" },
    ]);
    expect(general.matched[0]?.strength).toBeCloseTo(0.6, 10);
    expect(general.matched[0]?.via).toBe("data_visualization");
  });

  it("demotes professional skills to preferred and never lists them as gaps", () => {
    // Live card 2026-07-29: "Gaps: Teamwork, Problem Solving" was noise.
    const match = scoreJob(new Map([["cybersecurity", 1.0]]), [
      { skillId: "cybersecurity", importance: "required" },
      { skillId: "teamwork", importance: "required" },
      { skillId: "problem_solving", importance: "required" },
    ]);
    // Only the demonstrable skill counts as required...
    expect(match.totalRequired).toBe(1);
    expect(match.coveredRequired).toBe(1);
    // ...professional ones weigh in at preferred (0.5 each, uncovered):
    // (1x1.0 + 0.5x0 + 0.5x0) / 2.0 = 0.5.
    expect(match.score).toBeCloseTo(0.5, 10);
    expect(match.gaps).toEqual([]);
  });

  it("still shows covered professional skills in the matched list", () => {
    const match = scoreJob(new Map([["communication", 0.9]]), [
      { skillId: "communication", importance: "required" },
    ]);
    expect(match.matched.map((m) => m.skillId)).toEqual(["communication"]);
    expect(match.totalRequired).toBe(0);
  });

  it("a job with no tagged skills scores zero without dividing by zero", () => {
    const match = scoreJob(new Map(), []);
    expect(match.score).toBe(0);
    expect(match.totalRequired).toBe(0);
    expect(match.band).toBe("stretch");
  });
});
