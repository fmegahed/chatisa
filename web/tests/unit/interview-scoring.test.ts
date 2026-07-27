import { describe, expect, it } from "vitest";
import {
  answerRatio,
  bandFor,
  bandForAnswer,
  isEmptyAnswer,
  normaliseVerdicts,
  rollUpInterview,
  type CriterionResult,
  type Verdict,
} from "@/lib/interview/scoring";
import { INTERVIEW_CRITERIA } from "@/lib/prompts/interview-mentor";

function results(...verdicts: Verdict[]): CriterionResult[] {
  return INTERVIEW_CRITERIA.map((c, i) => ({
    id: c.id,
    label: c.label,
    verdict: verdicts[i] ?? "not_met",
  }));
}

const ALL_MET = results("met", "met", "met", "met");
const ALL_MISSED = results("not_met", "not_met", "not_met", "not_met");

describe("verdict normalisation", () => {
  it("keeps the server's rubric, not the model's", () => {
    const normalised = normaliseVerdicts([
      { id: "answered_the_question", verdict: "met" },
      { id: "a_criterion_the_model_invented", verdict: "met" },
    ]);
    expect(normalised).toHaveLength(INTERVIEW_CRITERIA.length);
    expect(normalised.map((r) => r.id)).toEqual(
      INTERVIEW_CRITERIA.map((c) => c.id),
    );
    expect(normalised.find((r) => r.id === "answered_the_question")?.verdict).toBe(
      "met",
    );
  });

  it("counts anything the model omitted as not met", () => {
    // Otherwise a model that returns half a response quietly improves the
    // student's result by saying less.
    const normalised = normaliseVerdicts([
      { id: "answered_the_question", verdict: "met" },
    ]);
    expect(normalised.filter((r) => r.verdict === "not_met")).toHaveLength(
      INTERVIEW_CRITERIA.length - 1,
    );
  });

  it("rejects verdict values outside the fixed set", () => {
    const normalised = normaliseVerdicts([
      { id: "structure", verdict: "excellent" },
      { id: "reasoning", verdict: "10/10" },
    ]);
    expect(normalised.every((r) => r.verdict === "not_met")).toBe(true);
  });

  it("survives null, empty, and malformed input", () => {
    for (const input of [null, undefined, [], [{}], [{ verdict: "met" }]]) {
      const normalised = normaliseVerdicts(
        input as Array<{ id?: string; verdict?: string }>,
      );
      expect(normalised).toHaveLength(INTERVIEW_CRITERIA.length);
      expect(normalised.every((r) => r.verdict === "not_met")).toBe(true);
    }
  });
});

describe("per-answer bands", () => {
  it("scores a fully met answer as strong and an empty one as needs work", () => {
    expect(bandForAnswer(ALL_MET)).toBe("strong");
    expect(bandForAnswer(ALL_MISSED)).toBe("needs work");
  });

  it("treats partly as half credit rather than nearly full", () => {
    const allPartly = results("partly", "partly", "partly", "partly");
    expect(answerRatio(allPartly)).toBeCloseTo(0.5, 5);
    expect(bandForAnswer(allPartly)).toBe("developing");
  });

  it("uses the same thresholds as the rest of the app", () => {
    expect(bandFor(0.8)).toBe("strong");
    expect(bandFor(0.79)).toBe("developing");
    expect(bandFor(0.5)).toBe("developing");
    expect(bandFor(0.49)).toBe("needs work");
  });
});

describe("whole-interview rollup", () => {
  it("does not score a skipped question as a failure", () => {
    // A student who ran out of time has not demonstrated weakness, and folding
    // that in would make the report say something untrue.
    const rollup = rollUpInterview([ALL_MET, ALL_MET, null]);
    expect(rollup.answeredCount).toBe(2);
    expect(rollup.skippedCount).toBe(1);
    expect(rollup.overallBand).toBe("strong");
  });

  it("reports no band at all when nothing was answered", () => {
    const rollup = rollUpInterview([null, null]);
    expect(rollup.overallBand).toBeNull();
    expect(rollup.answeredCount).toBe(0);
  });

  it("identifies the criteria the student was actually weakest on", () => {
    // Strong on answering the question, weak on evidence throughout.
    const weakEvidence = results("met", "not_met", "met", "met");
    const rollup = rollUpInterview([weakEvidence, weakEvidence, weakEvidence]);

    expect(rollup.weakest[0].id).toBe("specific_evidence");
    expect(rollup.weakest[0].band).toBe("needs work");
    expect(rollup.weakest.map((c) => c.id)).not.toContain(
      "answered_the_question",
    );
  });

  it("counts each verdict per criterion so the report can show detail", () => {
    const rollup = rollUpInterview([
      results("met", "met", "met", "met"),
      results("partly", "not_met", "met", "met"),
    ]);
    const answered = rollup.byCriterion.find(
      (c) => c.id === "answered_the_question",
    )!;
    expect(answered.met).toBe(1);
    expect(answered.partly).toBe(1);
    expect(answered.notMet).toBe(0);

    const evidence = rollup.byCriterion.find(
      (c) => c.id === "specific_evidence",
    )!;
    expect(evidence.met).toBe(1);
    expect(evidence.notMet).toBe(1);
  });

  it("never reports a percentage anywhere in the rollup", () => {
    // ADR-016: bands and per-criterion detail, never a percentage, because a
    // percentage claims a precision this cannot support.
    const rollup = rollUpInterview([ALL_MET, ALL_MISSED]);
    const serialised = JSON.stringify(rollup);
    expect(serialised).not.toMatch(/percent|"score"|"total"|%/i);
  });
});

describe("empty answers", () => {
  it("recognises answers with nothing in them", () => {
    expect(isEmptyAnswer("")).toBe(true);
    expect(isEmptyAnswer("   ")).toBe(true);
    expect(isEmptyAnswer(null)).toBe(true);
    expect(isEmptyAnswer(undefined)).toBe(true);
    expect(isEmptyAnswer("No.")).toBe(false);
  });
});
