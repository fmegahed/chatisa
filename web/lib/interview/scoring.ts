import {
  INTERVIEW_CRITERIA,
  type CriterionId,
} from "@/lib/prompts/interview-mentor";

/**
 * Interview scoring.
 *
 * Every number here is computed from stored judgements by ordinary arithmetic.
 * The model is never asked for a score, a percentage, or a total. The legacy
 * module asked the model to compute "(sum of points / (3 * number of
 * questions)) * 100, rounded" partway through a long spoken conversation while
 * also being told to be encouraging, then showed the result to the student as
 * their interview score. That number measured nothing.
 *
 * Following ADR-016, the student sees bands and per-criterion detail rather
 * than a percentage, because a band is an honest summary of what this can
 * actually judge.
 */

export type Verdict = "met" | "partly" | "not_met";

/** How much credit each verdict earns. Partly is worth half, not nearly all. */
const CREDIT: Record<Verdict, number> = {
  met: 1,
  partly: 0.5,
  not_met: 0,
};

export type Band = "strong" | "developing" | "needs work";

export interface CriterionResult {
  id: CriterionId;
  label: string;
  verdict: Verdict;
}

/** Same thresholds as Exam Ally, so a band means the same thing everywhere. */
export function bandFor(ratio: number): Band {
  if (ratio >= 0.8) return "strong";
  if (ratio >= 0.5) return "developing";
  return "needs work";
}

/** Fraction of criteria met across one answer, in the rubric's own order. */
export function answerRatio(results: CriterionResult[]): number {
  if (results.length === 0) return 0;
  const earned = results.reduce((sum, r) => sum + CREDIT[r.verdict], 0);
  return earned / results.length;
}

export function bandForAnswer(results: CriterionResult[]): Band {
  return bandFor(answerRatio(results));
}

/**
 * Normalises whatever the model returned into the fixed rubric.
 *
 * The rubric is the server's, not the model's. Anything the model invents is
 * dropped and anything it omits counts as not met, so a model that returns
 * half a response cannot quietly raise a student's result by saying less.
 */
export function normaliseVerdicts(
  raw: Array<{ id?: string; verdict?: string }> | null | undefined,
): CriterionResult[] {
  const byId = new Map<string, Verdict>();
  for (const item of raw ?? []) {
    if (!item?.id) continue;
    const verdict = item.verdict;
    if (verdict === "met" || verdict === "partly" || verdict === "not_met") {
      byId.set(item.id, verdict);
    }
  }
  return INTERVIEW_CRITERIA.map((criterion) => ({
    id: criterion.id,
    label: criterion.label,
    verdict: byId.get(criterion.id) ?? "not_met",
  }));
}

export interface CriterionRollup {
  id: CriterionId;
  label: string;
  /** Answers where this was fully met. */
  met: number;
  partly: number;
  notMet: number;
  band: Band;
}

export interface InterviewRollup {
  answeredCount: number;
  skippedCount: number;
  overallBand: Band | null;
  byCriterion: CriterionRollup[];
  /** Criteria the student was weakest on, worst first. Drives the study advice. */
  weakest: CriterionRollup[];
}

/**
 * Aggregates a whole interview.
 *
 * Skipped answers are counted and excluded from the bands rather than scored
 * as zero. A student who ran out of time on the last question has not
 * demonstrated weakness there, and folding that into their result would make
 * the report say something untrue.
 */
export function rollUpInterview(
  perAnswer: Array<CriterionResult[] | null>,
): InterviewRollup {
  const answered = perAnswer.filter(
    (r): r is CriterionResult[] => r !== null && r.length > 0,
  );
  const skippedCount = perAnswer.length - answered.length;

  const byCriterion: CriterionRollup[] = INTERVIEW_CRITERIA.map(
    (criterion, index) => {
      let met = 0;
      let partly = 0;
      let notMet = 0;
      for (const results of answered) {
        const verdict = results[index]?.verdict ?? "not_met";
        if (verdict === "met") met++;
        else if (verdict === "partly") partly++;
        else notMet++;
      }
      const total = answered.length || 1;
      const ratio = (met + partly * CREDIT.partly) / total;
      return {
        id: criterion.id,
        label: criterion.label,
        met,
        partly,
        notMet,
        band: bandFor(ratio),
      };
    },
  );

  const overallRatio =
    answered.length === 0
      ? 0
      : answered.reduce((sum, r) => sum + answerRatio(r), 0) / answered.length;

  const weakest = [...byCriterion]
    .sort((a, b) => {
      const score = (c: CriterionRollup) =>
        (c.met + c.partly * CREDIT.partly) / (answered.length || 1);
      return score(a) - score(b);
    })
    .filter((c) => c.band !== "strong");

  return {
    answeredCount: answered.length,
    skippedCount,
    overallBand: answered.length === 0 ? null : bandFor(overallRatio),
    byCriterion,
    weakest,
  };
}

/** True when an answer is too empty to judge. */
export function isEmptyAnswer(text: string | null | undefined): boolean {
  return (text ?? "").trim().length < 2;
}
