import {
  bandForAnswer,
  rollUpInterview,
  type CriterionResult,
  type InterviewRollup,
} from "@/lib/interview/scoring";

/**
 * The single place interview state becomes browser-visible.
 *
 * Everything the student sees goes through here, so there is one file to check
 * when asking "can the browser see X". Two things are deliberately withheld
 * while the interview is running: the judgement of an answer, and the closing
 * summary. Showing a student that their last answer scored badly, mid
 * interview, would change how they answer the rest and make the practice less
 * like the real thing. It is all released the moment they finish.
 */

export interface PublicTurn {
  ordinal: number;
  question: string;
  topic: string | null;
  answerText: string | null;
  answerSource: string | null;
  answered: boolean;
  /** Present only once the interview is complete. */
  criteria?: CriterionResult[];
  band?: string;
  strength?: string | null;
  improvement?: string | null;
}

export interface PublicInterview {
  id: string;
  status: string;
  jobTitle: string;
  interviewType: string;
  plannedQuestions: number;
  askedCount: number;
  answeredCount: number;
  /** True when background material was summarised for the interviewer. */
  hasBrief: boolean;
  createdAt: string;
  completedAt: string | null;
  turns: PublicTurn[];
  results?: {
    rollup: InterviewRollup;
    didWell: string[];
    workOn: string[];
    overall: string;
  };
}

interface TurnRow {
  ordinal: number;
  question: string;
  topic: string | null;
  answerText: string | null;
  answerSource: string | null;
  answeredAt: string | null;
  criteriaJson: string | null;
  strength: string | null;
  improvement: string | null;
}

interface InterviewRow {
  id: string;
  status: string;
  jobTitle: string;
  interviewType: string;
  plannedQuestions: number;
  askedCount: number;
  roleBrief: string | null;
  candidateBrief: string | null;
  summaryJson: string | null;
  createdAt: string;
  completedAt: string | null;
}

export function parseCriteria(json: string | null): CriterionResult[] | null {
  if (!json) return null;
  try {
    const parsed = JSON.parse(json);
    return Array.isArray(parsed) ? (parsed as CriterionResult[]) : null;
  } catch {
    return null;
  }
}

export function projectInterview(
  interview: InterviewRow,
  turns: TurnRow[],
): PublicInterview {
  const complete = interview.status === "completed";

  const publicTurns: PublicTurn[] = turns.map((turn) => {
    const base: PublicTurn = {
      ordinal: turn.ordinal,
      question: turn.question,
      topic: turn.topic,
      answerText: turn.answerText,
      answerSource: turn.answerSource,
      answered: turn.answeredAt !== null,
    };
    if (!complete) return base;

    const criteria = parseCriteria(turn.criteriaJson);
    return {
      ...base,
      criteria: criteria ?? undefined,
      band: criteria ? bandForAnswer(criteria) : undefined,
      strength: turn.strength,
      improvement: turn.improvement,
    };
  });

  const projected: PublicInterview = {
    id: interview.id,
    status: interview.status,
    jobTitle: interview.jobTitle,
    interviewType: interview.interviewType,
    plannedQuestions: interview.plannedQuestions,
    askedCount: interview.askedCount,
    answeredCount: turns.filter((t) => t.answeredAt !== null).length,
    hasBrief:
      Boolean(interview.roleBrief) || Boolean(interview.candidateBrief),
    createdAt: interview.createdAt,
    completedAt: interview.completedAt,
    turns: publicTurns,
  };

  if (complete && interview.summaryJson) {
    try {
      const summary = JSON.parse(interview.summaryJson) as {
        didWell?: string[];
        workOn?: string[];
        overall?: string;
      };
      projected.results = {
        rollup: rollUpInterview(turns.map((t) => parseCriteria(t.criteriaJson))),
        didWell: summary.didWell ?? [],
        workOn: summary.workOn ?? [],
        overall: summary.overall ?? "",
      };
    } catch {
      // A corrupt summary must not hide the rubric results, which are computed
      // from stored judgements and do not depend on it.
      projected.results = {
        rollup: rollUpInterview(turns.map((t) => parseCriteria(t.criteriaJson))),
        didWell: [],
        workOn: [],
        overall: "",
      };
    }
  }

  return projected;
}
