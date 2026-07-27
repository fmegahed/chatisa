import type { getExamQuestions } from "@/lib/db";

/**
 * The single place a question is turned into something the browser may see.
 *
 * Answer keys, rubrics, model answers and explanations must never reach a
 * student before they have answered. Everything goes through here so a future
 * change cannot leak them by accident, and a test snapshots the exact key set.
 */
type StoredQuestion = ReturnType<typeof getExamQuestions>[number];

export interface ClientQuestion {
  id: string;
  position: number;
  type: string;
  stem: string;
  options: string[] | null;
  topic: string;
  pointsPossible: number;
  correctIndex?: number | null;
  modelAnswer?: string;
  explanation?: string;
  sourceQuote?: string;
  sourcePage?: number;
}

/**
 * `reveal` is decided on the server from stored state, never from the request.
 */
export function toClientQuestion(
  question: StoredQuestion,
  reveal: boolean,
): ClientQuestion {
  const base: ClientQuestion = {
    id: question.id,
    position: question.position,
    type: question.type,
    stem: question.stem,
    options: question.optionsJson
      ? (JSON.parse(question.optionsJson) as string[])
      : null,
    topic: question.topic,
    pointsPossible: question.pointsPossible,
  };
  if (!reveal) return base;
  return {
    ...base,
    correctIndex: question.correctIndex,
    modelAnswer: question.modelAnswer,
    explanation: question.explanation,
    sourceQuote: question.sourceQuote,
    sourcePage: question.sourcePage,
  };
}
