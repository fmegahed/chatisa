import { repositionAnswers } from "@/lib/exam/answer-positions";
import { logger } from "@/lib/log";
import { randomUUID } from "node:crypto";
import { generateObject } from "ai";
import { temperatureFor } from "@/lib/config/models";
import { getLanguageModel } from "@/lib/providers";
import { getMockModel } from "@/lib/providers/mock";
import { EXAM_INSTRUCTIONS, buildExamPrompt } from "@/lib/prompts/exam-ally";
import { inputCharBudget, planBatches } from "./budget";
import { selectExcerpts, type CoverageReport, type SourcePage } from "./chunking";
import { checkGrounding, type GroundingStatus } from "./grounding";
import { checkQuality } from "./quality";
import {
  examGenerationSchema,
  strictQuestionSchema,
  type GeneratedQuestion,
  type QuestionType,
} from "./schemas";

/**
 * Builds an exam from a document: choose excerpts, generate in batches sized
 * to the model, then keep only questions that pass validation, grounding and
 * quality. Nothing is padded to hit a target count; a short exam is reported
 * honestly rather than filled with invention.
 */

/** Attempts to make up a shortfall before giving up. */
const MAX_REGENERATION_ATTEMPTS = 2;
/** Below this share of the requested count, the exam is not worth offering. */
const MIN_ACCEPTABLE_RATIO = 0.6;

export interface AcceptedQuestion extends GeneratedQuestion {
  groundingStatus: GroundingStatus;
  pointsPossible: number;
}

export interface GenerationResult {
  questions: AcceptedQuestion[];
  coverage: CoverageReport;
  dropped: { reason: string; count: number }[];
  /** True when too little survived to be worth presenting. */
  failed: boolean;
}

function model(modelId: string) {
  return process.env.CHATISA_MOCK_LLM === "1"
    ? getMockModel()
    : getLanguageModel(modelId);
}

export async function generateExam(params: {
  modelId: string;
  questionType: QuestionType;
  count: number;
  pages: SourcePage[];
  fromPage: number;
  toPage: number;
  /** Topics to concentrate on, used when a student retries weak areas. */
  focusTopics?: string[];
  /** Stems already asked, so a retry produces genuinely new questions. */
  excludeStems?: string[];
}): Promise<GenerationResult> {
  const { excerpts, coverage } = selectExcerpts({
    pages: params.pages,
    fromPage: params.fromPage,
    toPage: params.toPage,
    charBudget: inputCharBudget(params.modelId),
    questionCount: params.count,
  });

  const pagesForGrounding = params.pages.filter(
    (p) => p.pageNumber >= params.fromPage && p.pageNumber <= params.toPage,
  );

  const accepted: AcceptedQuestion[] = [];
  const dropCounts = new Map<string, number>();
  const drop = (reason: string) =>
    dropCounts.set(reason, (dropCounts.get(reason) ?? 0) + 1);

  if (excerpts.length === 0) {
    return {
      questions: [],
      coverage,
      dropped: [{ reason: "no_readable_pages", count: 1 }],
      failed: true,
    };
  }

  async function runBatch(size: number): Promise<void> {
    const { object } = await generateObject({
      model: model(params.modelId),
      schema: examGenerationSchema,
      instructions: EXAM_INSTRUCTIONS,
      // Explicit: the SDK caps unrecognized (freshly released) model ids at
      // 4096 output tokens, which would silently truncate a batch of
      // questions. A batch fits well inside 16k on every catalog model.
      maxOutputTokens: 16000,
      temperature: temperatureFor(params.modelId, 0.25),
      messages: [
        {
          role: "user",
          content: buildExamPrompt({
            questionType: params.questionType,
            count: size,
            excerpts,
            existingStems: [
              ...(params.excludeStems ?? []),
              ...accepted.map((q) => q.stem),
            ],
            focusTopics: params.focusTopics,
            nonce: randomUUID().slice(0, 8),
          }),
        },
      ],
    });

    for (const raw of object.questions) {
      if (accepted.length >= params.count) return;

      const parsed = strictQuestionSchema.safeParse(raw);
      if (!parsed.success) {
        drop("invalid_shape");
        continue;
      }
      const question = parsed.data;

      const quality = checkQuality(question, accepted);
      if (!quality.keep) {
        drop(quality.reason ?? "quality");
        continue;
      }

      const grounding = checkGrounding(
        question.sourceQuote,
        question.sourcePage,
        pagesForGrounding,
      );
      if (!grounding.grounded) {
        drop(grounding.reason === "too_short" ? "quote_too_short" : "ungrounded");
        continue;
      }

      accepted.push({
        ...question,
        sourcePage: grounding.page ?? question.sourcePage,
        groundingStatus: grounding.status ?? "verified",
        pointsPossible: 10,
      });
    }
  }

  for (const size of planBatches(params.modelId, params.count)) {
    if (accepted.length >= params.count) break;
    await runBatch(size);
  }

  // One or two focused attempts at whatever is missing, then stop: an
  // unbounded retry loop is a cost hazard.
  for (
    let attempt = 0;
    attempt < MAX_REGENERATION_ATTEMPTS && accepted.length < params.count;
    attempt += 1
  ) {
    const before = accepted.length;
    await runBatch(params.count - accepted.length);
    if (accepted.length === before) break; // making no progress
  }

  const dropped = [...dropCounts.entries()].map(([reason, count]) => ({
    reason,
    count,
  }));

  // Decide where each correct answer sits, after generation rather than by
  // asking the model. Models cluster the correct option in one position, which
  // makes an exam guessable once a student notices the pattern.
  const positioned = repositionAnswers(accepted);
  if (positioned.skipped > 0) {
    logger.info(
      { skipped: positioned.skipped },
      "answer positions left alone for questions whose options reference each other",
    );
  }

  return {
    questions: positioned.questions,
    coverage,
    dropped,
    failed: accepted.length < Math.ceil(params.count * MIN_ACCEPTABLE_RATIO),
  };
}

/** Plain sentence about what was dropped, for the student. */
export function describeShortfall(
  requested: number,
  delivered: number,
): string | null {
  if (delivered >= requested) return null;
  const missing = requested - delivered;
  return `We built ${delivered} of the ${requested} questions you asked for. ${missing === 1 ? "One question was" : `${missing} questions were`} left out because we could not trace ${missing === 1 ? "it" : "them"} back to your document.`;
}
