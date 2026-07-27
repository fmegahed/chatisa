import { isReasoningModel } from "@/lib/chat/budget";
import { MODELS } from "@/lib/config/models";

/**
 * Sizing generation to each model's real limits.
 *
 * Both ends matter. Several models in the catalog cap output at 4,096 tokens,
 * which is fewer than a ten-question exam with rubrics and quotes needs, so
 * generation is split into batches. Others have small context windows, which
 * bounds how much of the document can be shown at once.
 */

/** Measured cost of one question: stem, options, rubric, quote, explanation. */
export const TOKENS_PER_QUESTION = 380;
/** Instructions plus the JSON schema the provider is given. */
const PROMPT_OVERHEAD_TOKENS = 1200;
/** Conservative for mixed prose and code. */
const CHARS_PER_TOKEN = 3.6;
const OUTPUT_SAFETY = 0.85;
const INPUT_SAFETY = 0.8;
/**
 * Reasoning models spend output tokens thinking, so they yield fewer questions
 * per call. Read from catalog tags rather than a hardcoded id list: the list
 * this replaces named "deepseek-ai/DeepSeek-R1-0528", which left the catalog in
 * the 2026-07-21 refresh, so it had silently matched nothing since.
 */

export function maxQuestionsPerCall(modelId: string): number {
  const cfg = MODELS[modelId];
  if (!cfg) return 1;
  // A reasoning model spends output tokens on thinking before it writes, so
  // each question costs more of the allowance. Modelled as a higher per-question
  // cost rather than a flat question cap: a flat cap throttled models with
  // 128k of output room just as hard as one with 8k, which is the wrong shape.
  const perQuestion = TOKENS_PER_QUESTION * (isReasoningModel(modelId) ? 2.5 : 1);
  const raw = Math.floor((cfg.maxTokens * OUTPUT_SAFETY) / perQuestion);
  return Math.max(1, Math.min(raw, 20));
}

/** How much document text this model can be shown in one call. */
export function inputCharBudget(modelId: string): number {
  const cfg = MODELS[modelId];
  if (!cfg) return 4_000;
  const reservedOutput = Math.min(
    cfg.maxTokens,
    maxQuestionsPerCall(modelId) * TOKENS_PER_QUESTION,
  );
  const usableTokens =
    (cfg.contextWindow - reservedOutput) * INPUT_SAFETY - PROMPT_OVERHEAD_TOKENS;
  return Math.max(4_000, Math.floor(usableTokens * CHARS_PER_TOKEN));
}

/**
 * Splits a question count into batches this model can actually emit,
 * rebalanced so no batch is a lone straggler: a one-question batch costs a
 * full prompt and produces noticeably weaker questions.
 */
export function planBatches(modelId: string, count: number): number[] {
  const perCall = maxQuestionsPerCall(modelId);
  if (count <= perCall) return [count];
  const batches = Math.ceil(count / perCall);
  const base = Math.floor(count / batches);
  const remainder = count % batches;
  return Array.from({ length: batches }, (_, i) => base + (i < remainder ? 1 : 0));
}

/** True when this model can hold the given amount of document text. */
export function canHoldDocument(modelId: string, chars: number): boolean {
  return inputCharBudget(modelId) >= Math.min(chars, 4_000);
}

/** Rough token estimate, used for the pre-generation cost estimate. */
export function estimateTokens(chars: number): number {
  return Math.ceil(chars / CHARS_PER_TOKEN);
}
