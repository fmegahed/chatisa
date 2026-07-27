import { MODELS } from "@/lib/config/models";

/**
 * How many output tokens a call may use.
 *
 * The legacy app set `max_num_tokens = 1000` and then silently overrode it with
 * the model maximum, so the number never bit. The port set 1000 and actually
 * enforced it, which broke reasoning models outright: measured on 2026-07-21,
 * Ternary Bonsai 27B emitted **zero** visible characters at a 1000 cap, having
 * spent the entire budget on hidden reasoning, and Gemini 3.1 Pro was cut off
 * mid-sentence. The same mistake had already produced 14 false failures in
 * scripts/verify-models.ts.
 *
 * The correction is to treat a cap as a runaway guard rather than a spend
 * control. A model stops when it has finished; a higher ceiling costs nothing
 * unless the work genuinely needs it. Spend is controlled by rate limits and by
 * model choice, both of which act before a single token is generated.
 */

/** Comfortable ceiling for a chat reply, including any hidden reasoning. */
export const CHAT_OUTPUT_TOKENS = 8_000;

/**
 * Reasoning models spend part of the budget on tokens the student never sees,
 * so the visible answer competes with the thinking for the same allowance.
 * Detected from catalog metadata rather than a hardcoded id list: the previous
 * list in lib/exam/budget.ts named a model that no longer exists and therefore
 * matched nothing, which is the failure mode a name list always trends toward.
 */
export function isReasoningModel(modelId: string): boolean {
  return MODELS[modelId]?.tags.includes("reasoning") ?? false;
}

/**
 * Output ceiling for one call, never above what the model can actually emit.
 *
 * @param desired the ceiling this feature would like
 */
export function outputTokenBudget(modelId: string, desired: number): number {
  const cfg = MODELS[modelId];
  const headroom = isReasoningModel(modelId) ? 2 : 1;
  const wanted = desired * headroom;
  if (!cfg) return wanted;
  // Asking for more than the model's own ceiling is rejected by some providers
  // rather than clamped, so clamp here.
  return Math.min(wanted, cfg.maxTokens);
}

/**
 * Why a response arrived with nothing a student can read.
 *
 * Distinguishing these matters because the advice differs: a truncated answer
 * can be continued, whereas an answer entirely consumed by reasoning needs a
 * different model or a shorter question.
 */
export type EmptyReason = "truncated_before_text" | "no_text_returned";

export function describeEmptyResponse(reason: EmptyReason): string {
  return reason === "truncated_before_text"
    ? "This model spent its whole response thinking and ran out of room before writing an answer. Try asking again, or pick a different model."
    : "This model returned an empty response. Try asking again, or pick a different model.";
}

/** Shown after an answer that was cut off with text already delivered. */
export const TRUNCATION_NOTICE =
  "This answer was cut off because it reached the length limit. Ask a follow-up to continue it.";
