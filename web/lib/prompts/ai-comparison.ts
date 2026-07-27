/**
 * Both models in a comparison receive this identical, neutral instruction, so
 * the student compares the models themselves and not two different prompts.
 * Deliberately minimal: no tutor persona and no coding-style block, both of
 * which would shape the answers and blunt the comparison.
 */
export const AI_COMPARISON_SYSTEM_PROMPT =
  "You are a helpful assistant for an undergraduate business analytics student. Answer the question clearly and directly. If code helps, keep it short and correct.";
