import { describe, expect, it } from "vitest";
import {
  CHAT_OUTPUT_TOKENS,
  TRUNCATION_NOTICE,
  describeEmptyResponse,
  isReasoningModel,
  outputTokenBudget,
} from "@/lib/chat/budget";
import { MODELS } from "@/lib/config/models";

/**
 * Guards the fix for the defect reported on 2026-07-21: switching to Ternary
 * Bonsai produced empty reply bubbles. Measured cause, not inferred: at a
 * 1000-token cap that model emitted 0 characters of text and 1000 reasoning
 * parts, finishing with reason "length". The same cap truncated a Gemini answer
 * mid-sentence.
 */

describe("output budget", () => {
  it("gives chat far more room than the legacy 1000", () => {
    expect(CHAT_OUTPUT_TOKENS).toBeGreaterThanOrEqual(4_000);
  });

  it("gives reasoning models extra room for hidden thinking", () => {
    const reasoning = Object.keys(MODELS).find((id) => isReasoningModel(id))!;
    const plain = Object.keys(MODELS).find((id) => !isReasoningModel(id))!;
    expect(reasoning).toBeDefined();
    expect(plain).toBeDefined();
    expect(outputTokenBudget(reasoning, 1_000)).toBeGreaterThan(
      outputTokenBudget(plain, 1_000),
    );
  });

  it("never asks a model for more than it can emit", () => {
    // Some providers reject an over-large request rather than clamping it.
    for (const [id, cfg] of Object.entries(MODELS)) {
      expect(outputTokenBudget(id, 1_000_000), id).toBeLessThanOrEqual(
        cfg.maxTokens,
      );
    }
  });

  it("leaves every catalog model a workable chat allowance", () => {
    for (const id of Object.keys(MODELS)) {
      expect(outputTokenBudget(id, CHAT_OUTPUT_TOKENS), id).toBeGreaterThanOrEqual(
        1_000,
      );
    }
  });

  it("reads reasoning from catalog tags, not a hardcoded id list", () => {
    // The list this replaces named a model that had left the catalog, so it
    // silently matched nothing. Tags cannot rot the same way, because a model
    // without tags fails the catalog integrity tests.
    expect(isReasoningModel("no-such-model")).toBe(false);
    const tagged = Object.entries(MODELS).filter(([, c]) =>
      c.tags.includes("reasoning"),
    );
    expect(tagged.length).toBeGreaterThan(0);
    for (const [id] of tagged) expect(isReasoningModel(id)).toBe(true);
  });
});

describe("explaining an unreadable response", () => {
  it("distinguishes running out of room from returning nothing", () => {
    // The advice differs: one is worth retrying or switching model, the other
    // means the question was too big for the room the model had.
    const truncated = describeEmptyResponse("truncated_before_text");
    const empty = describeEmptyResponse("no_text_returned");
    expect(truncated).not.toBe(empty);
    expect(truncated).toMatch(/thinking|room/i);
  });

  it("always tells the student what they can do next", () => {
    for (const reason of ["truncated_before_text", "no_text_returned"] as const) {
      expect(describeEmptyResponse(reason)).toMatch(/try again|different model/i);
    }
    expect(TRUNCATION_NOTICE).toMatch(/follow-up|continue/i);
  });

  it("never blames the student or leaks provider internals", () => {
    const all = [
      describeEmptyResponse("truncated_before_text"),
      describeEmptyResponse("no_text_returned"),
      TRUNCATION_NOTICE,
    ];
    for (const message of all) {
      expect(message).not.toMatch(/token|finishReason|API|error code/i);
    }
  });
});
