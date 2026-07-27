import { describe, expect, it } from "vitest";
import { CHAT_MODULES } from "@/lib/chat/config";
import { getPageModels, MODELS } from "@/lib/config/models";
import { AI_COMPARISON_SYSTEM_PROMPT } from "@/lib/prompts/ai-comparison";
import {
  DEFAULT_TRIALS,
  MAX_TRIALS,
  COMPARISON_MODULE_KEY,
  COMPARISON_MODULE_SLUG,
} from "@/lib/comparison/config";

describe("comparison constants", () => {
  it("defaults to one trial and caps at five", () => {
    expect(DEFAULT_TRIALS).toBe(1);
    expect(MAX_TRIALS).toBe(5);
  });
  it("keys and slug match the catalog and the route segment", () => {
    expect(COMPARISON_MODULE_KEY).toBe("ai_comparisons");
    expect(COMPARISON_MODULE_SLUG).toBe("ai-comparison");
  });
});

describe("the chat route accepts the comparison module", () => {
  it("registers ai_comparisons in CHAT_MODULES", () => {
    const cfg = CHAT_MODULES[COMPARISON_MODULE_KEY];
    expect(cfg).toBeDefined();
    expect(cfg.slug).toBe(COMPARISON_MODULE_SLUG);
    expect(cfg.systemPrompt).toBe(AI_COMPARISON_SYSTEM_PROMPT);
    expect(cfg.temperature).toBeGreaterThanOrEqual(0);
    expect(cfg.maxOutputTokens).toBeGreaterThan(0);
  });

  it("uses a neutral prompt with no em dashes and no tutor persona", () => {
    expect(AI_COMPARISON_SYSTEM_PROMPT.length).toBeGreaterThan(20);
    expect(AI_COMPARISON_SYSTEM_PROMPT).not.toContain("—"); // em dash
    expect(AI_COMPARISON_SYSTEM_PROMPT.toLowerCase()).not.toContain("tutor");
  });
});

describe("the comparison offers at least two real models", () => {
  it("lists two or more known models", () => {
    const list = getPageModels(COMPARISON_MODULE_KEY);
    expect(list.length).toBeGreaterThanOrEqual(2);
    for (const id of list) expect(MODELS[id]).toBeDefined();
  });
});
