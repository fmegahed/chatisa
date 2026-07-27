import { describe, expect, it } from "vitest";
import {
  MODELS,
  getPageModels,
  buildModelOptions,
} from "@/lib/config/models";
import { CHAT_MODULES } from "@/lib/chat/config";
import { getModule } from "@/lib/modules";

// Anthropic + OpenAI only (2026-07-24): both providers accept the same native
// file parts, so chats with attachments stay model-switchable.
const ROSTER = [
  "gpt-5.6-sol",
  "gpt-5.6-terra",
  "gpt-5.6-luna",
  "claude-opus-5",
  "claude-sonnet-5",
];

describe("Ask Anything roster", () => {
  it("offers exactly the curated Anthropic + OpenAI five", () => {
    expect(getPageModels("ask_anything").sort()).toEqual([...ROSTER].sort());
  });

  it("every roster model has vision, tools, and structured output", () => {
    for (const id of ROSTER) {
      const m = MODELS[id];
      expect(m.supportsVision, id).toBe(true);
      expect(m.supportsFunctionCalling, id).toBe(true);
      expect(m.supportsStructuredOutput, id).toBe(true);
    }
  });

  it("every roster model is served by Anthropic or OpenAI", () => {
    for (const id of ROSTER) {
      expect(["anthropic", "openai"], id).toContain(MODELS[id].provider);
    }
  });

  it("defaults to Claude Sonnet 5", () => {
    const { defaultModelId } = buildModelOptions("ask_anything");
    expect(defaultModelId).toBe("claude-sonnet-5");
  });

  it("registers the chat module under the ask-anything slug", () => {
    const mod = CHAT_MODULES.ask_anything;
    expect(mod?.key).toBe("ask_anything");
    expect(mod?.slug).toBe("ask-anything");
    expect(mod?.name).toBe("Ask Anything");
    expect(mod?.systemPrompt.length).toBeGreaterThan(100);
  });

  it("renames the module tile to Ask Anything", () => {
    expect(getModule("ask-anything")?.name).toBe("Ask Anything");
  });
});
