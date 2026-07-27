import { describe, expect, it } from "vitest";
import {
  CHAT_MODULES,
  chatRequestSchema,
  getChatModuleBySlug,
  textFromParts,
} from "@/lib/chat/config";
import { CODING_COMPANION_SYSTEM_PROMPT } from "@/lib/prompts/coding-companion";
import { CHAT_OUTPUT_TOKENS } from "@/lib/chat/budget";
import { DEFAULT_MODELS, MODELS, getPageModels } from "@/lib/config/models";

describe("coding companion prompt parity", () => {
  const prompt = CODING_COMPANION_SYSTEM_PROMPT;

  it("keeps the legacy pedagogy instructions verbatim", () => {
    expect(prompt).toContain(
      "You are an upbeat, encouraging tutor who helps undergraduate students majoring in business analytics",
    );
    expect(prompt).toContain("Only ask one question at a time.");
    expect(prompt).toContain("library_name::function_name()");
    expect(prompt).toContain("use the native pipe |> as your pipe operator");
    expect(prompt).toContain(
      "do NOT write df.groupby('Region')['Sales'].agg('sum') on one line",
    );
    expect(prompt).toContain("if(require(library)==FALSE) install.packages(library)");
  });

  it("keeps the legacy opening user message", () => {
    expect(CHAT_MODULES.coding_companion.openingUserMessage).toBe(
      "Hi, I am an undergraduate student studying business analytics.",
    );
  });

  it("keeps the legacy temperature and enforces an output cap", () => {
    // Legacy pages/01_coding_companion.py: TEMPERATURE = 0, max 1000 tokens.
    expect(CHAT_MODULES.coding_companion.temperature).toBe(0);
    // Raised from the legacy 1000 on 2026-07-21: reasoning models spent that
    // entire budget on hidden thinking and emitted nothing visible, and it also
    // truncated ordinary answers mid-sentence. A cap is a runaway guard here,
    // not a spend control.
    expect(CHAT_MODULES.coding_companion.maxOutputTokens).toBe(CHAT_OUTPUT_TOKENS);
    expect(CHAT_OUTPUT_TOKENS).toBeGreaterThanOrEqual(4000);
  });

  it("resolves the module by its route slug", () => {
    expect(getChatModuleBySlug("coding-tutor")?.key).toBe(
      "coding_companion",
    );
    expect(getChatModuleBySlug("nope")).toBeUndefined();
  });
});

describe("chatRequestSchema", () => {
  const valid = {
    module: "coding_companion",
    modelId: "claude-sonnet-5",
    messages: [{ role: "user", parts: [{ type: "text", text: "hello" }] }],
  };

  it("accepts a well-formed request", () => {
    expect(chatRequestSchema.safeParse(valid).success).toBe(true);
  });

  it("rejects an empty message list", () => {
    expect(
      chatRequestSchema.safeParse({ ...valid, messages: [] }).success,
    ).toBe(false);
  });

  it("rejects an unknown role", () => {
    expect(
      chatRequestSchema.safeParse({
        ...valid,
        messages: [{ role: "root", parts: [] }],
      }).success,
    ).toBe(false);
  });

  it("rejects oversized message text", () => {
    expect(
      chatRequestSchema.safeParse({
        ...valid,
        messages: [
          { role: "user", parts: [{ type: "text", text: "x".repeat(50_001) }] },
        ],
      }).success,
    ).toBe(false);
  });
});

describe("textFromParts", () => {
  it("concatenates text parts and ignores other part types", () => {
    expect(
      textFromParts([
        { type: "text", text: "Hello " },
        { type: "step-start" },
        { type: "text", text: "world" },
      ]),
    ).toBe("Hello world");
  });

  it("returns an empty string when there are no parts", () => {
    expect(textFromParts(undefined)).toBe("");
  });
});

describe("module model policy", () => {
  it("offers the chat modules real models and no speech-only ones", () => {
    const allowed = getPageModels("coding_companion");
    expect(allowed.length).toBeGreaterThan(0);
    for (const id of allowed) {
      expect(MODELS[id]).toBeDefined();
      expect(MODELS[id].realtimeOnly ?? false).toBe(false);
    }
    // The default has to be one of the offered models, or the picker opens on
    // a model the server would then reject.
    expect(allowed).toContain(DEFAULT_MODELS.coding_companion);
  });
});
