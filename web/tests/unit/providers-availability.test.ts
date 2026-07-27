import { describe, expect, it } from "vitest";
import { describeProviderAvailability } from "@/lib/providers";
import { MODELS } from "@/lib/config/models";

const TOTAL = Object.keys(MODELS).length;

describe("describeProviderAvailability", () => {
  it("reports nothing hidden when every provider key is present", () => {
    const result = describeProviderAvailability([]);
    expect(result.hiddenModelCount).toBe(0);
    expect(result.availableModelCount).toBe(TOTAL);
  });

  it("hides every open-weight model when HF_TOKEN is absent", () => {
    // Counted from the catalog: the open-weight set is a teaching requirement
    // and is expected to grow, so a hard-coded count would fail on every
    // future addition without indicating a real problem.
    const openWeight = Object.keys(MODELS).filter((id) => MODELS[id].openWeight);
    const result = describeProviderAvailability(["HF_TOKEN"]);
    expect(result.hiddenModelCount).toBe(openWeight.length);
    expect(result.availableModelCount).toBe(TOTAL - openWeight.length);
    expect(new Set(result.hiddenModels)).toEqual(new Set(openWeight));
    // Commercial models stay available.
    expect(result.hiddenModels).not.toContain("claude-sonnet-5");
  });

  it("hides the OpenAI models when OPENAI_API_KEY is absent", () => {
    const result = describeProviderAvailability(["OPENAI_API_KEY"]);
    expect(result.hiddenModels).toContain("gpt-5.6-sol");
    expect(result.hiddenModels).toContain("gpt-5.6-luna");
    // OpenAI's open-weight models served through the HuggingFace router are a
    // different provider and a different credential.
    expect(result.hiddenModels).not.toContain("openai/gpt-oss-20b:groq");
    expect(result.hiddenModels).not.toContain("openai/gpt-oss-120b:cerebras");
  });

  it("hides everything when no provider keys are configured", () => {
    const result = describeProviderAvailability([
      "OPENAI_API_KEY",
      "ANTHROPIC_API_KEY",
      "GOOGLE_API_KEY",
      "COHERE_API_KEY",
      "GROQ_API_KEY",
      "HF_TOKEN",
    ]);
    expect(result.hiddenModelCount).toBe(TOTAL);
    expect(result.availableModelCount).toBe(0);
  });
});
