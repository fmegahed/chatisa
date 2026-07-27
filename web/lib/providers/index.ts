import type { LanguageModel } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAICompatible } from "@ai-sdk/openai-compatible";
import { MODELS, type ProviderId } from "@/lib/config/models";

/**
 * Server-only provider registry. API keys are read here and never leave the
 * server. Open-weight HuggingFace models are served through HF's
 * OpenAI-compatible router; keeping them is a pedagogical requirement, not a
 * cost measure (PROJECT_MEMORY, user instruction 2026-07-19).
 */

const HF_BASE_URL = "https://router.huggingface.co/v1";

/** Environment variable that supplies each provider's credential. */
export const PROVIDER_ENV_KEY: Record<ProviderId, string> = {
  openai: "OPENAI_API_KEY",
  anthropic: "ANTHROPIC_API_KEY",
  google: "GOOGLE_API_KEY",
  huggingface_inference: "HF_TOKEN",
};

export class ProviderNotConfiguredError extends Error {
  constructor(public readonly provider: ProviderId) {
    super(`No credential configured for provider "${provider}".`);
    this.name = "ProviderNotConfiguredError";
  }
}

/** True when the credential for this model's provider is present. */
export function isModelAvailable(modelId: string): boolean {
  const cfg = MODELS[modelId];
  if (!cfg) return false;
  if (process.env.CHATISA_MOCK_LLM === "1") return true;
  return Boolean(process.env[PROVIDER_ENV_KEY[cfg.provider]]);
}

/** Model ids from `candidates` whose provider credential is configured. */
export function filterAvailableModels(candidates: string[]): string[] {
  return candidates.filter(isModelAvailable);
}

/**
 * Which models are unavailable because their provider has no key, so the
 * operator can see the impact of a missing variable at a glance.
 */
export function describeProviderAvailability(missingEnvKeys: string[]): {
  hiddenModels: string[];
  hiddenModelCount: number;
  availableModelCount: number;
} {
  const missing = new Set(missingEnvKeys);
  const hiddenModels = Object.entries(MODELS)
    .filter(([, cfg]) => missing.has(PROVIDER_ENV_KEY[cfg.provider]))
    .map(([id]) => id);
  return {
    hiddenModels,
    hiddenModelCount: hiddenModels.length,
    availableModelCount: Object.keys(MODELS).length - hiddenModels.length,
  };
}

export function getLanguageModel(modelId: string): LanguageModel {
  const cfg = MODELS[modelId];
  if (!cfg) throw new Error(`Unknown model: ${modelId}`);

  const apiKey = process.env[PROVIDER_ENV_KEY[cfg.provider]];
  if (!apiKey) throw new ProviderNotConfiguredError(cfg.provider);

  switch (cfg.provider) {
    case "openai":
      return createOpenAI({ apiKey })(modelId);
    case "anthropic":
      return createAnthropic({ apiKey })(modelId);
    case "google":
      return createGoogleGenerativeAI({ apiKey })(modelId);
    case "huggingface_inference":
      return createOpenAICompatible({
        name: "huggingface",
        apiKey,
        baseURL: HF_BASE_URL,
      })(modelId);
  }
}
