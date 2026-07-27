import { CHAT_OUTPUT_TOKENS } from "@/lib/chat/budget";
import { z } from "zod";
import { CODING_COMPANION_SYSTEM_PROMPT } from "@/lib/prompts/coding-companion";
import { SANDBOX_CHAT_SYSTEM_PROMPT } from "@/lib/prompts/sandbox-chat";
import { AI_COMPARISON_SYSTEM_PROMPT } from "@/lib/prompts/ai-comparison";
import { ASK_ANYTHING_SYSTEM_PROMPT } from "@/lib/prompts/ask-anything";

/**
 * Per-module chat settings. Temperature and output caps come from the legacy
 * app's per-page constants. Unlike the legacy code, maxOutputTokens is
 * actually enforced (legacy lib/chatgeneration.py:52 silently overrode the
 * caller's value with the model maximum, allowing 128k-token replies to a
 * tutoring question). This is an approved intentional change.
 */
export interface ChatModuleConfig {
  /** Key used by the model catalog and analytics. */
  key: string;
  /** Route segment. */
  slug: string;
  name: string;
  systemPrompt: string;
  /** Fixed opening user turn preserved from the legacy app, if any. */
  openingUserMessage?: string;
  temperature: number;
  maxOutputTokens: number;
  placeholder: string;
}

export const CHAT_MODULES: Record<string, ChatModuleConfig> = {
  coding_companion: {
    key: "coding_companion",
    slug: "coding-tutor",
    name: "Coding Tutor",
    systemPrompt: CODING_COMPANION_SYSTEM_PROMPT,
    openingUserMessage:
      "Hi, I am an undergraduate student studying business analytics.",
    // Legacy pages/01_coding_companion.py used TEMPERATURE = 0 and
    // max_num_tokens = 1000. The temperature is kept; the 1000 cap is not.
    // Legacy never enforced it (lib/chatgeneration.py overwrote it with the
    // model maximum), and enforcing it broke reasoning models, which spent the
    // whole budget on hidden reasoning and returned nothing. See lib/chat/budget.ts.
    temperature: 0,
    maxOutputTokens: CHAT_OUTPUT_TOKENS,
    placeholder:
      "Ask me to help you with your code or to explain analytical concepts.",
  },
  sandbox_chat: {
    key: "sandbox_chat",
    slug: "coding-studio",
    name: "Sandbox assistant",
    systemPrompt: SANDBOX_CHAT_SYSTEM_PROMPT,
    // No fixed opening turn: it is a side helper, not a fresh tutoring session.
    temperature: 0,
    maxOutputTokens: CHAT_OUTPUT_TOKENS,
    placeholder: "Ask about your code, an error, or your data.",
  },
  ai_comparisons: {
    key: "ai_comparisons",
    slug: "ai-comparison",
    name: "AI Comparison",
    systemPrompt: AI_COMPARISON_SYSTEM_PROMPT,
    // No fixed opening turn: each trial is a single, self-contained question.
    // A moderate temperature so answers are natural rather than clipped; both
    // models get the same value, so the comparison stays fair.
    temperature: 0.7,
    maxOutputTokens: CHAT_OUTPUT_TOKENS,
    placeholder: "Ask both models the same question.",
  },
  ask_anything: {
    key: "ask_anything",
    slug: "ask-anything",
    name: "Ask Anything",
    systemPrompt: ASK_ANYTHING_SYSTEM_PROMPT,
    // No fixed opening turn: an open-ended assistant, not a persona session.
    // 0.7 keeps general answers natural (the tutoring modules pin 0 for
    // determinism; this module is closer to AI Comparison's register).
    temperature: 0.7,
    maxOutputTokens: CHAT_OUTPUT_TOKENS,
    placeholder:
      "Ask anything: a question, a draft to improve, a problem to work through.",
  },
};

export function getChatModuleBySlug(slug: string): ChatModuleConfig | undefined {
  return Object.values(CHAT_MODULES).find((m) => m.slug === slug);
}

/** Request body accepted by POST /api/chat. */
export const chatRequestSchema = z.object({
  module: z.string().min(1).max(64),
  modelId: z.string().min(1).max(128),
  // Ephemeral per-request context (the Sandbox sends the student's current
  // script, last result, and variable list). Never persisted; injected into the
  // system instructions for this one turn only.
  context: z.string().max(20_000).optional(),
  messages: z
    .array(
      z.object({
        id: z.string().optional(),
        role: z.enum(["user", "assistant", "system"]),
        parts: z
          .array(
            z.object({
              type: z.string(),
              text: z.string().max(50_000).optional(),
            }),
          )
          .max(200)
          .optional(),
      }),
    )
    .min(1)
    .max(200),
});

export type ChatRequest = z.infer<typeof chatRequestSchema>;

/** Concatenated text of a UI message's text parts. */
export function textFromParts(
  parts?: { type: string; text?: string }[],
): string {
  if (!parts) return "";
  return parts
    .filter((p) => p.type === "text" && typeof p.text === "string")
    .map((p) => p.text as string)
    .join("");
}
