/**
 * Inline code-completion helpers for the Sandbox editor. Pure functions shared
 * by the /api/complete route and its tests; no model or network here.
 */
/** Default completion model. Inline completions need to be fast (sub-second)
 * and are triggered constantly, so the default is on a fast inference route
 * (cerebras). Larger than the 20B and still quick. */
export const COMPLETION_DEFAULT_MODEL = "openai/gpt-oss-120b:cerebras";

/** Code-capable models the completion route will accept as an override. */
// v6.3.0: Qwen3.6-35B left the catalog and its replacement's only route is far
// too slow for inline completion (10-17s to answer), so Muse Glimmer takes the
// small-multimodal slot (0.6s first token); Kimi K3 replaces Kimi K2.7 Code on
// a faster route (0.5s first token).
export const COMPLETION_MODELS = [
  "openai/gpt-oss-20b:groq",
  "openai/gpt-oss-120b:cerebras",
  "google/gemma-4-31B-it:cerebras",
  "meta-models/Muse-Glimmer-30B:together",
  "moonshotai/Kimi-K3:baseten",
];

/** The languages the Sandbox editor can request completions for. */
export const COMPLETION_LANGUAGES = ["python", "r", "sql"] as const;

/** A terse instruction: small completion models do best with a short, directive
 * prompt. A compact style hint keeps completions consistent with the Tutor and
 * chat without the long tutor block, which made this model over-refuse. */
export const COMPLETION_SYSTEM_PROMPT = `You are a code autocomplete engine inside an editor. Continue the code at the cursor with the single most likely next code. Output ONLY the text to insert at the cursor: no explanation, no markdown fences, no backticks, and do not repeat the code that is already there. Keep it short, usually one line, and match the surrounding style (in R prefer pkg::fn() and the native pipe |>; in Python avoid one-line method chains).`;

/** The user prompt: the code with an explicit cursor marker. */
export function buildCompletionPrompt(
  language: string,
  prefix: string,
  suffix: string,
): string {
  return `The code is written in ${language}. Complete the code at the cursor, which is marked <CURSOR>. Return only the text that should be inserted at <CURSOR>.\n\n${prefix}<CURSOR>${suffix}`;
}

/** Longest completion we will show, so ghost text stays a suggestion, not a page. */
const MAX_COMPLETION_CHARS = 240;
const MAX_COMPLETION_LINES = 8;

/**
 * Cleans a model's raw reply into text safe to insert at the cursor: strips a
 * wrapping code fence and a stray leading newline, caps the length and line
 * count, and trims trailing whitespace (leading whitespace is kept because it
 * is indentation).
 */
export function parseCompletion(raw: string): string {
  let text = raw ?? "";

  const fenced = /^```[a-zA-Z0-9]*\n([\s\S]*?)\n```$/.exec(text.trim());
  if (fenced) text = fenced[1];

  text = text.replace(/^\n/, "");
  text = text.split("\n").slice(0, MAX_COMPLETION_LINES).join("\n");
  if (text.length > MAX_COMPLETION_CHARS) text = text.slice(0, MAX_COMPLETION_CHARS);
  text = text.replace(/\s+$/, "");

  return text;
}
