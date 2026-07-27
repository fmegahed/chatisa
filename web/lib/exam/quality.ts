import type { GeneratedQuestion } from "./schemas";
import { normalize } from "./grounding";

/**
 * Quality gates applied after generation. These catch the failures a schema
 * cannot: unanswerable multiple choice, repeated questions, and text from the
 * document trying to issue instructions.
 */

export type RejectReason =
  | "duplicate"
  | "mcq_options"
  | "banned_option"
  | "injection"
  | "not_self_contained";

export interface QualityVerdict {
  keep: boolean;
  reason?: RejectReason;
}

/** Options that make a multiple choice question untestable. */
const BANNED_OPTION = /^(all|none) of the above|both [ab] and [ab]$/i;

/**
 * Text that reads as an instruction rather than course content. The nonce
 * delimiter around the document is the first defence; this is the last, and
 * it matters because a question stem is shown to the student verbatim.
 */
const INJECTION = /(ignore\s+(all\s+)?(the\s+)?previous|disregard\s+(all\s+)?previous|full marks|award\s+\S+\s+(full|all)|system prompt|you are an? (ai|assistant|language model))/i;

/**
 * Questions that point at the source material instead of testing the idea.
 *
 * A student answers with the document closed, so "in the X table, which
 * option..." is a lookup exercise, not an exam question. The grounding rule
 * pulls generation towards hugging the source text, so this gate is what stops
 * that turning into document-referential questions.
 */
const DOCUMENT_REFERENCE = [
  // "in the table", "from the figure", "within the passage"
  /\b(in|on|from|per|within)\s+the\s+(above\s+|below\s+|following\s+|preceding\s+)?(table|figure|exhibit|chart|diagram|graph|passage|text|document|reading|slide|section|chapter|appendix|worksheet|excerpt|handout)\b/i,
  /\baccording to the\s+(table|figure|text|document|passage|reading|author|slide|section|chapter|material|notes)\b/i,
  /\bbased on the\s+(table|figure|text|document|passage|reading|slide|section|chapter|material|notes)\b/i,
  /\bas\s+(described|shown|stated|defined|listed|presented|discussed|outlined|noted)\s+(in|on)\b/i,
  // A named exhibit, for example the "Three Possible Options..." table
  /["“][^"”]{4,120}["”]\s*(table|figure|exhibit|section|chart|diagram)\b/i,
  /\bthe\s+(table|figure|exhibit|chart|diagram)\s+(titled|named|called|labell?ed)\b/i,
  /\b(this|the)\s+(document|reading|passage|excerpt|handout)\b/i,
];

/**
 * A stem may legitimately carry its own data, for example "using the table
 * below". That is self contained, so it is not a reference to the source.
 */
const SELF_CONTAINED_DATA = /\b(table|data|dataset|code|snippet|output|figure)\s+below\b/i;

export function referencesSourceDocument(stem: string): boolean {
  if (SELF_CONTAINED_DATA.test(stem)) return false;
  return DOCUMENT_REFERENCE.some((pattern) => pattern.test(stem));
}

function jaccard(a: Set<string>, b: Set<string>): number {
  if (a.size === 0 || b.size === 0) return 0;
  let shared = 0;
  for (const item of a) if (b.has(item)) shared += 1;
  return shared / (a.size + b.size - shared);
}

function wordSet(text: string): Set<string> {
  return new Set(normalize(text).split(" ").filter((w) => w.length > 3));
}

/** Two questions this similar are the same question in different words. */
export const DUPLICATE_THRESHOLD = 0.7;

export function checkQuality(
  question: GeneratedQuestion,
  kept: GeneratedQuestion[],
): QualityVerdict {
  const combined = `${question.stem} ${question.explanation} ${question.modelAnswer}`;
  if (INJECTION.test(combined)) return { keep: false, reason: "injection" };

  // The question must make sense with the document closed.
  if (referencesSourceDocument(question.stem)) {
    return { keep: false, reason: "not_self_contained" };
  }

  if (question.type === "multiple_choice") {
    const options = question.options ?? [];
    const distinct = new Set(options.map((o) => normalize(o)));
    if (options.length !== 4 || distinct.size !== options.length) {
      return { keep: false, reason: "mcq_options" };
    }
    if (options.some((o) => BANNED_OPTION.test(o.trim()))) {
      return { keep: false, reason: "banned_option" };
    }
  }

  const stemWords = wordSet(question.stem);
  for (const other of kept) {
    if (jaccard(stemWords, wordSet(other.stem)) >= DUPLICATE_THRESHOLD) {
      return { keep: false, reason: "duplicate" };
    }
    if (
      question.type === "multiple_choice" &&
      other.type === "multiple_choice" &&
      question.topic === other.topic &&
      question.correctIndex !== null &&
      other.correctIndex !== null &&
      normalize(question.options?.[question.correctIndex] ?? "") ===
        normalize(other.options?.[other.correctIndex] ?? "")
    ) {
      return { keep: false, reason: "duplicate" };
    }
  }

  return { keep: true };
}
