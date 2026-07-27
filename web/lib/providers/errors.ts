import { APICallError } from "ai";

/**
 * Why a model call failed, from the student's point of view.
 *
 * The distinction that matters is whether retrying can possibly work. A
 * previous version told the student "you can try again" for every failure,
 * including an exhausted billing account and a model the provider does not
 * serve at all. Both of those are permanent, so that advice sent students into
 * a loop that could never succeed while hiding a fault only an operator can
 * fix.
 */
export type FailureKind =
  | "rate_limited" // Busy right now. Waiting genuinely helps.
  | "model_unavailable" // This model cannot serve requests. Another model will.
  | "account_problem" // Billing or credentials. No student action helps.
  | "context_too_long" // The conversation outgrew the model's window.
  | "transient" // Network or server blip. Retrying is reasonable.
  | "unknown";

export type ClassifiedFailure = {
  kind: FailureKind;
  /** Shown to the student. No provider internals, no keys, no raw errors. */
  message: string;
  /** True only when retrying the same model could actually succeed. */
  retryable: boolean;
  /** True when an operator must act. Drives log severity, never shown. */
  operatorAction: boolean;
};

/** Billing and credential wording varies by provider, so match on meaning. */
const ACCOUNT_PATTERNS = [
  /credit balance is too low/i,
  /insufficient[_\s]?(quota|funds|credit)/i,
  /billing/i,
  /payment required/i,
  /exceeded your current quota/i,
  /invalid[_\s]api[_\s]key/i,
  /incorrect api key/i,
  /authentication[_\s]error/i,
  /unauthorized/i,
];

const UNAVAILABLE_PATTERNS = [
  /not supported by any provider/i,
  /model[_\s]not[_\s]found/i,
  /does not exist or you do not have access/i,
  /no (inference )?provider/i,
  /model is currently (loading|unavailable)/i,
  /has been (deprecated|retired|decommissioned)/i,
];

const CONTEXT_PATTERNS = [
  /context[_\s]length[_\s]exceeded/i,
  /maximum context length/i,
  /too many tokens/i,
  /input is too long/i,
  /prompt is too long/i,
];

function matches(patterns: RegExp[], text: string): boolean {
  return patterns.some((p) => p.test(text));
}

export function classifyProviderFailure(error: unknown): ClassifiedFailure {
  const status = APICallError.isInstance(error) ? error.statusCode : undefined;
  // The provider's own explanation usually lives in the body, not the message.
  const text = [
    error instanceof Error ? error.message : String(error),
    APICallError.isInstance(error) ? (error.responseBody ?? "") : "",
  ].join(" ");

  if (matches(CONTEXT_PATTERNS, text)) {
    return {
      kind: "context_too_long",
      message:
        "This conversation is now longer than this model can hold. Start a new chat, or pick a model with a larger context window.",
      retryable: false,
      operatorAction: false,
    };
  }

  // Order matters: a 401 or 402 carrying billing wording is an account problem,
  // and checking that before the generic status codes keeps the message honest.
  if (matches(ACCOUNT_PATTERNS, text) || status === 401 || status === 402) {
    return {
      kind: "account_problem",
      message:
        "This model is not available on ChatISA right now. Pick another model from the list. Someone has been notified.",
      retryable: false,
      operatorAction: true,
    };
  }

  if (matches(UNAVAILABLE_PATTERNS, text) || status === 404) {
    return {
      kind: "model_unavailable",
      message:
        "This model is not being served at the moment. Pick another model from the list. Someone has been notified.",
      retryable: false,
      operatorAction: true,
    };
  }

  if (status === 429) {
    return {
      kind: "rate_limited",
      message:
        "This model is busy right now. Wait a few seconds and try again, or pick another model.",
      retryable: true,
      operatorAction: false,
    };
  }

  if ((status !== undefined && status >= 500) || isNetworkError(error)) {
    return {
      kind: "transient",
      message:
        "The model provider had a problem. Your message was kept, so you can try again.",
      retryable: true,
      operatorAction: false,
    };
  }

  return {
    kind: "unknown",
    message:
      "The model couldn't complete that response. Your message was kept, so you can try again.",
    retryable: true,
    operatorAction: false,
  };
}
function isNetworkError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  const code = (error as NodeJS.ErrnoException).code;
  return (
    code === "ECONNRESET" ||
    code === "ETIMEDOUT" ||
    code === "ENOTFOUND" ||
    code === "EAI_AGAIN" ||
    /fetch failed|network error|socket hang up/i.test(error.message)
  );
}
