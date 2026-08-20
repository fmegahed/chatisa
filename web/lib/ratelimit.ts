/**
 * Fixed-window rate limiter, in-process. ChatISA runs as a single Node
 * process on one server, so an in-memory limiter is sufficient and adds no
 * infrastructure. If the app is ever scaled out, replace this with a shared
 * store; the interface stays the same.
 */
export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  /** Seconds until the current window resets. */
  retryAfterSeconds: number;
}

interface Bucket {
  count: number;
  resetAt: number;
}

const buckets = new Map<string, Bucket>();

export function checkRateLimit(
  key: string,
  opts: { limit: number; windowMs: number },
  now: number = Date.now(),
): RateLimitResult {
  const existing = buckets.get(key);
  if (!existing || now >= existing.resetAt) {
    buckets.set(key, { count: 1, resetAt: now + opts.windowMs });
    return {
      allowed: true,
      remaining: opts.limit - 1,
      retryAfterSeconds: Math.ceil(opts.windowMs / 1000),
    };
  }
  if (existing.count >= opts.limit) {
    return {
      allowed: false,
      remaining: 0,
      retryAfterSeconds: Math.max(1, Math.ceil((existing.resetAt - now) / 1000)),
    };
  }
  existing.count += 1;
  return {
    allowed: true,
    remaining: opts.limit - existing.count,
    retryAfterSeconds: Math.max(1, Math.ceil((existing.resetAt - now) / 1000)),
  };
}

/** Test helper: clears all counters. */
export function resetRateLimits(): void {
  buckets.clear();
}

/** Chat requests per user per minute. Raised from 20 to 60 on 2026-07-24
 * (professor: keep limits non-restrictive): the agentic loop legitimately
 * sends several requests per turn, and 60/min stays far above any human pace
 * while still stopping a stuck client or a leaked session from running free.
 * Env-overridable per deployment; the e2e suite sets its own higher value
 * because the whole parallel run shares one account. */
export const CHAT_RATE_LIMIT = {
  limit: Number(process.env.CHATISA_CHAT_LIMIT_PER_MINUTE ?? 60),
  windowMs: 60_000,
};

/**
 * Inline editor completions fire far more often than chat turns (on typing
 * pauses), so the ceiling is higher, but still bounded so a stuck client cannot
 * hammer the provider. Overridable for tests and busy terms.
 */
export const COMPLETION_RATE_LIMIT = {
  limit: Number(process.env.CHATISA_COMPLETION_LIMIT_PER_MINUTE ?? 120),
  windowMs: 60_000,
};

/**
 * Uploads are the most expensive thing a student can ask for, so they are
 * limited more tightly than chat turns. Set generously enough that someone
 * uploading several chapters in a row is not blocked, and overridable so
 * automated tests and busy terms can be tuned without a code change.
 */
export const EXAM_UPLOAD_RATE_LIMIT = {
  limit: Number(process.env.CHATISA_UPLOAD_LIMIT_PER_MINUTE ?? 30),
  windowMs: 60_000,
};

/**
 * Building an exam is a handful of model calls, so it is limited separately
 * from ordinary chat turns and from uploads.
 */
export const EXAM_GENERATE_RATE_LIMIT = {
  limit: Number(process.env.CHATISA_EXAM_LIMIT_PER_MINUTE ?? 20),
  windowMs: 60_000,
};

/**
 * Job Scout project generation (scaffold and polish share one bucket: both
 * are "design me a project" model calls). Was an inline {limit: 4} in each
 * route until v6.3.0, when the e2e suite's parallel projects started
 * legitimately exceeding it from one shared account; now overridable like
 * every other limit here.
 */
export const SCOUT_PROJECT_RATE_LIMIT = {
  limit: Number(process.env.CHATISA_SCOUT_PROJECT_LIMIT_PER_MINUTE ?? 4),
  windowMs: 60_000,
};

/**
 * Minting a speech token is cheap, but the browser re-mints whenever it
 * reconnects, so this is set to tolerate a flaky network without becoming a
 * free way to farm credentials.
 */
export const SPEECH_TOKEN_RATE_LIMIT = {
  limit: Number(process.env.CHATISA_SPEECH_TOKEN_LIMIT_PER_MINUTE ?? 20),
  windowMs: 60_000,
};

/**
 * Speech synthesis is limited harder than anything else, because Deepgram's
 * text-to-speech concurrency ceiling is the narrowest resource in the system
 * and one student holding it open costs every other student in the class.
 */
export const SPEECH_SYNTHESIS_RATE_LIMIT = {
  limit: Number(process.env.CHATISA_SPEECH_TTS_LIMIT_PER_MINUTE ?? 30),
  windowMs: 60_000,
};
