import "server-only";
import { createHash } from "node:crypto";
import { DeepgramClient } from "@deepgram/sdk";
import { logger } from "@/lib/log";

/**
 * Server-side Deepgram access.
 *
 * The account key lives here and never crosses the network boundary to a
 * browser. The browser needs to talk to Deepgram directly (relaying every
 * student's audio through our Windows Server would not survive a class of 80),
 * so it gets a short-lived scoped token minted here instead.
 *
 * The legacy Streamlit page did the opposite: it embedded the OpenAI client
 * secret in the page HTML and in the iframe URL fragment. That is the mistake
 * this module exists to avoid.
 */

/** Deepgram's default token lifetime is 30s; 60s absorbs a slow handshake. */
export const TOKEN_TTL_SECONDS = 60;

/**
 * Transcription model. Nova-3 is chosen over the newer turn-based Flux because
 * it emits true interim results, which is what live captions are built from.
 * Captions are an accessibility requirement here, not a nicety, so the model
 * that produces them wins.
 */
export const STT_MODEL = "nova-3";

/** Interviewer voice. Aura-2, natural and even-paced rather than bright. */
export const TTS_MODEL = "aura-2-thalia-en";

export class SpeechNotConfiguredError extends Error {
  constructor() {
    super("No Deepgram credential is configured on this server.");
    this.name = "SpeechNotConfiguredError";
  }
}

function apiKey(): string {
  const key = process.env.DEEPGRAM_TOKEN;
  if (!key || key.trim() === "") throw new SpeechNotConfiguredError();
  return key;
}

export function isSpeechConfigured(): boolean {
  const key = process.env.DEEPGRAM_TOKEN;
  return typeof key === "string" && key.trim() !== "";
}

function client(): DeepgramClient {
  return new DeepgramClient({ apiKey: apiKey() });
}

export interface SpeechGrant {
  accessToken: string;
  expiresInSeconds: number;
  sttModel: string;
}

/**
 * Mints a short-lived token for one browser session.
 *
 * The token only has to be valid at the moment the WebSocket handshake
 * completes; the connection then survives its expiry. So a short TTL costs
 * nothing and keeps the blast radius small if one leaks. Callers must
 * authenticate the student before calling this.
 */
export async function grantSpeechToken(): Promise<SpeechGrant> {
  const result = await client().auth.v1.tokens.grant({
    ttl_seconds: TOKEN_TTL_SECONDS,
  });
  return {
    accessToken: result.access_token,
    expiresInSeconds: result.expires_in ?? TOKEN_TTL_SECONDS,
    sttModel: STT_MODEL,
  };
}

/** What a readiness probe found. `state` is what a human should act on. */
export interface SpeechProbe {
  /**
   * "ok": the credential works, so Interview Mentor will speak.
   * "not-configured": DEEPGRAM_TOKEN is absent. A deliberate choice on a server
   *   that does not want speech, and the module degrades to typed answers.
   * "broken": a token IS set but Deepgram rejected or failed it. Always a
   *   problem, and the one state that looks identical to "ok" from outside.
   */
  state: "ok" | "not-configured" | "broken";
  /** A short, credential-free explanation. Safe to show in a health payload. */
  detail: string;
}

/**
 * Asks Deepgram whether our credential actually works.
 *
 * This exists because "the interviewer has no voice in production" was
 * undiagnosable from outside the server (reported 2026-07-25). Every candidate
 * looked equally plausible: an unset token, a revoked token, an account out of
 * credit, a blocked outbound port. /api/health already reported whether
 * DEEPGRAM_TOKEN is PRESENT, which distinguishes none of those, because presence
 * is not validity.
 *
 * Token minting is the probe rather than synthesis: it exercises the same
 * credential, the same SDK, and the same outbound path, and it is free. It does
 * not prove the text-to-speech product is enabled on the account, which is a
 * deliberate limit, stated here so nobody reads more into a green result.
 */
export async function probeSpeech(): Promise<SpeechProbe> {
  if (!isSpeechConfigured()) {
    return {
      state: "not-configured",
      detail:
        "DEEPGRAM_TOKEN is not set, so Interview Mentor will not speak. Students can still type answers.",
    };
  }
  try {
    const grant = await grantSpeechToken();
    if (!grant.accessToken) {
      return {
        state: "broken",
        detail: "Deepgram accepted the request but returned no token.",
      };
    }
    return {
      state: "ok",
      detail: `Deepgram minted a ${grant.expiresInSeconds}s token. Speech should work.`,
    };
  } catch (err) {
    // Never echo the provider's message verbatim: it can quote the credential
    // back. A status code is the useful, safe part.
    const status =
      typeof err === "object" && err !== null && "status" in err
        ? String((err as { status: unknown }).status)
        : null;
    logger.error({ err: String(err), status }, "speech probe failed");
    return {
      state: "broken",
      detail: status
        ? `DEEPGRAM_TOKEN is set but Deepgram refused it (HTTP ${status}). Check the key and the account's credit.`
        : "DEEPGRAM_TOKEN is set but Deepgram could not be reached. Check the key, the account's credit, and outbound HTTPS from this server.",
    };
  }
}

/**
 * Spoken audio for one interview question.
 *
 * Cached by exact text, which matters more than it looks: Deepgram's
 * text-to-speech concurrency ceiling is far lower than its transcription
 * ceiling, so a class starting together is throttled by speech synthesis long
 * before transcription. Interview questions repeat heavily across students, so
 * caching turns the common case into no request at all.
 */
const audioCache = new Map<string, { audio: Buffer; lastUsed: number }>();
const MAX_CACHE_ENTRIES = 500;

export function speechCacheKey(text: string, model: string): string {
  // Separator written as an explicit escape rather than embedded as a raw
  // byte, so it is visible to a reader. A null is a good separator here
  // because it cannot occur in a model id or in question text, so two
  // different pairs can never collide into one key.
  return createHash("sha256")
    .update(`${model}\u0000${text}`)
    .digest("hex");
}

function rememberAudio(key: string, audio: Buffer): void {
  if (audioCache.size >= MAX_CACHE_ENTRIES) {
    // Evict least recently used. A plain Map keeps this simple and the cache
    // is small enough that a linear scan on eviction is not worth optimising.
    let oldestKey: string | null = null;
    let oldest = Infinity;
    for (const [k, v] of audioCache) {
      if (v.lastUsed < oldest) {
        oldest = v.lastUsed;
        oldestKey = k;
      }
    }
    if (oldestKey) audioCache.delete(oldestKey);
  }
  audioCache.set(key, { audio, lastUsed: Date.now() });
}

export function clearSpeechCache(): void {
  audioCache.clear();
}

export interface SynthesisResult {
  audio: Buffer;
  cached: boolean;
  contentType: string;
}

/**
 * Turns question text into MP3.
 *
 * `encoding` is set explicitly because Deepgram's own documentation disagrees
 * with itself about the REST default, and because the streaming endpoint can
 * only emit raw PCM, which an <audio> element cannot play directly.
 */
export async function synthesizeSpeech(text: string): Promise<SynthesisResult> {
  const trimmed = text.trim();
  if (trimmed === "") throw new Error("Nothing to speak.");

  const key = speechCacheKey(trimmed, TTS_MODEL);
  const hit = audioCache.get(key);
  if (hit) {
    hit.lastUsed = Date.now();
    return { audio: hit.audio, cached: true, contentType: "audio/mpeg" };
  }

  const response = await client().speak.v1.audio.generate({
    text: trimmed,
    model: TTS_MODEL,
    encoding: "mp3",
  });

  // `.stream()` yields a web ReadableStream, which is not async-iterable in
  // this Node version, so it is drained through a reader rather than for-await.
  const stream = response.stream();
  if (!stream) throw new Error("Deepgram returned no audio stream.");

  const chunks: Uint8Array[] = [];
  const reader = stream.getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }
  const audio = Buffer.concat(chunks);
  if (audio.byteLength === 0) {
    throw new Error("Deepgram returned no audio.");
  }

  rememberAudio(key, audio);
  logger.info(
    { chars: trimmed.length, bytes: audio.byteLength, model: TTS_MODEL },
    "speech synthesized",
  );
  return { audio, cached: false, contentType: "audio/mpeg" };
}
