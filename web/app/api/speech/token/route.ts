import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import {
  SpeechNotConfiguredError,
  describeErrorChain,
  grantSpeechToken,
} from "@/lib/speech/deepgram";
import { SPEECH_TOKEN_RATE_LIMIT, checkRateLimit } from "@/lib/ratelimit";

// The Deepgram SDK depends on `ws`, a native Node module, so this cannot run
// on the edge runtime.
export const runtime = "nodejs";
// A credential must never be cached or revalidated from a shared cache.
export const dynamic = "force-dynamic";

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

/**
 * Mints a short-lived Deepgram token for the signed-in student's browser.
 *
 * The account key stays on the server. What the browser receives is scoped to
 * the voice APIs, cannot reach Deepgram's management endpoints, and expires in
 * about a minute. It is only required to be valid at the moment the WebSocket
 * handshake completes, so a short life does not shorten the interview.
 *
 * POST rather than GET on purpose: this has a side effect at Deepgram, and it
 * must never be reachable by prefetch, link preview, or browser cache.
 */
export async function POST() {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  const limit = checkRateLimit(`speech-token:${userEmail}`, SPEECH_TOKEN_RATE_LIMIT);
  if (!limit.allowed) {
    return errorResponse(
      429,
      "Too many attempts to start speech in a short time. Wait a moment and try again.",
      { retryAfterSeconds: limit.retryAfterSeconds },
    );
  }

  try {
    const grant = await grantSpeechToken();
    return NextResponse.json(grant, {
      headers: { "cache-control": "no-store" },
    });
  } catch (err) {
    if (err instanceof SpeechNotConfiguredError) {
      logger.error({}, "speech requested but DEEPGRAM_TOKEN is not set");
      return errorResponse(
        503,
        "Speech is not set up on this server. You can still type your answers.",
      );
    }
    // Never surface the provider's message: it can echo credentials back.
    // The cause chain goes to the server log only, where it names the real
    // failure (ETIMEDOUT vs ENOTFOUND) instead of fetch's opaque wrapper.
    logger.error({ err: describeErrorChain(err) }, "speech token mint failed");
    return errorResponse(
      502,
      "Speech could not be started just now. You can still type your answers.",
    );
  }
}
