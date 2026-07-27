import { NextResponse } from "next/server";
import { z } from "zod";
import { auth } from "@/lib/auth";
import { logger } from "@/lib/log";
import {
  SpeechNotConfiguredError,
  synthesizeSpeech,
} from "@/lib/speech/deepgram";
import {
  SPEECH_SYNTHESIS_RATE_LIMIT,
  checkRateLimit,
} from "@/lib/ratelimit";

export const runtime = "nodejs";

/** One interview question. Long enough for a case prompt, bounded so a pasted
 * document cannot be turned into an expensive audiobook. */
const MAX_SPEAK_CHARS = 1200;

const speakSchema = z.object({
  text: z.string().min(1).max(MAX_SPEAK_CHARS),
});

function errorResponse(status: number, message: string, extra?: object) {
  return NextResponse.json({ error: message, ...extra }, { status });
}

/**
 * Speaks a line of interviewer text.
 *
 * Audio is returned as MP3 so the browser can play it from an ordinary <audio>
 * element, which keeps the player accessible (real controls, real keyboard
 * support) rather than a bespoke Web Audio widget.
 */
export async function POST(req: Request) {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) return errorResponse(401, "Sign in to continue.");

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return errorResponse(400, "Request body must be JSON.");
  }

  const parsed = speakSchema.safeParse(body);
  if (!parsed.success) {
    return errorResponse(400, "That request wasn't valid.");
  }

  const limit = checkRateLimit(
    `speech-tts:${userEmail}`,
    SPEECH_SYNTHESIS_RATE_LIMIT,
  );
  if (!limit.allowed) {
    return errorResponse(
      429,
      "Too much speech requested in a short time. Wait a moment, or read the question instead.",
      { retryAfterSeconds: limit.retryAfterSeconds },
    );
  }

  try {
    const result = await synthesizeSpeech(parsed.data.text);
    return new NextResponse(new Uint8Array(result.audio), {
      headers: {
        "content-type": result.contentType,
        "content-length": String(result.audio.byteLength),
        // Private: the audio is one student's interview question. Cacheable in
        // their own browser so replaying a question costs nothing, never in a
        // shared cache.
        "cache-control": "private, max-age=3600",
        "x-speech-cache": result.cached ? "hit" : "miss",
      },
    });
  } catch (err) {
    if (err instanceof SpeechNotConfiguredError) {
      return errorResponse(
        503,
        "Speech is not set up on this server. The question is shown as text.",
      );
    }
    logger.error({ err: String(err) }, "speech synthesis failed");
    return errorResponse(
      502,
      "That question could not be read aloud. It is shown as text.",
    );
  }
}
