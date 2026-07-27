/**
 * Browser-side live transcription against Deepgram.
 *
 * Runs in the student's tab and talks to Deepgram directly. Relaying audio for
 * a class of 80 through the Windows Server would make the server the
 * bottleneck, so the browser holds the socket and the server only mints the
 * short-lived credential.
 *
 * Everything here is an enhancement. The interview is fully answerable by
 * typing, so every failure path in this file degrades to "keep typing" rather
 * than blocking the student.
 */

export type TranscriptionStatus =
  | "idle"
  | "requesting-microphone"
  | "connecting"
  | "listening"
  | "stopped"
  | "error";

export interface TranscriptionCallbacks {
  /** Provisional text. Will be revised, so render it as tentative. */
  onInterim(text: string): void;
  /** Text that will not change. Append it to the answer. */
  onFinal(text: string): void;
  /** Deepgram detected the end of an utterance. */
  onUtteranceEnd?(): void;
  onStatus(status: TranscriptionStatus): void;
  /** Plain-language, already safe to show a student. */
  onError(message: string): void;
}

/**
 * Deepgram closes an idle socket after about ten seconds. A student thinking
 * about a hard interview question passes that easily, and losing their
 * connection mid-thought would be the single most annoying possible bug, so a
 * keepalive runs well inside the window.
 */
const KEEPALIVE_INTERVAL_MS = 4_000;

/** Small enough to feel live, large enough not to flood the socket. */
const AUDIO_TIMESLICE_MS = 250;

/**
 * How long a pause has to last before Deepgram calls the utterance finished.
 *
 * Deepgram's default fires almost immediately, which cuts a student off while
 * they are recalling an example. How long a person naturally pauses varies far
 * too much between people for one value to suit everyone, so this is a dial the
 * student controls rather than a constant we guess at.
 */
export const MIN_ENDPOINTING_MS = 2_000;
export const MAX_ENDPOINTING_MS = 30_000;
export const DEFAULT_ENDPOINTING_MS = 10_000;

export function clampEndpointing(ms: number): number {
  if (!Number.isFinite(ms)) return DEFAULT_ENDPOINTING_MS;
  return Math.min(MAX_ENDPOINTING_MS, Math.max(MIN_ENDPOINTING_MS, Math.round(ms)));
}

const DEEPGRAM_LIVE_URL = "wss://api.deepgram.com/v1/listen";

/** Preference order; Safari has historically supported fewer of these. */
const CANDIDATE_MIME_TYPES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/ogg;codecs=opus",
  "audio/mp4",
];

export function pickSupportedMimeType(): string | null {
  if (typeof MediaRecorder === "undefined") return null;
  return (
    CANDIDATE_MIME_TYPES.find((type) => MediaRecorder.isTypeSupported(type)) ??
    null
  );
}

export function isSpeechInputSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getUserMedia === "function" &&
    typeof MediaRecorder !== "undefined" &&
    typeof WebSocket !== "undefined" &&
    pickSupportedMimeType() !== null
  );
}

/** Turns a getUserMedia rejection into something a student can act on. */
export function describeMicrophoneError(err: unknown): string {
  const name = (err as { name?: string })?.name ?? "";
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Your browser blocked microphone access. Allow the microphone for this site in your browser settings, or type your answer instead.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No microphone was found. Plug one in or type your answer instead.";
    case "NotReadableError":
      return "Your microphone is in use by another app. Close that app, or type your answer instead.";
    default:
      return "The microphone could not be started. You can type your answer instead.";
  }
}

export interface LiveTranscriptionHandle {
  stop(): Promise<void>;
  readonly status: TranscriptionStatus;
}

interface StartOptions extends TranscriptionCallbacks {
  /** Mints a fresh token. Called per connection attempt, never cached, because
   * a reconnect after a long answer needs a token that has not expired. */
  fetchToken(): Promise<{ accessToken: string; sttModel: string }>;
  /** Silence before an answer counts as finished. Student-controlled. */
  endpointingMs?: number;
}

export async function startLiveTranscription(
  options: StartOptions,
): Promise<LiveTranscriptionHandle> {
  let status: TranscriptionStatus = "idle";
  const setStatus = (next: TranscriptionStatus) => {
    status = next;
    options.onStatus(next);
  };

  const mimeType = pickSupportedMimeType();
  if (!mimeType) {
    throw new Error("This browser cannot record audio.");
  }

  setStatus("requesting-microphone");
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch (err) {
    setStatus("error");
    throw new Error(describeMicrophoneError(err));
  }

  const cleanupStream = () => stream.getTracks().forEach((t) => t.stop());

  setStatus("connecting");
  let token: { accessToken: string; sttModel: string };
  try {
    token = await options.fetchToken();
  } catch (err) {
    cleanupStream();
    setStatus("error");
    throw err instanceof Error ? err : new Error("Speech could not be started.");
  }

  // Containerised audio carries its own encoding and sample rate, so passing
  // those explicitly alongside a container is a documented way to get
  // confusing failures. They are deliberately absent.
  const params = new URLSearchParams({
    model: token.sttModel,
    language: "en",
    interim_results: "true",
    punctuate: "true",
    smart_format: "true",
    // Student-controlled: how long a silence counts as "I have finished". This
    // is the sole pause knob. Deepgram's separate `utterance_end_ms` was tried
    // and rejected: it caps below 10500 (verified 2026-07-21, the handshake
    // 400s above it), and because our client treats its UtteranceEnd event the
    // same as speech_final, a capped value would have fired early and defeated
    // the dial. `endpointing` accepts the full 2 to 30 second range and drives
    // speech_final, which is exactly the "pause means done" signal.
    endpointing: String(
      clampEndpointing(options.endpointingMs ?? DEFAULT_ENDPOINTING_MS),
    ),
  });

  // Browsers cannot set headers on a WebSocket, so Deepgram takes the
  // credential through the subprotocol array instead.
  const socket = new WebSocket(`${DEEPGRAM_LIVE_URL}?${params}`, [
    "bearer",
    token.accessToken,
  ]);

  let recorder: MediaRecorder | null = null;
  let keepAlive: ReturnType<typeof setInterval> | null = null;
  let stopped = false;

  const teardown = () => {
    if (keepAlive !== null) {
      clearInterval(keepAlive);
      keepAlive = null;
    }
    if (recorder && recorder.state !== "inactive") recorder.stop();
    recorder = null;
    cleanupStream();
  };

  socket.addEventListener("open", () => {
    setStatus("listening");

    recorder = new MediaRecorder(stream, { mimeType });
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0 && socket.readyState === WebSocket.OPEN) {
        socket.send(event.data);
      }
    });
    recorder.start(AUDIO_TIMESLICE_MS);

    keepAlive = setInterval(() => {
      if (socket.readyState !== WebSocket.OPEN) return;
      // Must be a text frame. Sent as binary it is treated as audio.
      socket.send(JSON.stringify({ type: "KeepAlive" }));
    }, KEEPALIVE_INTERVAL_MS);
  });

  socket.addEventListener("message", (event) => {
    if (typeof event.data !== "string") return;
    let message: unknown;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }

    const parsed = message as {
      type?: string;
      is_final?: boolean;
      speech_final?: boolean;
      channel?: { alternatives?: Array<{ transcript?: string }> };
    };

    if (parsed.type === "UtteranceEnd") {
      options.onUtteranceEnd?.();
      return;
    }
    if (parsed.type !== "Results") return;

    const text = parsed.channel?.alternatives?.[0]?.transcript ?? "";
    if (text.trim() === "") return;

    if (parsed.is_final) options.onFinal(text);
    else options.onInterim(text);

    if (parsed.speech_final) options.onUtteranceEnd?.();
  });

  socket.addEventListener("error", () => {
    if (stopped) return;
    // The browser deliberately withholds WebSocket error detail, so there is
    // nothing more specific to offer here honestly.
    options.onError(
      "The speech connection dropped. You can start it again, or type your answer.",
    );
    setStatus("error");
    teardown();
  });

  socket.addEventListener("close", () => {
    if (stopped) return;
    setStatus("stopped");
    teardown();
  });

  return {
    get status() {
      return status;
    },
    async stop() {
      if (stopped) return;
      stopped = true;

      // Flush before closing. A bare close can drop the tail of the last
      // sentence, which is exactly the part a student just finished saying.
      if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: "CloseStream" }));
        await new Promise<void>((resolve) => {
          const done = () => resolve();
          socket.addEventListener("close", done, { once: true });
          setTimeout(done, 1_500);
        });
      }
      teardown();
      if (socket.readyState !== WebSocket.CLOSED) socket.close();
      setStatus("stopped");
    },
  };
}
