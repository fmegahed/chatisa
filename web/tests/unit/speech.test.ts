import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const generate = vi.fn();
const grant = vi.fn();

vi.mock("@deepgram/sdk", () => ({
  DeepgramClient: class {
    auth = { v1: { tokens: { grant } } };
    speak = { v1: { audio: { generate } } };
  },
}));

function audioResponse(bytes: Uint8Array) {
  return {
    stream: () =>
      new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(bytes);
          controller.close();
        },
      }),
  };
}

describe("Deepgram server layer", () => {
  beforeEach(async () => {
    vi.resetModules();
    generate.mockReset();
    grant.mockReset();
    process.env.DEEPGRAM_TOKEN = "test-key-not-a-real-credential";
    const { clearSpeechCache } = await import("@/lib/speech/deepgram");
    clearSpeechCache();
  });

  afterEach(() => {
    delete process.env.DEEPGRAM_TOKEN;
  });

  it("reports speech as unconfigured when the key is absent or blank", async () => {
    const { isSpeechConfigured } = await import("@/lib/speech/deepgram");
    expect(isSpeechConfigured()).toBe(true);
    process.env.DEEPGRAM_TOKEN = "   ";
    expect(isSpeechConfigured()).toBe(false);
    delete process.env.DEEPGRAM_TOKEN;
    expect(isSpeechConfigured()).toBe(false);
  });

  it("refuses to work rather than calling Deepgram with no credential", async () => {
    delete process.env.DEEPGRAM_TOKEN;
    const { synthesizeSpeech, SpeechNotConfiguredError } = await import(
      "@/lib/speech/deepgram"
    );
    await expect(synthesizeSpeech("hello")).rejects.toBeInstanceOf(
      SpeechNotConfiguredError,
    );
    expect(generate).not.toHaveBeenCalled();
  });

  it("mints a short-lived token, not a long-lived one", async () => {
    grant.mockResolvedValue({ access_token: "jwt", expires_in: 60 });
    const { grantSpeechToken, TOKEN_TTL_SECONDS } = await import(
      "@/lib/speech/deepgram"
    );

    const result = await grantSpeechToken();
    expect(grant).toHaveBeenCalledWith({ ttl_seconds: TOKEN_TTL_SECONDS });
    // A token only has to survive the handshake, so a long life buys nothing
    // and widens the blast radius if it leaks.
    expect(TOKEN_TTL_SECONDS).toBeLessThanOrEqual(120);
    expect(result.accessToken).toBe("jwt");
  });

  it("asks for MP3 explicitly rather than trusting the default", async () => {
    // Deepgram's own docs disagree about the REST default encoding, and the
    // streaming endpoint emits raw PCM that an <audio> element cannot play.
    generate.mockResolvedValue(audioResponse(new Uint8Array([1, 2, 3])));
    const { synthesizeSpeech } = await import("@/lib/speech/deepgram");

    const result = await synthesizeSpeech("Tell me about yourself.");
    expect(generate).toHaveBeenCalledWith(
      expect.objectContaining({ encoding: "mp3" }),
    );
    expect(result.contentType).toBe("audio/mpeg");
    expect(result.cached).toBe(false);
  });

  it("serves repeated questions from cache without calling Deepgram again", async () => {
    // This is the load-bearing one: Deepgram's TTS concurrency ceiling is the
    // narrowest resource in the system, and interview questions repeat heavily
    // across students, so a class starting together must not mean a synthesis
    // request per student per question.
    generate.mockResolvedValue(audioResponse(new Uint8Array([9, 9, 9])));
    const { synthesizeSpeech } = await import("@/lib/speech/deepgram");

    const first = await synthesizeSpeech("Why this role?");
    const second = await synthesizeSpeech("Why this role?");
    const spaced = await synthesizeSpeech("  Why this role?  ");

    expect(first.cached).toBe(false);
    expect(second.cached).toBe(true);
    expect(spaced.cached).toBe(true);
    expect(generate).toHaveBeenCalledTimes(1);
  });

  it("treats different questions as different audio", async () => {
    generate.mockResolvedValue(audioResponse(new Uint8Array([4])));
    const { synthesizeSpeech } = await import("@/lib/speech/deepgram");
    await synthesizeSpeech("Question one?");
    await synthesizeSpeech("Question two?");
    expect(generate).toHaveBeenCalledTimes(2);
  });

  it("fails loudly rather than caching an empty recording", async () => {
    generate.mockResolvedValue(audioResponse(new Uint8Array([])));
    const { synthesizeSpeech } = await import("@/lib/speech/deepgram");
    await expect(synthesizeSpeech("silence")).rejects.toThrow(/no audio/i);

    // A cached empty file would be silently un-listenable forever after.
    generate.mockResolvedValue(audioResponse(new Uint8Array([1])));
    const retry = await synthesizeSpeech("silence");
    expect(retry.cached).toBe(false);
    expect(retry.audio.byteLength).toBe(1);
  });

  it("never puts the account key in the cache key", async () => {
    const { speechCacheKey } = await import("@/lib/speech/deepgram");
    const key = speechCacheKey("hello", "aura-2-thalia-en");
    expect(key).not.toContain("test-key-not-a-real-credential");
    expect(key).toMatch(/^[a-f0-9]{64}$/);
  });
});

describe("microphone failure messages", () => {
  it("tells the student what to do for each browser failure", async () => {
    const { describeMicrophoneError } = await import(
      "@/lib/speech/live-transcription"
    );

    const denied = describeMicrophoneError({ name: "NotAllowedError" });
    expect(denied).toMatch(/allow the microphone/i);

    const missing = describeMicrophoneError({ name: "NotFoundError" });
    expect(missing).toMatch(/no microphone/i);

    const busy = describeMicrophoneError({ name: "NotReadableError" });
    expect(busy).toMatch(/in use by another app/i);
  });

  it("always offers typing as a way out", async () => {
    const { describeMicrophoneError } = await import(
      "@/lib/speech/live-transcription"
    );
    // The legacy page left a student with a denied microphone no path through
    // the feature at all. Every message here names the alternative.
    for (const name of [
      "NotAllowedError",
      "NotFoundError",
      "NotReadableError",
      "SecurityError",
      "SomethingUnexpected",
    ]) {
      expect(describeMicrophoneError({ name })).toMatch(/type your answer/i);
    }
  });
});
