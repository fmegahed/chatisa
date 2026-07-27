"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import {
  isSpeechInputSupported,
  startLiveTranscription,
  type LiveTranscriptionHandle,
  type TranscriptionStatus,
} from "@/lib/speech/live-transcription";

/** Support never changes within a session, so there is nothing to subscribe to. */
function subscribeNever(): () => void {
  return () => {};
}

/**
 * Answering by voice.
 *
 * Speech is strictly an enhancement here. The textarea below is always present,
 * always editable, and is what actually gets submitted. Dictation just types
 * into it. The legacy module had no text path at all, so a student who could
 * not speak, had no microphone, or was sitting in a shared room had no way
 * through the feature.
 *
 * Because the textarea is the source of truth, a wrong transcription is a typo
 * the student can fix rather than a wrong answer they are stuck with.
 */
export function SpeechAnswer(props: {
  value: string;
  /** Accepts an updater so dictation can append to the latest value without
   * reading a ref during render. */
  onChange: (update: (previous: string) => string) => void;
  disabled: boolean;
  /** Told when speech contributed, so the answer can be recorded as spoken. */
  onSpokenChange: (spoken: boolean) => void;
  textareaId: string;
}) {
  const [status, setStatus] = useState<TranscriptionStatus>("idle");
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);
  const handleRef = useRef<LiveTranscriptionHandle | null>(null);

  // Microphone support is a client-only fact. useSyncExternalStore gives the
  // server a definite `false` and the client the real answer, so hydration
  // matches without setting state from an effect.
  const supported = useSyncExternalStore(
    subscribeNever,
    isSpeechInputSupported,
    () => false,
  );

  useEffect(() => {
    return () => {
      void handleRef.current?.stop();
    };
  }, []);

  async function start() {
    setError(null);
    setInterim("");
    try {
      const handle = await startLiveTranscription({
        fetchToken: async () => {
          const res = await fetch("/api/speech/token", { method: "POST" });
          const body = await res.json().catch(() => ({}));
          if (!res.ok) {
            throw new Error(
              body.error ?? "Speech could not be started. You can type instead.",
            );
          }
          return { accessToken: body.accessToken, sttModel: body.sttModel };
        },
        onInterim: (text) => setInterim(text),
        onFinal: (text) => {
          setInterim("");
          // Append to whatever is in the box right now, including anything the
          // student typed while speaking.
          props.onChange((previous) =>
            previous.trim() === "" ? text : `${previous} ${text}`,
          );
          props.onSpokenChange(true);
        },
        onStatus: setStatus,
        onError: (message) => setError(message),
      });
      handleRef.current = handle;
    } catch (err) {
      setStatus("error");
      setError(
        err instanceof Error
          ? err.message
          : "Speech could not be started. You can type your answer instead.",
      );
    }
  }

  async function stop() {
    await handleRef.current?.stop();
    handleRef.current = null;
    setInterim("");
  }

  const listening = status === "listening";
  const busy = status === "requesting-microphone" || status === "connecting";

  return (
    <div>
      <label htmlFor={props.textareaId} className="block font-bold">
        Your answer
      </label>
      <p className="mt-1 text-sm text-dark-tan">
        Type it, or use the microphone and edit what it hears. Either way, this
        box is what gets submitted.
      </p>

      <textarea
        id={props.textareaId}
        value={props.value}
        onChange={(e) => {
          const next = e.target.value;
          props.onChange(() => next);
        }}
        disabled={props.disabled}
        rows={7}
        className="mt-2 w-full rounded-card border border-medium-tan bg-paper p-3 leading-relaxed"
        placeholder="Answer in your own words, as you would out loud."
      />

      {supported ? (
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={listening ? stop : start}
            disabled={props.disabled || busy}
            className="rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan disabled:cursor-not-allowed disabled:border-medium-gray disabled:text-medium-gray"
          >
            {listening
              ? "Stop dictating"
              : busy
                ? "Starting..."
                : "Dictate your answer"}
          </button>

          {/*
            Status is text, not colour alone, so it does not depend on being
            able to see a coloured dot.
          */}
          <p role="status" className="text-sm">
            {listening
              ? "Listening. Speak normally, and pause when you are done."
              : busy
                ? "Starting the microphone..."
                : "Microphone off."}
          </p>
        </div>
      ) : (
        <p role="status" className="mt-2 text-sm text-dark-tan">
          Dictation is not available in this browser, so type your answer.
        </p>
      )}

      {/*
        Interim words appear here rather than being written straight into the
        textarea, because they get revised as you speak and watching a textarea
        rewrite itself is disorienting. Marked aria-live="polite" so a screen
        reader is not interrupted on every syllable.
      */}
      {interim ? (
        <p
          aria-live="polite"
          className="mt-2 rounded-card bg-light-tan px-3 py-2 text-sm italic"
        >
          Hearing: {interim}
        </p>
      ) : null}

      {error ? (
        <p role="alert" className="mt-2 text-sm font-bold text-miami-red">
          {error}
        </p>
      ) : null}
    </div>
  );
}
