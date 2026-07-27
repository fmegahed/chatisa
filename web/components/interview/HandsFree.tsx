"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  isSpeechInputSupported,
  startLiveTranscription,
  type LiveTranscriptionHandle,
} from "@/lib/speech/live-transcription";
import { PauseDial } from "@/components/interview/PauseDial";

/**
 * Hands-free interviewing.
 *
 * The question plays, the microphone opens by itself when it finishes, the
 * student's pause ends the answer, and it submits. No buttons in the normal
 * flow, because pressing a button between every sentence is not what an
 * interview feels like.
 *
 * Three things keep that from being maddening:
 *
 * 1. A pause before submitting. Deepgram calling the end of an utterance is a
 *    guess, and a cough or a long think can trigger it. The pause is the window
 *    in which the student can keep talking or edit, and it is why this is safe
 *    to run without confirmation.
 * 2. Nothing is ever only hands-free. Typing and manual control are always one
 *    click away, so a student in a shared room or without a microphone is never
 *    stuck. This is the same requirement that made the legacy module unusable
 *    when it was missing.
 * 3. Autoplay is attempted, never assumed. Browsers block audio that does not
 *    follow a user gesture, so a rejected play() falls back to a visible
 *    control rather than silently stalling the loop.
 */

/** How long the student has to intervene before the answer is sent. */
export const AUTO_SUBMIT_DELAY_MS = 5_000;

export type HandsFreePhase =
  | "idle"
  | "speaking"
  | "listening"
  | "confirming"
  | "submitting";

/**
 * The parent mounts this with a `key` per question, so each question gets fresh
 * state by construction rather than by resetting state inside an effect. That
 * keeps the effect a pure mount/unmount lifecycle and avoids a cascading render
 * on every question change.
 */
export function HandsFreeInterview(props: {
  questionText: string;
  /** Current answer text; owned by the parent so typing and speech agree. */
  answer: string;
  onAnswerChange: (update: (previous: string) => string) => void;
  onSubmit: () => void;
  onDisable: () => void;
  disabled: boolean;
  /** Student-controlled silence threshold, in milliseconds. */
  pauseMs: number;
  onPauseChange: (ms: number) => void;
}) {
  // Starts speaking immediately: this component only exists while a question
  // is being asked.
  const [phase, setPhase] = useState<HandsFreePhase>("speaking");
  const [interim, setInterim] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [countdown, setCountdown] = useState(0);
  const [audioBlocked, setAudioBlocked] = useState(false);
  /**
   * Why the question was not read aloud, when it was not.
   *
   * Added 2026-07-26. This path used to be entirely silent: a failed
   * /api/speech/speak simply fell through to opening the microphone, so a
   * student got no voice and no reason, and the professor's report of "the
   * interview mentor does not produce voice in production" was indistinguishable
   * from the feature not existing. Silence in the UI also meant silence in every
   * bug report, which is what made it undiagnosable.
   */
  const [voiceProblem, setVoiceProblem] = useState<string | null>(null);

  const handleRef = useRef<LiveTranscriptionHandle | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Kept in a ref so the end-of-utterance callback can read the latest answer
  // without being rebuilt on every keystroke. Synced in an effect rather than
  // during render, because writing a ref during render is not safe under
  // concurrent rendering.
  const answerRef = useRef(props.answer);
  useEffect(() => {
    answerRef.current = props.answer;
  }, [props.answer]);

  const pauseRef = useRef(props.pauseMs);
  useEffect(() => {
    pauseRef.current = props.pauseMs;
  }, [props.pauseMs]);

  // The end-of-utterance and transcript callbacks need the live phase, and they
  // are created once, so it is read from a ref rather than closed over.
  const phaseRef = useRef(phase);
  useEffect(() => {
    phaseRef.current = phase;
  }, [phase]);

  const clearTimer = useCallback(() => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
    if (tickRef.current !== null) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
    setCountdown(0);
  }, []);

  const stopListening = useCallback(async () => {
    await handleRef.current?.stop();
    handleRef.current = null;
    setInterim("");
  }, []);

  /**
   * Called when Deepgram thinks the student has stopped talking.
   *
   * The microphone stays open through the countdown. This is the fix for the
   * "speech connection dropped" error: the previous version closed the socket
   * here and reopened a fresh one if the student wanted to add more, and that
   * reconnect was what failed. Keeping the one connection open means "I have
   * more to say" and simply resuming both need no reconnect at all.
   */
  const onUtteranceEnd = useCallback(() => {
    if (answerRef.current.trim() === "") return; // silence, not an answer
    if (phaseRef.current === "confirming") return; // already counting down
    setPhase("confirming");

    let remaining = Math.ceil(AUTO_SUBMIT_DELAY_MS / 1000);
    setCountdown(remaining);
    tickRef.current = setInterval(() => {
      remaining -= 1;
      setCountdown(remaining);
      if (remaining <= 0 && tickRef.current) clearInterval(tickRef.current);
    }, 1000);

    timerRef.current = setTimeout(() => {
      if (tickRef.current) clearInterval(tickRef.current);
      setPhase("submitting");
      void stopListening();
      props.onSubmit();
    }, AUTO_SUBMIT_DELAY_MS);
  }, [props, stopListening]);

  const listen = useCallback(async () => {
    setError(null);
    try {
      handleRef.current = await startLiveTranscription({
        fetchToken: async () => {
          const res = await fetch("/api/speech/token", { method: "POST" });
          const body = await res.json().catch(() => ({}));
          if (!res.ok) throw new Error(body.error ?? "Speech could not start.");
          return { accessToken: body.accessToken, sttModel: body.sttModel };
        },
        // Read at connect time, so moving the dial applies to the next
        // question rather than to a connection already open.
        endpointingMs: pauseRef.current,
        onInterim: (text) => {
          // The student started talking again during the countdown: cancel it
          // and go back to listening, no reconnect needed.
          if (phaseRef.current === "confirming") {
            clearTimer();
            setPhase("listening");
          }
          setInterim(text);
        },
        onFinal: (text) => {
          setInterim("");
          if (phaseRef.current === "confirming") {
            clearTimer();
            setPhase("listening");
          }
          props.onAnswerChange((previous) =>
            previous.trim() === "" ? text : `${previous} ${text}`,
          );
        },
        onUtteranceEnd,
        onStatus: (status) => {
          if (status === "listening") setPhase("listening");
        },
        onError: (message) => {
          setError(message);
          setPhase("idle");
        },
      });
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Speech could not be started.",
      );
      setPhase("idle");
    }
  }, [onUtteranceEnd, clearTimer, props]);

  // Play the question, then open the microphone when it finishes.
  useEffect(() => {
    let cancelled = false;

    void (async () => {
      try {
        const res = await fetch("/api/speech/speak", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ text: props.questionText }),
        });
        if (cancelled) return;
        if (!res.ok) {
          // No audio is survivable: the question is on screen already, and the
          // interview continues by microphone. But say what happened, and
          // distinguish the two cases, because they need different actions from
          // different people. 503 means this server has no Deepgram credential,
          // which only the maintainers can fix; anything else is transient and
          // worth retrying.
          setVoiceProblem(
            res.status === 503
              ? "Spoken questions are not set up on this server, so this one is text only. The interview still works: read the question and answer out loud."
              : "This question could not be read aloud just now, so it is text only. The interview still works: read it and answer out loud.",
          );
          await listen();
          return;
        }
        const url = URL.createObjectURL(await res.blob());
        const audio = new Audio(url);
        audioRef.current = audio;
        audio.addEventListener("ended", () => {
          if (!cancelled) void listen();
        });
        try {
          await audio.play();
        } catch {
          // Blocked because it did not follow a gesture. Show a control.
          if (!cancelled) setAudioBlocked(true);
        }
      } catch {
        // The request itself never completed: offline, or the server went away.
        // Same reasoning as the !res.ok branch, which is that a student must not
        // be left guessing why the interviewer is mute.
        if (cancelled) return;
        setVoiceProblem(
          "The spoken question could not be loaded, so this one is text only. The interview still works: read it and answer out loud.",
        );
        await listen();
      }
    })();

    return () => {
      cancelled = true;
      clearTimer();
      void stopListening();
      audioRef.current?.pause();
      audioRef.current = null;
    };
    // Mount and unmount only: a new question is a new component instance.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function keepTalking() {
    // The microphone never closed, so this only cancels the pending submit.
    clearTimer();
    setPhase("listening");
  }

  function sendNow() {
    clearTimer();
    setPhase("submitting");
    void stopListening();
    props.onSubmit();
  }

  return (
    <div className="rounded-card border-2 border-miami-red bg-paper p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        {/* State is words, never colour alone. */}
        <p role="status" className="font-bold">
          {phase === "speaking"
            ? "The interviewer is asking the question."
            : phase === "listening"
              ? "Listening. Take your time, and pause when you are done."
              : phase === "confirming"
                ? `Sending your answer in ${countdown}. Keep talking or edit it to stop.`
                : phase === "submitting"
                  ? "Sending your answer."
                  : "Voice mode is ready."}
        </p>
        <button
          type="button"
          onClick={() => {
            clearTimer();
            void stopListening();
            props.onDisable();
          }}
          className="rounded-card border border-medium-tan px-3 py-1.5 text-sm font-bold hover:bg-light-tan"
        >
          Switch to typing
        </button>
      </div>

      {audioBlocked ? (
        <button
          type="button"
          onClick={() => {
            setAudioBlocked(false);
            void audioRef.current?.play().catch(() => setAudioBlocked(true));
          }}
          className="mt-3 rounded-card bg-miami-red px-4 py-2 font-bold text-paper"
        >
          Play the question
        </button>
      ) : null}

      {/* Why there is no voice. role="status" rather than "alert": the interview
          is continuing normally by microphone, so this is information, not an
          error to interrupt the student with. */}
      {voiceProblem ? (
        <p role="status" className="mt-3 text-sm text-dark-tan">
          {voiceProblem}
        </p>
      ) : null}

      {interim ? (
        <p aria-live="polite" className="mt-3 text-sm italic text-dark-tan">
          Hearing: {interim}
        </p>
      ) : null}

      {phase === "confirming" ? (
        <div className="mt-3 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={keepTalking}
            className="rounded-card border-2 border-miami-red px-3 py-1.5 text-sm font-bold text-miami-red hover:bg-light-tan"
          >
            Wait, I have more to say
          </button>
          <button
            type="button"
            onClick={sendNow}
            disabled={props.disabled}
            className="rounded-card bg-miami-red px-3 py-1.5 text-sm font-bold text-paper hover:bg-accent-red"
          >
            Send it now
          </button>
        </div>
      ) : null}

      <PauseDial
        valueMs={props.pauseMs}
        onChange={props.onPauseChange}
        liveNow={phase === "listening"}
      />

      {error ? (
        <p role="alert" className="mt-3 text-sm font-bold text-miami-red">
          {error} You can switch to typing above.
        </p>
      ) : null}
    </div>
  );
}

export function handsFreeAvailable(): boolean {
  return isSpeechInputSupported();
}
