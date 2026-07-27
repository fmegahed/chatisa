"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Reads the interviewer's question aloud.
 *
 * The question is always on screen as text. Audio is an addition to it, never
 * a replacement, which is the opposite of the legacy module: there the question
 * was spoken only, with no transcript anywhere, so a deaf or hard-of-hearing
 * student could not use the feature at all.
 *
 * A real <audio controls> element is used rather than a custom player, so
 * pause, seek, volume and keyboard support are the browser's and work with
 * assistive technology without us reimplementing them.
 */
export function QuestionAudio(props: { text: string }) {
  const [src, setSrc] = useState<string | null>(null);
  const [state, setState] = useState<
    "idle" | "loading" | "ready" | "failed" | "unavailable"
  >("idle");
  const objectUrlRef = useRef<string | null>(null);

  // The parent remounts this component per question via a `key`, so there is
  // no stale audio to clear here: a new question gets fresh state by
  // construction rather than by resetting state in an effect. This only has to
  // release the object URL when the component goes away.
  useEffect(() => {
    return () => {
      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
        objectUrlRef.current = null;
      }
    };
  }, []);

  async function load() {
    setState("loading");
    try {
      const res = await fetch("/api/speech/speak", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ text: props.text }),
      });
      if (!res.ok) {
        // 503 is the server saying it has no Deepgram credential at all, which
        // is permanent until a maintainer fixes it, so "try again" would be
        // false advice. Anything else is worth another press.
        setState(res.status === 503 ? "unavailable" : "failed");
        return;
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      objectUrlRef.current = url;
      setSrc(url);
      setState("ready");
    } catch {
      setState("failed");
    }
  }

  if (state === "ready" && src) {
    return (
      <div className="mt-3">
        {/* autoPlay is correct here, despite the usual rule against it, and the
            reason is the click that got us here: this element only mounts after
            the student presses "Hear this question", so playback follows a user
            gesture. That is both what they asked for and what browser autoplay
            policy permits. Autoplay would be wrong on a question that appeared
            without being asked for, which is why the hands-free path handles a
            blocked play() explicitly instead.

            The comment this replaces read "Not autoplay", directly contradicting
            the attribute beside it, which made it a plausible suspect while the
            missing-voice report was being diagnosed. It was never the cause. */}
        <audio
          src={src}
          controls
          autoPlay
          className="w-full"
          aria-label="The interviewer reading this question aloud"
        />
      </div>
    );
  }

  return (
    <div className="mt-3">
      <button
        type="button"
        onClick={load}
        disabled={state === "loading" || state === "unavailable"}
        className="rounded-card border border-medium-tan px-3 py-1.5 text-sm font-bold hover:bg-light-tan disabled:cursor-not-allowed disabled:text-medium-gray"
      >
        {state === "loading" ? "Loading audio..." : "Hear this question"}
      </button>
      {state === "failed" ? (
        <p role="status" className="mt-1 text-sm text-dark-tan">
          The audio could not be loaded. Try again, or read the question above.
        </p>
      ) : null}
      {state === "unavailable" ? (
        <p role="status" className="mt-1 text-sm text-dark-tan">
          Spoken questions are not set up on this server. The question is written
          above.
        </p>
      ) : null}
    </div>
  );
}
