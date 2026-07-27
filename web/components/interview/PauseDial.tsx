"use client";

import { useSyncExternalStore } from "react";
import {
  DEFAULT_ENDPOINTING_MS,
  MAX_ENDPOINTING_MS,
  MIN_ENDPOINTING_MS,
  clampEndpointing,
} from "@/lib/speech/live-transcription";

const STORAGE_KEY = "chatisa.interview.pauseMs";

/**
 * How long a silence counts as "I have finished answering".
 *
 * This is a dial rather than a constant because the right value is a fact about
 * the person, not about the software. Some people think in silence for five
 * seconds mid-answer; others stop and wait. A single tuned default would cut
 * the first group off constantly and bore the second.
 *
 * Remembered across sessions, because a student who has found their setting
 * should not have to find it again every interview.
 */
const listeners = new Set<() => void>();

/** Cached so the store snapshot is stable between reads, which
 * useSyncExternalStore requires: returning a freshly parsed value each call
 * would loop forever. */
let cached: number | null = null;

function readStored(): number {
  if (cached !== null) return cached;
  try {
    const stored = window.localStorage.getItem(STORAGE_KEY);
    cached = stored ? clampEndpointing(Number(stored)) : DEFAULT_ENDPOINTING_MS;
  } catch {
    // Private browsing or blocked storage: the default is fine.
    cached = DEFAULT_ENDPOINTING_MS;
  }
  return cached;
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function usePauseSetting(): [number, (ms: number) => void] {
  // The server has no localStorage, so it gets the default and the client gets
  // the stored value. Reading storage in an effect and setting state would
  // work but triggers a cascading render on every mount.
  const pauseMs = useSyncExternalStore(
    subscribe,
    readStored,
    () => DEFAULT_ENDPOINTING_MS,
  );

  function update(ms: number) {
    const clamped = clampEndpointing(ms);
    cached = clamped;
    try {
      window.localStorage.setItem(STORAGE_KEY, String(clamped));
    } catch {
      // Not being able to remember it is not worth telling the student about.
    }
    for (const listener of listeners) listener();
  }

  return [pauseMs, update];
}

export function PauseDial(props: {
  valueMs: number;
  onChange: (ms: number) => void;
  /** True while a microphone session is open, since the setting applies to the
   * next one rather than the current connection. */
  liveNow: boolean;
}) {
  const seconds = Math.round(props.valueMs / 1000);

  return (
    <div className="mt-3">
      <label htmlFor="pause-dial" className="block text-sm font-bold">
        How long a pause means you have finished: {seconds} seconds
      </label>
      <input
        id="pause-dial"
        type="range"
        min={MIN_ENDPOINTING_MS}
        max={MAX_ENDPOINTING_MS}
        step={1000}
        value={props.valueMs}
        onChange={(e) => props.onChange(Number(e.target.value))}
        aria-describedby="pause-dial-help"
        // The value is in the label too, so it is announced on change without
        // relying on the slider's own value being read out usefully.
        aria-valuetext={`${seconds} seconds`}
        className="mt-1 w-full max-w-sm"
      />
      <div className="flex max-w-sm justify-between text-sm text-dark-tan">
        <span>2s, quicker</span>
        <span>30s, lots of thinking time</span>
      </div>
      <p id="pause-dial-help" className="mt-1 text-sm text-dark-tan">
        Raise this if you are being cut off mid-answer. Lower it if the wait
        between questions feels long.
        {props.liveNow ? " Your change applies to the next question." : ""}
      </p>
    </div>
  );
}
