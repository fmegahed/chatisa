"use client";

import { useId } from "react";

/**
 * Back / Next footer shared by every wizard step, with the step counter.
 * `requirement` says what the step needs before Next opens; it shows while
 * Next is unavailable and describes the button, so the reason is stated
 * rather than left to a greyed-out control (#31). While `busy` (generating),
 * Next stays focusable with aria-disabled so focus is not dropped (#27).
 */
export function StepNav(props: {
  index: number;
  total: number;
  canContinue: boolean;
  busy?: boolean;
  nextLabel?: string;
  requirement?: string;
  onBack: (() => void) | null;
  onNext: () => void;
}) {
  const hintId = useId();
  const showHint = !props.canContinue && !!props.requirement;
  return (
    <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-medium-tan pt-4">
      <div>
        <p className="text-dark-tan">Step {props.index} of {props.total}</p>
        {showHint ? (
          <p id={hintId} className="text-sm text-dark-tan">
            {props.requirement}
          </p>
        ) : null}
      </div>
      <div className="flex gap-3">
        {props.onBack ? (
          <button
            type="button"
            onClick={props.onBack}
            className="rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan"
          >
            Back
          </button>
        ) : null}
        <button
          type="button"
          disabled={!props.canContinue}
          aria-disabled={props.busy || undefined}
          aria-describedby={showHint ? hintId : undefined}
          onClick={() => {
            if (!props.busy) props.onNext();
          }}
          className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray aria-disabled:bg-medium-gray"
        >
          {props.nextLabel ?? "Next"}
        </button>
      </div>
    </div>
  );
}
