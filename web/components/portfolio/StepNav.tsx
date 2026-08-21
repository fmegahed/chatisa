"use client";

/** Back / Next footer shared by every wizard step, with the step counter. */
export function StepNav(props: {
  index: number;
  total: number;
  canContinue: boolean;
  busy?: boolean;
  nextLabel?: string;
  onBack: (() => void) | null;
  onNext: () => void;
}) {
  return (
    <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-medium-tan pt-4">
      <p className="text-dark-tan">Step {props.index} of {props.total}</p>
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
          disabled={!props.canContinue || props.busy}
          onClick={props.onNext}
          className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
        >
          {props.nextLabel ?? "Next"}
        </button>
      </div>
    </div>
  );
}
