"use client";

/**
 * A file input a student can actually see: the native control renders as
 * browser-default "Choose File" text with no affordance (user feedback,
 * 2026-07-29), so the input is visually hidden INSIDE a real
 * secondary-style button label. Keyboard focus lands on the input, and the
 * has-[:focus-visible] ring mirrors the app's global 3px Miami Red outline
 * so the button visibly holds focus.
 */
export function FilePick(props: {
  label: string;
  accept: string;
  fileName: string | null;
  onChange: (file: File | null) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3">
      <label
        className={
          props.disabled
            ? "inline-block cursor-not-allowed rounded-card border-2 border-medium-gray px-4 py-2 font-bold text-medium-gray"
            : "inline-block cursor-pointer rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-miami-red has-[:focus-visible]:outline-offset-2"
        }
      >
        <input
          type="file"
          accept={props.accept}
          disabled={props.disabled}
          className="sr-only"
          onChange={(e) => props.onChange(e.target.files?.[0] ?? null)}
        />
        {props.label}
      </label>
      <span aria-live="polite">{props.fileName ?? "No file chosen yet"}</span>
    </div>
  );
}
