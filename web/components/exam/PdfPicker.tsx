"use client";

import { useRef, useState } from "react";

const MAX_UPLOAD_MB = 25;

/**
 * The upload control for course material.
 *
 * A bare <input type="file"> renders as the browser's default "Choose file /
 * no file chosen", which a student reported as unclear: it does not look like
 * the primary action of the screen and does not say what file is wanted. This
 * gives the action a real button, states the accepted format and size limit up
 * front, and adds drag and drop.
 *
 * Drag and drop is an enhancement only. The visible label is a real <label> for
 * a real file input, so the keyboard and screen reader path is the ordinary
 * native one and never depends on the drop zone.
 */
export function PdfPicker(props: {
  hasDocument: boolean;
  onUpload: (file: File) => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [rejected, setRejected] = useState<string | null>(null);

  function accept(file: File | undefined) {
    if (!file) return;
    if (file.type !== "application/pdf" && !/\.pdf$/i.test(file.name)) {
      setRejected(
        `${file.name} is not a PDF. Export your notes or slides as a PDF and try again.`,
      );
      return;
    }
    setRejected(null);
    props.onUpload(file);
  }

  return (
    <div className="mt-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          accept(e.dataTransfer.files?.[0]);
        }}
        className={[
          "rounded-card border-2 border-dashed p-6 text-center transition-colors",
          dragging
            ? "border-miami-red bg-light-tan"
            : "border-medium-tan bg-warm-white",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          id="exam-file"
          type="file"
          accept="application/pdf,.pdf"
          className="peer sr-only"
          onChange={(e) => {
            accept(e.target.files?.[0]);
            // Let the student pick the same file again after an error.
            e.target.value = "";
          }}
        />
        <label
          htmlFor="exam-file"
          className="inline-block cursor-pointer rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-miami-red"
        >
          {props.hasDocument ? "Choose a different PDF" : "Choose a PDF"}
        </label>
        <p className="mt-3 text-sm">
          or drag it onto this box. PDF only, up to {MAX_UPLOAD_MB} MB.
        </p>
      </div>

      {rejected ? (
        <p role="alert" className="mt-2 text-sm font-bold text-miami-red">
          {rejected}
        </p>
      ) : null}
    </div>
  );
}
