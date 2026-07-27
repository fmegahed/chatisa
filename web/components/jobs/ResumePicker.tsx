"use client";

import { useRef, useState } from "react";

const MAX_MB = 25;

/**
 * A visible, unmistakable control for choosing a resume PDF.
 *
 * The browser's default file input renders an unstyled "Choose file" that reads
 * as plain text rather than a button, which a student flagged both here and at
 * the Exam Ally upload. This gives it a real red button and a drop zone, states
 * the format and limit, and shows the chosen file name.
 *
 * It only captures the file; the caller uploads it. Drag and drop is an
 * enhancement over a real labelled file input, so the keyboard and screen
 * reader path stays the native one.
 */
export function ResumePicker(props: {
  file: File | null;
  onChoose: (file: File | null) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [rejected, setRejected] = useState<string | null>(null);

  function accept(file: File | undefined) {
    if (!file) return;
    if (file.type !== "application/pdf" && !/\.pdf$/i.test(file.name)) {
      setRejected(`${file.name} is not a PDF. Export your resume as a PDF and try again.`);
      return;
    }
    setRejected(null);
    props.onChoose(file);
  }

  return (
    <div className="mt-2">
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
          "rounded-card border-2 border-dashed p-5 text-center transition-colors",
          dragging ? "border-miami-red bg-light-tan" : "border-medium-tan bg-warm-white",
        ].join(" ")}
      >
        <input
          ref={inputRef}
          id="resume-file"
          type="file"
          accept="application/pdf,.pdf"
          className="peer sr-only"
          disabled={props.disabled}
          onChange={(e) => {
            accept(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
        <label
          htmlFor="resume-file"
          className="inline-block cursor-pointer rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-miami-red"
        >
          {props.file ? "Choose a different PDF" : "Choose your resume (PDF)"}
        </label>
        <p className="mt-2 text-sm">
          or drag it onto this box. PDF only, up to {MAX_MB} MB.
        </p>
        {props.file ? (
          <p role="status" className="mt-2 text-sm font-bold">
            Chosen: {props.file.name}
          </p>
        ) : null}
      </div>

      {rejected ? (
        <p role="alert" className="mt-2 text-sm font-bold text-miami-red">
          {rejected}
        </p>
      ) : null}
    </div>
  );
}
