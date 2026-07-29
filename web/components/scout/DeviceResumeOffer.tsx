"use client";

import { useEffect, useState } from "react";
import {
  getResume,
  resumeAsFile,
  type DeviceResume,
} from "@/lib/scout/device-files";

/**
 * One-click reuse of the resume a student saved on this device in Job
 * Scout's profile (user feedback 2026-07-29: the handoff carried the job
 * but not the resume). Renders nothing when no device resume exists or a
 * file is already chosen, so students who never used Job Scout see no
 * change. The file still uploads per request; the server stores nothing new.
 */
export function DeviceResumeOffer(props: {
  currentFile: File | null;
  disabled?: boolean;
  onUse: (file: File) => void;
}) {
  const [stored, setStored] = useState<DeviceResume | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    // Async IIFE, setState after await (the InterviewMentor pattern).
    void (async () => {
      const r = await getResume();
      if (r) setStored(r);
    })();
  }, []);

  if (!stored || props.currentFile) return null;

  return (
    <p role="status" className="mt-2 rounded-card bg-light-tan p-3">
      Job Scout has your resume on this device:{" "}
      <strong>{stored.name}</strong>.{" "}
      <button
        type="button"
        disabled={props.disabled}
        className="font-bold underline disabled:text-medium-gray"
        onClick={() => {
          void (async () => {
            const file = await resumeAsFile();
            if (file) props.onUse(file);
            else setFailed(true);
          })();
        }}
      >
        Use it here
      </button>{" "}
      or choose another below.
      {failed ? " It could not be read from this device; choose the file instead." : ""}
    </p>
  );
}
