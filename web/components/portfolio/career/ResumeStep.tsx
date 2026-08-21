"use client";

import { useEffect } from "react";
import { FilePick } from "@/components/scout/FilePick";
import { DeviceResumeOffer } from "@/components/scout/DeviceResumeOffer";
import { resumeAsFile } from "@/lib/scout/device-files";
import type { StepProps } from "@/lib/portfolio/draft";
import { StepNav } from "../StepNav";

/**
 * Step 1 of the career wizard. A resume the student already saved in Job
 * Scout is picked up from this device on mount; the offer below covers the
 * case where the stored copy cannot be rehydrated, and every student can
 * choose a different PDF. The file is uploaded per request and never stored
 * on the server.
 */
export function ResumeStep({ draft, patch, nav }: StepProps) {
  useEffect(() => {
    if (draft.resume) return;
    void resumeAsFile().then((f) => {
      if (f) patch({ resume: f });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section className="rounded-card border border-medium-tan bg-paper p-5">
      <h2 className="text-2xl">Your resume</h2>
      <p className="mt-1 text-dark-tan">
        The page is written from your resume: experience, education, and the skills it shows. PDF
        only. It is read once and not kept on the server.
      </p>
      <div className="mt-4">
        <DeviceResumeOffer currentFile={draft.resume} onUse={(f) => patch({ resume: f })} />
        <FilePick
          label={draft.resume ? "Choose a different PDF" : "Choose your resume PDF"}
          accept="application/pdf"
          fileName={draft.resume?.name ?? null}
          onChange={(f) => patch({ resume: f })}
        />
      </div>
      <StepNav {...nav} canContinue={draft.resume !== null} />
    </section>
  );
}
