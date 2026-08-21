"use client";

import { useEffect } from "react";
import { loadProfile } from "@/lib/scout/profile-store";
import type { StepProps } from "@/lib/portfolio/draft";
import { CoursePicker } from "../CoursePicker";
import { StepNav } from "../StepNav";

/**
 * Step 2 of the career wizard. Job Scout's profile already knows which
 * courses this student has taken, so the picker starts from it rather than
 * asking twice; anything chosen here stays local to the draft.
 */
export function ClassesStep({ draft, patch, nav }: StepProps) {
  useEffect(() => {
    if (draft.courses.length > 0) return;
    const profile = loadProfile();
    if (profile?.courses.length) patch({ courses: profile.courses });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section className="rounded-card border border-medium-tan bg-paper p-5">
      <h2 className="text-2xl">Classes you have taken</h2>
      <p className="mt-1 text-dark-tan">
        Pick the ISA courses you have completed. The page highlights the ones that best support
        your story.
      </p>
      <div className="mt-4">
        <CoursePicker selected={draft.courses} onChange={(courses) => patch({ courses })} />
      </div>
      <StepNav {...nav} canContinue={draft.courses.length > 0} />
    </section>
  );
}
