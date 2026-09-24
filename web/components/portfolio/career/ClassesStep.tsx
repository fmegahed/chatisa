"use client";

import { useEffect } from "react";
import { loadProfile } from "@/lib/scout/profile-store";
import type { ChecklistState } from "@/lib/scout/checklist";
import type { Draft, StepProps } from "@/lib/portfolio/draft";
import { CourseChecklist } from "@/components/scout/CourseChecklist";
import { StepNav } from "../StepNav";
import { GuestCoursesStep } from "./GuestCoursesStep";

/**
 * Step 2 of the career wizard. Job Scout's profile already knows which
 * courses this student has taken, so the checklist starts from it rather
 * than asking twice; anything chosen here stays local to the draft.
 */
export function ClassesStep(props: StepProps & { isGuest: boolean }) {
  // Guests have no Miami courses: they type their own, or skip (v6.6.0).
  return props.isGuest ? <GuestCoursesStep {...props} /> : <MiamiClassesStep {...props} />;
}

/** Where the checklist starts: this draft's own, then its course list, then Job Scout's. */
function startingPlan(draft: Draft): ChecklistState {
  if (draft.coursePlan) return draft.coursePlan;
  const profile = loadProfile();
  if (draft.courses.length > 0) {
    // A draft from before v6.7.0, or a restored site: rebuild from its list.
    const now = new Set(draft.inProgress ?? []);
    return {
      programs: profile?.programs ?? [],
      courses: draft.courses.map((code) => ({ code, status: now.has(code) ? "now" : "done" })),
      removedPrereqs: [],
    };
  }
  return {
    programs: profile?.programs ?? [],
    courses: profile?.courses ?? [],
    removedPrereqs: profile?.removedPrereqs ?? [],
  };
}

/** The draft fields the checklist drives. */
function fromPlan(plan: ChecklistState): Pick<Draft, "coursePlan" | "courses" | "inProgress"> {
  return {
    coursePlan: plan,
    courses: plan.courses.map((c) => c.code),
    inProgress: plan.courses.filter((c) => c.status === "now").map((c) => c.code),
  };
}

function MiamiClassesStep({ draft, patch, nav }: StepProps) {
  useEffect(() => {
    if (!draft.coursePlan) patch(fromPlan(startingPlan(draft)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <section className="rounded-card border border-medium-tan bg-paper p-5">
      <h2 className="text-2xl">Your classes</h2>
      <p className="mt-1 text-dark-tan">
        Mark the courses you have finished or are taking now. Courses you are taking now show as
        &quot;(in progress)&quot; on your page. The page highlights the ones that best support
        your story.
      </p>
      <div className="mt-4">
        {/* Mounted once the starting plan is in the draft, so a returning
            student's filled-in sections open folded like Job Scout's. */}
        {draft.coursePlan ? (
          <CourseChecklist value={draft.coursePlan} onChange={(next) => patch(fromPlan(next))} />
        ) : null}
      </div>
      <StepNav {...nav} canContinue={draft.courses.length > 0} />
    </section>
  );
}
