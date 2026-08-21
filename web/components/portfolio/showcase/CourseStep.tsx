"use client";

import { useState } from "react";
import type { StepProps } from "@/lib/portfolio/draft";
import { CoursePicker } from "../CoursePicker";
import { StepNav } from "../StepNav";

/**
 * Step 1 of the showcase wizard: which course the project was for, plus the
 * two facts a reader wants in the header. Topics and internship courses have
 * no fixed subject, and they need no extra question here: the model titles
 * the page from the files and the story.
 *
 * The team field keeps its raw text in local state and parses into the draft
 * on every change. Binding the input to the parsed array instead would feed
 * back a re-joined string mid-typing, which eats the space after a comma and
 * makes a second name impossible to type.
 */
export function CourseStep({ draft, patch, nav }: StepProps) {
  const [teamText, setTeamText] = useState(draft.team.join(", "));
  const selected = draft.course ? [draft.course] : [];
  return (
    <section className="rounded-card border border-medium-tan bg-paper p-5">
      <h2 className="text-2xl">Which course was this for?</h2>
      <div className="mt-4">
        <CoursePicker single selected={selected} onChange={(c) => patch({ course: c[0] ?? "" })} />
      </div>
      <div className="mt-4 grid gap-3 md:grid-cols-2">
        <label className="block">
          Semester (optional)
          <input
            value={draft.semester}
            onChange={(e) => patch({ semester: e.target.value })}
            placeholder="Spring 2026"
            className="mt-1 w-full rounded-card border border-medium-tan p-2"
          />
        </label>
        <label className="block">
          Team members (optional, comma separated)
          <input
            value={teamText}
            onChange={(e) => {
              setTeamText(e.target.value);
              patch({
                team: e.target.value.split(",").map((s) => s.trim()).filter(Boolean).slice(0, 8),
              });
            }}
            placeholder="Ann Lee, Bo Chen"
            className="mt-1 w-full rounded-card border border-medium-tan p-2"
          />
        </label>
      </div>
      <StepNav {...nav} canContinue={draft.course.length > 0} />
    </section>
  );
}
