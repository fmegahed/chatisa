"use client";

import { useEffect, useRef, useState } from "react";
import type { StepProps } from "@/lib/portfolio/draft";
import { originOf, type ProjectOrigin } from "@/lib/portfolio/origin";
import { CoursePicker } from "../CoursePicker";
import { StepNav } from "../StepNav";

/**
 * Step 1 of the showcase wizard: where the project came from, plus the two
 * facts a reader wants in the header. Topics and internship courses have
 * no fixed subject, and they need no extra question here: the model titles
 * the page from the files and the story.
 *
 * Origins (v6.6.0): a Miami course (the picker, as before), a course at
 * another school (typed), self-study, or a personal project. Guests never
 * see the Miami option. Each origin keeps its own course while the step is
 * open, so text typed for one origin never leaks into another's header or
 * README, and arrowing through the options never loses a pick.
 *
 * The team field keeps its raw text in local state and parses into the draft
 * on every change. Binding the input to the parsed array instead would feed
 * back a re-joined string mid-typing, which eats the space after a comma and
 * makes a second name impossible to type.
 */

/** The route clips the course to this; mirrored so typing stops there. */
const MAX_COURSE = 80;

const CHOICES: { value: ProjectOrigin; label: string; hint?: string }[] = [
  { value: "miami", label: "A Miami course" },
  { value: "other", label: "A course at another school" },
  { value: "self", label: "Self-study", hint: "Working through a book, an online course, or your own plan." },
  { value: "personal", label: "A hobby or personal project" },
];

export function CourseStep({ draft, patch, nav, isGuest }: StepProps & { isGuest: boolean }) {
  const [teamText, setTeamText] = useState(draft.team.join(", "));
  const origin = originOf(draft.origin);
  const remembered = useRef<Partial<Record<ProjectOrigin, string>>>({ [origin]: draft.course });
  // A guest landing here with the Miami default moves to "another school";
  // a Miami course means nothing to someone who never took one.
  useEffect(() => {
    if (isGuest && origin === "miami") patch({ origin: "other", course: "" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isGuest, origin]);

  const choices = isGuest ? CHOICES.filter((c) => c.value !== "miami") : CHOICES;
  const selected = draft.course ? [draft.course] : [];
  const needsCourse = origin === "miami" || origin === "other";
  const canContinue = !needsCourse || draft.course.trim().length > 0;

  return (
    <section className="rounded-card border border-medium-tan bg-paper p-5">
      <fieldset>
        {/* A heading inside the legend: every step announces itself with an
            h2 for screen-reader heading navigation, and the legend still
            names the radio group. */}
        <legend>
          <h2 className="text-2xl">Where did this project come from?</h2>
        </legend>
        <div className="mt-3 space-y-2">
          {choices.map((c) => (
            <label key={c.value} className="flex items-start gap-2">
              <input
                type="radio"
                name="project-origin"
                value={c.value}
                checked={origin === c.value}
                onChange={() => {
                  // Arrow keys move a radio group's selection, so each origin
                  // keeps its own course: browsing the options never loses a
                  // pick, and text typed for one origin never reaches another.
                  remembered.current[origin] = draft.course;
                  patch({ origin: c.value, course: remembered.current[c.value] ?? "" });
                }}
                className="mt-1.5"
              />
              <span>
                {c.label}
                {c.hint ? <span className="block text-dark-tan">{c.hint}</span> : null}
              </span>
            </label>
          ))}
        </div>
      </fieldset>
      {origin === "miami" ? (
        <div className="mt-4">
          <CoursePicker single selected={selected} onChange={(c) => patch({ course: c[0] ?? "" })} />
        </div>
      ) : null}
      {origin === "other" ? (
        <label className="mt-4 block font-bold">
          Course and school
          <input
            value={draft.course}
            maxLength={MAX_COURSE}
            onChange={(e) => patch({ course: e.target.value })}
            placeholder="STAT 4520, Ohio State"
            className="mt-1 w-full rounded-card border border-medium-tan p-2 font-normal"
          />
        </label>
      ) : null}
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
      <StepNav {...nav} canContinue={canContinue} />
    </section>
  );
}
