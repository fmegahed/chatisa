"use client";

import type { OtherCourse, StepProps } from "@/lib/portfolio/draft";
import { StepNav } from "../StepNav";

/**
 * The Classes step for guests (v6.6.0). Guests have no Miami courses, so
 * they may type up to five from their own school, or skip. Skipping leaves
 * no trace on the page: the route sends the model no course lines and the
 * page has no Coursework section.
 */

const MAX_ROWS = 5;
/** The route clips each part to this; mirrored so typing stops there. */
const MAX_TEXT = 80;

export function GuestCoursesStep({ draft, patch, nav }: StepProps) {
  // At least one row to type into; an empty row is simply not sent.
  const rows: OtherCourse[] = draft.otherCourses?.length ? draft.otherCourses : [{ name: "", school: "" }];
  const set = (i: number, p: Partial<OtherCourse>) =>
    patch({ otherCourses: rows.map((r, j) => (j === i ? { ...r, ...p } : r)) });
  const remove = (i: number) => patch({ otherCourses: rows.filter((_, j) => j !== i) });
  const filled = rows.some((r) => r.name.trim().length > 0);

  return (
    <section className="rounded-card border border-medium-tan bg-paper p-5">
      <h2 className="text-2xl">Relevant courses (optional)</h2>
      <p className="mt-1 text-dark-tan">
        Add four or five courses that fit the work you want to show. They help the page tell your
        story. You can skip this.
      </p>
      <ol className="mt-4 space-y-3">
        {rows.map((r, i) => (
          <li key={i} className="grid gap-2 md:grid-cols-[1fr_1fr_auto] md:items-end">
            <label className="block font-bold">
              Course {i + 1}
              <input
                value={r.name}
                maxLength={MAX_TEXT}
                onChange={(e) => set(i, { name: e.target.value })}
                placeholder={i === 0 ? "Applied Regression" : undefined}
                className="mt-1 w-full rounded-card border border-medium-tan p-2 font-normal"
              />
            </label>
            <label className="block font-bold">
              School {i + 1} (optional)
              <input
                value={r.school}
                maxLength={MAX_TEXT}
                onChange={(e) => set(i, { school: e.target.value })}
                placeholder={i === 0 ? "Ohio State" : undefined}
                className="mt-1 w-full rounded-card border border-medium-tan p-2 font-normal"
              />
            </label>
            {rows.length > 1 ? (
              <button type="button" onClick={() => remove(i)} className="justify-self-start py-2 underline">
                Remove course {i + 1}
              </button>
            ) : null}
          </li>
        ))}
      </ol>
      {rows.length < MAX_ROWS ? (
        <button
          type="button"
          className="mt-3 underline"
          onClick={() => patch({ otherCourses: [...rows, { name: "", school: "" }] })}
        >
          Add a course
        </button>
      ) : null}
      <StepNav {...nav} canContinue nextLabel={filled ? "Continue" : "Continue without courses"} />
    </section>
  );
}
