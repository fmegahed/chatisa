"use client";

import { useState, useSyncExternalStore } from "react";
import { COURSES, POPULAR_CODES, getCourse, matchesCourse, type CourseDef } from "@/lib/scout/courses";
import { courseCodes, loadProfile } from "@/lib/scout/profile-store";

/**
 * The showcase's one Miami course (v6.7.0): the student's own courses from
 * Job Scout first, then a search across every FSB course, including
 * previous titles. Without a Job Scout profile it offers the ISA courses
 * students take most.
 */

const MAX_RESULTS = 12;
const noSubscribe = () => () => {};
/** A primitive snapshot, so React sees the same value until the profile changes. */
const profileCodes = () => {
  const profile = loadProfile();
  return profile ? courseCodes(profile).join("|") : "";
};

export function CoursePicker(props: { selected: string; onChange: (code: string) => void }) {
  const [query, setQuery] = useState("");
  const mine = useSyncExternalStore(noSubscribe, profileCodes, () => "");
  const own = mine
    ? mine.split("|").flatMap((c) => (getCourse(c) ? [getCourse(c)!] : []))
    : [];
  const suggested = own.length
    ? own
    : Object.values(POPULAR_CODES).flat().flatMap((c) => (getCourse(c) ? [getCourse(c)!] : []));
  const q = query.trim();
  const matches = q.length >= 2 ? COURSES.filter((c) => matchesCourse(c, q)) : [];

  const chip = (course: CourseDef) => {
    const on = props.selected === course.code;
    return (
      <button
        key={course.code}
        type="button"
        aria-pressed={on}
        title={course.title}
        onClick={() => props.onChange(on ? "" : course.code)}
        className={
          on
            ? "rounded-card bg-miami-red px-3 py-1 font-bold text-paper"
            : "rounded-card border-2 border-medium-tan px-3 py-1 hover:bg-light-tan"
        }
      >
        {course.code}
        <span className="sr-only"> {course.title}</span>
      </button>
    );
  };

  return (
    <div>
      <fieldset className="min-w-0">
        <legend className="font-bold">{own.length ? "Your courses" : "Courses students often showcase"}</legend>
        {own.length ? null : (
          <p className="text-dark-tan">Mark your courses in Job Scout and they appear here first.</p>
        )}
        <div className="mt-2 flex flex-wrap gap-2">{suggested.map(chip)}</div>
      </fieldset>
      <label className="mt-4 block font-bold" htmlFor="course-search">Find a course</label>
      <input
        id="course-search"
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Code or title, for example 401 or forecasting"
        className="mt-1 w-full rounded-card border border-medium-tan p-2"
      />
      {q.length >= 2 ? (
        <>
          <p role="status" className="mt-2 text-dark-tan">
            {matches.length === 0
              ? "No FSB course matches that."
              : matches.length > MAX_RESULTS
                ? `Showing ${MAX_RESULTS} of ${matches.length}. Keep typing to narrow it down.`
                : `${matches.length} ${matches.length === 1 ? "course matches" : "courses match"}.`}
          </p>
          <div className="mt-2 flex flex-wrap gap-2">{matches.slice(0, MAX_RESULTS).map(chip)}</div>
        </>
      ) : null}
      {props.selected ? (
        <p className="mt-3 text-dark-tan">
          Selected: {props.selected}
          {getCourse(props.selected) ? ` ${getCourse(props.selected)!.title}` : ""}
        </p>
      ) : null}
    </div>
  );
}
