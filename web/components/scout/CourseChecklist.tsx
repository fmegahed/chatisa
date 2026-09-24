"use client";

import { useEffect, useId, useRef, useState } from "react";
import { COURSES, getCourse, matchesCourse } from "@/lib/scout/courses";
import { BULLETIN_YEAR, PROGRAMS, getProgram, type ProgramGroup } from "@/lib/scout/programs";
import {
  addedFor,
  markRequiredDone,
  setCourseStatus,
  statusOf,
  type ChecklistState,
  type RowStatus,
} from "@/lib/scout/checklist";

/**
 * The major-first course checklist (v6.7.0), shared by Job Scout's profile
 * and the Portfolio Builder's Classes step. The student picks what they
 * study, then marks each course Done, Taking now or Not yet in the groups
 * the Bulletin uses. Prerequisites of what they mark come in automatically,
 * labelled with the course that caused them, and each can be removed.
 *
 * It records what a student took; it never audits a degree.
 */

const CORE = "business-core";
const MAX_RESULTS = 12;
const STATUS_LABEL: Record<RowStatus, string> = { done: "Done", now: "Taking now", none: "Not yet" };
const KIND_LABEL = { major: "Majors", comajor: "Co-majors", minor: "Minors" } as const;

const titleOf = (code: string) => getCourse(code)?.title ?? null;

export function CourseChecklist(props: {
  value: ChecklistState;
  onChange: (next: ChecklistState) => void;
}) {
  const uid = useId();
  const rootRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  // Where focus goes if a change removes the element that had it (a row in
  // "Your other courses", a Remove button): a course code, or the search box.
  const pendingFocus = useRef<string | null>(null);
  useEffect(() => {
    const target = pendingFocus.current;
    if (!target) return;
    pendingFocus.current = null;
    const active = document.activeElement;
    if (active && active !== document.body && rootRef.current?.contains(active)) return;
    const row = target === "search"
      ? null
      : rootRef.current?.querySelector<HTMLInputElement>(`fieldset[data-course="${target}"] input[type=radio]:checked`);
    (row ?? searchRef.current)?.focus();
  });
  // A returning student sees each section they already filled in folded to
  // its "n of m marked" line, so the page opens short; a new one sees all.
  const [closed, setClosed] = useState<Set<string>>(() => {
    const marked = new Set(props.value.courses.map((c) => c.code));
    return new Set(
      [CORE, ...props.value.programs].filter((key) =>
        (getProgram(key)?.groups ?? []).some((g) => g.items.some((i) => i.codes.some((c) => marked.has(c)))),
      ),
    );
  });
  const [query, setQuery] = useState("");
  const [announcement, setAnnouncement] = useState("");
  const { value } = props;

  const sections = [CORE, ...PROGRAMS.filter((p) => p.key !== CORE && value.programs.includes(p.key)).map((p) => p.key)];
  const shownCodes = new Set(
    sections.flatMap((key) => (getProgram(key)?.groups ?? []).flatMap((g) => g.items.flatMap((i) => i.codes))),
  );
  const others = value.courses.filter((c) => !shownCodes.has(c.code));

  function announce(root: string, added: string[], status: RowStatus) {
    if (status === "none") return setAnnouncement(`${root} removed.`);
    if (added.length === 0) return setAnnouncement("");
    setAnnouncement(
      `Added ${added.join(", ")} as Done, because you ${status === "now" ? "are taking" : "took"} ${root}.`,
    );
  }

  function setStatus(code: string, status: RowStatus) {
    if (status === "none") pendingFocus.current = "search";
    const { state, added } = setCourseStatus(value, code, status);
    props.onChange(state);
    announce(code, added, status);
  }

  /** An "or" row switched options: the status moves with it, in one change. */
  function moveStatus(from: string, to: string, status: RowStatus) {
    const cleared = setCourseStatus(value, from, "none").state;
    const { state, added } = setCourseStatus(cleared, to, status);
    props.onChange(state);
    announce(to, added, status);
  }

  function toggleProgram(key: string, on: boolean) {
    const programs = on ? [...value.programs, key] : value.programs.filter((k) => k !== key);
    props.onChange({ ...value, programs });
  }

  const row = (codes: string[], rowKey: string) => (
    <CourseRow key={rowKey} name={`${uid}-${rowKey}`} codes={codes} value={value} onStatus={setStatus} onMove={moveStatus}
      onRemove={(code, root) => {
        setStatus(code, "none");
        pendingFocus.current = root;
      }}
    />
  );

  const q = query.trim().toLowerCase();
  const matches = q.length >= 2 ? COURSES.filter((c) => matchesCourse(c, q)) : [];

  return (
    <div ref={rootRef}>
      <fieldset className="min-w-0">
        <legend className="font-bold">What are you studying?</legend>
        <p className="text-dark-tan">Choose any that apply. The business core is always shown.</p>
        {(["major", "comajor", "minor"] as const).map((kind) => (
          <div key={kind} className="mt-2">
            <p className="font-bold">{KIND_LABEL[kind]}</p>
            <div className="mt-1 grid gap-1 sm:grid-cols-2">
              {PROGRAMS.filter((p) => p.kind === kind).map((p) => (
                <label key={p.key} className="flex items-start gap-2">
                  <input
                    type="checkbox"
                    className="mt-1 accent-miami-red"
                    checked={value.programs.includes(p.key)}
                    onChange={(e) => toggleProgram(p.key, e.target.checked)}
                  />
                  <span>{p.name.replace(/ \((co-major|minor)\)$/, "")}</span>
                </label>
              ))}
            </div>
          </div>
        ))}
      </fieldset>

      <p className="mt-4 text-dark-tan">
        Grouped as in the {BULLETIN_YEAR} Bulletin. Took something different? Search for it below.
      </p>

      {sections.map((key) => {
        const program = getProgram(key);
        if (!program) return null;
        const isOpen = !closed.has(key);
        const rows = program.groups.flatMap((g) => g.items);
        const marked = rows.filter((i) => i.codes.some((c) => statusOf(value, c) !== "none")).length;
        const panelId = `${uid}-${key}`;
        return (
          <section key={key} className="mt-4 rounded-card border border-medium-tan bg-paper">
            <h3 className="text-lg">
              <button
                type="button"
                aria-expanded={isOpen}
                aria-controls={panelId}
                onClick={() => {
                  const next = new Set(closed);
                  if (isOpen) next.add(key);
                  else next.delete(key);
                  setClosed(next);
                }}
                className="flex w-full flex-wrap items-baseline justify-between gap-2 p-3 text-left font-bold hover:bg-light-tan"
              >
                <span>
                  <span aria-hidden="true">{isOpen ? "▾ " : "▸ "}</span>
                  {program.name}
                </span>
                <span className="text-base font-normal text-dark-tan">
                  {marked} of {rows.length} marked
                </span>
              </button>
            </h3>
            <div id={panelId} hidden={!isOpen} className="px-3 pb-3">
              {key === CORE ? (
                <button
                  type="button"
                  onClick={() => {
                    const { state, added } = markRequiredDone(value, CORE);
                    props.onChange(state);
                    setAnnouncement(
                      added.length ? `Marked ${added.length} required core courses Done.` : "The required core courses were already marked.",
                    );
                  }}
                  className="mb-2 rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
                >
                  I&apos;ve finished the required core courses
                </button>
              ) : null}
              {key === CORE ? (
                <p className="mb-2 text-dark-tan">
                  Marks the courses every FSB student takes. Choose your math course and capstone yourself.
                </p>
              ) : null}
              {program.groups.map((g, gi) => (
                <div key={gi} className="mt-3">
                  {groupHeading(g, program.groups[gi - 1]) ? (
                    <h4 className="font-bold">{groupHeading(g, program.groups[gi - 1])}</h4>
                  ) : null}
                  {g.instruction ? <p className="text-dark-tan">{g.instruction}</p> : null}
                  {g.notes.map((n) => (
                    <p key={n} className="text-dark-tan">{n}</p>
                  ))}
                  <ul className="mt-1">{g.items.map((item, ii) => row(item.codes, `${key}-${gi}-${ii}`))}</ul>
                </div>
              ))}
            </div>
          </section>
        );
      })}

      {others.length > 0 ? (
        <section aria-labelledby={`${uid}-others`} className="mt-4 rounded-card border border-medium-tan bg-paper p-3">
          <h3 id={`${uid}-others`} className="text-lg font-bold">Your other courses</h3>
          <p className="text-dark-tan">Courses you added from search, and prerequisites outside the groups above.</p>
          <ul className="mt-1">{others.map((c) => row([c.code], `other-${c.code}`))}</ul>
        </section>
      ) : null}

      <div className="mt-4">
        <label htmlFor={`${uid}-search`} className="block font-bold">
          Search all FSB courses
        </label>
        <input
          id={`${uid}-search`}
          ref={searchRef}
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Code or title, for example 401 or forecasting"
          className="mt-1 w-full rounded-card border border-medium-tan bg-paper p-2"
        />
        {q.length >= 2 ? (
          <>
            <p className="mt-2 text-dark-tan" role="status">
              {matches.length === 0
                ? "No FSB course matches that."
                : matches.length > MAX_RESULTS
                  ? `Showing ${MAX_RESULTS} of ${matches.length}. Keep typing to narrow it down.`
                  : `${matches.length} ${matches.length === 1 ? "course matches" : "courses match"}.`}
            </p>
            <ul>{matches.slice(0, MAX_RESULTS).map((c) => row([c.code], `search-${c.code}`))}</ul>
          </>
        ) : null}
      </div>

      <p aria-live="polite" className="sr-only">
        {announcement}
      </p>
    </div>
  );
}

/** A group's heading, skipped when it repeats the one just above it. */
function groupHeading(g: ProgramGroup, prev: ProgramGroup | undefined): string | null {
  const text = (x: ProgramGroup) => (x.subtitle ? `${x.title}: ${x.subtitle}` : x.title);
  return prev && text(prev) === text(g) ? null : text(g);
}

/**
 * One checklist row: a course, or an "or" row whose options sit in a small
 * select. The radios apply to the selected option. Prerequisites this course
 * brought in are listed under it, each removable.
 */
function CourseRow(props: {
  name: string;
  codes: string[];
  value: ChecklistState;
  onStatus: (code: string, status: RowStatus) => void;
  onMove: (from: string, to: string, status: RowStatus) => void;
  onRemove: (code: string, root: string) => void;
}) {
  const { codes, value } = props;
  const present = codes.find((c) => statusOf(value, c) !== "none");
  const [picked, setPicked] = useState(codes[0]);
  const code = present ?? picked;
  const status = statusOf(value, code);
  const entry = value.courses.find((c) => c.code === code);
  const course = getCourse(code);
  const added = addedFor(value, code);
  const isOr = codes.length > 1;

  return (
    <li className="border-t border-light-tan py-2 first:border-t-0">
      <fieldset className="min-w-0" data-course={code}>
        <legend className="w-full">
          {isOr ? (
            <span className="text-dark-tan">One of {codes.join(" or ")}: </span>
          ) : null}
          <strong>{code}</strong>
          {course ? ` ${course.title}` : ""}
          {course?.retired ? <span className="text-dark-tan"> (no longer offered)</span> : null}
          {course?.previousTitles.length ? (
            <span className="text-dark-tan"> (formerly {course.previousTitles.join("; ")})</span>
          ) : null}
        </legend>
        {isOr ? (
          <div className="mt-1">
            <label htmlFor={`${props.name}-pick`} className="mr-2">
              Course
            </label>
            <select
              id={`${props.name}-pick`}
              value={code}
              onChange={(e) => {
                const next = e.target.value;
                if (present) props.onMove(present, next, status);
                setPicked(next);
              }}
              className="max-w-full rounded-card border border-medium-tan bg-paper p-1"
            >
              {codes.map((c) => (
                <option key={c} value={c}>
                  {c}
                  {titleOf(c) ? ` ${titleOf(c)}` : ""}
                </option>
              ))}
            </select>
          </div>
        ) : null}
        <div className="mt-1 flex flex-wrap gap-x-4 gap-y-1">
          {(["done", "now", "none"] as const).map((s) => (
            <label key={s} className="flex items-center gap-1">
              <input
                type="radio"
                name={props.name}
                className="accent-miami-red"
                checked={status === s}
                onChange={() => props.onStatus(code, s)}
              />
              {STATUS_LABEL[s]}
            </label>
          ))}
        </div>
        {entry?.addedBecause ? (
          <p className="mt-1 text-dark-tan">Added automatically for {entry.addedBecause}.</p>
        ) : null}
      </fieldset>
      {added.length > 0 ? (
        <div className="mt-1 text-dark-tan">
          <p>
            Added because you {status === "now" ? "are taking" : "took"} {code}:
          </p>
          <ul className="flex flex-wrap gap-x-4">
            {added.map((a) => (
              <li key={a}>
                {a}{" "}
                <button
                  type="button"
                  className="underline"
                  aria-label={`Remove ${a}, added because of ${code}`}
                  onClick={() => props.onRemove(a, code)}
                >
                  Remove
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </li>
  );
}
