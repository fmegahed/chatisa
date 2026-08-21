"use client";

import { useMemo, useState } from "react";
import { COURSES } from "@/lib/scout/courses";
import { buildTiers } from "@/lib/scout/course-tiers";

/**
 * The course chips, popular first by tier, with a search box for everything
 * else. `single` turns the picker into a one-of choice for the showcase step.
 */
export function CoursePicker(props: {
  selected: string[];
  onChange: (codes: string[]) => void;
  single?: boolean;
}) {
  const [query, setQuery] = useState("");
  const [openTiers, setOpenTiers] = useState<Set<string>>(new Set());
  const tiers = useMemo(() => buildTiers(), []);
  const q = query.trim().toLowerCase();
  const matches = q
    ? COURSES.filter((c) => c.code.toLowerCase().includes(q) || c.title.toLowerCase().includes(q))
    : null;
  const toggle = (code: string) => {
    if (props.single) return props.onChange(props.selected[0] === code ? [] : [code]);
    props.onChange(
      props.selected.includes(code)
        ? props.selected.filter((c) => c !== code)
        : [...props.selected, code],
    );
  };
  const chip = (course: { code: string; title: string }) => {
    const on = props.selected.includes(course.code);
    return (
      <button
        key={course.code}
        type="button"
        aria-pressed={on}
        title={course.title}
        onClick={() => toggle(course.code)}
        className={
          on
            ? "rounded-card bg-miami-red px-3 py-1 font-bold text-paper"
            : "rounded-card border-2 border-medium-tan px-3 py-1 hover:bg-light-tan"
        }
      >
        {course.code}
      </button>
    );
  };
  return (
    <div>
      <label className="block font-bold" htmlFor="course-search">Find a course</label>
      <input
        id="course-search"
        type="search"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Code or title, for example 401 or forecasting"
        className="mt-1 w-full rounded-card border border-medium-tan p-2"
      />
      {matches ? (
        <div className="mt-3 flex flex-wrap gap-2">{matches.map(chip)}</div>
      ) : (
        tiers.map((tier) => {
          // Popular chips are always visible; the long tail sits behind a
          // toggle, the way Job Scout's profile shows the same tiers.
          const isOpen =
            openTiers.has(tier.name) || (!tier.collapsedByDefault && tier.more.length === 0);
          return (
            <fieldset key={tier.name} className="mt-4">
              <legend className="font-bold">{tier.name}</legend>
              {tier.popular.length > 0 ? (
                <div className="mt-2 flex flex-wrap gap-2">{tier.popular.map(chip)}</div>
              ) : null}
              {tier.more.length > 0 ? (
                <>
                  {isOpen ? (
                    <div className="mt-2 flex flex-wrap gap-2">{tier.more.map(chip)}</div>
                  ) : null}
                  <button
                    type="button"
                    aria-expanded={isOpen}
                    onClick={() => {
                      const next = new Set(openTiers);
                      if (isOpen) next.delete(tier.name);
                      else next.add(tier.name);
                      setOpenTiers(next);
                    }}
                    className="mt-2 underline"
                  >
                    {isOpen
                      ? "Show fewer"
                      : tier.popular.length === 0
                        ? `Show ${tier.more.length} graduate courses`
                        : `Show ${tier.more.length} more`}
                  </button>
                </>
              ) : null}
            </fieldset>
          );
        })
      )}
      {props.selected.length > 0 ? (
        <p className="mt-3 text-dark-tan">Selected: {props.selected.join(", ")}</p>
      ) : null}
    </div>
  );
}
