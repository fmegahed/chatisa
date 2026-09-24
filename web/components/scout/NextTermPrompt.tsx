"use client";

import { useState } from "react";
import { currentTerm, staleTakingNow, type ScoutProfile } from "@/lib/scout/profile-store";
import { finishCourses, keepTaking } from "@/lib/scout/checklist";

/**
 * The next-term prompt (v6.7.0): courses marked Taking now in an earlier
 * term are asked about once, above every Job Scout tab, so a returning
 * student sees it wherever they land. "Not now" hides it for this visit.
 */
export function NextTermPrompt(props: {
  profile: ScoutProfile;
  onChange: (next: ScoutProfile) => void;
}) {
  const [dismissed, setDismissed] = useState(false);
  const now = new Date();
  const stale = staleTakingNow(props.profile, now);
  if (dismissed || stale.length === 0) return null;

  const plan = { programs: props.profile.programs, courses: props.profile.courses, removedPrereqs: props.profile.removedPrereqs };
  return (
    <section
      aria-labelledby="next-term-heading"
      className="mb-4 rounded-card border-2 border-miami-red bg-paper p-3"
    >
      <h2 id="next-term-heading" className="text-lg font-bold">
        It&apos;s {currentTerm(now)}. Did you finish these?
      </h2>
      <ul className="mt-2">
        {stale.map((c) => (
          <li key={c.code} className="mt-1 flex flex-wrap items-center gap-3">
            <span>
              <strong>{c.code}</strong>
              {c.term ? ` (Taking now since ${c.term})` : ""}
            </span>
            <button
              type="button"
              aria-label={`Finished ${c.code}`}
              onClick={() => props.onChange({ ...props.profile, ...finishCourses(plan, [c.code]) })}
              className="rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
            >
              Finished
            </button>
            <button
              type="button"
              aria-label={`Still taking ${c.code}`}
              onClick={() => props.onChange({ ...props.profile, ...keepTaking(plan, [c.code], now) })}
              className="underline"
            >
              Still taking
            </button>
          </li>
        ))}
      </ul>
      <button type="button" onClick={() => setDismissed(true)} className="mt-2 underline">
        Not now
      </button>
    </section>
  );
}
