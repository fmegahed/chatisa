"use client";

import { useCallback, useMemo, useState } from "react";
import Link from "next/link";
import { scoreJob, type JobMatch, type JobSkill } from "@/lib/scout/matching";
import { getSkill } from "@/lib/scout/taxonomy";
import { COURSE_SKILLS } from "@/lib/scout/course-skills";
import type {
  SavedSnapshot,
  SavedState,
  ScoutProfile,
} from "@/lib/scout/profile-store";
import type { FeedFreshness, FeedPosting } from "@/lib/scout/feed-types";

/**
 * The Jobs tab. Matches are computed HERE, in the browser, from the
 * postings' public tags and the local profile: the server never sees who
 * matches what (local-first decision, 2026-07-28). Bands are text, never
 * colour alone; the signature element is the serif coverage fraction.
 * The feed index arrives via props (fetched once by JobScout and shared
 * across tabs).
 */

const BAND_LABEL: Record<JobMatch["band"], string> = {
  strong: "Strong match",
  good: "Good match",
  stretch: "Stretch",
};

const CATEGORY_LABEL = {
  fulltime: "Full-time",
  internship: "Internship",
  federal: "Federal",
} as const;

const PAGE_SIZE = 25;
/** States shown as one-click chips before "More states". */
const TOP_STATES = 6;

/** Full names for the type-ahead, so "ken" finds KY. */
const STATE_NAMES: Record<string, string> = {
  AL: "Alabama", AK: "Alaska", AZ: "Arizona", AR: "Arkansas",
  CA: "California", CO: "Colorado", CT: "Connecticut", DE: "Delaware",
  DC: "District of Columbia", FL: "Florida", GA: "Georgia", HI: "Hawaii",
  ID: "Idaho", IL: "Illinois", IN: "Indiana", IA: "Iowa", KS: "Kansas",
  KY: "Kentucky", LA: "Louisiana", ME: "Maine", MD: "Maryland",
  MA: "Massachusetts", MI: "Michigan", MN: "Minnesota", MS: "Mississippi",
  MO: "Missouri", MT: "Montana", NE: "Nebraska", NV: "Nevada",
  NH: "New Hampshire", NJ: "New Jersey", NM: "New Mexico", NY: "New York",
  NC: "North Carolina", ND: "North Dakota", OH: "Ohio", OK: "Oklahoma",
  OR: "Oregon", PA: "Pennsylvania", RI: "Rhode Island",
  SC: "South Carolina", SD: "South Dakota", TN: "Tennessee", TX: "Texas",
  UT: "Utah", VT: "Vermont", VA: "Virginia", WA: "Washington",
  WV: "West Virginia", WI: "Wisconsin", WY: "Wyoming",
};

/**
 * State filter: top states by demand as one-click chips, the tail behind a
 * type-ahead. Deliberately NOT a Power BI-style Ctrl-click slicer: hidden
 * modifier keys fail on touch devices and screen readers, and a dropdown
 * of fifty checkboxes still scans badly (user discussion, 2026-07-29).
 */
function StateFilter(props: {
  stateCounts: [string, number][];
  selected: Set<string>;
  onChange: (next: Set<string>) => void;
  open: boolean;
  onToggleOpen: () => void;
}) {
  const [query, setQuery] = useState("");
  const top = props.stateCounts.slice(0, TOP_STATES);
  const topIds = new Set(top.map(([s]) => s));
  // Selected tail states surface as chips too, so removal is one click.
  const chipStates = [
    ...top,
    ...props.stateCounts.filter(
      ([s]) => props.selected.has(s) && !topIds.has(s),
    ),
  ];
  const tail = props.stateCounts.filter(([s]) => !topIds.has(s));
  const q = query.trim().toLowerCase();
  const tailFiltered = q
    ? tail.filter(
        ([s]) =>
          s.toLowerCase().startsWith(q) ||
          (STATE_NAMES[s] ?? "").toLowerCase().includes(q),
      )
    : tail;

  const toggle = (state: string, on: boolean) => {
    const next = new Set(props.selected);
    if (on) next.add(state);
    else next.delete(state);
    props.onChange(next);
  };

  return (
    <fieldset>
      <legend className="font-bold">
        States
        {props.selected.size > 0 ? ` (${props.selected.size} selected)` : ""}
      </legend>
      <div className="mt-1 flex flex-wrap items-center gap-2">
        {chipStates.map(([state, count]) => {
          const on = props.selected.has(state);
          return (
            <label
              key={state}
              title={STATE_NAMES[state] ?? state}
              className={
                on
                  ? "cursor-pointer rounded-card border-2 border-miami-red bg-paper px-2 py-1 font-bold text-miami-red has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-miami-red has-[:focus-visible]:outline-offset-2"
                  : "cursor-pointer rounded-card border border-medium-tan bg-paper px-2 py-1 hover:bg-light-tan has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-miami-red has-[:focus-visible]:outline-offset-2"
              }
            >
              <input
                type="checkbox"
                className="sr-only"
                checked={on}
                aria-label={`${STATE_NAMES[state] ?? state} (${count === 1 ? "1 posting" : `${count} postings`})`}
                onChange={(e) => toggle(state, e.target.checked)}
              />
              {on ? "✓ " : ""}
              {state} ({count})
            </label>
          );
        })}
        {tail.length > 0 ? (
          <button
            type="button"
            aria-expanded={props.open}
            aria-controls="state-more"
            onClick={props.onToggleOpen}
            className="underline"
          >
            {props.open ? "Fewer states" : `More states (${tail.length})`}
          </button>
        ) : null}
        {props.selected.size > 0 ? (
          <button
            type="button"
            onClick={() => props.onChange(new Set())}
            className="underline"
          >
            Clear
          </button>
        ) : null}
      </div>
      {props.open && tail.length > 0 ? (
        <div
          id="state-more"
          className="mt-2 max-w-md rounded-card border border-medium-tan bg-paper p-3"
        >
          <label htmlFor="state-search" className="block font-bold">
            Type a state
          </label>
          <input
            id="state-search"
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="ken finds Kentucky"
            className="mt-1 w-40 rounded-card border border-medium-tan bg-paper px-2 py-1"
          />
          <div className="mt-2 flex max-h-40 flex-wrap gap-x-4 gap-y-1 overflow-y-auto">
            {tailFiltered.map(([state, count]) => (
              <label key={state} className="flex items-center gap-1">
                <input
                  type="checkbox"
                  checked={props.selected.has(state)}
                  aria-label={`${STATE_NAMES[state] ?? state} (${count === 1 ? "1 posting" : `${count} postings`})`}
                  onChange={(e) => toggle(state, e.target.checked)}
                />
                <span>
                  {STATE_NAMES[state] ?? state} ({count})
                </span>
              </label>
            ))}
            {tailFiltered.length === 0 ? (
              <p className="text-dark-tan">No state matches that.</p>
            ) : null}
          </div>
        </div>
      ) : null}
    </fieldset>
  );
}

export function JobFeed(props: {
  postings: FeedPosting[];
  freshness: FeedFreshness | null;
  loadState: "loading" | "ready" | "failed";
  strengths: Map<string, number>;
  profile: ScoutProfile;
  saved: SavedState;
  onToggleSaved: (snapshot: Omit<SavedSnapshot, "savedAt">) => void;
  onHide: (id: string) => void;
  onBuildSkills: (skillIds: string[]) => void;
  onRetry: () => void;
}) {
  const [category, setCategory] = useState<
    "all" | "fulltime" | "internship" | "federal"
  >("all");
  const [selectedStates, setSelectedStates] = useState<Set<string>>(new Set());
  const [statesOpen, setStatesOpen] = useState(false);
  const [remoteOnly, setRemoteOnly] = useState(false);
  const [hideNoSponsorship, setHideNoSponsorship] = useState(false);
  const [visible, setVisible] = useState(PAGE_SIZE);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [details, setDetails] = useState<Record<string, string>>({});

  /** Which of the student's courses evidence a skill, for honest provenance. */
  const provenance = useCallback(
    (skillId: string): string[] => {
      const fromCourses = COURSE_SKILLS.filter(
        (l) =>
          l.skillId === skillId && props.profile.courses.includes(l.course),
      ).map((l) => l.course);
      const fromExtras = props.profile.extras
        .filter((e) => e.skillId === skillId)
        .map((e) =>
          e.source === "resume" ? "your resume" : "your experience",
        );
      return [...fromCourses, ...fromExtras];
    },
    [props.profile],
  );

  const matched = useMemo(() => {
    const hidden = new Set(props.saved.hiddenIds);
    const rows = props.postings
      .filter((p) => !hidden.has(p.id))
      .map((p) => {
        let skills: JobSkill[] = [];
        try {
          skills = JSON.parse(p.skillsJson);
        } catch {
          // A malformed tag row matches nothing rather than crashing the feed.
        }
        return { posting: p, match: scoreJob(props.strengths, skills) };
      });
    // Ranking (user request to document, 2026-07-29): postings sort by the
    // requirement-coverage match score (design §2.4 — how much of what the
    // job asks for the student covers, required skills weighted double),
    // shrunk by how much evidence the tag carries so a thin one-skill 1/1
    // never outranks a broad 6/7 match, with newer postings breaking ties.
    // Filters below narrow but never reorder, so the best fits are always
    // at the top of any view.
    rows.sort(
      (a, b) =>
        b.match.rank - a.match.rank ||
        (b.posting.postedAt ?? "").localeCompare(a.posting.postedAt ?? ""),
    );
    return rows;
  }, [props.postings, props.strengths, props.saved.hiddenIds]);

  // Sorted by posting count: demand concentrates in a few states, so the
  // top handful become one-click chips and the tail hides behind a
  // type-ahead (user feedback 2026-07-29 — fifty checkboxes scan badly;
  // same popular-first pattern as the course picker).
  const stateCounts = useMemo(() => {
    const counts = new Map<string, number>();
    for (const p of props.postings) {
      if (p.locationState)
        counts.set(p.locationState, (counts.get(p.locationState) ?? 0) + 1);
    }
    return [...counts.entries()].sort(
      (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]),
    );
  }, [props.postings]);

  const filtered = matched.filter(({ posting }) => {
    if (category !== "all" && posting.category !== category) return false;
    if (
      selectedStates.size > 0 &&
      (!posting.locationState || !selectedStates.has(posting.locationState))
    )
      return false;
    if (remoteOnly && !posting.remote) return false;
    if (hideNoSponsorship && posting.visaSponsorship === "no_sponsorship")
      return false;
    return true;
  });

  async function expand(id: string) {
    if (expanded === id) {
      setExpanded(null);
      return;
    }
    setExpanded(id);
    if (!details[id]) {
      try {
        const res = await fetch(`/api/scout/postings/${id}`);
        if (res.ok) {
          const body = await res.json();
          setDetails((d) => ({ ...d, [id]: body.posting.description }));
        } else {
          setDetails((d) => ({
            ...d,
            [id]: "This posting's full text is no longer available.",
          }));
        }
      } catch {
        setDetails((d) => ({
          ...d,
          [id]: "The full text did not load. Use the apply link to read it on the employer's site.",
        }));
      }
    }
  }

  const updatedDate = props.freshness?.updatedAt
    ? new Date(props.freshness.updatedAt).toLocaleDateString("en-US", {
        weekday: "long",
        month: "long",
        day: "numeric",
      })
    : null;
  const sourceNotes: string[] = [];
  if (props.freshness?.sourceErrors?.activejobs)
    sourceNotes.push("some employer boards were unavailable last harvest");
  if (props.freshness?.sourceErrors?.usajobs)
    sourceNotes.push("federal listings were unavailable last harvest");

  return (
    <section aria-labelledby="feed-heading">
      <h2 id="feed-heading" className="text-2xl">
        This week&apos;s jobs
      </h2>
      <p role="status" className="mt-1 text-dark-tan">
        {props.loadState === "loading"
          ? "Loading this week's postings..."
          : props.freshness && props.freshness.totalActive > 0
            ? `Updated ${updatedDate ?? "recently"} · ${props.freshness.totalActive} postings from employer career sites and USAJobs${sourceNotes.length ? ` · Note: ${sourceNotes.join("; ")}` : ""}`
            : "No postings yet. The feed fills after the first Sunday harvest; ask the ChatISA maintainers if this persists."}
      </p>
      <p className="text-dark-tan">
        Sorted by how well each posting matches your profile, best first;
        newer postings break ties.
      </p>
      {props.loadState === "failed" ? (
        <div
          role="alert"
          className="mt-3 rounded-card border-2 border-miami-red bg-paper p-3"
        >
          <p className="font-bold text-miami-red">The feed did not load.</p>
          <button
            type="button"
            onClick={props.onRetry}
            className="mt-2 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red"
          >
            Try again
          </button>
        </div>
      ) : null}

      <div className="mt-4 flex flex-wrap items-start gap-x-6 gap-y-3">
        <fieldset>
          <legend className="font-bold">Type</legend>
          <div className="mt-1 flex gap-3">
            {(["all", "fulltime", "internship", "federal"] as const).map((c) => (
              <label key={c} className="flex items-center gap-1">
                <input
                  type="radio"
                  name="scout-category"
                  checked={category === c}
                  onChange={() => setCategory(c)}
                />
                <span>{c === "all" ? "All" : CATEGORY_LABEL[c]}</span>
              </label>
            ))}
          </div>
        </fieldset>

        <StateFilter
          stateCounts={stateCounts}
          selected={selectedStates}
          onChange={setSelectedStates}
          open={statesOpen}
          onToggleOpen={() => setStatesOpen(!statesOpen)}
        />

        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={remoteOnly}
            onChange={(e) => setRemoteOnly(e.target.checked)}
          />
          <span>Remote only</span>
        </label>
        <label className="flex items-center gap-2">
          <input
            type="checkbox"
            checked={hideNoSponsorship}
            onChange={(e) => setHideNoSponsorship(e.target.checked)}
          />
          <span>Hide &quot;no visa sponsorship&quot; postings</span>
        </label>
      </div>

      <ul className="mt-4 space-y-3">
        {filtered.slice(0, visible).map(({ posting, match }) => {
          const isOpen = expanded === posting.id;
          const isSaved = props.saved.saved.some((s) => s.id === posting.id);
          const location = posting.remote
            ? "Remote"
            : [posting.locationCity, posting.locationState]
                .filter(Boolean)
                .join(", ") || "Location not listed";
          return (
            <li
              key={posting.id}
              className="rounded-card border border-medium-tan bg-paper p-4"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <h3 className="text-xl">{posting.title}</h3>
                  <p className="text-dark-tan">
                    {posting.company} · {location} ·{" "}
                    {CATEGORY_LABEL[posting.category]}
                    {posting.postedAt ? ` · Posted ${posting.postedAt}` : ""}
                  </p>
                  {posting.visaSponsorship !== "unknown" ? (
                    // Only what the posting itself said; silence means the
                    // ad did not say, not that the answer is no.
                    <p className="font-bold">
                      {posting.visaSponsorship === "sponsors"
                        ? "Posting mentions visa sponsorship"
                        : "Posting says no visa sponsorship"}
                    </p>
                  ) : null}
                </div>
                <p className="text-right">
                  {match.totalRequired > 0 ? (
                    <span className="font-display text-2xl">
                      {match.coveredRequired}/{match.totalRequired}
                    </span>
                  ) : null}
                  <span className="block font-bold">{BAND_LABEL[match.band]}</span>
                  {match.totalRequired > 0 ? (
                    <span className="block text-dark-tan">
                      required skills covered
                    </span>
                  ) : null}
                </p>
              </div>

              {match.matched.length > 0 ? (
                <p className="mt-2">
                  <strong>You bring:</strong>{" "}
                  {match.matched
                    .map((m) => getSkill(m.skillId)?.label ?? m.skillId)
                    .join(", ")}
                </p>
              ) : null}
              {match.gaps.length > 0 ? (
                <p className="mt-1">
                  <strong>Gaps:</strong>{" "}
                  {match.gaps
                    .map(
                      (g) =>
                        `${getSkill(g.skillId)?.label ?? g.skillId}${g.importance === "preferred" ? " (preferred)" : ""}`,
                    )
                    .join(", ")}
                </p>
              ) : null}

              <div className="mt-3 flex flex-wrap gap-3">
                <button
                  type="button"
                  aria-expanded={isOpen}
                  aria-controls={`posting-${posting.id}`}
                  onClick={() => void expand(posting.id)}
                  className="rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
                >
                  {isOpen ? "Hide details" : "Details"}
                </button>
                <a
                  href={posting.applyUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-card bg-miami-red px-3 py-1 font-bold text-paper hover:bg-accent-red"
                >
                  Apply on employer site
                </a>
                <Link
                  href={`/jobapp-drafter?job=${posting.id}`}
                  className="rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
                >
                  Draft my resume and cover letter
                </Link>
                <button
                  type="button"
                  onClick={() =>
                    props.onToggleSaved({
                      id: posting.id,
                      title: posting.title,
                      company: posting.company,
                      applyUrl: posting.applyUrl,
                    })
                  }
                  className="underline"
                >
                  {isSaved ? "Unsave" : "Save"}
                </button>
                <button
                  type="button"
                  onClick={() => props.onHide(posting.id)}
                  className="underline"
                >
                  Hide
                </button>
              </div>

              {isOpen ? (
                <div id={`posting-${posting.id}`} className="mt-3">
                  {match.matched.length > 0 ? (
                    <div className="rounded-card bg-light-tan p-3">
                      <p className="font-bold">Where your skills come from</p>
                      <ul className="mt-1 list-inside list-disc">
                        {match.matched.map((m) => (
                          <li key={m.skillId}>
                            {getSkill(m.skillId)?.label ?? m.skillId}:{" "}
                            {provenance(m.via ?? m.skillId).join(", ") ||
                              "your profile"}
                            {m.via
                              ? ` (via ${getSkill(m.via)?.label ?? m.via})`
                              : ""}
                          </li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                  {match.gaps.length > 0 ? (
                    <button
                      type="button"
                      onClick={() =>
                        props.onBuildSkills(match.gaps.map((g) => g.skillId))
                      }
                      className="mt-3 rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
                    >
                      Close these gaps with a portfolio project
                    </button>
                  ) : null}
                  <p className="mt-3 max-h-96 overflow-y-auto whitespace-pre-wrap border-t border-medium-tan pt-3">
                    {details[posting.id] ?? "Loading the full posting..."}
                  </p>
                </div>
              ) : null}
            </li>
          );
        })}
      </ul>
      {filtered.length === 0 && props.loadState === "ready" ? (
        <p className="mt-4 rounded-card border border-medium-tan bg-light-tan p-4">
          Nothing matches these filters this week. Widen a filter, or check
          back after Sunday&apos;s refresh.
        </p>
      ) : null}
      {filtered.length > visible ? (
        <button
          type="button"
          onClick={() => setVisible(visible + PAGE_SIZE)}
          className="mt-4 rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan"
        >
          Show more ({filtered.length - visible} left)
        </button>
      ) : null}
    </section>
  );
}
