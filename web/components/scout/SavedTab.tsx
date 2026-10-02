"use client";

import { useEffect, useRef } from "react";
import Link from "next/link";
import { announce, focusElement } from "@/lib/a11y/announce";
import type { SavedSnapshot, SavedState } from "@/lib/scout/profile-store";
import { postingName, type FeedPosting } from "@/lib/scout/feed-types";

const savedTitleId = (id: string) => `saved-title-${id}`;

/**
 * Saved Jobs: the home the user asked for (2026-07-29). Still-active saves
 * render full cards from the feed index; retired ones fall back to the
 * snapshot taken at save time, honestly labelled, because a posting that
 * left the weekly feed may still be open on the employer's site.
 */
export function SavedTab(props: {
  saved: SavedState;
  postings: FeedPosting[];
  onToggleSaved: (snapshot: Omit<SavedSnapshot, "savedAt">) => void;
  onGoJobs: () => void;
}) {
  const byId = new Map(props.postings.map((p) => [p.id, p]));
  const active = props.saved.saved.filter((s) => byId.has(s.id));
  const retired = props.saved.saved.filter((s) => !byId.has(s.id));

  // Unsave removes the card that had focus; the next one's title gets it
  // instead, or the heading, or the empty state (#35).
  const pendingFocus = useRef<string | null>(null);
  useEffect(() => {
    const target = pendingFocus.current;
    if (!target) return;
    pendingFocus.current = null;
    focusElement(
      document.getElementById(target) ??
        document.getElementById("saved-heading") ??
        document.getElementById("saved-empty"),
    );
  });

  function remove(s: SavedSnapshot) {
    const order = [...active, ...retired];
    const at = order.findIndex((x) => x.id === s.id);
    const neighbour = order[at + 1] ?? order[at - 1];
    pendingFocus.current = neighbour ? savedTitleId(neighbour.id) : "saved-heading";
    props.onToggleSaved({
      id: s.id,
      title: s.title,
      company: s.company,
      applyUrl: s.applyUrl,
    });
    announce(`Removed ${postingName(s)} from saved jobs.`);
  }

  if (props.saved.saved.length === 0) {
    return (
      <div
        id="saved-empty"
        className="rounded-card border border-medium-tan bg-light-tan p-5"
      >
        <p>
          Nothing saved yet. Save jobs from the weekly feed and they collect
          here, even after they leave the feed.
        </p>
        <button
          type="button"
          onClick={props.onGoJobs}
          className="mt-3 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red"
        >
          Browse this week&apos;s jobs
        </button>
      </div>
    );
  }

  return (
    <section aria-labelledby="saved-heading">
      <h2 id="saved-heading" className="text-2xl">
        Saved jobs
      </h2>

      {active.length > 0 ? (
        <ul className="mt-3 space-y-3">
          {active.map((s) => {
            const posting = byId.get(s.id)!;
            const name = postingName(posting);
            const location = posting.remote
              ? "Remote"
              : [posting.locationCity, posting.locationState]
                  .filter(Boolean)
                  .join(", ") || "Location not listed";
            return (
              <li
                key={s.id}
                className="rounded-card border border-medium-tan bg-paper p-4"
              >
                <h3 id={savedTitleId(s.id)} className="text-xl">
                  {posting.title}
                </h3>
                <p className="text-dark-tan">
                  {posting.company} · {location}
                  {s.savedAt ? ` · Saved ${s.savedAt.slice(0, 10)}` : ""}
                </p>
                {/* Repeated controls name their job (#34). */}
                <div className="mt-3 flex flex-wrap gap-3">
                  <a
                    href={posting.applyUrl}
                    target="_blank"
                    rel="noopener noreferrer"
                    aria-label={`Apply on employer site for ${name} (opens in a new tab)`}
                    className="rounded-card bg-miami-red px-3 py-1 font-bold text-paper hover:bg-accent-red"
                  >
                    Apply on employer site
                  </a>
                  <Link
                    href={`/jobapp-drafter?job=${posting.id}`}
                    aria-label={`Draft my resume and cover letter for ${name}`}
                    className="rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
                  >
                    Draft my resume and cover letter
                  </Link>
                  <button
                    type="button"
                    aria-label={`Unsave ${name}`}
                    onClick={() => remove(s)}
                    className="underline"
                  >
                    Unsave
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
      ) : null}

      {retired.length > 0 ? (
        <div className="mt-6">
          <h3 className="text-xl">No longer in the weekly feed</h3>
          <p className="text-dark-tan">
            These left our feed (postings expire after about a month), but
            the employer&apos;s listing may still be open.
          </p>
          <ul className="mt-2 space-y-2">
            {retired.map((s) => {
              const name = postingName(s);
              return (
                <li
                  key={s.id}
                  className="rounded-card border border-medium-tan bg-light-tan p-3"
                >
                  <p id={savedTitleId(s.id)}>
                    <strong>{s.title || "Saved posting"}</strong>
                    {s.company ? ` · ${s.company}` : ""}
                    {s.savedAt ? ` · Saved ${s.savedAt.slice(0, 10)}` : ""}
                  </p>
                  <div className="mt-2 flex flex-wrap gap-3">
                    {s.applyUrl ? (
                      <a
                        href={s.applyUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        aria-label={`Check the employer's listing for ${name} (opens in a new tab)`}
                        className="underline"
                      >
                        Check the employer&apos;s listing
                      </a>
                    ) : null}
                    <button
                      type="button"
                      aria-label={`Remove ${name} from saved jobs`}
                      onClick={() => remove(s)}
                      className="underline"
                    >
                      Remove
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
