"use client";

import Link from "next/link";
import type { SavedSnapshot, SavedState } from "@/lib/scout/profile-store";
import type { FeedPosting } from "@/lib/scout/feed-types";

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

  if (props.saved.saved.length === 0) {
    return (
      <div className="rounded-card border border-medium-tan bg-light-tan p-5">
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
                <h3 className="text-xl">{posting.title}</h3>
                <p className="text-dark-tan">
                  {posting.company} · {location}
                  {s.savedAt ? ` · Saved ${s.savedAt.slice(0, 10)}` : ""}
                </p>
                <div className="mt-3 flex flex-wrap gap-3">
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
                        id: s.id,
                        title: s.title,
                        company: s.company,
                        applyUrl: s.applyUrl,
                      })
                    }
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
            {retired.map((s) => (
              <li
                key={s.id}
                className="rounded-card border border-medium-tan bg-light-tan p-3"
              >
                <p>
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
                      className="underline"
                    >
                      Check the employer&apos;s listing
                    </a>
                  ) : null}
                  <button
                    type="button"
                    onClick={() =>
                      props.onToggleSaved({
                        id: s.id,
                        title: s.title,
                        company: s.company,
                        applyUrl: s.applyUrl,
                      })
                    }
                    className="underline"
                  >
                    Remove
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
