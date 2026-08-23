"use client";

import type { SiteRecord } from "@/lib/portfolio/store";
import type { Wip } from "@/lib/portfolio/wip";

/** The front door: pick which kind of site to build, or reopen one you made. */
export function ModeStep(props: {
  sites: SiteRecord[];
  wip: Wip | null;
  onResume: () => void;
  onDiscard: () => void;
  onPick: (mode: "career" | "showcase") => void;
  onOpen: (site: SiteRecord) => void;
  onRemove: (site: SiteRecord) => void;
}) {
  const card = (mode: "career" | "showcase", title: string, body: string) => (
    <button
      type="button"
      onClick={() => props.onPick(mode)}
      className="rounded-card border-2 border-medium-tan bg-paper p-5 text-left hover:border-miami-red"
    >
      <h2 className="text-2xl">{title}</h2>
      <p className="mt-2">{body}</p>
    </button>
  );
  return (
    <div>
      {props.wip ? (
        <section
          aria-label="Unfinished site"
          className="mb-6 flex flex-wrap items-center justify-between gap-3 rounded-card border-2 border-miami-red bg-paper p-4"
        >
          <p>
            <strong>You have an unfinished {props.wip.mode === "career" ? "portfolio" : "showcase"}</strong> saved
            in this browser{" "}
            {new Date(props.wip.savedAt).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}.
          </p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={props.onResume}
              className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:opacity-90"
            >
              Continue
            </button>
            <button type="button" onClick={props.onDiscard} className="rounded-card px-3 py-2 underline">
              Discard
            </button>
          </div>
        </section>
      ) : null}
      <div className="grid gap-4 md:grid-cols-2">
        {card(
          "career",
          "Career portfolio",
          "One page about you: resume, the classes you took, up to five projects, an optional photo. Published as your portfolio repository.",
        )}
        {card(
          "showcase",
          "Project showcase",
          "One finished course project, organized into a clean repository with a landing page that tells its story. Make as many as you like.",
        )}
      </div>
      {props.sites.length > 0 ? (
        <section className="mt-8">
          <h2 className="text-xl">Your sites</h2>
          <ul className="mt-2 space-y-2">
            {props.sites.map((s) => (
              <li
                key={s.id}
                className="flex flex-wrap items-center justify-between gap-2 rounded-card border border-medium-tan bg-paper p-3"
              >
                <div>
                  <strong>{s.title}</strong>{" "}
                  <span className="text-dark-tan">({s.kind === "career" ? "portfolio" : "showcase"})</span>
                  {s.pagesUrl ? (
                    <>
                      {" "}
                      &middot;{" "}
                      <a href={s.pagesUrl} target="_blank" rel="noopener noreferrer" className="underline">
                        View
                      </a>
                    </>
                  ) : null}
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => props.onOpen(s)}
                    className="rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
                  >
                    Update
                  </button>
                  <button
                    type="button"
                    onClick={() => props.onRemove(s)}
                    className="rounded-card px-3 py-1 underline"
                  >
                    Forget
                  </button>
                </div>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
