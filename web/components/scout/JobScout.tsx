"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelOption } from "@/lib/config/models";
import {
  useScoutProfile,
  useScoutProjects,
  useScoutSaved,
} from "@/lib/scout/use-scout-store";
import { profileStrengths } from "@/lib/scout/matching";
import { projectExtras } from "@/lib/scout/profile-store";
import type { FeedIndex } from "@/lib/scout/feed-types";
import { ProfileTab } from "./ProfileTab";
import { ProjectsTab } from "./ProjectsTab";
import { JobFeed } from "./JobFeed";
import { SavedTab } from "./SavedTab";
import { PortfolioTab } from "./PortfolioTab";

/**
 * Job Scout's client root: four tabs (user flow decision, 2026-07-29,
 * with My Projects deliberately BEFORE the feed). The feed index is
 * fetched once here and shared: the Jobs tab renders it, the Profile tab
 * derives demand analytics from it, the Saved tab joins against it.
 * Everything personal stays in this browser.
 */

const TABS = [
  { id: "profile", label: "My Profile" },
  { id: "projects", label: "My Projects" },
  { id: "jobs", label: "This Week's Jobs" },
  { id: "saved", label: "Saved Jobs" },
  // After Saved on purpose: the portfolio consumes saved jobs (v6.3.0).
  { id: "portfolio", label: "Portfolio Site" },
] as const;

type TabId = (typeof TABS)[number]["id"];

function tabFromUrl(): TabId | null {
  if (typeof window === "undefined") return null;
  const raw = new URLSearchParams(window.location.search).get("tab");
  return TABS.some((t) => t.id === raw) ? (raw as TabId) : null;
}

export function JobScout(props: {
  models: ModelOption[];
  defaultModelId: string;
  /** GitHub OAuth is configured server-side, so push/publish is offered. */
  githubEnabled: boolean;
  /** The signed-in student's display name, seeding the portfolio site. */
  studentName: string;
}) {
  const [profile, setProfile] = useScoutProfile();
  const { saved, toggle, hide } = useScoutSaved();
  const projectsStore = useScoutProjects();
  // null until the student navigates; the effective tab falls back to
  // profile-aware defaults below. Safe to read the URL lazily: the tab UI
  // only renders client-side (profile === "unknown" during SSR).
  const [tab, setTab] = useState<TabId | null>(() => tabFromUrl());
  const [seedSkills, setSeedSkills] = useState<string[]>([]);
  const [feed, setFeed] = useState<FeedIndex | null>(null);
  const [feedState, setFeedState] = useState<"loading" | "ready" | "failed">(
    "loading",
  );
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  // Retriable without a page refresh (user hit a failed load, 2026-07-29).
  // reloadNonce bumps re-run the effect; all setState happens after awaits
  // (the InterviewMentor pattern, keeps react-hooks/set-state-in-effect
  // clean).
  const [reloadNonce, setReloadNonce] = useState(0);
  useEffect(() => {
    void (async () => {
      try {
        const res = await fetch("/api/scout/feed?shape=index");
        if (!res.ok) throw new Error(String(res.status));
        const body = (await res.json()) as FeedIndex;
        setFeed(body);
        setFeedState("ready");
      } catch {
        setFeedState("failed");
      }
    })();
  }, [reloadNonce]);

  const strengths = useMemo(() => {
    if (profile === "unknown" || profile === false) return new Map<string, number>();
    return profileStrengths(
      profile.courses,
      [...profile.extras, ...projectExtras(projectsStore.projects)],
      profile.overrides ?? [],
    );
  }, [profile, projectsStore.projects]);

  if (profile === "unknown") {
    // Only the server render sees this; hydration swaps in the stored value.
    return <p role="status">Loading your profile from this browser...</p>;
  }

  const activeTab: TabId = tab ?? (profile === false ? "profile" : "jobs");

  function switchTab(next: TabId) {
    setTab(next);
    const url = new URL(window.location.href);
    url.searchParams.set("tab", next);
    window.history.replaceState(null, "", url);
  }

  function onTabKeyDown(e: React.KeyboardEvent, index: number) {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    e.preventDefault();
    const next =
      (index + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length;
    tabRefs.current[next]?.focus();
    switchTab(TABS[next].id);
  }

  const postings = feed?.postings ?? [];

  return (
    <div>
      <div
        role="tablist"
        aria-label="Job Scout sections"
        className="flex flex-wrap gap-1 border-b-2 border-medium-tan"
      >
        {TABS.map((t, i) => {
          const selected = activeTab === t.id;
          return (
            <button
              key={t.id}
              ref={(el) => {
                tabRefs.current[i] = el;
              }}
              role="tab"
              id={`tab-${t.id}`}
              aria-selected={selected}
              aria-controls={`panel-${t.id}`}
              tabIndex={selected ? 0 : -1}
              onClick={() => switchTab(t.id)}
              onKeyDown={(e) => onTabKeyDown(e, i)}
              className={
                selected
                  ? "rounded-t-lg border-2 border-b-0 border-medium-tan bg-paper px-4 py-2 font-bold text-miami-red"
                  : "rounded-t-lg px-4 py-2 hover:bg-light-tan"
              }
            >
              {t.label}
              {t.id === "saved" && saved.saved.length > 0
                ? ` (${saved.saved.length})`
                : ""}
            </button>
          );
        })}
      </div>

      <div
        role="tabpanel"
        id={`panel-${activeTab}`}
        aria-labelledby={`tab-${activeTab}`}
        className="pt-6"
      >
        {activeTab === "profile" ? (
          <ProfileTab
            models={props.models}
            defaultModelId={props.defaultModelId}
            profile={profile === false ? null : profile}
            onSave={(next) => {
              setProfile(next);
            }}
            onSeeJobs={() => switchTab("jobs")}
            strengths={strengths}
            projects={projectsStore.projects.projects}
            postings={postings}
          />
        ) : null}

        {activeTab === "projects" ? (
          profile === false ? (
            <EmptyState onGoProfile={() => switchTab("profile")} />
          ) : (
            <ProjectsTab
              key={seedSkills.join(",")}
              models={props.models}
              defaultModelId={props.defaultModelId}
              profile={profile}
              store={projectsStore}
              seedSkills={seedSkills}
              githubEnabled={props.githubEnabled}
            />
          )
        ) : null}

        {activeTab === "jobs" ? (
          profile === false ? (
            <EmptyState onGoProfile={() => switchTab("profile")} />
          ) : (
            <JobFeed
              postings={postings}
              freshness={feed?.freshness ?? null}
              loadState={feedState}
              onRetry={() => {
                setFeedState("loading");
                setReloadNonce((n) => n + 1);
              }}
              strengths={strengths}
              profile={profile}
              saved={saved}
              onToggleSaved={toggle}
              onHide={hide}
              onBuildSkills={(skillIds) => {
                setSeedSkills(skillIds);
                switchTab("projects");
              }}
            />
          )
        ) : null}

        {activeTab === "saved" ? (
          <SavedTab
            saved={saved}
            postings={postings}
            onToggleSaved={toggle}
            onGoJobs={() => switchTab("jobs")}
          />
        ) : null}

        {activeTab === "portfolio" ? (
          profile === false ? (
            <EmptyState onGoProfile={() => switchTab("profile")} />
          ) : (
            <PortfolioTab
              models={props.models}
              defaultModelId={props.defaultModelId}
              profile={profile}
              saved={saved}
              projects={projectsStore.projects}
              githubEnabled={props.githubEnabled}
              studentName={props.studentName}
              onGoJobs={() => switchTab("jobs")}
            />
          )
        ) : null}
      </div>
    </div>
  );
}

function EmptyState(props: { onGoProfile: () => void }) {
  return (
    <div className="rounded-card border border-medium-tan bg-light-tan p-5">
      <p>
        Start with your profile: check off the ISA courses you have taken so
        the matching has something to work with.
      </p>
      <button
        type="button"
        onClick={props.onGoProfile}
        className="mt-3 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red"
      >
        Set up my profile
      </button>
    </div>
  );
}
