"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { ModelOption } from "@/lib/config/models";
import {
  useScoutProfile,
  useScoutProjects,
  useScoutSaved,
} from "@/lib/scout/use-scout-store";
import { profileStrengths } from "@/lib/scout/matching";
import { focusElement } from "@/lib/a11y/announce";
import { projectExtras, publishedExtras } from "@/lib/scout/profile-store";
import { usePublishedWork } from "@/lib/portfolio/published";
import type { FeedIndex } from "@/lib/scout/feed-types";
import { ProfileTab } from "./ProfileTab";
import { ProjectsTab } from "./ProjectsTab";
import { JobFeed } from "./JobFeed";
import { SavedTab } from "./SavedTab";
import { NextTermPrompt } from "./NextTermPrompt";

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
}) {
  const [profile, setProfile] = useScoutProfile();
  const { saved, toggle, hide, unhide } = useScoutSaved();
  const projectsStore = useScoutProjects();
  // Sites published with the Portfolio Builder are real, built work, so
  // their skills feed matching exactly like a pushed project does.
  const published = usePublishedWork();
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
  const panelRef = useRef<HTMLDivElement>(null);
  // Set when a button inside a panel switches tabs: the button unmounts
  // with its panel, so focus moves to the new panel's heading (#38).
  const focusPanel = useRef(false);
  useEffect(() => {
    if (!focusPanel.current) return;
    focusPanel.current = false;
    const panel = panelRef.current;
    focusElement(panel?.querySelector<HTMLElement>("h2") ?? panel);
  });

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
      [],
      [
        ...profile.extras,
        ...projectExtras(projectsStore.projects),
        ...publishedExtras(published),
      ],
      profile.overrides ?? [],
      profile.courses,
    );
  }, [profile, projectsStore.projects, published]);

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

  /** Tab switches started from inside a panel also move focus (#38). */
  function goToTab(next: TabId) {
    focusPanel.current = true;
    switchTab(next);
  }

  // APG tabs keyboard: arrows wrap, Home/End jump; selection follows focus.
  function onTabKeyDown(e: React.KeyboardEvent, index: number) {
    let next: number;
    if (e.key === "ArrowRight") next = (index + 1) % TABS.length;
    else if (e.key === "ArrowLeft") next = (index + TABS.length - 1) % TABS.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = TABS.length - 1;
    else return;
    e.preventDefault();
    tabRefs.current[next]?.focus();
    switchTab(TABS[next].id);
  }

  const postings = feed?.postings ?? [];

  return (
    <div>
      {profile !== false ? (
        <NextTermPrompt
          profile={profile}
          // The profile tab adopts the changed plan itself, keeping any
          // suggestions and text still in progress there.
          onChange={(next) => setProfile(next)}
        />
      ) : null}
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
              // Only the selected panel is rendered, so only its tab
              // points at a panel that exists.
              aria-controls={selected ? `panel-${t.id}` : undefined}
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
        ref={panelRef}
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
            onSeeJobs={() => goToTab("jobs")}
            strengths={strengths}
            projects={projectsStore.projects.projects}
            postings={postings}
          />
        ) : null}

        {activeTab === "projects" ? (
          profile === false ? (
            <EmptyState onGoProfile={() => goToTab("profile")} />
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
            <EmptyState onGoProfile={() => goToTab("profile")} />
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
              onUnhide={unhide}
              onBuildSkills={(skillIds) => {
                setSeedSkills(skillIds);
                goToTab("projects");
              }}
            />
          )
        ) : null}

        {activeTab === "saved" ? (
          <SavedTab
            saved={saved}
            postings={postings}
            onToggleSaved={toggle}
            onGoJobs={() => goToTab("jobs")}
          />
        ) : null}
      </div>
    </div>
  );
}

function EmptyState(props: { onGoProfile: () => void }) {
  return (
    <div className="rounded-card border border-medium-tan bg-light-tan p-5">
      <p>
        Start with your profile: check off the courses you have taken so the
        matching has something to work with.
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
