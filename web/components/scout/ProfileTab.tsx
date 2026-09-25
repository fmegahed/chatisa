"use client";

import { useMemo, useRef, useState } from "react";
import Link from "next/link";
import type { ModelOption } from "@/lib/config/models";
import { ModelChooser } from "@/components/ModelChooser";
import { getSkill } from "@/lib/scout/taxonomy";
import type { CourseSkillLevel } from "@/lib/scout/course-skills";
import type { ChecklistState } from "@/lib/scout/checklist";
import type {
  ProfileExtra,
  ProjectRecord,
  ScoutProfile,
  SkillOverride,
} from "@/lib/scout/profile-store";
import {
  deleteResume,
  getResume,
  putResume,
  type DeviceResume,
} from "@/lib/scout/device-files";
import type { FeedPosting } from "@/lib/scout/feed-types";
import { FilePick } from "./FilePick";
import { SkillsPanel } from "./SkillsPanel";
import { CourseChecklist } from "./CourseChecklist";
import { GithubSkills } from "./GithubSkills";
import { useSyncExternalStore } from "react";

/**
 * My Profile: courses (popular-first chips), the live skills panel, and the
 * resume. The pick-a-course -> see-your-skills loop the first version was
 * missing (user feedback, 2026-07-29) is the whole point of this layout:
 * the skills panel recomputes on every toggle.
 */

const LEVEL_HELP: Record<CourseSkillLevel, string> = {
  anchor: "I can show real work with this",
  applied: "I have used it repeatedly",
  exposure: "I know the basics",
};

interface Suggestion {
  skillId: string;
  level: CourseSkillLevel;
  evidence: string;
  source: "resume" | "freeform";
}

/** Device-resume mirror (async IndexedDB behind a tiny external store). */
let resumeCache: DeviceResume | null | undefined;
const resumeListeners = new Set<() => void>();
function readResumeSnapshot(): DeviceResume | null {
  if (resumeCache === undefined) {
    resumeCache = null;
    void getResume().then((r) => {
      resumeCache = r;
      for (const l of resumeListeners) l();
    });
  }
  return resumeCache;
}
function setResumeCache(r: DeviceResume | null) {
  resumeCache = r;
  for (const l of resumeListeners) l();
}

/** The saved profile from this tab's checklist and skills. */
function toProfile(
  plan: ChecklistState,
  extras: ProfileExtra[],
  overrides: SkillOverride[],
): ScoutProfile {
  return { v: 2, ...plan, extras, overrides };
}

export function ProfileTab(props: {
  models: ModelOption[];
  defaultModelId: string;
  profile: ScoutProfile | null;
  onSave: (profile: ScoutProfile) => void;
  onSeeJobs: () => void;
  strengths: Map<string, number>;
  projects: ProjectRecord[];
  postings: FeedPosting[];
}) {
  const isFirstRun = props.profile === null;
  const [draftPlan, setDraftPlan] = useState<ChecklistState>(() => ({
    programs: props.profile?.programs ?? [],
    courses: props.profile?.courses ?? [],
    removedPrereqs: props.profile?.removedPrereqs ?? [],
  }));
  // A plan change made outside this tab (the next-term prompt) is adopted
  // here, without a remount that would clear unconfirmed suggestions and
  // typed text. Compared by content: the store may hand back a fresh object.
  const planKey = JSON.stringify([props.profile?.programs, props.profile?.courses, props.profile?.removedPrereqs]);
  const [seenPlanKey, setSeenPlanKey] = useState(planKey);
  if (planKey !== seenPlanKey) {
    setSeenPlanKey(planKey);
    if (props.profile) {
      setDraftPlan({
        programs: props.profile.programs,
        courses: props.profile.courses,
        removedPrereqs: props.profile.removedPrereqs,
      });
    }
  }
  const draftCourses = useMemo(
    () => new Set(draftPlan.courses.map((c) => c.code)),
    [draftPlan],
  );
  const [draftExtras, setDraftExtras] = useState<ProfileExtra[]>(
    () => props.profile?.extras ?? [],
  );
  const [draftOverrides, setDraftOverrides] = useState<SkillOverride[]>(
    () => props.profile?.overrides ?? [],
  );
  const [modelId, setModelId] = useState(props.defaultModelId);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [freeText, setFreeText] = useState("");
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [busy, setBusy] = useState<"resume" | "text" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const errorRef = useRef<HTMLParagraphElement>(null);
  const deviceResume = useSyncExternalStore(
    (l) => {
      resumeListeners.add(l);
      return () => resumeListeners.delete(l);
    },
    readResumeSnapshot,
    () => null,
  );

  /** First run keeps a draft; an existing profile live-saves every change. */
  function commit(
    plan: ChecklistState,
    extras: ProfileExtra[],
    overrides: SkillOverride[] = draftOverrides,
  ) {
    setDraftPlan(plan);
    setDraftExtras(extras);
    setDraftOverrides(overrides);
    if (!isFirstRun) {
      props.onSave(toProfile(plan, extras, overrides));
    }
  }

  const fail = (message: string) => {
    setError(message);
    setTimeout(() => errorRef.current?.focus(), 0);
  };

  async function extract(kind: "resume" | "text") {
    setError(null);
    setBusy(kind);
    try {
      const form = new FormData();
      form.set("modelId", modelId);
      if (kind === "resume") {
        if (!resumeFile) {
          fail("Choose a resume PDF first.");
          return;
        }
        form.set("resume", resumeFile);
      } else {
        if (freeText.trim().length < 10) {
          fail("Describe what you worked on in a sentence or two first.");
          return;
        }
        form.set("text", freeText.trim());
      }
      const res = await fetch("/api/scout/resume-skills", {
        method: "POST",
        body: form,
      });
      const body = await res.json();
      if (!res.ok) {
        fail(body.error ?? "Skill extraction did not complete. Try again.");
        return;
      }
      const source = kind === "resume" ? ("resume" as const) : ("freeform" as const);
      const have = new Set([
        ...draftExtras.map((e) => e.skillId),
        ...suggestions.map((s) => s.skillId),
      ]);
      const fresh = (body.skills as Omit<Suggestion, "source">[])
        .filter((s) => !have.has(s.skillId))
        .map((s) => ({ ...s, source }));
      setSuggestions((prev) => [...prev, ...fresh]);
      if (kind === "resume" && resumeFile) {
        // Keep the PDF on this device so JobApp Drafter and Interview
        // Mentor can reuse it (user decision, 2026-07-29). Never uploaded
        // anywhere by this call.
        const dataUrl = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => resolve(String(reader.result));
          reader.onerror = reject;
          reader.readAsDataURL(resumeFile);
        });
        if (await putResume(resumeFile.name, dataUrl)) {
          setResumeCache({
            name: resumeFile.name,
            dataUrl,
            addedAt: new Date().toISOString(),
          });
        }
      }
    } catch {
      fail("Skill extraction did not complete. Try again.");
    } finally {
      setBusy(null);
    }
  }

  function acceptSuggestion(s: Suggestion, level: CourseSkillLevel) {
    commit(draftPlan, [
      ...draftExtras,
      { skillId: s.skillId, level, source: s.source, evidence: s.evidence },
    ]);
    setSuggestions(suggestions.filter((x) => x.skillId !== s.skillId));
  }

  return (
    <div>
      {error ? (
        <p
          ref={errorRef}
          role="alert"
          tabIndex={-1}
          className="mb-4 rounded-card border-2 border-miami-red bg-paper p-3 font-bold text-miami-red"
        >
          {error}
        </p>
      ) : null}

      <p className="mt-4 rounded-card border border-medium-tan bg-light-tan p-4">
        Want a site that shows this off?{" "}
        <Link href="/portfolio?mode=career" className="font-bold underline">
          Build your portfolio
        </Link>{" "}
        with the Portfolio Builder. Published sites count toward your skills
        here.
      </p>

      <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* ------------------------------------------------ courses column */}
        <section aria-labelledby="profile-courses" className="min-w-0">
          <h2 id="profile-courses" className="text-2xl">
            Your courses
          </h2>
          <p className="mt-1 text-dark-tan">
            Mark what you have taken or are taking now. The skills panel
            updates as you go. Cross-listed codes count automatically.
          </p>
          <div className="mt-4">
            <CourseChecklist
              value={draftPlan}
              onChange={(plan) => commit(plan, draftExtras)}
            />
          </div>
        </section>

        {/* ------------------------------------------------- skills column */}
        <SkillsPanel
          strengths={props.strengths}
          draftCourses={draftCourses}
          draftStatusCourses={draftPlan.courses}
          draftExtras={draftExtras}
          overrides={draftOverrides}
          isFirstRun={isFirstRun}
          projects={props.projects}
          postings={props.postings}
          onAddManual={(skillId, level) =>
            commit(draftPlan, [
              ...draftExtras.filter((e) => e.skillId !== skillId),
              { skillId, level, source: "manual" },
            ])
          }
          onRemoveExtra={(skillId) =>
            commit(
              draftPlan,
              draftExtras.filter((e) => e.skillId !== skillId),
            )
          }
          onSetOverride={(skillId, level) =>
            commit(
              draftPlan,
              draftExtras,
              level === null
                ? draftOverrides.filter((o) => o.skillId !== skillId)
                : [
                    ...draftOverrides.filter((o) => o.skillId !== skillId),
                    { skillId, level },
                  ],
            )
          }
        />
      </div>

      {/* ---------------------------------------------------- experience */}
      <section aria-labelledby="profile-resume" className="mt-8">
        <h2 id="profile-resume" className="text-2xl">
          Your resume and experience (optional)
        </h2>
        <p className="mt-1 max-w-2xl text-dark-tan">
          An internship or your own projects can matter as much as a course.
          Your resume is read once to suggest skills, then kept only on this
          device so JobApp Drafter and Interview Mentor can reuse it. You
          confirm every suggestion.
        </p>

        <div className="mt-3 max-w-xl">
          <ModelChooser
            options={props.models}
            value={modelId}
            onChange={setModelId}
            help="Used to suggest skills from your resume or description."
          />
        </div>

        <div className="mt-4 flex flex-wrap items-center gap-3">
          <FilePick
            label="Choose a resume PDF"
            accept="application/pdf"
            fileName={resumeFile?.name ?? null}
            onChange={setResumeFile}
            disabled={busy !== null}
          />
          <button
            type="button"
            disabled={busy !== null || !resumeFile}
            onClick={() => void extract("resume")}
            className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
          >
            {busy === "resume" ? "Reading your resume..." : "Suggest skills from it"}
          </button>
        </div>
        {deviceResume && !resumeFile ? (
          <p role="status" className="mt-2">
            Resume on this device: <strong>{deviceResume.name}</strong>
            {deviceResume.addedAt
              ? ` (added ${deviceResume.addedAt.slice(0, 10)})`
              : ""}{" "}
            <button
              type="button"
              className="underline"
              onClick={() => {
                void deleteResume();
                setResumeCache(null);
              }}
            >
              Remove from this device
            </button>
          </p>
        ) : null}

        <div className="mt-4 max-w-xl">
          <label htmlFor="scout-freeform" className="block font-bold">
            Internship, ISA 340/480/481, or independent work
          </label>
          <textarea
            id="scout-freeform"
            rows={2}
            value={freeText}
            onChange={(e) => setFreeText(e.target.value)}
            placeholder="One or two lines about what you actually did."
            className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
          />
          <button
            type="button"
            disabled={busy !== null}
            onClick={() => void extract("text")}
            className="mt-2 rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan disabled:border-medium-gray disabled:text-medium-gray"
          >
            {busy === "text" ? "Mapping to skills..." : "Suggest skills from this"}
          </button>
        </div>
        {busy ? (
          <p role="status" className="mt-2 text-dark-tan">
            Working. This usually takes a few seconds.
          </p>
        ) : null}

        {suggestions.length > 0 ? (
          <div className="mt-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h3 className="text-xl">Suggested skills to confirm</h3>
              <button
                type="button"
                className="underline"
                onClick={() => {
                  commit(draftPlan, [
                    ...draftExtras,
                    ...suggestions.map((s) => ({
                      skillId: s.skillId,
                      level: s.level,
                      source: s.source,
                      evidence: s.evidence,
                    })),
                  ]);
                  setSuggestions([]);
                }}
              >
                Add all as suggested
              </button>
            </div>
            <ul className="mt-2 grid gap-3 sm:grid-cols-2">
              {suggestions.map((s) => (
                <SuggestionCard
                  key={s.skillId}
                  suggestion={s}
                  onAccept={(level) => acceptSuggestion(s, level)}
                  onDismiss={() =>
                    setSuggestions(suggestions.filter((x) => x.skillId !== s.skillId))
                  }
                />
              ))}
            </ul>
          </div>
        ) : null}
      </section>

      <GithubSkills
        models={props.models}
        extras={draftExtras}
        onExtras={(next) => commit(draftPlan, next)}
      />

      {isFirstRun ? (
        <div className="mt-8">
          <button
            type="button"
            disabled={draftCourses.size === 0 && draftExtras.length === 0}
            onClick={() => {
              props.onSave(toProfile(draftPlan, draftExtras, draftOverrides));
              props.onSeeJobs();
            }}
            className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
          >
            Save profile and see this week&apos;s jobs
          </button>
          {draftCourses.size === 0 && draftExtras.length === 0 ? (
            <p className="mt-2 text-dark-tan">
              Check at least one course, or confirm at least one skill, so
              there is something to match on.
            </p>
          ) : null}
        </div>
      ) : (
        <p role="status" className="mt-8 text-dark-tan">
          Changes save to this browser as you make them.
        </p>
      )}
    </div>
  );
}

function SuggestionCard(props: {
  suggestion: Suggestion;
  onAccept: (level: CourseSkillLevel) => void;
  onDismiss: () => void;
}) {
  const [level, setLevel] = useState<CourseSkillLevel>(props.suggestion.level);
  return (
    <li className="rounded-card border border-medium-tan bg-paper p-3">
      <p className="font-bold">
        {getSkill(props.suggestion.skillId)?.label ?? props.suggestion.skillId}
      </p>
      {props.suggestion.evidence ? (
        <p className="mt-1 text-dark-tan">
          &quot;{props.suggestion.evidence}&quot;
        </p>
      ) : null}
      <div className="mt-2 flex flex-wrap gap-3">
        {(Object.keys(LEVEL_HELP) as CourseSkillLevel[]).map((l) => (
          <label key={l} className="flex items-center gap-1">
            <input
              type="radio"
              name={`suggest-${props.suggestion.skillId}`}
              checked={level === l}
              onChange={() => setLevel(l)}
            />
            <span>{LEVEL_HELP[l]}</span>
          </label>
        ))}
      </div>
      <div className="mt-2 flex gap-3">
        <button
          type="button"
          onClick={() => props.onAccept(level)}
          className="rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
        >
          Add to my skills
        </button>
        <button type="button" onClick={props.onDismiss} className="underline">
          Dismiss
        </button>
      </div>
    </li>
  );
}
