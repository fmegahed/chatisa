"use client";

import { useMemo, useRef, useState } from "react";
import Link from "next/link";
import type { ModelOption } from "@/lib/config/models";
import { ModelChooser } from "@/components/ModelChooser";
import { getSkill, SKILLS } from "@/lib/scout/taxonomy";
import { COURSE_SKILLS } from "@/lib/scout/course-skills";
import type {
  ProjectRecord,
  ProjectsState,
  ScoutProfile,
} from "@/lib/scout/profile-store";
import {
  deleteScaffold,
  getScaffold,
  putScaffold,
} from "@/lib/scout/device-files";
import { scaffoldFileSet } from "@/lib/scout/github";
import { GithubConnect } from "@/components/scout/GithubConnect";
import { PushToGithubButton } from "@/components/scout/PushToGithubButton";

/**
 * My Projects: generate a job-agnostic portfolio scaffold, and keep every
 * generated project as an artifact with a home (user feedback, 2026-07-29:
 * "where does the GitHub from projects go?"). Records live in localStorage,
 * scaffold JSON in IndexedDB, so a push to GitHub works long after the
 * original visit (zip downloads were removed 2026-08-20: GitHub is the
 * only destination). Once a repo URL is added, the project's skills count
 * toward the profile (gated on the repo existing; an unbuilt scaffold
 * never inflates a student).
 */

interface Scaffold {
  repoName: string;
  summary: string;
  readme: string;
  files: { path: string; contents: string }[];
  instructions: string[];
  resumeBullets: string[];
}

const MAX_SKILLS = 6;

export function ProjectsTab(props: {
  models: ModelOption[];
  defaultModelId: string;
  profile: ScoutProfile;
  store: {
    projects: ProjectsState;
    add: (record: ProjectRecord) => void;
    setRepoUrl: (id: string, repoUrl: string | null) => void;
    remove: (id: string) => void;
  };
  /** Gap skills handed over by a job card's "close these gaps" button. */
  seedSkills: string[];
  /** GitHub OAuth is configured on this server, so pushing is offered. */
  githubEnabled: boolean;
}) {
  const [selected, setSelected] = useState<string[]>(() =>
    props.seedSkills.slice(0, MAX_SKILLS),
  );
  const [modelId, setModelId] = useState(props.defaultModelId);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scaffold, setScaffold] = useState<Scaffold | null>(null);
  const [scaffoldRecordId, setScaffoldRecordId] = useState<string | null>(null);
  const errorRef = useRef<HTMLParagraphElement>(null);

  /** The student's own evidence for the selected skills, sent for grounding. */
  const evidence = useMemo(
    () =>
      COURSE_SKILLS.filter(
        (l) =>
          props.profile.courses.includes(l.course) &&
          selected.includes(l.skillId) &&
          l.evidence,
      )
        .map((l) => l.evidence as string)
        .slice(0, 12),
    [props.profile.courses, selected],
  );

  async function generate() {
    setError(null);
    setBusy(true);
    setScaffold(null);
    try {
      const res = await fetch("/api/scout/project", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ modelId, skillIds: selected, evidence }),
      });
      const body = await res.json();
      if (!res.ok) {
        setError(body.error ?? "The scaffold did not generate. Try again.");
        setTimeout(() => errorRef.current?.focus(), 0);
        return;
      }
      const generated = body.scaffold as Scaffold;
      setScaffold(generated);
      setScaffoldRecordId(null);
      // The artifact gets a permanent local home immediately.
      const record: ProjectRecord = {
        id: crypto.randomUUID(),
        repoName: generated.repoName,
        summary: generated.summary,
        skillIds: [...selected],
        createdAt: new Date().toISOString(),
        mode: "scaffold",
        repoUrl: null,
      };
      props.store.add(record);
      setScaffoldRecordId(record.id);
      void putScaffold(record.id, generated);
    } catch {
      setError("The scaffold did not generate. Try again.");
      setTimeout(() => errorRef.current?.focus(), 0);
    } finally {
      setBusy(false);
    }
  }

  const projects = props.store.projects.projects;

  return (
    <div>
      <p className="mb-4 rounded-card border border-medium-tan bg-light-tan p-4">
        Already built a course project?{" "}
        <Link href="/portfolio?mode=project" className="font-bold underline">
          Publish it as a showcase
        </Link>{" "}
        in the Portfolio Builder: organized repository, landing page, and it
        counts toward your skills.
      </p>

      <section
        aria-labelledby="generate-heading"
        className="rounded-card border border-medium-tan bg-paper p-5"
      >
        <h2 id="generate-heading" className="text-2xl">
          Build a portfolio project
        </h2>
        <p className="mt-1 text-dark-tan">
          Pick up to {MAX_SKILLS} skills to demonstrate. The project is
          designed around skills, not any one job, so the repo works for
          every application. You build it; the scaffold gets you started.
        </p>

        {error ? (
          <p
            ref={errorRef}
            role="alert"
            tabIndex={-1}
            className="mt-3 rounded-card border-2 border-miami-red bg-paper p-3 font-bold text-miami-red"
          >
            {error}
          </p>
        ) : null}

        <div className="mt-3 flex flex-wrap gap-2">
          {selected.map((id) => (
            <button
              key={id}
              type="button"
              onClick={() => setSelected(selected.filter((s) => s !== id))}
              className="rounded-card border border-medium-tan bg-light-tan px-2 py-1"
              aria-label={`Remove ${getSkill(id)?.label ?? id}`}
            >
              {getSkill(id)?.label ?? id} ✕
            </button>
          ))}
          {selected.length < MAX_SKILLS ? (
            <label>
              <span className="sr-only">Add a skill</span>
              <select
                value=""
                onChange={(e) => {
                  if (e.target.value && !selected.includes(e.target.value)) {
                    setSelected([...selected, e.target.value]);
                  }
                }}
                className="rounded-card border border-medium-tan bg-paper px-2 py-1"
              >
                <option value="">Add a skill...</option>
                {SKILLS.filter((s) => !selected.includes(s.id)).map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label}
                  </option>
                ))}
              </select>
            </label>
          ) : null}
        </div>

        <div className="mt-4 max-w-xl">
          <ModelChooser
            options={props.models}
            value={modelId}
            onChange={setModelId}
            disabled={busy}
          />
        </div>

        <button
          type="button"
          disabled={busy || selected.length === 0}
          onClick={() => void generate()}
          className="mt-4 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
        >
          {busy ? "Designing your project..." : "Generate project scaffold"}
        </button>
        {busy ? (
          <p role="status" className="mt-2 text-dark-tan">
            Designing a project around your skills. This takes up to a minute.
          </p>
        ) : null}

        {scaffold ? (
          <div className="mt-5 border-t border-medium-tan pt-4">
            <h3 className="text-xl">{scaffold.repoName}</h3>
            <p className="mt-1">{scaffold.summary}</p>
            <h4 className="mt-4 font-bold">Put it on GitHub</h4>
            {props.githubEnabled ? (
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <GithubConnect returnPath="/job-scout" />
                {scaffoldRecordId ? (
                  <PushToGithubButton
                    repoName={scaffold.repoName}
                    getFiles={async () => scaffoldFileSet(scaffold)}
                    expectedRepoUrl={
                      props.store.projects.projects.find(
                        (p) => p.id === scaffoldRecordId,
                      )?.repoUrl ?? null
                    }
                    commitMessage="Project scaffold from ChatISA Job Scout"
                    onPushed={(repoUrl) =>
                      props.store.setRepoUrl(scaffoldRecordId, repoUrl)
                    }
                  />
                ) : null}
              </div>
            ) : null}
            {scaffold.resumeBullets.length > 0 ? (
              <>
                <h4 className="mt-4 font-bold">
                  Resume bullets to earn as you build
                </h4>
                <p className="text-dark-tan">
                  Bracketed numbers like [X%] are placeholders. Fill them in
                  only with what you actually measure; never guess a figure
                  you will be asked about in an interview.
                </p>
                <ul className="mt-1 list-inside list-disc">
                  {scaffold.resumeBullets.map((b) => (
                    <li key={b}>{b}</li>
                  ))}
                </ul>
              </>
            ) : null}
            <h4 className="mt-4 font-bold">README preview</h4>
            <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap rounded-card border border-medium-tan p-3">
              {scaffold.readme}
            </pre>
          </div>
        ) : null}
      </section>

      <section aria-labelledby="artifacts-heading" className="mt-8">
        <h2 id="artifacts-heading" className="text-2xl">
          Your projects
        </h2>
        {projects.length === 0 ? (
          <p className="mt-2 rounded-card border border-medium-tan bg-light-tan p-4">
            Projects you generate collect here, ready to push to GitHub.
            Once a project has a repository link, its skills count in your
            profile.
          </p>
        ) : (
          <ul className="mt-3 space-y-3">
            {projects.map((p) => (
              <ProjectCard
                key={p.id}
                project={p}
                githubEnabled={props.githubEnabled}
                onSetRepoUrl={(url) => props.store.setRepoUrl(p.id, url)}
                onRemove={() => {
                  props.store.remove(p.id);
                  void deleteScaffold(p.id);
                }}
              />
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

/** Loads a project's stored files and shapes them for a push, or null when
 * this device no longer holds them. */
async function storedPushFiles(projectId: string) {
  const stored = await getScaffold<Scaffold>(projectId);
  // A record stored by the retired Polish pane has no scaffold shape; the
  // button says the files are gone rather than pushing a broken tree.
  if (!stored || !Array.isArray(stored.files)) return null;
  return scaffoldFileSet(stored);
}

function ProjectCard(props: {
  project: ProjectRecord;
  githubEnabled: boolean;
  onSetRepoUrl: (url: string | null) => void;
  onRemove: () => void;
}) {
  const [urlDraft, setUrlDraft] = useState(props.project.repoUrl ?? "");
  const [editingUrl, setEditingUrl] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const p = props.project;

  return (
    <li className="rounded-card border border-medium-tan bg-paper p-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-xl">{p.repoName}</h3>
        <span className="text-dark-tan">
          Created {p.createdAt.slice(0, 10)}
        </span>
      </div>
      <p className="mt-1">{p.summary}</p>
      <p className="mt-1 text-dark-tan">
        Skills: {p.skillIds.map((id) => getSkill(id)?.label ?? id).join(", ")}
      </p>
      {p.repoUrl ? (
        <p className="mt-1">
          <a
            href={p.repoUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="underline"
          >
            {p.repoUrl}
          </a>{" "}
          <span className="font-bold">
            Built. Its skills count in your profile.
          </span>
        </p>
      ) : null}
      {note ? (
        <p role="status" className="mt-2 text-dark-tan">
          {note}
        </p>
      ) : null}
      <div className="mt-3 flex flex-wrap items-center gap-3">
        {/* Push is offered only while unlinked. Once a repo URL exists the
            student may have built real work there, and re-pushing the stored
            stubs would overwrite files of the same name. */}
        {props.githubEnabled && !p.repoUrl ? (
          <PushToGithubButton
            repoName={p.repoName}
            getFiles={() => storedPushFiles(p.id)}
            expectedRepoUrl={null}
            commitMessage="Project scaffold from ChatISA Job Scout"
            onPushed={(repoUrl) => props.onSetRepoUrl(repoUrl)}
          />
        ) : null}
        {editingUrl ? (
          <span className="flex flex-wrap items-center gap-2">
            <label htmlFor={`repo-${p.id}`} className="sr-only">
              GitHub repository URL
            </label>
            <input
              id={`repo-${p.id}`}
              type="url"
              value={urlDraft}
              onChange={(e) => setUrlDraft(e.target.value)}
              placeholder="https://github.com/you/repo"
              className="rounded-card border border-medium-tan bg-paper px-2 py-1"
            />
            <button
              type="button"
              onClick={() => {
                const trimmed = urlDraft.trim();
                if (!/^https:\/\/\S+\/\S+/.test(trimmed)) {
                  setNote("Paste the full https link to your repository.");
                  return;
                }
                props.onSetRepoUrl(trimmed);
                setEditingUrl(false);
                setNote(null);
              }}
              className="rounded-card bg-miami-red px-3 py-1 font-bold text-paper hover:bg-accent-red"
            >
              Save link
            </button>
          </span>
        ) : (
          <button
            type="button"
            onClick={() => setEditingUrl(true)}
            className="rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
          >
            {p.repoUrl ? "Change the repo link" : "I pushed it to GitHub"}
          </button>
        )}
        <button type="button" onClick={props.onRemove} className="underline">
          Remove
        </button>
      </div>
    </li>
  );
}
