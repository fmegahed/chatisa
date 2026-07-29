"use client";

import { useMemo, useRef, useState } from "react";
import { strToU8, zipSync } from "fflate";
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

/**
 * My Projects: generate a job-agnostic portfolio scaffold, and keep every
 * generated project as an artifact with a home (user feedback, 2026-07-29:
 * "where does the GitHub from projects go?"). Records live in localStorage,
 * scaffold JSON in IndexedDB, so "Download the zip again" works long after
 * the original visit. Once a repo URL is added, the project's skills count
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

/** The organize-only plan for a student's real project (2026-07-29). */
interface PolishPlan {
  repoName: string;
  summary: string;
  readme: string;
  gitignore: string;
  layout: { from: string; to: string }[];
  exclude: { name: string; reason: string }[];
  extraFiles: { path: string; contents: string }[];
  suggestions: string[];
  resumeBullets: string[];
  skillIds: string[];
}

/** What device-files stores for a polished project so re-download works.
 * Binary originals (PDFs) are not stored; the card says to re-add them. */
interface StoredPolish {
  mode: "polished";
  plan: PolishPlan;
  textFiles: { path: string; contents: string }[];
  binaryPaths: string[];
}

const TEXT_EXTENSIONS =
  /\.(py|r|ipynb|sql|md|txt|csv|qmd|rmd|js|ts|json|yml|yaml)$/i;
const MAX_POLISH_FILES = 15;

const MAX_SKILLS = 6;

function downloadScaffoldZip(scaffold: Scaffold) {
  const entries: Record<string, Uint8Array> = {
    "README.md": strToU8(scaffold.readme),
  };
  for (const file of scaffold.files) entries[file.path] = strToU8(file.contents);
  const blob = new Blob([zipSync(entries) as BlobPart], {
    type: "application/zip",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${scaffold.repoName}.zip`;
  a.click();
  URL.revokeObjectURL(url);
}

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
}) {
  const [selected, setSelected] = useState<string[]>(() =>
    props.seedSkills.slice(0, MAX_SKILLS),
  );
  const [modelId, setModelId] = useState(props.defaultModelId);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [scaffold, setScaffold] = useState<Scaffold | null>(null);
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
      void putScaffold(record.id, generated);
    } catch {
      setError("The scaffold did not generate. Try again.");
      setTimeout(() => errorRef.current?.focus(), 0);
    } finally {
      setBusy(false);
    }
  }

  const projects = props.store.projects.projects;
  const [mode, setMode] = useState<"polish" | "scratch">(
    props.seedSkills.length > 0 ? "scratch" : "polish",
  );

  return (
    <div>
      <fieldset className="mb-4">
        <legend className="sr-only">What kind of project help</legend>
        <div className="flex flex-wrap gap-3">
          <label className="flex items-center gap-1">
            <input
              type="radio"
              name="project-mode"
              checked={mode === "polish"}
              onChange={() => setMode("polish")}
            />
            <span className="font-bold">Polish a project I already built</span>
          </label>
          <label className="flex items-center gap-1">
            <input
              type="radio"
              name="project-mode"
              checked={mode === "scratch"}
              onChange={() => setMode("scratch")}
            />
            <span>Start something new</span>
          </label>
        </div>
      </fieldset>

      {mode === "polish" ? (
        <PolishPane
          models={props.models}
          defaultModelId={props.defaultModelId}
          onRecord={(record, stored) => {
            props.store.add(record);
            void putScaffold(record.id, stored);
          }}
        />
      ) : null}

      <section
        aria-labelledby="generate-heading"
        className="rounded-card border border-medium-tan bg-paper p-5"
        hidden={mode !== "scratch"}
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
            <div className="mt-3 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => downloadScaffoldZip(scaffold)}
                className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red"
              >
                Download {scaffold.repoName}.zip
              </button>
            </div>
            <h4 className="mt-4 font-bold">Put it on GitHub</h4>
            <p className="text-dark-tan">
              Unzip, then run these in the project folder (needs the GitHub
              CLI, <code>gh</code>, signed in):
            </p>
            <pre className="mt-2 overflow-x-auto rounded-card bg-light-tan p-3">
              {scaffold.instructions.join("\n")}
            </pre>
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
            Projects you generate collect here, with their zips ready to
            download again. Once you push one to GitHub and add the link,
            its skills count in your profile.
          </p>
        ) : (
          <ul className="mt-3 space-y-3">
            {projects.map((p) => (
              <ProjectCard
                key={p.id}
                project={p}
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

function ProjectCard(props: {
  project: ProjectRecord;
  onSetRepoUrl: (url: string | null) => void;
  onRemove: () => void;
}) {
  const [urlDraft, setUrlDraft] = useState(props.project.repoUrl ?? "");
  const [editingUrl, setEditingUrl] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const p = props.project;

  async function downloadAgain() {
    const stored = await getScaffold<Scaffold | StoredPolish>(p.id);
    if (!stored) {
      setNote(
        "The project files are no longer on this device (storage was cleared). Generate a fresh one.",
      );
      return;
    }
    if ("mode" in stored && stored.mode === "polished") {
      // Re-zip from the stored plan and text originals. Binary originals
      // (PDFs and the like) were never stored on this device, so the
      // student re-adds those; the note says exactly which.
      const entries: Record<string, Uint8Array> = {
        "README.md": strToU8(stored.plan.readme),
        ".gitignore": strToU8(stored.plan.gitignore),
      };
      for (const extra of stored.plan.extraFiles) {
        entries[extra.path] = strToU8(extra.contents);
      }
      for (const file of stored.textFiles) {
        entries[file.path] = strToU8(file.contents);
      }
      const blob = new Blob([zipSync(entries) as BlobPart], {
        type: "application/zip",
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${stored.plan.repoName}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      setNote(
        stored.binaryPaths.length > 0
          ? `Re-add these files yourself; they were not kept on this device: ${stored.binaryPaths.join(", ")}.`
          : null,
      );
      return;
    }
    downloadScaffoldZip(stored as Scaffold);
    setNote(null);
  }

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
        <button
          type="button"
          onClick={() => void downloadAgain()}
          className="rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
        >
          Download the zip again
        </button>
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

/** Zips generated parts plus the student's ORIGINAL files, placed per plan. */
async function downloadPolishZip(
  plan: PolishPlan,
  originals: Map<string, File>,
) {
  const entries: Record<string, Uint8Array> = {
    "README.md": strToU8(plan.readme),
    ".gitignore": strToU8(plan.gitignore),
  };
  for (const extra of plan.extraFiles) {
    entries[extra.path] = strToU8(extra.contents);
  }
  for (const move of plan.layout) {
    const original = originals.get(move.from);
    if (original) {
      entries[move.to] = new Uint8Array(await original.arrayBuffer());
    }
  }
  const blob = new Blob([zipSync(entries) as BlobPart], {
    type: "application/zip",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${plan.repoName}.zip`;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * "Polish a project I already built": upload real coursework files, get an
 * organization plan back. The files are read transiently server-side; the
 * zip is assembled HERE from the student's own originals, so their code
 * ships verbatim (organize + suggest decision, 2026-07-29).
 */
function PolishPane(props: {
  models: ModelOption[];
  defaultModelId: string;
  onRecord: (record: ProjectRecord, stored: StoredPolish) => void;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [hint, setHint] = useState("");
  const [modelId, setModelId] = useState(props.defaultModelId);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [plan, setPlan] = useState<PolishPlan | null>(null);
  const errorRef = useRef<HTMLParagraphElement>(null);

  const fail = (message: string) => {
    setError(message);
    setTimeout(() => errorRef.current?.focus(), 0);
  };

  async function polish() {
    setError(null);
    setBusy(true);
    setPlan(null);
    try {
      const payload = await Promise.all(
        files.map(async (f) =>
          TEXT_EXTENSIONS.test(f.name) && f.size <= 400_000
            ? {
                kind: "text" as const,
                name: f.name,
                content: (await f.text()).slice(0, 30_000),
              }
            : { kind: "binary" as const, name: f.name, sizeBytes: f.size },
        ),
      );
      const res = await fetch("/api/scout/polish", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          modelId,
          projectHint: hint.trim(),
          files: payload,
        }),
      });
      const body = await res.json();
      if (!res.ok) {
        fail(body.error ?? "The organization plan did not generate. Try again.");
        return;
      }
      const nextPlan = body.polish as PolishPlan;
      setPlan(nextPlan);

      const contentByName = new Map(
        payload.flatMap((f) =>
          f.kind === "text" ? [[f.name, f.content] as const] : [],
        ),
      );
      const record: ProjectRecord = {
        id: crypto.randomUUID(),
        repoName: nextPlan.repoName,
        summary: nextPlan.summary,
        skillIds: nextPlan.skillIds,
        createdAt: new Date().toISOString(),
        mode: "polished",
        repoUrl: null,
      };
      props.onRecord(record, {
        mode: "polished",
        plan: nextPlan,
        textFiles: nextPlan.layout.flatMap((m) => {
          const contents = contentByName.get(m.from);
          return contents === undefined ? [] : [{ path: m.to, contents }];
        }),
        binaryPaths: nextPlan.layout
          .filter((m) => !contentByName.has(m.from))
          .map((m) => m.to),
      });
    } catch {
      fail("The organization plan did not generate. Try again.");
    } finally {
      setBusy(false);
    }
  }

  const originals = new Map(files.map((f) => [f.name, f]));

  return (
    <section
      aria-labelledby="polish-heading"
      className="rounded-card border border-medium-tan bg-paper p-5"
    >
      <h2 id="polish-heading" className="text-2xl">
        Polish a project you already built
      </h2>
      <p className="mt-1 text-dark-tan">
        Upload the real files from a course project (code, notebooks, your
        report). You get back a clean repo layout, a grounded README, and a
        list of improvements. Your files are never rewritten and never
        stored on the server; the zip is built right here from your
        originals.
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

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <label className="inline-block cursor-pointer rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-miami-red has-[:focus-visible]:outline-offset-2">
          <input
            type="file"
            multiple
            className="sr-only"
            disabled={busy}
            onChange={(e) => {
              const picked = [...(e.target.files ?? [])];
              const merged = [...files];
              for (const f of picked) {
                if (!merged.some((m) => m.name === f.name)) merged.push(f);
              }
              setFiles(merged.slice(0, MAX_POLISH_FILES));
            }}
          />
          Choose project files
        </label>
        <span aria-live="polite">
          {files.length === 0
            ? "No files chosen yet"
            : `${files.length} ${files.length === 1 ? "file" : "files"} chosen`}
        </span>
      </div>
      {files.length > 0 ? (
        <ul className="mt-2 flex flex-wrap gap-2">
          {files.map((f) => (
            <li key={f.name}>
              <button
                type="button"
                onClick={() => setFiles(files.filter((x) => x.name !== f.name))}
                className="rounded-card border border-medium-tan bg-light-tan px-2 py-1"
                aria-label={`Remove ${f.name}`}
              >
                {f.name} ✕
              </button>
            </li>
          ))}
        </ul>
      ) : null}

      <div className="mt-3 max-w-xl">
        <label htmlFor="polish-hint" className="block font-bold">
          One line about the project (optional)
        </label>
        <input
          id="polish-hint"
          value={hint}
          onChange={(e) => setHint(e.target.value)}
          placeholder="ISA 444 forecasting project on retail demand"
          className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
        />
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
        disabled={busy || files.length === 0}
        onClick={() => void polish()}
        className="mt-4 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
      >
        {busy ? "Organizing your project..." : "Organize my project"}
      </button>
      {busy ? (
        <p role="status" className="mt-2 text-dark-tan">
          Reading your files and drafting the plan. This takes up to a minute.
        </p>
      ) : null}

      {plan ? (
        <div className="mt-5 border-t border-medium-tan pt-4">
          <h3 className="text-xl">{plan.repoName}</h3>
          <p className="mt-1">{plan.summary}</p>
          <p className="mt-1 font-bold">
            Its skills now count in your profile: real work, already built.
          </p>

          <div className="mt-3 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => void downloadPolishZip(plan, originals)}
              className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red"
            >
              Download {plan.repoName}.zip
            </button>
          </div>

          <h4 className="mt-4 font-bold">How your files are organized</h4>
          <ul className="mt-1 list-inside list-disc">
            {plan.layout.map((m) => (
              <li key={m.from}>
                {m.from} <span aria-hidden="true">to</span>
                <span className="sr-only">moves to</span> {m.to}
              </li>
            ))}
            {plan.extraFiles.map((f) => (
              <li key={f.path}>{f.path} (new)</li>
            ))}
          </ul>

          {plan.exclude.length > 0 ? (
            <div className="mt-3 rounded-card border-2 border-miami-red bg-paper p-3">
              <h4 className="font-bold text-miami-red">Left out on purpose</h4>
              <ul className="mt-1 list-inside list-disc">
                {plan.exclude.map((e) => (
                  <li key={e.name}>
                    <strong>{e.name}</strong>: {e.reason}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}

          {plan.suggestions.length > 0 ? (
            <>
              <h4 className="mt-4 font-bold">Suggested improvements</h4>
              <p className="text-dark-tan">
                Listed in the README too. Your files were not changed; these
                are yours to make.
              </p>
              <ul className="mt-1 list-inside list-disc">
                {plan.suggestions.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </>
          ) : null}

          {plan.resumeBullets.length > 0 ? (
            <>
              <h4 className="mt-4 font-bold">Resume bullets this work earns</h4>
              <ul className="mt-1 list-inside list-disc">
                {plan.resumeBullets.map((b) => (
                  <li key={b}>{b}</li>
                ))}
              </ul>
            </>
          ) : null}

          <h4 className="mt-4 font-bold">README preview</h4>
          <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap rounded-card border border-medium-tan p-3">
            {plan.readme}
          </pre>
        </div>
      ) : null}
    </section>
  );
}
