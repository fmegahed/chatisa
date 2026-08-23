"use client";

import { useRef, useState } from "react";
import { guessRole, MAX_PROJECT_FILES, slugify, careerFileSet } from "@/lib/portfolio/files";
import { prepareFile, pushable } from "@/lib/portfolio/intake";
import { UploadLimits } from "@/components/portfolio/UploadLimits";
import { normalizeUrl } from "@/lib/portfolio/links";
import { loadProjects } from "@/lib/scout/profile-store";
import { usePublishedWork } from "@/lib/portfolio/published";
import type { CareerProject, StepProps } from "@/lib/portfolio/draft";
import { SizeMeter } from "../SizeMeter";
import { StepNav } from "../StepNav";

/**
 * Step 3 of the career wizard: one to five projects, each a set of files
 * and an optional link. Work already published from this device (a showcase
 * page, a Job Scout project repo) can be added as a link with one click.
 * Data files arrive unpublished and files too large to push cannot be
 * ticked at all.
 */

const MAX_PROJECTS = 5;
/** The generation route's cap, mirrored so a long title is a message here. */
const MAX_TITLE = 80;

export function ProjectsStep({ draft, patch, nav }: StepProps) {
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const errorRef = useRef<HTMLParagraphElement>(null);
  const published = usePublishedWork().filter((w) => w.kind === "showcase");
  const scoutProjects = loadProjects().projects.filter((p) => p.repoUrl);

  const update = (i: number, p: Partial<CareerProject>) =>
    patch({ projects: draft.projects.map((x, j) => (j === i ? { ...x, ...p } : x)) });
  const add = (p: CareerProject) => {
    if (draft.projects.length < MAX_PROJECTS) patch({ projects: [...draft.projects, p] });
  };
  const uniqueSlug = (base: string) => {
    let s = slugify(base);
    let n = 2;
    // The suffix must keep the slug inside the 60-character bound the route checks.
    while (draft.projects.some((p) => p.slug === s)) s = `${slugify(base).slice(0, 56)}-${n++}`;
    return s;
  };

  /**
   * A file the browser cannot read must not strand the step: the reader is
   * always released, the picker is cleared so the same file name fires change
   * again, and the student is told what happened instead of watching a button
   * sit on "Reading files..." forever.
   */
  async function addFiles(i: number, input: HTMLInputElement) {
    const list = input.files;
    if (!list) return;
    setError(null);
    setBusy(draft.projects[i].slug);
    try {
      const room = MAX_PROJECT_FILES - draft.projects[i].files.length;
      const prepared = await Promise.all(
        Array.from(list).slice(0, room).map((f) => prepareFile(f, guessRole(f.name))),
      );
      update(i, { files: [...draft.projects[i].files, ...prepared] });
    } catch {
      setError("One of those files could not be read. Try adding it again.");
      setTimeout(() => errorRef.current?.focus(), 0);
    } finally {
      setBusy(null);
      input.value = "";
    }
  }

  const valid =
    draft.projects.length >= 1 &&
    draft.projects.every((p) => p.files.some(pushable) || p.externalUrl.trim().length > 0);

  /**
   * A link is repaired before it leaves this step: "github.com/ada/churn" is
   * a link, so the scheme is added rather than the generation failing on it
   * later. Anything that still will not parse stops the step with the field
   * named, instead of turning into a 400 two steps on.
   */
  function onNext() {
    const fixed: CareerProject[] = [];
    for (const [i, p] of draft.projects.entries()) {
      if (p.title.trim().length > MAX_TITLE) {
        return report(`Project ${i + 1}: the title is longer than ${MAX_TITLE} characters.`);
      }
      const raw = p.externalUrl.trim();
      const url = raw ? normalizeUrl(raw) : "";
      if (url === null) {
        return report(`Project ${i + 1}: the link is not a web address. Check it and try again.`);
      }
      fixed.push({ ...p, title: p.title.trim(), externalUrl: url });
    }
    setError(null);
    patch({ projects: fixed });
    nav.onNext();
  }

  function report(message: string) {
    setError(message);
    setTimeout(() => errorRef.current?.focus(), 0);
  }
  const measured = careerFileSet({
    html: "",
    photoBase64: draft.photo?.base64 ?? null,
    resumeBase64: null,
    projects: draft.projects,
  });

  return (
    <section className="rounded-card border border-medium-tan bg-paper p-5">
      <h2 className="text-2xl">Projects (1 to 5)</h2>
      <p className="mt-1 text-dark-tan">
        Add the files from each project (code, notebooks, report, figures) and, if it already
        lives somewhere, its link. Files you add are published in your portfolio repository under
        projects/. Data files are left out unless you tick them. Unticking a file keeps it out of
        the next publish. A file that was already published stays in the repository and its
        history until you delete it on GitHub.
      </p>
      <UploadLimits perProject={MAX_PROJECT_FILES} />
      {error ? (
        <p
          ref={errorRef}
          role="alert"
          tabIndex={-1}
          className="mt-3 rounded-card border-2 border-miami-red p-3 font-bold text-miami-red"
        >
          {error}
        </p>
      ) : null}
      <ul className="mt-4 space-y-4">
        {draft.projects.map((p, i) => (
          <li key={p.slug} className="rounded-card border border-medium-tan p-4">
            <div className="grid gap-3 md:grid-cols-2">
              <label className="block">
                Title (optional)
                <input
                  value={p.title}
                  onChange={(e) => update(i, { title: e.target.value })}
                  className="mt-1 w-full rounded-card border border-medium-tan p-2"
                />
              </label>
              <label className="block">
                Link (repo or demo, optional)
                <input
                  type="url"
                  value={p.externalUrl}
                  onChange={(e) => update(i, { externalUrl: e.target.value })}
                  className="mt-1 w-full rounded-card border border-medium-tan p-2"
                />
              </label>
            </div>
            <label className="mt-3 inline-block cursor-pointer rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan">
              <input
                type="file"
                multiple
                className="sr-only"
                aria-label={`Add files to project ${i + 1}`}
                onChange={(e) => void addFiles(i, e.target)}
                disabled={busy !== null || p.files.length >= MAX_PROJECT_FILES}
              />
              {busy === p.slug ? "Reading files..." : `Add files (${p.files.length}/${MAX_PROJECT_FILES})`}
            </label>
            {p.files.length > 0 ? (
              <ul className="mt-2 space-y-1">
                {p.files.map((f, k) => (
                  <li key={f.name + k} className="flex flex-wrap items-center gap-3">
                    <label className="flex items-center gap-2">
                      <input
                        type="checkbox"
                        checked={f.publish}
                        disabled={!pushable(f)}
                        onChange={(e) =>
                          update(i, {
                            files: p.files.map((x, m) =>
                              m === k ? { ...x, publish: e.target.checked } : x,
                            ),
                          })
                        }
                      />
                      <span>{f.name}</span>
                    </label>
                    <span className="text-dark-tan">
                      {f.role}
                      {!pushable(f) ? ", too large to publish" : ""}
                    </span>
                    <button
                      type="button"
                      className="underline"
                      onClick={() => update(i, { files: p.files.filter((_, m) => m !== k) })}
                    >
                      Remove
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}
            <button
              type="button"
              className="mt-3 underline"
              onClick={() => patch({ projects: draft.projects.filter((_, j) => j !== i) })}
            >
              Remove this project
            </button>
          </li>
        ))}
      </ul>
      {draft.projects.length < MAX_PROJECTS ? (
        <div className="mt-4 flex flex-wrap gap-3">
          <button
            type="button"
            onClick={() =>
              add({
                slug: uniqueSlug(`project-${draft.projects.length + 1}`),
                title: "",
                externalUrl: "",
                files: [],
              })
            }
            className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red"
          >
            Add a project
          </button>
          {published.map((w) => (
            <button
              key={w.id}
              type="button"
              onClick={() =>
                add({
                  slug: uniqueSlug(w.title),
                  title: w.title,
                  externalUrl: w.pagesUrl ?? w.repoUrl,
                  files: [],
                })
              }
              className="rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan"
            >
              Add showcase: {w.title}
            </button>
          ))}
          {scoutProjects.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() =>
                add({
                  slug: uniqueSlug(p.repoName),
                  title: p.repoName,
                  externalUrl: p.repoUrl ?? "",
                  files: [],
                })
              }
              className="rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan"
            >
              Add from Job Scout: {p.repoName}
            </button>
          ))}
        </div>
      ) : null}
      <SizeMeter files={measured} />
      <StepNav {...nav} onNext={onNext} canContinue={valid && busy === null} />
    </section>
  );
}
