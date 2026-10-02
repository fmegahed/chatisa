"use client";

import { useEffect, useRef, useState } from "react";
import {
  guessRole, MAX_SHOWCASE_FILES, ROLE_LABELS, rolePath, showcaseFileSet, DEFAULT_GITIGNORE,
  type FileRole,
} from "@/lib/portfolio/files";
import { prepareFile, pushable, splitOversize } from "@/lib/portfolio/intake";
import { UploadLimits } from "@/components/portfolio/UploadLimits";
import type { StepProps } from "@/lib/portfolio/draft";
import { SizeMeter } from "../SizeMeter";
import { StepNav } from "../StepNav";
import { GithubImport } from "../GithubImport";
import { githubRepoUrl } from "@/lib/portfolio/github-import";
import { announce } from "@/lib/a11y/announce";

/**
 * Step 2 of the showcase wizard. Each file gets a guessed role, which is
 * both its folder in the repository and how the model is told to read it;
 * the guess is a select the student can correct. Data starts unpublished and
 * files too large to push cannot be published at all.
 */

const ROLES = Object.keys(ROLE_LABELS) as FileRole[];

export function FilesStep({ draft, patch, nav }: StepProps) {
  const [busy, setBusy] = useState(false);
  // The latest draft, for an import that finishes after other edits.
  const draftRef = useRef(draft);
  useEffect(() => {
    draftRef.current = draft;
  });
  const [error, setError] = useState<string | null>(null);
  const errorRef = useRef<HTMLParagraphElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  /**
   * A file the browser cannot read must not strand the step: the reader is
   * always released, the picker is cleared so the same file name fires change
   * again, and the student is told what happened instead of watching a button
   * sit on "Reading files..." forever.
   */
  async function addFiles(input: HTMLInputElement) {
    const list = input.files;
    if (!list) return;
    setError(null);
    setBusy(true);
    try {
      const room = MAX_SHOWCASE_FILES - draft.files.length;
      // Files over 25 MB are not added (one rule for uploads and imports).
      const { accepted, refused } = splitOversize(Array.from(list));
      const prepared = await Promise.all(
        accepted.slice(0, room).map((f) => prepareFile(f, guessRole(f.name))),
      );
      patch({ files: [...draft.files, ...prepared] });
      if (prepared.length > 0) {
        announce(`Added ${prepared.length} ${prepared.length === 1 ? "file" : "files"}.`);
      }
      if (refused) {
        setError(refused);
        setTimeout(() => errorRef.current?.focus(), 0);
      }
    } catch {
      setError("One of those files could not be read. Try adding it again.");
      setTimeout(() => errorRef.current?.focus(), 0);
    } finally {
      setBusy(false);
      input.value = "";
    }
  }

  const set = (i: number, p: Partial<(typeof draft.files)[number]>) =>
    patch({ files: draft.files.map((f, j) => (j === i ? { ...f, ...p } : f)) });
  const measured = showcaseFileSet({
    html: "", readme: "", gitignore: DEFAULT_GITIGNORE, files: draft.files,
  });
  const hasData = draft.files.some((f) => f.role === "data");
  const fileBlocked = busy || draft.files.length >= MAX_SHOWCASE_FILES;

  return (
    <section className="rounded-card border border-medium-tan bg-paper p-5">
      <h2 className="text-2xl">Project files</h2>
      <p className="mt-1 text-dark-tan">
        Add whatever the project has: data, code or notebooks, the written report, slides,
        figures. Not every project has all of these. Each file gets a role that decides its folder
        in the repository; change it if the guess is wrong. Your files are never rewritten. Every
        file you add is read to write the page, published or not. Unticking a file keeps it out of
        the next publish. A file that was already published stays in the repository and its
        history until you delete it on GitHub.
      </p>
      <UploadLimits />
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
      {/* The ring shows focus on the visually hidden input (#31). While
          reading or full, the input stays focusable (aria-disabled) and the
          click is refused, so focus is not dropped mid-upload (#27). */}
      <label className="mt-4 inline-block cursor-pointer rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-miami-red has-[:focus-visible]:outline-offset-2">
        <input
          type="file"
          multiple
          className="sr-only"
          ref={fileInputRef}
          aria-label="Add project files"
          onChange={(e) => void addFiles(e.target)}
          aria-disabled={fileBlocked || undefined}
          onClick={(e) => {
            if (fileBlocked) e.preventDefault();
          }}
        />
        {busy ? "Reading files..." : `Add files (${draft.files.length}/${MAX_SHOWCASE_FILES})`}
      </label>
      <GithubImport
        label="Import from GitHub"
        room={MAX_SHOWCASE_FILES - draft.files.length}
        existingNames={draft.files.map((f) => f.name)}
        disabled={busy}
        onImport={(files, repo) => {
          const latest = draftRef.current.files;
          const kept = [...latest, ...files].slice(0, MAX_SHOWCASE_FILES);
          patch({ files: kept, sourceRepoUrl: repo.url });
          return kept.length - latest.length;
        }}
      />
      {githubRepoUrl(draft.sourceRepoUrl) ? (
        <p className="mt-2">
          Original repository:{" "}
          <a href={githubRepoUrl(draft.sourceRepoUrl)!} className="underline" target="_blank" rel="noreferrer">{githubRepoUrl(draft.sourceRepoUrl)}</a>{" "}
          <button type="button" className="underline" onClick={() => patch({ sourceRepoUrl: undefined })}>Remove link</button>
        </p>
      ) : null}
      {hasData ? (
        <p className="mt-3 rounded-card bg-light-tan p-3">
          Data files start unpublished. Course datasets are often licensed or provided by an
          instructor; tick a data file only if you are sure it can be public.
        </p>
      ) : null}
      {draft.files.length > 0 ? (
        <table className="mt-3 w-full text-left">
          <thead>
            <tr>
              <th className="pr-3">Publish</th>
              <th className="pr-3">File</th>
              <th className="pr-3">Role</th>
              <th className="pr-3">Goes to</th>
              <th>
                <span className="sr-only">Remove</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {draft.files.map((f, i) => (
              <tr key={f.name + i} className="border-t border-medium-tan">
                <td className="py-1 pr-3">
                  <input
                    type="checkbox"
                    aria-label={`Publish ${f.name}`}
                    checked={f.publish}
                    disabled={!pushable(f)}
                    onChange={(e) => set(i, { publish: e.target.checked })}
                  />
                </td>
                <td className="py-1 pr-3">
                  {f.name}
                  {!pushable(f) ? <span className="text-dark-tan">, too large to publish</span> : null}
                </td>
                <td className="py-1 pr-3">
                  <select
                    aria-label={`Role for ${f.name}`}
                    value={f.role}
                    onChange={(e) => set(i, { role: e.target.value as FileRole })}
                    className="rounded-card border border-medium-tan p-1"
                  >
                    {ROLES.map((r) => (
                      <option key={r} value={r}>{ROLE_LABELS[r]}</option>
                    ))}
                  </select>
                </td>
                <td className="py-1 text-dark-tan">{rolePath(f.role, f.name)}</td>
                <td className="py-1">
                  <button
                    type="button"
                    className="underline"
                    aria-label={`Remove ${f.name}`}
                    onClick={() => {
                      patch({ files: draft.files.filter((_, j) => j !== i) });
                      announce(`Removed ${f.name}.`);
                      // The pressed button leaves with its row.
                      fileInputRef.current?.focus();
                    }}
                  >
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      <SizeMeter files={measured} />
      <StepNav
        {...nav}
        canContinue={!busy && draft.files.some((f) => f.publish && pushable(f))}
        requirement={busy ? undefined : "Add at least one file to publish to continue."}
      />
    </section>
  );
}
