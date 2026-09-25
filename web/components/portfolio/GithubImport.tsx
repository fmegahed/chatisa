"use client";

import { useEffect, useId, useRef, useState } from "react";
import { useGithubConnection } from "@/lib/scout/use-scout-store";
import { fetchRepoFile, listOwnRepos, readRepoTree, type ReadError } from "@/lib/scout/github-read";
import type { RepoListing } from "@/lib/scout/github-summary";
import { githubRepoUrl, importCandidates, importedName, type ImportCandidate } from "@/lib/portfolio/github-import";
import { formatSize, guessRole } from "@/lib/portfolio/files";
import type { PreparedFile } from "@/lib/portfolio/files";
import { prepareFile } from "@/lib/portfolio/intake";
import { GithubConnect } from "@/components/scout/GithubConnect";

/**
 * Import from GitHub (v6.9.0): pick one of your public repositories, tick
 * its files, and they arrive as if uploaded. The repository is only read;
 * the page links back to it. An inline disclosure, not a modal.
 */

const EXPIRED = "Your GitHub connection has expired. Connect again to continue.";
const SHOWN = 10;

function problem(e: ReadError, what: string): string {
  if (e.kind === "not-found") return `${what} could not be read. It may be private, renamed or deleted.`;
  if (e.kind === "rate-limit") return "GitHub asked us to slow down. Try again in a few minutes.";
  if (e.kind === "network") return `${what} could not be reached. Check your connection and try again.`;
  return `${what} could not be read. Try again later.`;
}

const size = formatSize;

export function GithubImport(props: {
  label: string;
  room: number;
  disabled?: boolean;
  /** Names already in the project, so an import never duplicates one. */
  existingNames: string[];
  /** Applies the files to the latest draft; returns how many were added. */
  onImport: (files: PreparedFile[], repo: { fullName: string; url: string }) => number;
}) {
  const uid = useId();
  const { connection, clear } = useGithubConnection();
  const [open, setOpen] = useState(false);
  const [repos, setRepos] = useState<RepoListing[] | null>(null);
  const [filter, setFilter] = useState("");
  const [repo, setRepo] = useState<RepoListing | null>(null);
  const [candidates, setCandidates] = useState<ImportCandidate[] | null>(null);
  const [ticked, setTicked] = useState<string[]>([]);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  /** A file list is loading: the repository choice is locked (review fix). */
  const [listing, setListing] = useState(false);
  const listSeq = useRef(0);
  const [error, setError] = useState<string | null>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const errorRef = useRef<HTMLParagraphElement>(null);

  const fail = (message: string) => {
    setError(message);
    setTimeout(() => errorRef.current?.focus(), 0);
  };
  const expire = () => {
    clear();
    setRepos(null);
    setRepo(null);
    setCandidates(null);
    fail(EXPIRED);
  };

  useEffect(() => {
    if (!open || !connection) return;
    let live = true;
    void (async () => {
      const out = await listOwnRepos(connection);
      if (!live) return;
      setError(null);
      if (out.ok) setRepos(out.repos);
      else if (out.error.kind === "auth") expire();
      else fail("Your repositories could not be listed. Try again in a few minutes.");
    })();
    return () => { live = false; };
    // expire and fail only use state setters and the store's stable clear.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, connection]);

  async function choose(r: RepoListing) {
    if (!connection) return;
    setRepo(r);
    setCandidates(null);
    setError(null);
    setStatus(`Listing the files in ${r.fullName}.`);
    // Only the latest choice may fill the list, whatever order GitHub answers in.
    const seq = ++listSeq.current;
    setListing(true);
    const out = await readRepoTree(connection, r);
    if (seq !== listSeq.current) return;
    setListing(false);
    if (!out.ok) {
      if (out.error.kind === "auth") return expire();
      return fail(problem(out.error, r.fullName));
    }
    const list = importCandidates(out.tree, props.room);
    setCandidates(list);
    setTicked(list.filter((c) => c.preselected).map((c) => c.path));
    setStatus(`${list.length} files listed${out.truncated ? " (GitHub shortened the list for this large repository)" : ""}.`);
  }

  function close() {
    setOpen(false);
    setTimeout(() => toggleRef.current?.focus(), 0);
  }

  async function doImport() {
    if (!connection || !repo || ticked.length === 0 || ticked.length > room) return;
    setBusy(true);
    setError(null);
    const files: PreparedFile[] = [];
    const failed: string[] = [];
    for (const [i, path] of ticked.entries()) {
      setStatus(`Reading ${i + 1} of ${ticked.length}: ${path}`);
      const out = await fetchRepoFile(connection, repo.fullName, path);
      if (!out.ok) {
        if (out.error.kind === "auth") { setBusy(false); return expire(); }
        failed.push(path);
        continue;
      }
      const name = importedName(path, ticked, [...props.existingNames, ...files.map((f) => f.name)]);
      try {
        files.push(await prepareFile(new File([out.bytes], name), guessRole(name)));
      } catch {
        failed.push(path);
      }
    }
    setBusy(false);
    const url = githubRepoUrl(repo.htmlUrl) ?? `https://github.com/${repo.fullName}`;
    const added = files.length ? props.onImport(files, { fullName: repo.fullName, url }) : 0;
    // What arrived is no longer ticked, so Import never brings it twice; a
    // failed file stays ticked for a retry (review fix).
    setTicked(failed);
    const dropped = files.length - added;
    const tail = dropped > 0 ? ` ${dropped} did not fit in the project.` : "";
    if (failed.length) {
      fail(`${failed.join(", ")} could not be read. ${added} other ${added === 1 ? "file was" : "files were"} imported.${tail}`);
      return;
    }
    setStatus(`Imported ${added} ${added === 1 ? "file" : "files"} from ${repo.fullName}.${tail}`);
    close();
  }

  const q = filter.trim().toLowerCase();
  const visible = (repos ?? []).filter((r) => !q || r.fullName.toLowerCase().includes(q));
  const panelId = `${uid}-panel`;
  const room = Math.max(0, props.room);

  return (
    <div className="mt-3">
      <button
        ref={toggleRef} type="button" aria-expanded={open} aria-controls={panelId}
        disabled={props.disabled}
        onClick={() => (open ? close() : setOpen(true))}
        className="rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan disabled:border-medium-gray disabled:text-medium-gray"
      >
        {props.label}
      </button>
      <div id={panelId} hidden={!open} className="mt-2 min-w-0 rounded-card border border-medium-tan bg-paper p-3">
        {error ? (
          <p ref={errorRef} role="alert" tabIndex={-1} className="mb-2 rounded-card border-2 border-miami-red p-2 font-bold text-miami-red">{error}</p>
        ) : null}
        {!connection ? (
          <GithubConnect returnPath="/portfolio" />
        ) : (
          <>
            <p className="text-dark-tan">Your repository is only read, never changed. The page links back to it.</p>
            {repos === null && !error ? <p role="status" className="mt-2">Loading your repositories...</p> : null}
            {repos && repos.length === 0 ? <p className="mt-2">You have no public repositories of your own yet.</p> : null}
            {repos && repos.length > SHOWN ? (
              <label className="mt-2 block max-w-md">
                <span className="block">Filter repositories</span>
                <input type="search" value={filter} onChange={(e) => setFilter(e.target.value)} className="mt-1 w-full rounded-card border border-medium-tan p-2" />
              </label>
            ) : null}
            {repos && repos.length > 0 ? (
              <fieldset className="mt-2 min-w-0">
                <legend className="font-bold">Repository</legend>
                <ul className="mt-1 space-y-1">
                  {(q ? visible : visible.slice(0, 50)).map((r) => (
                    <li key={r.fullName}>
                      <label className="flex items-start gap-2">
                        <input type="radio" name={`${uid}-repo`} className="mt-1 accent-miami-red" checked={repo?.fullName === r.fullName} disabled={busy || listing} onChange={() => void choose(r)} />
                        <span className="min-w-0 wrap-break-word"><strong>{r.fullName}</strong>{r.language ? ` (${r.language})` : ""}</span>
                      </label>
                    </li>
                  ))}
                </ul>
                {!q && visible.length > 50 ? <p className="mt-1 text-dark-tan">Showing 50 of {visible.length}. Filter to find the others.</p> : null}
              </fieldset>
            ) : null}
            {candidates ? (
              <fieldset className="mt-3 min-w-0">
                <legend className="font-bold">Files to import ({ticked.length} of up to {room})</legend>
                {candidates.length === 0 ? <p>This repository has no files yet.</p> : (
                  <ul className="mt-1 max-h-80 space-y-1 overflow-auto">
                    {candidates.map((c) => {
                      const on = ticked.includes(c.path);
                      return (
                        <li key={c.path}>
                          <label className="flex items-start gap-2">
                            <input
                              type="checkbox" className="mt-1 accent-miami-red" checked={on}
                              disabled={busy || !c.selectable || (!on && ticked.length >= room)}
                              onChange={() => setTicked(on ? ticked.filter((p) => p !== c.path) : [...ticked, c.path])}
                            />
                            <span className="min-w-0 wrap-break-word">
                              {c.path} <span className="text-dark-tan">({size(c.size)}, {c.role})</span>
                              {c.note ? <span className="block text-dark-tan">{c.note}</span> : null}
                            </span>
                          </label>
                        </li>
                      );
                    })}
                  </ul>
                )}
                {room === 0 ? <p className="mt-2">This project already has as many files as it can hold. Remove some to import more.</p> : null}
                {room > 0 && ticked.length > room ? (
                  <p className="mt-2">Untick {ticked.length - room}: there is room for {room} more {room === 1 ? "file" : "files"}.</p>
                ) : null}
                <button
                  type="button" disabled={busy || ticked.length === 0 || ticked.length > room} onClick={() => void doImport()}
                  className="mt-3 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
                >
                  {busy ? "Importing..." : `Import ${ticked.length} ${ticked.length === 1 ? "file" : "files"}`}
                </button>
              </fieldset>
            ) : null}
          </>
        )}
      </div>
      {/* Outside the panel: a confirmation must still be read after it closes. */}
      <p aria-live="polite" className="sr-only">{status}</p>
    </div>
  );
}
