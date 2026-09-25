"use client";

import { useEffect, useRef, useState } from "react";
import { useGithubConnection } from "@/lib/scout/use-scout-store";
import { listOwnRepos, readRepo, type ReadError } from "@/lib/scout/github-read";
import type { RepoListing } from "@/lib/scout/github-summary";
import type { RepoSuggestion } from "@/lib/scout/repo-guards";
import { mergeRepoExtras, type ProfileExtra } from "@/lib/scout/profile-store";
import type { CourseSkillLevel } from "@/lib/scout/course-skills";
import { getSkill } from "@/lib/scout/taxonomy";
import { GithubConnect } from "./GithubConnect";
import { ModelChooser } from "@/components/ModelChooser";
import { formatSize } from "@/lib/portfolio/files";
import type { ModelOption } from "@/lib/config/models";
import { REPO_SKILLS_DEFAULT_MODEL } from "@/lib/scout/repo-skills";

/**
 * "Your GitHub (optional)" in My Profile (v6.8.0): suggest skills from up
 * to five of the student's own public repositories. The browser reads
 * GitHub with the student's token; only summaries reach our server. The
 * guards set each card's starting level; the student may choose any level.
 */

export const MAX_REPOS = 5;
const SHOWN = 10;
const LEVEL_HELP: Record<CourseSkillLevel, string> = {
  anchor: "I can show real work with this",
  applied: "I have used it repeatedly",
  exposure: "I know the basics",
};

export function canPick(selected: string[], name: string): boolean {
  return selected.includes(name) || selected.length < MAX_REPOS;
}

type RepoOutcome =
  | { fullName: string; state: "reading" }
  | { fullName: string; state: "suggesting" }
  | { fullName: string; state: "done"; suggestions: RepoSuggestion[]; substantial: boolean; codeRead: boolean; authorship: { studentCommits: number; totalCommits: number }; skippedLarge: { path: string; size: number }[] }
  | { fullName: string; state: "error"; message: string };

function readErrorMessage(fullName: string, e: ReadError): string {
  if (e.kind === "not-found") return `${fullName} could not be read. It may be private, renamed or deleted.`;
  if (e.kind === "rate-limit") {
    const at = e.resetAt ? new Date(e.resetAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : null;
    return `GitHub asked us to slow down. Try again${at ? ` after ${at}` : " in a few minutes"}.`;
  }
  if (e.kind === "network") return `${fullName} could not be reached. Check your connection and try again.`;
  return `${fullName} could not be read (GitHub answered ${e.kind === "github" ? e.status : "an error"}). Try again later.`;
}

const EXPIRED = "Your GitHub connection has expired. Connect again to continue.";

/** Why a repository's skills start at the level they do, in plain words. */
export function ruleText(o: { substantial: boolean; codeRead: boolean; authorship: { studentCommits: number; totalCommits: number } }): string {
  const { studentCommits, totalCommits } = o.authorship;
  const share = totalCommits ? Math.round((studentCommits / totalCommits) * 100) : 0;
  if (!o.codeRead) return "No code or data files were found here, so its skills are suggested as basics only.";
  if (o.substantial) return `You wrote ${share}% of ${totalCommits} commits here, so this repository can support an anchor.`;
  if (totalCommits === 0) return "GitHub shows no commits by you here, so its skills are suggested as applied. Commits made under another email do not count toward you.";
  // Their own repository, just small (review fix: the email caveat sent
  // students chasing a problem they did not have).
  if (share >= 60 && studentCommits < 10) {
    const wrote = studentCommits === totalCommits ? `all ${studentCommits} commits` : `${share}% of ${totalCommits} commits`;
    return `You wrote ${wrote} here. A repository needs at least 10 commits of yours to support an anchor, so its skills are suggested as applied.`;
  }
  return `You wrote ${share}% of the commits here, so its skills are suggested as applied. Commits made under another email do not count toward you.`;
}

/** Replace the outcomes for the repositories just run; keep the others. */
function mergeOutcomes(prev: RepoOutcome[], next: RepoOutcome[]): RepoOutcome[] {
  const byName = new Map(next.map((o) => [o.fullName, o]));
  const merged = prev.map((o) => byName.get(o.fullName) ?? o);
  const seen = new Set(prev.map((o) => o.fullName));
  return [...merged, ...next.filter((o) => !seen.has(o.fullName))];
}

export function GithubSkills(props: { models: ModelOption[]; extras: ProfileExtra[]; onExtras: (next: ProfileExtra[]) => void }) {
  const { connection, clear } = useGithubConnection();
  const [modelId, setModelId] = useState(() =>
    props.models.some((m) => m.id === REPO_SKILLS_DEFAULT_MODEL) ? REPO_SKILLS_DEFAULT_MODEL : props.models[0]?.id ?? "",
  );
  const [repos, setRepos] = useState<RepoListing[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [outcomes, setOutcomes] = useState<RepoOutcome[]>([]);
  /** Files over 6 MB the student chose to read, per repository; they accumulate. */
  const [includes, setIncludes] = useState<Record<string, string[]>>({});
  const [busy, setBusy] = useState(false);
  const [alert, setAlert] = useState<string | null>(null);
  const alertRef = useRef<HTMLParagraphElement>(null);
  const resultsRef = useRef<HTMLHeadingElement>(null);

  /**
   * An expired or revoked token: forget it, so the block offers the real
   * Connect button (review fix: it used to say "connect again" beside
   * "Connected as ..."), and say so once.
   */
  function expire() {
    clear();
    setRepos(null);
    setOutcomes([]);
    setAlert(EXPIRED);
    setTimeout(() => alertRef.current?.focus(), 0);
  }

  useEffect(() => {
    if (!connection) return;
    let live = true;
    void (async () => {
      const out = await listOwnRepos(connection);
      if (!live) return;
      // A new (or renewed) connection starts the block afresh.
      setAlert(null);
      setListError(null);
      setOutcomes([]);
      setSelected([]);
      setIncludes({});
      if (out.ok) setRepos(out.repos);
      else if (out.error.kind === "auth") expire();
      else {
        setRepos(null);
        setListError("Your repositories could not be listed. Try again in a few minutes.");
      }
    })();
    return () => { live = false; };
    // expire only touches state setters and the stable clear from the store.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [connection]);

  /**
   * Suggest skills for `names`. The main button replaces all results; a
   * retry or "Read it anyway" updates only its own repository and keeps the
   * others' unconfirmed cards (review fix).
   */
  async function suggest(names: string[], include: Record<string, string[]>, replace: boolean) {
    if (!connection || !repos || busy) return;
    setBusy(true);
    setAlert(null);
    const show = (list: RepoOutcome[]) => setOutcomes((prev) => (replace ? list : mergeOutcomes(prev, list)));
    const picked = repos.filter((r) => names.includes(r.fullName));
    const initial: RepoOutcome[] = picked.map((r) => ({ fullName: r.fullName, state: "reading" }));
    show(initial);
    const summaries = [];
    const skipped = new Map<string, { path: string; size: number }[]>();
    const next = [...initial];
    for (const [i, r] of picked.entries()) {
      const read = await readRepo(connection, r, fetch, include[r.fullName] ?? []);
      if (!read.ok) {
        if (read.error.kind === "auth") {
          // One message, not one per repository.
          setBusy(false);
          expire();
          return;
        }
        next[i] = { fullName: r.fullName, state: "error", message: readErrorMessage(r.fullName, read.error) };
      } else {
        next[i] = { fullName: r.fullName, state: "suggesting" };
        skipped.set(r.fullName, read.summary.skippedLarge);
        summaries.push(read.summary);
      }
      show([...next]);
    }
    if (summaries.length) {
      try {
        const res = await fetch("/api/scout/repo-skills", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ modelId, repos: summaries }),
        });
        const body = await res.json();
        if (!res.ok) throw new Error(body.error ?? "failed");
        for (const r of body.results as ({ fullName: string; ok: true } & Omit<Extract<RepoOutcome, { state: "done" }>, "fullName" | "state" | "skippedLarge"> | { fullName: string; ok: false; error: string })[]) {
          const i = next.findIndex((o) => o.fullName === r.fullName);
          if (i === -1) continue;
          next[i] = r.ok
            ? { fullName: r.fullName, state: "done", suggestions: r.suggestions, substantial: r.substantial, codeRead: r.codeRead, authorship: r.authorship, skippedLarge: skipped.get(r.fullName) ?? [] }
            : { fullName: r.fullName, state: "error", message: `${r.fullName}: ${r.error}` };
        }
      } catch (err) {
        const message = err instanceof Error && err.message !== "failed" ? err.message : "Skill suggestions did not complete. Try again.";
        for (const [i, o] of next.entries()) if (o.state === "suggesting") next[i] = { fullName: o.fullName, state: "error", message };
      }
      show([...next]);
    }
    setBusy(false);
    setTimeout(() => resultsRef.current?.focus(), 0);
  }

  function readLarge(fullName: string, path: string) {
    const next = { ...includes, [fullName]: [...new Set([...(includes[fullName] ?? []), path])] };
    setIncludes(next);
    void suggest([fullName], next, false);
  }

  function confirm(fullName: string, cards: { skillId: string; level: CourseSkillLevel; suggested: CourseSkillLevel; evidence: string }[]) {
    // Confirming merges this repository's cards with any it already had.
    const existing = props.extras
      .filter((e) => e.source === "github" && e.repo === fullName && !cards.some((c) => c.skillId === e.skillId))
      .map((e) => ({ skillId: e.skillId, level: e.level, suggested: e.setByStudent ? "exposure" as const : e.level, evidence: e.evidence ?? "" }));
    props.onExtras(mergeRepoExtras(props.extras, fullName, [...existing, ...cards]));
    setOutcomes((prev) => prev.map((o) => o.fullName !== fullName || o.state !== "done" ? o
      : { ...o, suggestions: o.suggestions.filter((s) => !cards.some((c) => c.skillId === s.skillId)) }));
    // The confirmed card's button is gone; keep focus in the results
    // (review fix: it dropped to the page).
    setTimeout(() => resultsRef.current?.focus(), 0);
  }

  const q = filter.trim().toLowerCase();
  const visible = (repos ?? []).filter((r) => !q || `${r.fullName} ${r.description ?? ""} ${r.language ?? ""}`.toLowerCase().includes(q));
  const listed = showAll || q ? visible : visible.slice(0, SHOWN);

  return (
    <section aria-labelledby="profile-github" className="mt-8">
      <h2 id="profile-github" className="text-2xl">Your GitHub (optional)</h2>
      <p className="mt-1 max-w-2xl text-dark-tan">
        Suggest skills from your public repositories. Nothing about them is stored on our server, and you confirm every suggestion.
      </p>
      {alert ? (
        <p ref={alertRef} role="alert" tabIndex={-1} className="mt-3 rounded-card border-2 border-miami-red bg-paper p-3 font-bold text-miami-red">
          {alert}
        </p>
      ) : null}
      {!connection ? (
        <div className="mt-3"><GithubConnect returnPath="/job-scout?tab=profile" /></div>
      ) : (
        <>
          <p className="mt-3">Connected as <strong>{connection.login}</strong></p>
          {listError ? <p role="alert" className="mt-2 text-miami-red">{listError}</p> : null}
          {repos === null && !listError ? <p role="status" className="mt-2 text-dark-tan">Loading your repositories...</p> : null}
          {repos && repos.length === 0 ? <p className="mt-2 text-dark-tan">You have no public repositories of your own yet.</p> : null}
          {repos && repos.length > 0 ? (
            <fieldset className="mt-3 min-w-0">
              <legend className="font-bold">Pick up to {MAX_REPOS} repositories</legend>
              {repos.length > SHOWN ? (
                <label className="mt-2 block max-w-md">
                  <span className="block">Filter repositories</span>
                  <input type="search" value={filter} onChange={(e) => setFilter(e.target.value)} className="mt-1 w-full rounded-card border border-medium-tan bg-paper p-2" />
                </label>
              ) : null}
              <ul className="mt-2 space-y-1">
                {listed.map((r) => {
                  const on = selected.includes(r.fullName);
                  return (
                    <li key={r.fullName}>
                      <label className="flex items-start gap-2">
                        <input
                          type="checkbox" className="mt-1 accent-miami-red" checked={on}
                          disabled={!canPick(selected, r.fullName) || busy}
                          onChange={() => setSelected(on ? selected.filter((x) => x !== r.fullName) : [...selected, r.fullName])}
                        />
                        <span className="min-w-0">
                          <strong className="wrap-break-word">{r.fullName}</strong>
                          {r.language ? ` (${r.language})` : ""}
                          {r.description ? <span className="block text-dark-tan">{r.description}</span> : null}
                          <span className="block text-sm text-dark-tan">Updated {r.pushedAt.slice(0, 10)}</span>
                        </span>
                      </label>
                    </li>
                  );
                })}
              </ul>
              {!showAll && !q && visible.length > SHOWN ? (
                <button type="button" className="mt-2 underline" onClick={() => setShowAll(true)}>Show all {visible.length}</button>
              ) : null}
              {selected.length >= MAX_REPOS ? <p className="mt-2 text-dark-tan">Five is the most per round. Untick one to choose another.</p> : null}
            </fieldset>
          ) : null}
          <div className="mt-3 max-w-xl">
            <ModelChooser
              options={props.models} value={modelId} onChange={setModelId} disabled={busy}
              help="Used to suggest skills from your repositories."
            />
          </div>
          <button
            type="button" disabled={busy || selected.length === 0} onClick={() => void suggest(selected, includes, true)}
            className="mt-3 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
          >
            {`Suggest skills from ${selected.length} ${selected.length === 1 ? "repository" : "repositories"}`}
          </button>
          <div aria-live="polite" className="sr-only">
            {outcomes.map((o) => (o.state === "reading" ? `Reading ${o.fullName}. ` : o.state === "suggesting" ? `Suggesting skills for ${o.fullName}. ` : "")).join("")}
          </div>
          {outcomes.length > 0 ? (
            <div className="mt-4">
              <h3 ref={resultsRef} tabIndex={-1} className="text-xl">Suggested skills to confirm</h3>
              {outcomes.map((o) => (
                <RepoGroup
                  key={o.fullName} outcome={o} busy={busy}
                  onRetry={() => void suggest([o.fullName], includes, false)}
                  onReadLarge={(path) => readLarge(o.fullName, path)}
                  onConfirm={(cards) => confirm(o.fullName, cards)}
                />
              ))}
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}

function RepoGroup(props: { outcome: RepoOutcome; busy: boolean; onRetry: () => void; onReadLarge: (path: string) => void; onConfirm: (cards: { skillId: string; level: CourseSkillLevel; suggested: CourseSkillLevel; evidence: string }[]) => void }) {
  const o = props.outcome;
  const [levels, setLevels] = useState<Record<string, CourseSkillLevel>>({});
  if (o.state === "reading" || o.state === "suggesting") {
    return <p className="mt-3 text-dark-tan">{o.state === "reading" ? `Reading ${o.fullName}...` : `Suggesting skills for ${o.fullName}...`}</p>;
  }
  if (o.state === "error") {
    return (
      <p role="alert" className="mt-3 text-miami-red">
        {o.message}{" "}
        <button type="button" className="underline" disabled={props.busy} onClick={props.onRetry}>Try again</button>
      </p>
    );
  }
  const levelOf = (s: RepoSuggestion) => levels[s.skillId] ?? s.suggested;
  return (
    <fieldset className="mt-4 min-w-0 rounded-card border border-medium-tan bg-paper p-3">
      <legend className="px-1 font-bold wrap-break-word">{o.fullName}</legend>
      <p className="text-dark-tan">{ruleText(o)}</p>
      {o.skippedLarge.length > 0 ? (
        <div className="mt-2">
          <p>Not read because they are larger than 6 MB. The model sees at most the first 45,000 characters of a file&apos;s code either way, so this mostly helps notebooks full of plots.</p>
          <ul className="mt-1">
            {o.skippedLarge.map((f) => (
              <li key={f.path} className="flex flex-wrap items-center gap-2">
                <span className="wrap-break-word">{f.path} ({formatSize(f.size)})</span>
                <button
                  type="button" className="underline" disabled={props.busy} onClick={() => props.onReadLarge(f.path)}
                  aria-label={`Read it anyway: ${f.path}`}
                >
                  Read it anyway
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {o.suggestions.length === 0 ? <p className="mt-2">Nothing left to confirm here.</p> : (
        <>
          <button
            type="button" className="mt-2 underline"
            onClick={() => props.onConfirm(o.suggestions.map((s) => ({ skillId: s.skillId, level: s.suggested, suggested: s.suggested, evidence: s.evidence })))}
          >
            Add all as suggested
          </button>
          <ul className="mt-2 grid gap-3 sm:grid-cols-2">
            {o.suggestions.map((s) => (
              <li key={s.skillId} className="rounded-card border border-medium-tan p-3">
                <p className="font-bold">{getSkill(s.skillId)?.label ?? s.skillId}</p>
                {s.evidence ? <p className="mt-1 text-dark-tan">&quot;{s.evidence}&quot;</p> : null}
                <div role="radiogroup" aria-label={`Level for ${getSkill(s.skillId)?.label ?? s.skillId}`} className="mt-2 flex flex-wrap gap-3">
                  {(Object.keys(LEVEL_HELP) as CourseSkillLevel[]).map((l) => (
                    <label key={l} className="flex items-center gap-1">
                      <input type="radio" name={`${o.fullName}-${s.skillId}`} checked={levelOf(s) === l} onChange={() => setLevels({ ...levels, [s.skillId]: l })} />
                      <span>{LEVEL_HELP[l]}{l === s.suggested ? " (suggested)" : ""}</span>
                    </label>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={() => props.onConfirm([{ skillId: s.skillId, level: levelOf(s), suggested: s.suggested, evidence: s.evidence }])}
                  className="mt-2 rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
                >
                  Add to my skills
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </fieldset>
  );
}
