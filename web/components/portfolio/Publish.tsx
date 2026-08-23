"use client";

import { useEffect, useRef, useState } from "react";
import { GithubConnect } from "@/components/scout/GithubConnect";
import { useGithubConnection } from "@/lib/scout/use-scout-store";
import { enablePages, pushToRepo, type PushError, type PushProgress } from "@/lib/scout/github";
import { SLUG } from "@/lib/portfolio/content";
import { fileToBase64 } from "@/lib/portfolio/intake";
import { buildPublishPlan } from "@/lib/portfolio/publish-plan";
import { putDraft, upsertSite, type SiteRecord } from "@/lib/portfolio/store";
import { clearWip } from "@/lib/portfolio/wip";
import { upsertPublished } from "@/lib/portfolio/published";
import { measure, showcaseRepoName } from "@/lib/portfolio/files";
import { resolveSkillId } from "@/lib/scout/taxonomy";
import type { Draft } from "@/lib/portfolio/draft";
import type { CareerContent } from "@/lib/portfolio/content";

/**
 * Publishing (2026-08-20). The push runs in the browser with the student's
 * own GitHub token, so the site never passes through the server; the server
 * only learns that a publish happened, as a count, through the event beacon.
 */

function copy(error: PushError): string {
  switch (error.kind) {
    case "auth": return "GitHub no longer accepts this connection. Connect GitHub again and retry.";
    case "rate-limit":
      return error.resetAt
        ? `GitHub is rate limiting your account. Try again after ${new Date(error.resetAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}.`
        : "GitHub is rate limiting your account. Try again in a few minutes.";
    case "cancelled": return "Publishing was cancelled. Nothing changed on GitHub.";
    case "name-taken": return `A repository with that name already exists on your account and was not created by ChatISA.${error.suggestion ? ` Try the name ${error.suggestion}.` : " Pick another name."}`;
    case "too-large": return "The site is too large to publish from the browser. Untick some files and try again.";
    case "network": return "Could not reach GitHub. Check your connection and try again.";
    default: return "GitHub refused the publish. Try again in a minute.";
  }
}

/**
 * The career generation route returns content, not skill ids, so a career
 * site would otherwise land in the published store with an empty skill list
 * and match nothing. The skills the model wrote are already the student's
 * own words for the taxonomy, so resolve them here.
 */
function careerSkillIds(content: CareerContent): string[] {
  const raw = [
    ...content.skillGroups.flatMap((g) => g.skills),
    ...content.projects.flatMap((p) => p.skills),
  ];
  const ids = raw.map(resolveSkillId).filter((id): id is string => id !== null);
  return Array.from(new Set(ids)).slice(0, 25);
}

export function Publish(props: {
  draft: Draft;
  githubEnabled: boolean;
  site: SiteRecord | null;
  onPublished: (site: SiteRecord) => void;
}) {
  const { connection } = useGithubConnection();
  const isCareer = props.draft.content?.kind === "career";
  const [repoName, setRepoName] = useState(
    props.site?.repoName ??
      (isCareer
        ? "portfolio"
        : showcaseRepoName(
            props.draft.course,
            props.draft.content?.kind === "showcase" ? props.draft.content.content.title : "project",
          )),
  );
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<PushProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  // A publish in flight is abandoned (not left running blind) if the student
  // navigates away; Cancel does the same on purpose.
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => () => abortRef.current?.abort(), []);
  const [note, setNote] = useState<{ text: string; link: string | null } | null>(null);
  const nameOk = SLUG.test(repoName);

  async function publish() {
    if (!connection || !props.draft.content) return;
    setBusy(true); setError(null); setNote(null); setProgress(null);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      const resumeBase64 =
        props.draft.resumeLink && props.draft.resume ? await fileToBase64(props.draft.resume) : null;
      // Before the first publish the name in the field is the name to use,
      // in either mode; after it, the repository the site already lives in.
      const plan = buildPublishPlan(props.draft, connection.login, {
        resumeBase64,
        existingRepoName: props.site?.repoUrl ? props.site.repoName : repoName,
      });
      const m = measure(plan.files);
      if (!m.ok) {
        setError("The site is over the repository limits. Go back and untick some files.");
        return;
      }
      const pushed = await pushToRepo(connection, plan.repoName, plan.files, {
        message: props.site?.publishedAt
          ? "Update site from ChatISA Portfolio Builder"
          : "Publish site from ChatISA Portfolio Builder",
        expectedRepoUrl: props.site?.repoUrl ?? null,
        onProgress: setProgress,
        signal: controller.signal,
      });
      if (!pushed.ok) {
        // A free name from GitHub is one click away from working: put it in
        // the field so the student presses the button again rather than
        // guessing a name themselves.
        if (pushed.error.kind === "name-taken" && pushed.error.suggestion) {
          setRepoName(pushed.error.suggestion);
        }
        setError(copy(pushed.error));
        return;
      }
      const pages = await enablePages(connection, plan.repoName, pushed.defaultBranch);
      const title =
        props.draft.content.kind === "career"
          ? props.draft.content.content.siteTitle || props.draft.name
          : props.draft.content.content.title;
      const publishedAt = new Date().toISOString();
      const skillIds = props.draft.skillIds.length > 0
        ? props.draft.skillIds
        : props.draft.content.kind === "career"
          ? careerSkillIds(props.draft.content.content)
          : [];
      const record: SiteRecord = {
        v: 1, id: props.draft.siteId, kind: props.draft.content.kind, title, repoName: plan.repoName,
        repoUrl: pushed.repoUrl, pagesUrl: pages.ok ? pages.pagesUrl : (props.site?.pagesUrl ?? null),
        generatedAt: props.site?.generatedAt ?? publishedAt, publishedAt,
      };
      upsertSite(record);
      // The published draft lives under the site now; the in-progress copy would
      // only nag on the next visit.
      void clearWip();
      upsertPublished({
        id: record.id, kind: record.kind, title,
        summary: props.draft.content.kind === "career"
          ? props.draft.content.content.headline
          : props.draft.content.content.tagline,
        skillIds, repoUrl: pushed.repoUrl, pagesUrl: record.pagesUrl,
        publishedAt,
      });
      void putDraft(record.id, {
        v: 1, content: props.draft.content, html: plan.html,
        student: isCareer
          ? { name: props.draft.name, links: props.draft.links, courses: props.draft.courses }
          : null,
        showcaseMeta: isCareer
          ? null
          : { course: props.draft.course, semester: props.draft.semester, team: props.draft.team },
        files: [
          ...props.draft.files.map((f) => ({ ...f, projectSlug: null })),
          ...props.draft.projects.flatMap((p) => p.files.map((f) => ({ ...f, projectSlug: p.slug }))),
        ],
        photoBase64: props.draft.photo?.base64 ?? null,
        resumeBase64: null,
        resumeLink: props.draft.resumeLink,
        readme: props.draft.readme,
        skillIds,
      });
      void fetch("/api/portfolio/event", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ kind: record.kind }),
      });
      setNote(pages.ok
        ? {
            text: `Your site is live at ${pages.pagesUrl}. GitHub takes a few minutes to build it, so the link can show a 404 at first.`,
            link: null,
          }
        : {
            // A link the student clicks, not a popup: a window.open here is
            // one user gesture removed from the click and browsers block it.
            text: "The files were pushed. One more step on GitHub: open your repository's Pages settings and turn Pages on for the main branch.",
            link: pages.settingsUrl,
          });
      props.onPublished(record);
    } catch {
      setError("Something went wrong while publishing. Try again in a minute.");
    } finally {
      abortRef.current = null;
      setBusy(false);
      setProgress(null);
    }
  }

  const progressText = (p: PushProgress | null): string => {
    if (!p) return "Publishing...";
    if (p.stage === "upload") return `Uploading file ${p.done} of ${p.total}...`;
    if (p.stage === "commit") return "Saving the commit...";
    return "Preparing the repository...";
  };

  if (!props.githubEnabled) {
    return <p className="mt-3 rounded-card bg-light-tan p-3">Publishing to GitHub is not configured on this server.</p>;
  }
  return (
    <div className="mt-3">
      {error ? <p role="alert" className="rounded-card border-2 border-miami-red p-3 font-bold text-miami-red">{error}</p> : null}
      {note ? (
        <p role="status" className="rounded-card bg-light-tan p-3">
          {note.text}
          {note.link ? (
            <>
              {" "}
              <a href={note.link} target="_blank" rel="noopener noreferrer" className="underline">
                Open the Pages settings for your repository
              </a>
            </>
          ) : null}
        </p>
      ) : null}
      {!props.site?.repoUrl ? (
        <label className="mt-2 block font-bold">
          Repository name
          <input
            value={repoName}
            onChange={(e) => setRepoName(e.target.value.toLowerCase())}
            className="mt-1 w-full rounded-card border border-medium-tan p-2 font-normal"
            aria-invalid={!nameOk}
          />
          {!nameOk ? (
            <span className="block font-normal text-miami-red">Lowercase letters, digits, and hyphens, 3 to 60 characters.</span>
          ) : null}
          {isCareer ? (
            <span className="block font-normal text-dark-tan">
              Your site is published to this repository. Change it if your account already uses
              the name for something else.
            </span>
          ) : null}
        </label>
      ) : null}
      <div className="mt-3 flex flex-wrap items-center gap-3">
        <GithubConnect returnPath="/portfolio" />
        {connection ? (
          <button
            type="button"
            disabled={busy || !nameOk}
            onClick={() => void publish()}
            className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
          >
            {busy ? progressText(progress) : props.site?.publishedAt ? "Publish the update" : "Publish to GitHub Pages"}
          </button>
        ) : null}
        {busy ? (
          <button type="button" onClick={() => abortRef.current?.abort()} className="rounded-card px-3 py-2 underline">
            Cancel
          </button>
        ) : null}
      </div>
      {busy ? (
        <p role="status" aria-live="polite" className="mt-2 text-dark-tan">
          {progressText(progress)} Keep this tab open.
        </p>
      ) : null}
      {props.site?.pagesUrl && !note ? (
        <p className="mt-2">
          Published at{" "}
          <a href={props.site.pagesUrl} target="_blank" rel="noopener noreferrer" className="underline">
            {props.site.pagesUrl}
          </a>
        </p>
      ) : null}
    </div>
  );
}
