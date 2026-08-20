"use client";

import { useEffect, useRef, useState } from "react";
import type { ModelOption } from "@/lib/config/models";
import { ModelChooser } from "@/components/ModelChooser";
import { DeviceResumeOffer } from "@/components/scout/DeviceResumeOffer";
import { GithubConnect } from "@/components/scout/GithubConnect";
import {
  clearPortfolio,
  loadPortfolio,
  savePortfolio,
  type PortfolioRecord,
  type ProjectsState,
  type SavedState,
  type ScoutProfile,
} from "@/lib/scout/profile-store";
import {
  deletePortfolio,
  getPortfolio,
  putPortfolio,
  resumeAsFile,
} from "@/lib/scout/device-files";
import {
  enablePages,
  portfolioFileSet,
  pushToRepo,
  type PushError,
} from "@/lib/scout/github";
import { useGithubConnection } from "@/lib/scout/use-scout-store";
import type { PortfolioContent } from "@/lib/scout/portfolio-html";

/**
 * Portfolio Site tab (v6.3.0). The student picks up to five saved jobs,
 * ChatISA drafts a one-page site optimized across that range, and one click
 * publishes it to GitHub Pages. One site per student by design: retailoring
 * regenerates and republishes the same repo, and the history lives in git.
 */

const MAX_JOBS = 5;
const MAX_LINKS = 4;

interface StoredPortfolio {
  html: string;
  content: PortfolioContent;
  focusNotes: { jobTitle: string; company: string; how: string }[];
}

function pushErrorCopy(error: PushError): string {
  switch (error.kind) {
    case "auth":
      return "GitHub no longer accepts this connection. Connect GitHub again and retry.";
    case "rate-limit":
      return "GitHub is rate limiting your account. Try again in a few minutes.";
    case "name-taken":
      return "A repository called portfolio already exists on your account and was not created by ChatISA. Rename or remove it on GitHub, then publish again.";
    case "too-large":
      return "The site is too large to publish from the browser.";
    case "network":
      return "Could not reach GitHub. Check your connection and try again.";
    default:
      return "GitHub refused the publish. Try again in a minute.";
  }
}

export function PortfolioTab(props: {
  models: ModelOption[];
  defaultModelId: string;
  profile: ScoutProfile;
  saved: SavedState;
  projects: ProjectsState;
  githubEnabled: boolean;
  studentName: string;
  onGoJobs: () => void;
}) {
  const { connection } = useGithubConnection();
  const [selectedJobs, setSelectedJobs] = useState<string[]>([]);
  const [name, setName] = useState(props.studentName);
  const [links, setLinks] = useState<{ label: string; url: string }[]>([]);
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [modelId, setModelId] = useState(props.defaultModelId);
  const [busy, setBusy] = useState(false);
  const [publishBusy, setPublishBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [portfolio, setPortfolio] = useState<StoredPortfolio | null>(null);
  const [record, setRecord] = useState<PortfolioRecord | null>(null);
  const errorRef = useRef<HTMLParagraphElement>(null);

  // Restore a previous generation so republishing works across visits.
  useEffect(() => {
    void (async () => {
      const stored = await getPortfolio<StoredPortfolio>();
      const rec = loadPortfolio();
      if (stored) setPortfolio(stored);
      if (rec) {
        setRecord(rec);
        setSelectedJobs(rec.jobIds);
      }
    })();
  }, []);

  // The profile's stored resume is the default; offer it once.
  useEffect(() => {
    void (async () => {
      if (resumeFile) return;
      const file = await resumeAsFile();
      if (file) setResumeFile(file);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const fail = (message: string) => {
    setError(message);
    setTimeout(() => errorRef.current?.focus(), 0);
  };

  /** Projects that can appear on the site: linked or polished (real work). */
  const eligibleProjects = props.projects.projects.filter(
    (p) => p.repoUrl !== null,
  );

  async function generate() {
    setError(null);
    setNote(null);
    setBusy(true);
    try {
      const jobs = props.saved.saved
        .filter((s) => selectedJobs.includes(s.id))
        .map((s) => ({ id: s.id, title: s.title, company: s.company }));
      const payload = {
        jobs,
        profile: {
          courses: props.profile.courses,
          skills: (props.profile.extras ?? [])
            .slice(0, 25)
            .map((e) => ({ skillId: e.skillId, level: e.level })),
        },
        projects: eligibleProjects.slice(0, 6).map((p) => ({
          repoName: p.repoName,
          summary: p.summary,
          skillIds: p.skillIds,
          repoUrl: p.repoUrl as string,
        })),
        student: {
          name: name.trim() || "Miami University student",
          links: links.filter((l) => l.label.trim() && l.url.trim()).slice(0, MAX_LINKS),
        },
      };
      const form = new FormData();
      form.append("modelId", modelId);
      form.append("payload", JSON.stringify(payload));
      if (resumeFile) form.append("resume", resumeFile);

      const res = await fetch("/api/scout/portfolio", { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) {
        fail(body.error ?? "The portfolio did not generate. Try again.");
        return;
      }
      const generated: StoredPortfolio = {
        html: body.portfolio.html,
        content: body.portfolio.content,
        focusNotes: body.portfolio.focusNotes ?? [],
      };
      setPortfolio(generated);
      void putPortfolio(generated);
      const nextRecord: PortfolioRecord = {
        v: 1,
        repoName: record?.repoName ?? "portfolio",
        repoUrl: record?.repoUrl ?? null,
        pagesUrl: record?.pagesUrl ?? null,
        generatedAt: new Date().toISOString(),
        publishedAt: record?.publishedAt ?? null,
        jobIds: selectedJobs,
      };
      setRecord(savePortfolio(nextRecord));
    } catch {
      fail("The portfolio did not generate. Try again.");
    } finally {
      setBusy(false);
    }
  }

  async function publish() {
    if (!connection || !portfolio) return;
    setPublishBusy(true);
    setError(null);
    setNote(null);
    try {
      const repoName = record?.repoName ?? "portfolio";
      const pushed = await pushToRepo(
        connection,
        repoName,
        portfolioFileSet(portfolio.html),
        {
          message: record?.publishedAt
            ? "Update portfolio site"
            : "Publish portfolio site",
          expectedRepoUrl: record?.repoUrl ?? null,
        },
      );
      if (!pushed.ok) {
        fail(pushErrorCopy(pushed.error));
        return;
      }
      const pages = await enablePages(connection, repoName, pushed.defaultBranch);
      const nextRecord: PortfolioRecord = {
        v: 1,
        repoName,
        repoUrl: pushed.repoUrl,
        pagesUrl: pages.ok ? pages.pagesUrl : (record?.pagesUrl ?? null),
        generatedAt: record?.generatedAt ?? new Date().toISOString(),
        publishedAt: new Date().toISOString(),
        jobIds: record?.jobIds ?? selectedJobs,
      };
      setRecord(savePortfolio(nextRecord));
      setNote(
        pages.ok
          ? `Your site is live at ${pages.pagesUrl}. GitHub takes a few minutes to build it, so the link can show a 404 at first.`
          : "The site was pushed. One more step on GitHub: open your repo settings and turn on Pages for the main branch.",
      );
      if (!pages.ok) {
        window.open(pages.settingsUrl, "_blank", "noopener");
      }
    } finally {
      setPublishBusy(false);
    }
  }

  return (
    <div>
      <section className="rounded-card border border-medium-tan bg-paper p-5">
        <h2 className="text-2xl">Build your portfolio site</h2>
        <p className="mt-1 text-dark-tan">
          A one-page site with your story, skills, and best projects,
          published free on GitHub Pages. Pick up to {MAX_JOBS} saved jobs
          you care about and the page is written to speak to that whole
          range, not just one posting.
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

        <h3 className="mt-4 font-bold">Jobs to speak to</h3>
        {props.saved.saved.length === 0 ? (
          <p className="mt-1 rounded-card bg-light-tan p-3">
            You have no saved jobs yet. Save a few from the weekly board
            first, so the site can be aimed at them.{" "}
            <button type="button" onClick={props.onGoJobs} className="font-bold underline">
              See this week&apos;s jobs
            </button>
          </p>
        ) : (
          <ul className="mt-2 space-y-1">
            {props.saved.saved.map((s) => {
              const checked = selectedJobs.includes(s.id);
              return (
                <li key={s.id}>
                  <label className="flex items-baseline gap-2">
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={!checked && selectedJobs.length >= MAX_JOBS}
                      onChange={() =>
                        setSelectedJobs(
                          checked
                            ? selectedJobs.filter((id) => id !== s.id)
                            : [...selectedJobs, s.id],
                        )
                      }
                    />
                    <span>
                      {s.title}
                      {s.company ? ` at ${s.company}` : ""}
                    </span>
                  </label>
                </li>
              );
            })}
          </ul>
        )}

        <h3 className="mt-4 font-bold">About you</h3>
        <div className="mt-1 max-w-xl">
          <label htmlFor="pf-name" className="block">
            Name shown on the site
          </label>
          <input
            id="pf-name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
          />
        </div>
        <DeviceResumeOffer
          currentFile={resumeFile}
          disabled={busy}
          onUse={setResumeFile}
        />
        <div className="mt-2 flex flex-wrap items-center gap-3">
          <label className="inline-block cursor-pointer rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan has-[:focus-visible]:outline-3 has-[:focus-visible]:outline-miami-red has-[:focus-visible]:outline-offset-2">
            <input
              type="file"
              accept=".pdf"
              className="sr-only"
              disabled={busy}
              onChange={(e) => setResumeFile(e.target.files?.[0] ?? null)}
            />
            {resumeFile ? "Choose a different resume" : "Add your resume (PDF)"}
          </label>
          <span aria-live="polite">
            {resumeFile ? `Using ${resumeFile.name}` : "No resume chosen (optional but recommended)"}
          </span>
        </div>

        <h3 className="mt-4 font-bold">Links (optional)</h3>
        <ul className="mt-1 space-y-2">
          {links.map((l, i) => (
            <li key={i} className="flex flex-wrap items-center gap-2">
              <label className="sr-only" htmlFor={`pf-link-label-${i}`}>
                Link label
              </label>
              <input
                id={`pf-link-label-${i}`}
                value={l.label}
                placeholder="LinkedIn"
                onChange={(e) =>
                  setLinks(links.map((x, j) => (j === i ? { ...x, label: e.target.value } : x)))
                }
                className="rounded-card border border-medium-tan bg-paper px-2 py-1"
              />
              <label className="sr-only" htmlFor={`pf-link-url-${i}`}>
                Link URL
              </label>
              <input
                id={`pf-link-url-${i}`}
                type="url"
                value={l.url}
                placeholder="https://www.linkedin.com/in/you"
                onChange={(e) =>
                  setLinks(links.map((x, j) => (j === i ? { ...x, url: e.target.value } : x)))
                }
                className="w-72 max-w-full rounded-card border border-medium-tan bg-paper px-2 py-1"
              />
              <button
                type="button"
                onClick={() => setLinks(links.filter((_, j) => j !== i))}
                className="underline"
                aria-label={`Remove link ${l.label || i + 1}`}
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
        {links.length < MAX_LINKS ? (
          <button
            type="button"
            onClick={() => setLinks([...links, { label: "", url: "" }])}
            className="mt-2 rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
          >
            Add a link
          </button>
        ) : null}

        {eligibleProjects.length === 0 ? (
          <p className="mt-4 rounded-card bg-light-tan p-3">
            No projects with a repo link yet. The site is stronger with 2 or
            3 built projects from the My Projects tab; you can still generate
            without them.
          </p>
        ) : null}

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
          disabled={busy || selectedJobs.length === 0}
          onClick={() => void generate()}
          className="mt-4 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
        >
          {busy
            ? "Writing your site..."
            : portfolio
              ? "Regenerate the site"
              : "Generate my site"}
        </button>
        {busy ? (
          <p role="status" className="mt-2 text-dark-tan">
            Reading your material and writing the page. This takes up to a
            minute.
          </p>
        ) : null}
        {selectedJobs.length === 0 && props.saved.saved.length > 0 ? (
          <p className="mt-2 text-sm text-dark-tan">
            Pick at least one saved job above to aim the site at.
          </p>
        ) : null}
      </section>

      {portfolio ? (
        <section className="mt-6 rounded-card border border-medium-tan bg-paper p-5">
          <h2 className="text-2xl">Preview</h2>
          <iframe
            sandbox=""
            srcDoc={portfolio.html}
            title="Portfolio site preview"
            className="mt-3 h-[32rem] w-full rounded-card border border-medium-tan bg-white"
          />
          {portfolio.focusNotes.length > 0 ? (
            <>
              <h3 className="mt-4 font-bold">
                How this speaks to the jobs you picked
              </h3>
              <p className="text-dark-tan">
                Private notes for you; they are not part of the published site.
              </p>
              <ul className="mt-1 list-inside list-disc">
                {portfolio.focusNotes.map((n) => (
                  <li key={`${n.jobTitle}-${n.company}`}>
                    <strong>
                      {n.jobTitle}
                      {n.company ? ` at ${n.company}` : ""}
                    </strong>
                    : {n.how}
                  </li>
                ))}
              </ul>
            </>
          ) : null}

          <h3 className="mt-4 font-bold">Publish</h3>
          {note ? (
            <p role="status" className="mt-1 rounded-card bg-light-tan p-3">
              {note}
            </p>
          ) : null}
          {record?.pagesUrl && !note ? (
            <p className="mt-1">
              Published at{" "}
              <a
                href={record.pagesUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="underline"
              >
                {record.pagesUrl}
              </a>
            </p>
          ) : null}
          {props.githubEnabled ? (
            <div className="mt-2 flex flex-wrap items-center gap-3">
              <GithubConnect returnPath="/job-scout?tab=portfolio" />
              {connection ? (
                <button
                  type="button"
                  disabled={publishBusy}
                  onClick={() => void publish()}
                  className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
                >
                  {publishBusy
                    ? "Publishing..."
                    : record?.publishedAt
                      ? "Republish the site"
                      : "Publish to GitHub Pages"}
                </button>
              ) : null}
            </div>
          ) : (
            <p className="mt-1 text-dark-tan">
              GitHub publishing is not configured on this server. Download
              the page instead: save the preview as index.html in a public
              repository and turn on GitHub Pages in that repo&apos;s
              settings.
            </p>
          )}
          <p className="mt-3">
            <button
              type="button"
              className="underline"
              onClick={() => {
                setPortfolio(null);
                setRecord(null);
                setNote(null);
                clearPortfolio();
                void deletePortfolio();
              }}
            >
              Remove this draft from the device
            </button>{" "}
            <span className="text-dark-tan">
              (an already published site stays on GitHub)
            </span>
          </p>
        </section>
      ) : null}
    </div>
  );
}
