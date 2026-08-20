"use client";

import { useState } from "react";
import type { PushError, PushFile } from "@/lib/scout/github";
import { pushToRepo } from "@/lib/scout/github";
import { useGithubConnection } from "@/lib/scout/use-scout-store";

/**
 * One-click push of a project's files to the student's GitHub (v6.3.0).
 * Renders nothing without a connection: the caller pairs it with
 * GithubConnect. On a name collision it offers the first free suffixed
 * name rather than ever overwriting a repo this app did not create.
 */

function errorCopy(error: PushError): string {
  switch (error.kind) {
    case "auth":
      return "GitHub no longer accepts this connection. Connect GitHub again and retry.";
    case "rate-limit":
      return error.resetAt
        ? `GitHub is rate limiting your account. Try again after ${new Date(error.resetAt).toLocaleTimeString()}.`
        : "GitHub is rate limiting your account. Try again in a few minutes.";
    case "too-large":
      return "This project is too large to push from the browser. Download the zip and push it yourself.";
    case "network":
      return "Could not reach GitHub. Check your connection and try again.";
    case "name-taken":
      return "That repository name is already taken on your account.";
    default:
      return "GitHub refused the push. Try again in a minute.";
  }
}

export function PushToGithubButton(props: {
  repoName: string;
  /** Loads the files at click time (they may live in IndexedDB). Null when
   * the device no longer has them; the button explains instead of pushing. */
  getFiles: () => Promise<PushFile[] | null>;
  /** The repo URL a previous push recorded, so updates go to OUR repo only. */
  expectedRepoUrl: string | null;
  commitMessage: string;
  onPushed: (repoUrl: string, defaultBranch: string) => void;
  label?: string;
}) {
  const { connection, clear } = useGithubConnection();
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [suggestion, setSuggestion] = useState<string | null>(null);

  if (!connection) return null;

  async function push(repoName: string) {
    if (!connection) return;
    setBusy(true);
    setNote(null);
    setSuggestion(null);
    try {
      const files = await props.getFiles();
      if (!files) {
        setNote(
          "The project files are no longer on this device (storage was cleared). Generate a fresh one, then push.",
        );
        return;
      }
      const result = await pushToRepo(connection, repoName, files, {
        message: props.commitMessage,
        expectedRepoUrl: props.expectedRepoUrl,
      });
      if (result.ok) {
        setNote(`Pushed to ${repoName}.`);
        props.onPushed(result.repoUrl, result.defaultBranch);
        return;
      }
      if (result.error.kind === "auth") clear();
      if (result.error.kind === "name-taken" && result.error.suggestion) {
        setSuggestion(result.error.suggestion);
      }
      setNote(errorCopy(result.error));
    } finally {
      setBusy(false);
    }
  }

  return (
    <span className="inline-flex flex-wrap items-center gap-2">
      <button
        type="button"
        disabled={busy}
        onClick={() => void push(props.repoName)}
        className="rounded-card bg-miami-red px-3 py-1 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
      >
        {busy ? "Pushing..." : (props.label ?? "Push to GitHub")}
      </button>
      {note ? (
        <span role="status" className="text-dark-tan">
          {note}
        </span>
      ) : null}
      {suggestion ? (
        <button
          type="button"
          disabled={busy}
          onClick={() => void push(suggestion)}
          className="rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
        >
          Push as {suggestion} instead
        </button>
      ) : null}
    </span>
  );
}
