"use client";

import { useEffect, useState } from "react";
import { saveGithubConnection } from "@/lib/scout/github-store";
import { safeReturnPath } from "@/lib/scout/github-state";

/**
 * Receives the OAuth result from the URL fragment, stores the connection,
 * tells the opener, and gets out of the way. The hash is cleared before
 * anything else so the token never survives in history, and it never
 * appears in any request (fragments are not sent to servers).
 */

const ERROR_COPY: Record<string, string> = {
  denied:
    "You said no on GitHub, so nothing was connected. You can close this window and try again any time.",
  state:
    "This connection attempt could not be verified, so it was not completed. Close this window and click Connect GitHub again.",
  exchange:
    "GitHub did not complete the connection. Close this window and try again in a minute.",
};

export function GithubConnected() {
  const [status, setStatus] = useState<
    { kind: "working" } | { kind: "done" } | { kind: "error"; message: string }
  >({ kind: "working" });

  useEffect(() => {
    const hash = window.location.hash;
    // Clear immediately: the token must not sit in the address bar or history.
    window.history.replaceState(null, "", window.location.pathname + window.location.search);

    // Async IIFE with setState only after an await (the InterviewMentor
    // pattern; the house lint rule forbids synchronous setState in effects).
    void (async () => {
      await Promise.resolve();

      const errorMatch = /^#gh-error=([a-z]+)$/.exec(hash);
      if (errorMatch) {
        setStatus({
          kind: "error",
          message: ERROR_COPY[errorMatch[1]] ?? ERROR_COPY.exchange,
        });
        return;
      }

      const payloadMatch = /^#gh=([A-Za-z0-9_-]+)$/.exec(hash);
      if (!payloadMatch) {
        setStatus({ kind: "error", message: ERROR_COPY.exchange });
        return;
      }
      try {
        const decoded = JSON.parse(
          atob(payloadMatch[1].replace(/-/g, "+").replace(/_/g, "/")),
        ) as { token?: string; login?: string };
        if (!decoded.token || !decoded.login) throw new Error("incomplete");
        saveGithubConnection({ token: decoded.token, login: decoded.login });
      } catch {
        setStatus({ kind: "error", message: ERROR_COPY.exchange });
        return;
      }

      if (window.opener) {
        (window.opener as Window).postMessage(
          { type: "chatisa:github-connected" },
          window.location.origin,
        );
        setStatus({ kind: "done" });
        window.close();
      } else {
        // Popup-blocked path: this was a full-page navigation, go back to work.
        const params = new URLSearchParams(window.location.search);
        window.location.replace(safeReturnPath(params.get("return")));
      }
    })();
  }, []);

  if (status.kind === "error") {
    return (
      <p role="alert" className="rounded-card border-2 border-miami-red bg-paper p-4 font-bold text-miami-red">
        {status.message}
      </p>
    );
  }
  return (
    <p role="status" className="rounded-card border border-medium-tan bg-light-tan p-4">
      {status.kind === "done"
        ? "You are connected to GitHub. You can close this window."
        : "Finishing the GitHub connection..."}
    </p>
  );
}
