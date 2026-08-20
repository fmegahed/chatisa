"use client";

import { useGithubConnection } from "@/lib/scout/use-scout-store";

/**
 * Connect-to-GitHub affordance (v6.3.0). Opens the OAuth flow in a popup so
 * the student keeps their place; a blocked popup degrades to a full-page
 * navigation that returns to `returnPath` afterwards. The connected state
 * re-renders via the store hook, which listens for the popup's postMessage
 * and storage signals.
 */
export function GithubConnect(props: { returnPath: string }) {
  const { connection, clear } = useGithubConnection();

  if (connection) {
    return (
      <p className="flex flex-wrap items-center gap-2">
        <span className="rounded-card border border-medium-tan bg-light-tan px-2 py-1">
          Connected to GitHub as <strong>{connection.login}</strong>
        </span>
        <button
          type="button"
          onClick={clear}
          className="underline"
          title="Forgets the connection on this device. To fully revoke access, remove ChatISA under Settings, Applications on github.com."
        >
          Disconnect
        </button>
      </p>
    );
  }

  const startUrl = `/api/scout/github/start?return=${encodeURIComponent(props.returnPath)}`;
  return (
    <button
      type="button"
      onClick={() => {
        const popup = window.open(startUrl, "chatisa-github", "width=900,height=700");
        if (!popup) window.location.assign(startUrl);
      }}
      className="rounded-card border-2 border-miami-red px-3 py-1 font-bold text-miami-red hover:bg-light-tan"
    >
      Connect GitHub
    </button>
  );
}
