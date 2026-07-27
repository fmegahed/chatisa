"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

/**
 * Trash button for an owned project. Deleting is irreversible and cascades to
 * the team and all deliverables, so it asks for confirmation inline (no blocking
 * browser dialog) before calling the owner-only DELETE route.
 */
export function DeleteProjectButton({
  projectId,
  projectName,
}: {
  projectId: string;
  projectName: string;
}) {
  const router = useRouter();
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  async function del() {
    setBusy(true);
    const res = await fetch(`/api/project-assistant/${projectId}`, { method: "DELETE" });
    if (res.ok) {
      router.refresh();
    } else {
      setBusy(false);
      setConfirming(false);
    }
  }

  if (confirming) {
    return (
      <span className="flex items-center gap-1 rounded-card bg-paper/90 px-1 text-xs">
        <span className="sr-only">Delete {projectName}?</span>
        <button
          type="button"
          onClick={del}
          disabled={busy}
          className="font-bold text-miami-red hover:underline disabled:opacity-60"
        >
          Delete
        </button>
        <button
          type="button"
          onClick={() => setConfirming(false)}
          className="text-dark-tan hover:underline"
        >
          Cancel
        </button>
      </span>
    );
  }

  return (
    <button
      type="button"
      onClick={() => setConfirming(true)}
      aria-label={`Delete ${projectName}`}
      className="rounded-card p-1 text-dark-tan hover:text-miami-red focus-visible:outline focus-visible:outline-2"
    >
      <svg
        width="18"
        height="18"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d="M3 6h18" />
        <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
        <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
        <path d="M10 11v6" />
        <path d="M14 11v6" />
      </svg>
    </button>
  );
}
