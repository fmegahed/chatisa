"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { announce, focusElement } from "@/lib/a11y/announce";

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
  const confirmRef = useRef<HTMLButtonElement>(null);
  const trashRef = useRef<HTMLButtonElement>(null);
  // Swapping the trash button for the confirm pair (and back) removes the
  // focused button, so focus follows the swap. Not on first render.
  const swapped = useRef(false);

  useEffect(() => {
    if (!swapped.current) return;
    if (confirming) confirmRef.current?.focus();
    else trashRef.current?.focus();
  }, [confirming]);

  function setConfirm(next: boolean) {
    swapped.current = true;
    setConfirming(next);
  }

  async function del() {
    if (busy) return;
    setBusy(true);
    const res = await fetch(`/api/project-assistant/${projectId}`, { method: "DELETE" });
    if (res.ok) {
      announce(`Deleted ${projectName}.`);
      // The card is about to disappear; land on the list heading.
      focusElement(document.getElementById("my-projects-heading"));
      router.refresh();
    } else {
      setBusy(false);
      setConfirm(false);
      announce(`Could not delete ${projectName}. Try again.`, "assertive");
    }
  }

  if (confirming) {
    return (
      <span className="flex items-center gap-1 rounded-card bg-paper/90 px-1 text-xs">
        <span id={`delete-q-${projectId}`} className="sr-only">
          Delete {projectName}?
        </span>
        <button
          ref={confirmRef}
          type="button"
          onClick={del}
          aria-describedby={`delete-q-${projectId}`}
          aria-disabled={busy || undefined}
          className="font-bold text-miami-red hover:underline aria-disabled:opacity-60"
        >
          Delete
        </button>
        <button
          type="button"
          onClick={() => setConfirm(false)}
          className="text-dark-tan hover:underline"
        >
          Cancel
        </button>
      </span>
    );
  }

  return (
    <button
      ref={trashRef}
      type="button"
      onClick={() => setConfirm(true)}
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
