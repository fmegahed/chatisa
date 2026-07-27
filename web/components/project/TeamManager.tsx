// components/project/TeamManager.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import type { MemberRow } from "@/lib/db/projects";

export function TeamManager({
  projectId,
  members,
  ownerEmail,
}: {
  projectId: string;
  members: MemberRow[];
  ownerEmail: string;
}) {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showHelp, setShowHelp] = useState(false);

  const base = `/api/project-assistant/${projectId}/members`;

  async function invite(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const value = email.trim();
    if (!value) return;
    setBusy(true);
    try {
      const res = await fetch(base, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: value }),
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        setError(data.error ?? "Could not add that teammate.");
        return;
      }
      setEmail("");
      router.refresh();
    } catch {
      setError("Could not reach the server. Try again.");
    } finally {
      setBusy(false);
    }
  }

  async function remove(memberEmail: string) {
    setError(null);
    const res = await fetch(base, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: memberEmail }),
    });
    if (res.ok) router.refresh();
    else setError("Could not remove that teammate.");
  }

  return (
    <div className="mt-3">
      <ul className="flex flex-wrap gap-2">
        {members.map((m) => (
          <li
            key={m.id}
            className="flex items-center gap-2 rounded-card border border-medium-tan bg-light-tan px-3 py-1 text-sm"
          >
            <span>
              {m.name ?? m.email}
              {m.email === ownerEmail.toLowerCase() ? " (lead)" : ""}
            </span>
            {m.email !== ownerEmail.toLowerCase() ? (
              <button
                type="button"
                onClick={() => remove(m.email)}
                aria-label={`Remove ${m.name ?? m.email}`}
                className="font-bold text-miami-red hover:underline"
              >
                Remove
              </button>
            ) : null}
          </li>
        ))}
      </ul>

      <form onSubmit={invite} className="mt-3 flex flex-wrap items-end gap-2">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <label htmlFor="invite-email" className="text-sm font-bold">
              Add a teammate by email
            </label>
            <button
              type="button"
              onClick={() => setShowHelp((s) => !s)}
              aria-expanded={showHelp}
              aria-controls="add-teammate-help"
              aria-label="What does adding a teammate do?"
              className="flex h-5 w-5 items-center justify-center rounded-full border border-medium-tan text-xs font-bold text-dark-tan hover:border-miami-red hover:text-miami-red"
            >
              ?
            </button>
          </div>
          <input
            id="invite-email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="name@miamioh.edu"
            className="rounded border border-medium-tan bg-paper p-2"
          />
        </div>
        <button
          type="submit"
          disabled={busy || email.trim().length === 0}
          className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
        >
          Add teammate
        </button>
      </form>
      {showHelp ? (
        <p
          id="add-teammate-help"
          className="mt-2 max-w-prose rounded-card border border-medium-tan bg-light-tan p-3 text-sm text-dark-tan"
        >
          Adding a teammate gives them access to this project. Enter the
          miamioh.edu email they sign in with. No email is sent, so tell them to
          sign in to ChatISA and open the project. It appears for them in their
          Shared with me list, and their name fills in once they open it. The
          email must match their Miami login, or they will not see the project.
        </p>
      ) : null}
      {error ? (
        <p role="alert" className="mt-2 text-miami-red">
          {error}
        </p>
      ) : null}
    </div>
  );
}
