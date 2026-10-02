// components/project/TeamManager.tsx
"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import type { MemberRow } from "@/lib/db/projects";
import { ALLOWED_EMAIL_DOMAIN } from "@/lib/auth/domain";
import { announce } from "@/lib/a11y/announce";
import { INVITE_EMAIL_HINT, inviteEmailError } from "@/lib/project/invite";

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
  // An error about the email field is shown under it; anything else (a
  // failed removal, the server unreachable) is shown after the form.
  const [fieldError, setFieldError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showHelp, setShowHelp] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const base = `/api/project-assistant/${projectId}/members`;

  async function invite(e: React.FormEvent) {
    e.preventDefault();
    // The button stays focusable while busy (aria-disabled), so a second
    // press is ignored here rather than by disabling it (#27).
    if (busy) return;
    setError(null);
    const value = email.trim();
    const problem = inviteEmailError(value);
    setFieldError(problem);
    if (problem) {
      announce(problem, "assertive");
      inputRef.current?.focus();
      return;
    }
    setBusy(true);
    try {
      const res = await fetch(base, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: value }),
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        const message = data.error ?? "Could not add that teammate.";
        setFieldError(message);
        announce(message, "assertive");
        return;
      }
      setEmail("");
      announce(`Added ${value} to the team.`);
      router.refresh();
    } catch {
      setError("Could not reach the server. Try again.");
    } finally {
      setBusy(false);
    }
  }

  async function remove(memberEmail: string, label: string) {
    setError(null);
    const res = await fetch(base, {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email: memberEmail }),
    });
    if (res.ok) {
      // The pressed button disappears with the member, so focus goes to the
      // invite field rather than the page body.
      inputRef.current?.focus();
      announce(`Removed ${label} from the team.`);
      router.refresh();
    } else setError("Could not remove that teammate.");
  }

  const describedBy = [
    "invite-email-hint",
    fieldError ? "invite-email-error" : null,
  ]
    .filter(Boolean)
    .join(" ");

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
                onClick={() => remove(m.email, m.name ?? m.email)}
                aria-label={`Remove ${m.name ?? m.email}`}
                className="font-bold text-miami-red hover:underline"
              >
                Remove
              </button>
            ) : null}
          </li>
        ))}
      </ul>

      <form
        onSubmit={invite}
        noValidate
        className="mt-3 flex flex-wrap items-end gap-2"
      >
        <div className="flex max-w-prose flex-col gap-1">
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
          {/* The explanation sits right after its button, before the field it
              explains, so it is next in reading order (#29). */}
          <p
            id="add-teammate-help"
            hidden={!showHelp}
            className="max-w-prose rounded-card border border-medium-tan bg-light-tan p-3 text-sm text-dark-tan"
          >
            Adding a teammate gives them access to this project. No email is
            sent, so tell them to sign in to ChatISA and open the project. It
            appears for them in their Shared with me list, and their name fills
            in once they open it. The email must match their Miami login, or
            they will not see the project.
          </p>
          {/* The requirement is stated before submitting (#26). */}
          <p id="invite-email-hint" className="text-sm text-dark-tan">
            {INVITE_EMAIL_HINT}
          </p>
          <input
            ref={inputRef}
            id="invite-email"
            type="email"
            autoComplete="off"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
              if (fieldError) setFieldError(null);
            }}
            placeholder={`name@${ALLOWED_EMAIL_DOMAIN}`}
            aria-describedby={describedBy}
            aria-invalid={fieldError ? true : undefined}
            className="rounded border border-medium-tan bg-paper p-2"
          />
          {fieldError ? (
            <p id="invite-email-error" className="text-sm text-miami-red">
              {fieldError}
            </p>
          ) : null}
        </div>
        <button
          type="submit"
          aria-disabled={busy || undefined}
          className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red aria-disabled:cursor-not-allowed aria-disabled:bg-medium-gray"
        >
          Add teammate
        </button>
      </form>
      {error ? (
        <p role="alert" className="mt-2 text-miami-red">
          {error}
        </p>
      ) : null}
    </div>
  );
}
