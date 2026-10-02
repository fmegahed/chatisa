// components/project/CoachSelector.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { COACHES, type CoachType } from "@/lib/project/coaches";
import { announce } from "@/lib/a11y/announce";

export function CoachSelector({
  projectId,
  enabled,
}: {
  projectId: string;
  enabled: CoachType[];
}) {
  const router = useRouter();
  const [selected, setSelected] = useState<CoachType[]>(enabled);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggle(type: CoachType) {
    setSelected((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type],
    );
  }

  async function save() {
    // Kept focusable while saving (aria-disabled), so guard here (#27).
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const res = await fetch(`/api/project-assistant/${projectId}/coaches`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ coachTypes: selected }),
      });
      if (!res.ok) {
        setError("Could not save the coaches. Try again.");
        return;
      }
      announce("Coaches saved.");
      router.refresh();
    } catch {
      setError("Could not reach the server. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <fieldset className="mt-3 rounded-card border border-medium-tan p-4">
      <legend className="px-1 font-bold">Choose coaches (lead only)</legend>
      <div className="mt-2 grid gap-2">
        {COACHES.map((c) => (
          <label key={c.type} className="flex items-start gap-2">
            <input
              type="checkbox"
              checked={selected.includes(c.type)}
              onChange={() => toggle(c.type)}
              className="mt-1"
            />
            <span>
              <span className="font-bold">{c.label}.</span> {c.blurb}
            </span>
          </label>
        ))}
      </div>
      <button
        type="button"
        onClick={save}
        aria-disabled={busy || undefined}
        className="mt-3 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red aria-disabled:bg-medium-gray"
      >
        Save coaches
      </button>
      {error ? (
        <p role="alert" className="mt-2 text-miami-red">
          {error}
        </p>
      ) : null}
    </fieldset>
  );
}
