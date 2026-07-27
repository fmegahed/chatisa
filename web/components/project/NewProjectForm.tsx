// components/project/NewProjectForm.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ISA_COURSES, courseLabel } from "@/lib/project/courses";
import { COACHES, type CoachType } from "@/lib/project/coaches";

export function NewProjectForm() {
  const router = useRouter();
  const [courseCode, setCourseCode] = useState("");
  const [name, setName] = useState("");
  const [organization, setOrganization] = useState("");
  const [coaches, setCoaches] = useState<CoachType[]>(["scoping"]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function toggleCoach(type: CoachType) {
    setCoaches((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type],
    );
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    if (!courseCode) {
      setError("Pick a course.");
      return;
    }
    if (!name.trim()) {
      setError("Give the project a name.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ courseCode, name, organization, coachTypes: coaches }),
      });
      if (!res.ok) {
        const data = (await res.json().catch(() => ({}))) as { error?: string };
        setError(data.error ?? "Could not create the project. Try again.");
        setSubmitting(false);
        return;
      }
      const { id } = (await res.json()) as { id: string };
      router.push(`/project-assistant/${id}`);
    } catch {
      setError("Could not reach the server. Check your connection and try again.");
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={onSubmit} className="mt-6 max-w-2xl">
      <div className="mb-5">
        <label htmlFor="course" className="block font-bold">
          Course
        </label>
        <select
          id="course"
          value={courseCode}
          onChange={(e) => setCourseCode(e.target.value)}
          className="mt-1 w-full rounded border border-medium-tan p-2"
          required
        >
          <option value="">Select a course</option>
          {ISA_COURSES.map((c) => (
            <option key={c.code} value={c.code}>
              {courseLabel(c)}
            </option>
          ))}
        </select>
      </div>

      <div className="mb-5">
        <label htmlFor="name" className="block font-bold">
          Project name
        </label>
        <input
          id="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="mt-1 w-full rounded border border-medium-tan p-2"
          maxLength={160}
          required
        />
      </div>

      <div className="mb-5">
        <label htmlFor="organization" className="block font-bold">
          Organization (optional)
        </label>
        <input
          id="organization"
          value={organization}
          onChange={(e) => setOrganization(e.target.value)}
          className="mt-1 w-full rounded border border-medium-tan p-2"
          maxLength={160}
        />
      </div>

      <fieldset className="mb-6">
        <legend className="font-bold">Coaches to include</legend>
        <p className="text-sm text-neutral-700">
          Pick the coaches this project will use. You can change this later.
        </p>
        <div className="mt-2 grid gap-2">
          {COACHES.map((c) => (
            <label key={c.type} className="flex items-start gap-2">
              <input
                type="checkbox"
                checked={coaches.includes(c.type)}
                onChange={() => toggleCoach(c.type)}
                className="mt-1"
              />
              <span>
                <span className="font-bold">{c.label}.</span> {c.blurb}
              </span>
            </label>
          ))}
        </div>
      </fieldset>

      {error ? (
        <p role="alert" className="mb-4 text-miami-red">
          {error}
        </p>
      ) : null}

      <button
        type="submit"
        disabled={submitting}
        className="rounded-card bg-miami-red px-5 py-2.5 font-bold text-white disabled:opacity-60"
      >
        {submitting ? "Creating..." : "Create project"}
      </button>
    </form>
  );
}
