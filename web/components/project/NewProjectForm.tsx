// components/project/NewProjectForm.tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ISA_COURSES, courseLabel } from "@/lib/project/courses";
import { COACHES, type CoachType } from "@/lib/project/coaches";
import { RequiredMark, RequiredNote } from "@/components/a11y/Required";

export function NewProjectForm() {
  const router = useRouter();
  const [courseCode, setCourseCode] = useState("");
  const [name, setName] = useState("");
  const [organization, setOrganization] = useState("");
  const [coaches, setCoaches] = useState<CoachType[]>(["scoping"]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Which field the client-side error is about, so it can be marked invalid
  // and described by the message.
  const [errorField, setErrorField] = useState<"course" | "name" | null>(null);

  function toggleCoach(type: CoachType) {
    setCoaches((prev) =>
      prev.includes(type) ? prev.filter((t) => t !== type) : [...prev, type],
    );
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault();
    // Kept focusable while submitting (aria-disabled), so guard here (#27).
    if (submitting) return;
    setError(null);
    setErrorField(null);
    if (!courseCode) {
      setError("Pick a course.");
      setErrorField("course");
      document.getElementById("course")?.focus();
      return;
    }
    if (!name.trim()) {
      setError("Give the project a name.");
      setErrorField("name");
      document.getElementById("name")?.focus();
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
    <form onSubmit={onSubmit} noValidate className="mt-6 max-w-2xl">
      <RequiredNote className="mb-4" />
      <div className="mb-5">
        <label htmlFor="course" className="block font-bold">
          Course
          <RequiredMark />
        </label>
        <select
          id="course"
          value={courseCode}
          onChange={(e) => setCourseCode(e.target.value)}
          className="mt-1 w-full rounded border border-medium-tan p-2"
          required
          aria-invalid={errorField === "course" || undefined}
          aria-describedby={errorField === "course" ? "new-project-error" : undefined}
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
          <RequiredMark />
        </label>
        <input
          id="name"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="mt-1 w-full rounded border border-medium-tan p-2"
          maxLength={160}
          required
          aria-invalid={errorField === "name" || undefined}
          aria-describedby={errorField === "name" ? "new-project-error" : undefined}
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
        <p id="new-project-error" role="alert" className="mb-4 text-miami-red">
          {error}
        </p>
      ) : null}

      <button
        type="submit"
        aria-disabled={submitting || undefined}
        className="rounded-card bg-miami-red px-5 py-2.5 font-bold text-white aria-disabled:opacity-60"
      >
        {submitting ? "Creating..." : "Create project"}
      </button>
    </form>
  );
}
