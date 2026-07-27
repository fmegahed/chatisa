// components/project/ProjectList.tsx
import Link from "next/link";
import type { ProjectRow } from "@/lib/db/projects";
import { courseLabel, findCourse } from "@/lib/project/courses";
import { DeleteProjectButton } from "@/components/project/DeleteProjectButton";

function courseName(code: string): string {
  const course = findCourse(code);
  return course ? courseLabel(course) : `ISA ${code}`;
}

/** `deletable` shows a trash button on each card, for the owner's own projects. */
export function ProjectList({
  projects,
  deletable = false,
}: {
  projects: ProjectRow[];
  deletable?: boolean;
}) {
  return (
    <ul className="mt-4 grid gap-4 sm:grid-cols-2">
      {projects.map((p) => (
        <li key={p.id} className="relative">
          <Link
            href={`/project-assistant/${p.id}`}
            className="block rounded-card border border-medium-tan bg-light-tan p-4 pr-10 hover:border-miami-red focus-visible:outline focus-visible:outline-2"
          >
            <p className="text-sm text-neutral-700">{courseName(p.courseCode)}</p>
            <p className="mt-1 text-lg font-bold">{p.name}</p>
            {p.organization ? (
              <p className="mt-1 text-sm text-neutral-700">{p.organization}</p>
            ) : null}
          </Link>
          {deletable ? (
            <div className="absolute right-2 top-2">
              <DeleteProjectButton projectId={p.id} projectName={p.name} />
            </div>
          ) : null}
        </li>
      ))}
    </ul>
  );
}
