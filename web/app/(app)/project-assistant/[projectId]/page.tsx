// app/(app)/project-assistant/[projectId]/page.tsx
import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import {
  getAccessibleProject,
  listProjectMembers,
  fillMemberName,
} from "@/lib/db/projects";
import { courseLabel, findCourse } from "@/lib/project/courses";
import { COACHES } from "@/lib/project/coaches";
import { TeamManager } from "@/components/project/TeamManager";
import { CoachSelector } from "@/components/project/CoachSelector";

export default async function ProjectWorkspacePage({
  params,
}: {
  params: Promise<{ projectId: string }>;
}) {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");
  const { projectId } = await params;

  const project = getAccessibleProject(projectId, session.user.email);
  // Access control: a non-member gets the same not-found a bad id gives.
  if (!project) notFound();

  // On first visit, fill the invited member's name from their authenticated
  // Google name (without overwriting one already stored).
  if (session.user.name) {
    fillMemberName(projectId, session.user.email, session.user.name);
  }
  const members = listProjectMembers(projectId);
  const isLead =
    members.find((m) => m.email === session.user!.email!.toLowerCase())?.role ===
    "lead";
  const course = findCourse(project.courseCode);
  const enabled = COACHES.filter((c) => project.coachTypes.includes(c.type));

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <Link href="/project-assistant" className="text-sm underline">
        Back to my projects
      </Link>

      <p className="ribbon mt-4">
        {course ? courseLabel(course) : `ISA ${project.courseCode}`}
      </p>
      <h1 className="mt-3 text-4xl">{project.name}</h1>
      {project.organization ? (
        <p className="mt-2 text-lg text-neutral-700">{project.organization}</p>
      ) : null}

      <div className="mt-4">
        <a
          href={`/api/project-assistant/${project.id}/export`}
          className="inline-block rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red"
        >
          Download all deliverables
        </a>
      </div>

      <section className="mt-8" aria-labelledby="team-heading">
        <h2 id="team-heading" className="text-2xl">
          Team
        </h2>
        {isLead ? (
          <TeamManager
            projectId={project.id}
            members={members}
            ownerEmail={project.ownerEmail}
          />
        ) : (
          <ul className="mt-3 flex flex-wrap gap-2">
            {members.map((m) => (
              <li
                key={m.id}
                className="rounded-card border border-medium-tan bg-light-tan px-3 py-1 text-sm"
              >
                {m.name ?? m.email}
                {m.email === project.ownerEmail.toLowerCase() ? " (lead)" : ""}
              </li>
            ))}
          </ul>
        )}
      </section>

      <section className="mt-10" aria-labelledby="coaches-heading">
        <h2 id="coaches-heading" className="text-2xl">
          Coaches
        </h2>
        {enabled.length === 0 ? (
          <p className="mt-3 text-neutral-700">
            No coaches are enabled for this project yet.
          </p>
        ) : (
          <ul className="mt-4 grid gap-4 sm:grid-cols-2">
            {enabled.map((c) => (
              <li key={c.type}>
                <Link
                  href={`/project-assistant/${project.id}/coach/${c.type}`}
                  className="block rounded-card border border-medium-tan bg-light-tan p-4 hover:border-miami-red focus-visible:outline focus-visible:outline-2"
                >
                  <p className="text-lg font-bold">{c.label}</p>
                  <p className="mt-1 text-sm text-neutral-700">{c.blurb}</p>
                </Link>
              </li>
            ))}
          </ul>
        )}
        {isLead ? (
          <CoachSelector projectId={project.id} enabled={project.coachTypes} />
        ) : null}
      </section>
    </div>
  );
}
