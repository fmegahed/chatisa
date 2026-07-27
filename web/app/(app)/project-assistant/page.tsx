// app/(app)/project-assistant/page.tsx
import type { Metadata } from "next";
import Link from "next/link";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { recordUsageEvent } from "@/lib/db";
import { listOwnedProjects, listSharedProjects } from "@/lib/db/projects";
import { ProjectList } from "@/components/project/ProjectList";

export const metadata: Metadata = { title: "Project Assistant" };

export default async function ProjectAssistantPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");
  const email = session.user.email;

  const owned = listOwnedProjects(email);
  const shared = listSharedProjects(email);

  recordUsageEvent({
    userEmail: email,
    module: "project_coach",
    eventType: "module_open",
  });

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">Project Assistant</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        Set up a team project for your course, invite your teammates, and work
        with AI coaches that help you scope, plan, and reflect. Each coach fills
        a deliverable you can edit together and export to Word.
      </p>

      <div className="mt-6">
        <Link
          href="/project-assistant/new"
          className="inline-block rounded-card bg-miami-red px-5 py-2.5 font-bold text-white"
        >
          New project
        </Link>
      </div>

      <section className="mt-10" aria-labelledby="my-projects-heading">
        <h2 id="my-projects-heading" className="text-2xl">
          My projects
        </h2>
        {owned.length === 0 ? (
          <p className="mt-3 text-neutral-700">
            You have not created a project yet. Start one with New project above.
          </p>
        ) : (
          <ProjectList projects={owned} deletable />
        )}
      </section>

      {shared.length > 0 ? (
        <section className="mt-10" aria-labelledby="shared-projects-heading">
          <h2 id="shared-projects-heading" className="text-2xl">
            Shared with me
          </h2>
          <ProjectList projects={shared} />
        </section>
      ) : null}
    </div>
  );
}
