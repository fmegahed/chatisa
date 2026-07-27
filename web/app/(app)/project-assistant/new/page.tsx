// app/(app)/project-assistant/new/page.tsx
import type { Metadata } from "next";
import Link from "next/link";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { NewProjectForm } from "@/components/project/NewProjectForm";

export const metadata: Metadata = { title: "New project" };

export default async function NewProjectPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <Link href="/project-assistant" className="text-sm underline">
        Back to my projects
      </Link>
      <h1 className="mt-4 text-3xl">New project</h1>
      <p className="mt-2 max-w-2xl text-neutral-700">
        Choose the course, name the project, and pick the coaches. You will be
        the team lead and can invite teammates from the project page.
      </p>
      <NewProjectForm />
    </div>
  );
}
