import type { Metadata } from "next";
import Link from "next/link";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { JobAppAssistant } from "@/components/jobs/JobAppAssistant";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";
import { recordUsageEvent } from "@/lib/db";

export const metadata: Metadata = { title: "JobApp Drafter" };

export default async function JobAppAssistantPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  const available = filterAvailableModels(getPageModels("jobapp_assistant"));
  const { options, defaultModelId } = buildModelOptions(
    "jobapp_assistant",
    available,
  );

  recordUsageEvent({
    userEmail: session.user.email,
    module: "jobapp_assistant",
    eventType: "module_open",
  });

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">JobApp Drafter</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        Tailor your resume and cover letter to a specific job, to Farmer School
        of Business standards. Then practise the interview for it.
      </p>

      <nav aria-label="JobApp Assistant stages" className="mt-6">
        <ol className="flex flex-wrap gap-4">
          <li className="font-bold">1. Tailor your application</li>
          <li>
            2.{" "}
            <Link href="/interview-mentor" className="underline">
              Practise the interview
            </Link>
          </li>
        </ol>
      </nav>

      {/* Said once, plainly, before a student starts. The tool selects and
          rewords what they wrote; it does not invent experience, and anything
          it cannot trace back to their resume is flagged for them to check. */}
      <p className="mt-6 rounded-card border border-medium-tan bg-light-tan p-4">
        Everything here is built from the resume you upload. Nothing is invented,
        and any line we cannot trace back to your own resume is flagged for you
        to check. You will be asked about every line of this in an interview, so
        read it before you send it.
      </p>

      {options.length === 0 ? (
        <div
          role="status"
          className="mt-8 rounded-card border-2 border-miami-red bg-paper p-5"
        >
          <h2 className="font-bold text-miami-red">No models are available</h2>
          <p className="mt-1">
            This server has no AI provider configured yet. Contact the ChatISA
            maintainers.
          </p>
        </div>
      ) : (
        <div className="mt-8">
          <JobAppAssistant
            models={options}
            defaultModelId={defaultModelId}
            studentName={session.user.name ?? ""}
            studentEmail={session.user.email}
          />
        </div>
      )}
    </div>
  );
}
