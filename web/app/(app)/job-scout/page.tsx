import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { JobScout } from "@/components/scout/JobScout";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";
import { recordUsageEvent } from "@/lib/db";
import { githubOauthConfigured } from "@/lib/scout/github-oauth";

export const metadata: Metadata = { title: "Job Scout" };

export default async function JobScoutPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  const available = filterAvailableModels(getPageModels("job_scout"));
  const { options, defaultModelId } = buildModelOptions("job_scout", available);

  recordUsageEvent({
    userEmail: session.user.email,
    module: "job_scout",
    eventType: "module_open",
  });

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">Job Scout</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        A fresh board of analytics, information systems, and security jobs
        every Sunday, matched to the ISA courses you have taken. Found one
        you want? Draft a customized resume and cover letter with JobApp
        Drafter.
      </p>

      {/* Said once, plainly: what stays on the device and what the feed is. */}
      <p className="mt-6 rounded-card border border-medium-tan bg-light-tan p-4">
        Your course list, confirmed skills, and saved jobs live only in this
        browser. The job postings are gathered weekly from employer career
        sites and USAJobs; applying always happens on the employer&apos;s
        own site.
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
          <JobScout
            models={options}
            defaultModelId={defaultModelId}
            githubEnabled={githubOauthConfigured()}
          />
        </div>
      )}
    </div>
  );
}
