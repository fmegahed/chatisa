import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { PortfolioBuilder } from "@/components/portfolio/PortfolioBuilder";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";
import { recordUsageEvent } from "@/lib/db";
import { githubOauthConfigured } from "@/lib/scout/github-oauth";

export const metadata: Metadata = { title: "Portfolio Builder" };

export default async function PortfolioPage(props: { searchParams: Promise<{ mode?: string }> }) {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");
  const { mode } = await props.searchParams;
  const available = filterAvailableModels(getPageModels("portfolio"));
  const { options, defaultModelId } = buildModelOptions("portfolio", available);
  recordUsageEvent({ userEmail: session.user.email, module: "portfolio", eventType: "module_open" });

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">Portfolio Builder</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        Turn your work into a site you can send to anyone: a portfolio of who you are and what you can
        do, or a showcase that tells the story of one project. You see and edit the page before it goes
        to GitHub Pages.
      </p>
      <p className="mt-6 rounded-card border border-medium-tan bg-light-tan p-4">
        Your files, photo, and drafts stay in this browser. The server reads uploads only to write the
        page and keeps nothing. Publishing sends the files to a public repository on your own GitHub
        account.
      </p>
      {options.length === 0 ? (
        <div role="status" className="mt-8 rounded-card border-2 border-miami-red bg-paper p-5">
          <h2 className="font-bold text-miami-red">No models are available</h2>
          <p className="mt-1">This server has no AI provider configured yet. Contact the ChatISA maintainers.</p>
        </div>
      ) : (
        <div className="mt-8">
          <PortfolioBuilder
            models={options}
            defaultModelId={defaultModelId}
            githubEnabled={githubOauthConfigured()}
            studentName={session.user.name ?? ""}
            initialMode={mode === "career" ? "career" : mode === "project" || mode === "showcase" ? "showcase" : null}
          />
        </div>
      )}
    </div>
  );
}
