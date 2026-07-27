import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";
import { recordUsageEvent } from "@/lib/db";
import { AiComparison } from "@/components/comparison/AiComparison";

export const metadata: Metadata = { title: "AI Comparison" };

export default async function AiComparisonsPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  const available = filterAvailableModels(getPageModels("ai_comparisons"));
  const { options } = buildModelOptions("ai_comparisons", available);

  recordUsageEvent({
    userEmail: session.user.email,
    module: "ai_comparisons",
    eventType: "module_open",
  });

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">AI Comparison</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        Put two AI models side by side on the same question and vote for the
        answer you prefer. The models stay hidden until the end.
      </p>

      {options.length < 2 ? (
        <div
          role="status"
          className="mt-8 rounded-card border-2 border-miami-red bg-paper p-5"
        >
          <h2 className="font-bold text-miami-red">Not enough models available</h2>
          <p className="mt-1">
            A comparison needs at least two models, and this server does not
            have two configured right now. Contact the ChatISA maintainers.
          </p>
        </div>
      ) : (
        <div className="mt-8">
          <AiComparison models={options} />
        </div>
      )}
    </div>
  );
}
