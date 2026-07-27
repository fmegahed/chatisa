import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { ExamAlly } from "@/components/exam/ExamAlly";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";
import { recordUsageEvent } from "@/lib/db";

export const metadata: Metadata = { title: "Exam Prep" };

export default async function ExamAllyPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  const available = filterAvailableModels(getPageModels("exam_ally"));
  const { options: models, defaultModelId } = buildModelOptions(
    "exam_ally",
    available,
  );

  // Records that a student opened the module, which the legacy app could not
  // measure: page visits were only ever instrumented for the home page.
  recordUsageEvent({
    userEmail: session.user.email,
    module: "exam_ally",
    eventType: "module_open",
  });

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">Exam Prep</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        Turn your course material into a practice exam. Every question is
        checked against your own document, and each answer points you back to
        the page it came from.
      </p>

      {models.length === 0 ? (
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
          <ExamAlly models={models} defaultModelId={defaultModelId} />
        </div>
      )}
    </div>
  );
}
