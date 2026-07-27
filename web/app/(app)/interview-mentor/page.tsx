import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { InterviewMentor } from "@/components/interview/InterviewMentor";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";
import { recordUsageEvent } from "@/lib/db";
import { isSpeechConfigured } from "@/lib/speech/deepgram";

export const metadata: Metadata = { title: "Interview Mentor" };

export default async function InterviewMentorPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  const available = filterAvailableModels(getPageModels("interview_mentor"));
  const { options: models, defaultModelId } = buildModelOptions(
    "interview_mentor",
    available,
  );

  const speechReady = isSpeechConfigured();

  recordUsageEvent({
    userEmail: session.user.email,
    module: "interview_mentor",
    eventType: "module_open",
  });

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">Interview Mentor</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        Practise a real interview, one question at a time. Speak your answers or
        type them, then get specific feedback on what landed and what to work
        on.
      </p>

      {!speechReady ? (
        // Speech is an enhancement, so its absence is a note rather than a
        // blocker. The module works fully by typing.
        <p
          role="status"
          className="mt-6 rounded-card border border-medium-tan bg-light-tan p-4"
        >
          Speaking your answers is not set up on this server yet, so answer by
          typing. Everything else works as normal.
        </p>
      ) : null}

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
          <InterviewMentor models={models} defaultModelId={defaultModelId} />
        </div>
      )}
    </div>
  );
}
