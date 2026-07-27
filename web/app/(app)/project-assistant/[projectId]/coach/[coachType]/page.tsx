// app/(app)/project-assistant/[projectId]/coach/[coachType]/page.tsx
import { notFound, redirect } from "next/navigation";
import type { UIMessage } from "ai";
import { auth } from "@/lib/auth";
import { recordUsageEvent } from "@/lib/db";
import { getAccessibleProject, getOrCreateDeliverable } from "@/lib/db/projects";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";
import { CoachSession } from "@/components/project/CoachSession";
import { getCoachEngine } from "@/lib/project/coach-engine";
import { getCoachSpec } from "@/lib/project/coach-specs";
import { coachLabel, isCoachType } from "@/lib/project/coaches";

export default async function CoachSessionPage({
  params,
}: {
  params: Promise<{ projectId: string; coachType: string }>;
}) {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");
  const { projectId, coachType } = await params;

  const engine = getCoachEngine(coachType);
  if (!engine || !isCoachType(coachType)) notFound();

  const project = getAccessibleProject(projectId, session.user.email);
  if (!project) notFound();

  const row = getOrCreateDeliverable(projectId, coachType);
  const content = engine.parseContent(row.contentJson);

  let initialMessages: UIMessage[] = [];
  try {
    const parsed = JSON.parse(row.transcriptJson);
    if (Array.isArray(parsed)) initialMessages = parsed as UIMessage[];
  } catch {
    initialMessages = [];
  }

  const available = filterAvailableModels(getPageModels("project_coach"));
  const { options, defaultModelId } = buildModelOptions("project_coach", available);

  recordUsageEvent({
    userEmail: session.user.email,
    module: "project_coach",
    eventType: "coach_open",
    outcome: coachType,
  });

  const common = {
    projectId,
    projectName: project.name,
    coachType,
    coachTitle: `${coachLabel(coachType)} Coach`,
    models: options,
    defaultModelId,
    initialContent: content,
    initialMessages,
    initialLastUpdatedBy: row.lastUpdatedBy,
  };
  const spec = getCoachSpec(coachType);
  return spec ? (
    <CoachSession {...common} kind="generic" spec={spec} />
  ) : (
    <CoachSession {...common} kind="scoping" />
  );
}
