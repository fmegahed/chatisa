import { coachLabel, isCoachType } from "@/lib/project/coaches";

/**
 * Page titles for project routes (#28). The root layout appends
 * " · ChatISA". Callers pass a project only after checking the signed-in
 * user can open it; without one the title stays generic, so a title never
 * reveals a project the viewer cannot see.
 */
export interface TitledProject {
  name: string;
  courseCode: string;
}

export function projectPageTitle(project: TitledProject | null | undefined): string {
  if (!project) return "Project";
  return `${project.name} (ISA ${project.courseCode})`;
}

export function coachPageTitle(
  coachType: string,
  project: TitledProject | null | undefined,
): string {
  const coach = isCoachType(coachType) ? `${coachLabel(coachType)} Coach` : "Coach";
  return project ? `${coach}: ${projectPageTitle(project)}` : coach;
}
