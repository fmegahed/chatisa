/**
 * What a publish actually pushes (2026-08-20). Pure so the exact repository
 * name, the exact rendered HTML, and the exact file set are unit-testable
 * without a browser or a GitHub token. The caller reads the resume off disk
 * and supplies the repository name a republish must keep.
 */

import type { PushFile } from "@/lib/scout/github";
import type { Draft } from "./draft";
import { renderCareer, renderShowcase } from "./html";
import {
  CAREER_REPO, DEFAULT_GITIGNORE, careerFileSet, dedupePaths, rolePath, showcaseFileSet,
  showcaseRepoName,
} from "./files";
import { pushable } from "./intake";

export function buildPublishPlan(
  draft: Draft,
  login: string,
  extras: { resumeBase64: string | null; existingRepoName: string | null },
): { repoName: string; files: PushFile[]; html: string; readme: string | null } {
  if (!draft.content) throw new Error("Nothing to publish yet.");
  if (draft.content.kind === "career") {
    const content = draft.content.content;
    const links = draft.links.filter((l) => l.label.trim() && l.url.trim());
    const projects = draft.projects.map((p) => ({ slug: p.slug, files: p.files.filter(pushable) }));
    // "portfolio" is the default, but a student whose account already owns
    // that name publishes under the one they picked, and a republish must
    // keep it. The rendered Files links have to point at the same repo.
    const repoName = extras.existingRepoName ?? CAREER_REPO;
    const html = renderCareer(content, {
      name: draft.name.trim(), links, hasPhoto: !!draft.photo,
      resumeLink: draft.resumeLink, login, repoName,
      folders: projects.filter((p) => p.files.length > 0).map((p) => p.slug),
    });
    const files = careerFileSet({
      html,
      photoBase64: draft.photo?.base64 ?? null,
      resumeBase64: draft.resumeLink ? extras.resumeBase64 : null,
      projects,
    });
    return { repoName, files, html, readme: null };
  }
  const content = draft.content.content;
  const repoName = extras.existingRepoName ?? showcaseRepoName(draft.course, content.title);
  const published = draft.files.filter((f) => f.publish && pushable(f));
  // The paths the push will really write (collisions suffixed), so the page
  // never links a deliverable or figure that is not in the repository.
  const deliverablePaths = dedupePaths(published.map((f) => rolePath(f.role, f.name)));
  const figures = deliverablePaths.filter((_, i) => published[i].role === "figure");
  const html = renderShowcase(content, {
    course: draft.course, semester: draft.semester, team: draft.team,
    repoUrl: `https://github.com/${login}/${repoName}`, figures, deliverablePaths,
  });
  const readme = draft.readme && draft.readme.trim().length > 0
    ? draft.readme
    : `# ${content.title}\n\n${content.tagline}\n\nBuilt for ${draft.course}. Published with ChatISA's Portfolio Builder.\n`;
  const files = showcaseFileSet({ html, readme, gitignore: DEFAULT_GITIGNORE, files: published });
  return { repoName, files, html, readme };
}
