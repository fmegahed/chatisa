"use client";

import { useEffect, useMemo, useState } from "react";
import type { ModelOption } from "@/lib/config/models";
import { renderCareer, renderShowcase } from "@/lib/portfolio/html";
import { CAREER_REPO, dedupePaths, rolePath } from "@/lib/portfolio/files";
import { pushable } from "@/lib/portfolio/intake";
import { loadSites, type SiteRecord } from "@/lib/portfolio/store";
import type { StepProps } from "@/lib/portfolio/draft";
import { ContentEditor } from "./ContentEditor";
import { Preview } from "./Preview";
import { Publish } from "./Publish";

/**
 * The last step (2026-08-20): the generated content on the left as editable
 * fields, the rendered page on the right, and publishing underneath it. The
 * preview re-renders from the same deterministic renderer the publish uses,
 * so what the student sees is what gets pushed.
 */

export function ReviewStep({ draft, patch, nav, githubEnabled, onPublished, onStartOver }: StepProps & {
  models: ModelOption[];
  defaultModelId: string;
  githubEnabled: boolean;
  onPublished: () => void;
  onStartOver: () => void;
}) {
  const [site, setSite] = useState<SiteRecord | null>(null);
  useEffect(() => {
    // Async IIFE with setState only after an await (the house pattern; the
    // lint rule forbids synchronous setState in an effect body).
    void (async () => {
      const found = loadSites().find((s) => s.id === draft.siteId) ?? null;
      await Promise.resolve();
      setSite(found);
    })();
  }, [draft.siteId]);

  const folders = useMemo(
    () => draft.projects.filter((p) => p.files.some(pushable)).map((p) => p.slug),
    [draft.projects],
  );
  // The paths the publish will really write (collisions suffixed), so the
  // preview links exactly the files that end up in the repository.
  const publishedPaths = useMemo(() => {
    const kept = draft.files.filter((f) => f.publish && pushable(f));
    const paths = dedupePaths(kept.map((f) => rolePath(f.role, f.name)));
    return { all: paths, figures: paths.filter((_, i) => kept[i].role === "figure") };
  }, [draft.files]);
  const figures = publishedPaths.figures;
  const html = useMemo(() => {
    if (!draft.content) return "";
    return draft.content.kind === "career"
      ? renderCareer(draft.content.content, {
          name: draft.name, links: draft.links, hasPhoto: !!draft.photo, resumeLink: draft.resumeLink,
          login: site?.repoUrl ? site.repoUrl.split("/")[3] ?? null : null, folders,
          repoName: site?.repoName ?? CAREER_REPO,
        })
      : renderShowcase(draft.content.content, {
          course: draft.course, semester: draft.semester, team: draft.team,
          repoUrl: site?.repoUrl ?? null, figures, deliverablePaths: publishedPaths.all,
        });
  }, [draft.content, draft.name, draft.links, draft.photo, draft.resumeLink, draft.course, draft.semester, draft.team, figures, publishedPaths, folders, site]);

  // The published page loads the photo from assets/photo.jpg and figures
  // from figures/<name>, files that exist only after the push. The preview
  // substitutes the bytes the browser already holds so the student never
  // sees a broken image next to their own name or finding.
  const previewHtml = useMemo(() => {
    let out = html;
    if (draft.photo) out = out.replaceAll('src="assets/photo.jpg"', `src="data:image/jpeg;base64,${draft.photo.base64}"`);
    const kept = draft.files.filter((f) => f.publish && pushable(f));
    const paths = dedupePaths(kept.map((f) => rolePath(f.role, f.name)));
    kept.forEach((f, i) => {
      if (f.role !== "figure" || !f.base64) return;
      const ext = f.name.toLowerCase().split(".").pop() ?? "";
      const mime = ext === "svg" ? "image/svg+xml" : ext === "jpg" ? "image/jpeg" : `image/${ext}`;
      out = out.replaceAll(`src="${paths[i]}"`, `src="data:${mime};base64,${f.base64}"`);
    });
    return out;
  }, [html, draft.photo, draft.files]);

  if (!draft.content) return null;
  return (
    <div className="grid gap-6 lg:grid-cols-2">
      <section className="rounded-card border border-medium-tan bg-paper p-5">
        <h2 className="text-2xl">Edit the page</h2>
        <p className="mt-1 text-dark-tan">Everything here is yours to change. The preview updates as you type.</p>
        <ContentEditor value={draft.content} onChange={(content) => patch({ content })} figures={figures} />
        {draft.content.kind === "showcase" ? (
          <label className="mt-4 block font-bold">
            README.md
            <textarea
              rows={8}
              value={draft.readme ?? ""}
              onChange={(e) => patch({ readme: e.target.value })}
              className="mt-1 w-full rounded-card border border-medium-tan p-2 font-mono text-sm font-normal"
            />
          </label>
        ) : null}
        <div className="mt-4 flex flex-wrap gap-3">
          {nav.onBack ? (
            <button
              type="button"
              onClick={nav.onBack}
              className="rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan"
            >
              Back to inputs
            </button>
          ) : null}
          <button type="button" onClick={onStartOver} className="underline">Start a different site</button>
        </div>
      </section>
      <section className="rounded-card border border-medium-tan bg-paper p-5 lg:sticky lg:top-4 lg:self-start">
        <h2 className="text-2xl">Preview</h2>
        <div className="mt-3"><Preview html={previewHtml} /></div>
        <h3 className="mt-4 font-bold">Publish</h3>
        <Publish
          draft={{ ...draft, html }}
          githubEnabled={githubEnabled}
          site={site}
          onPublished={(s) => { setSite(s); onPublished(); }}
        />
      </section>
    </div>
  );
}
