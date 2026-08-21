"use client";

import { useRef, useState } from "react";
import type { ModelOption } from "@/lib/config/models";
import { ModelChooser } from "@/components/ModelChooser";
import {
  DEFAULT_GITIGNORE, dedupePaths, measure, rolePath, showcaseFileSet,
} from "@/lib/portfolio/files";
import { pushable, toRoutePayloadFile } from "@/lib/portfolio/intake";
import { renderShowcase } from "@/lib/portfolio/html";
import { showcaseContentSchema } from "@/lib/portfolio/content";
import type { StepProps } from "@/lib/portfolio/draft";
import { StepNav } from "../StepNav";

/**
 * The last input step of the showcase wizard, and the one that generates.
 * Three optional questions carry what the files cannot say. The model only
 * ever sees, and may only reference, the paths that will actually be
 * published, so the page never links to a file that is not in the repo.
 */
export function StoryStep({
  draft, patch, nav, models, defaultModelId,
}: StepProps & { models: ModelOption[]; defaultModelId: string }) {
  const [modelId, setModelId] = useState(defaultModelId);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const errorRef = useRef<HTMLParagraphElement>(null);
  const fail = (m: string) => {
    setError(m);
    setTimeout(() => errorRef.current?.focus(), 0);
  };
  const setPrompt = (k: keyof typeof draft.prompts, v: string) =>
    patch({ prompts: { ...draft.prompts, [k]: v } });

  // The README and the rendered page are written by the model, so measure the
  // set with them empty: the files are what can put a repository over.
  const measured = measure(
    showcaseFileSet({ html: "", readme: "", gitignore: DEFAULT_GITIGNORE, files: draft.files }),
  );

  async function generate() {
    setError(null);
    setBusy(true);
    try {
      // The paths the push will really write, collisions suffixed: the model
      // may only name files that will exist in the repository.
      const publishedPaths = dedupePaths(
        draft.files.filter((f) => f.publish && pushable(f)).map((f) => rolePath(f.role, f.name)),
      );
      const payload = {
        course: draft.course, semester: draft.semester, team: draft.team, prompts: draft.prompts,
        files: draft.files.map((f) => ({ ...toRoutePayloadFile(f), role: f.role })),
        publishedPaths,
      };
      const form = new FormData();
      form.append("modelId", modelId);
      form.append("mode", "showcase");
      form.append("payload", JSON.stringify(payload));
      const res = await fetch("/api/portfolio/generate", { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) return fail(body.error ?? "The page did not generate. Try again.");
      const content = showcaseContentSchema.parse(body.content);
      const figures = publishedPaths.filter((p) => p.startsWith("figures/"));
      const html = renderShowcase(content, {
        course: draft.course, semester: draft.semester, team: draft.team,
        repoUrl: null, figures, deliverablePaths: publishedPaths,
      });
      patch({
        content: { kind: "showcase", content },
        readme: String(body.readme ?? ""),
        skillIds: Array.isArray(body.skillIds) ? body.skillIds : [],
        html,
        step: "review",
      });
    } catch {
      fail("The page did not generate. Try again.");
    } finally {
      setBusy(false);
    }
  }

  const field = (k: keyof typeof draft.prompts, label: string, hint: string) => (
    <label className="mt-4 block font-bold">
      {label}
      <span className="block font-normal text-dark-tan">{hint}</span>
      <textarea
        value={draft.prompts[k]}
        onChange={(e) => setPrompt(k, e.target.value.slice(0, 1000))}
        rows={3}
        className="mt-1 w-full rounded-card border border-medium-tan p-2 font-normal"
      />
    </label>
  );

  return (
    <section className="rounded-card border border-medium-tan bg-paper p-5">
      <h2 className="text-2xl">Tell the story (optional)</h2>
      <p className="mt-1 text-dark-tan">
        Three short answers help the page say what the files cannot. Skip any of them.
      </p>
      {error ? (
        <p
          ref={errorRef}
          role="alert"
          tabIndex={-1}
          className="mt-3 rounded-card border-2 border-miami-red p-3 font-bold text-miami-red"
        >
          {error}
        </p>
      ) : null}
      {field("problem", "What problem were you solving?", "One or two sentences, in plain words.")}
      {field("hardest", "What was the hardest part?", "Messy data, a method that did not work, a deadline.")}
      {field("next", "What would you do next?", "If you had another month.")}
      <div className="mt-4">
        <ModelChooser options={models} value={modelId} onChange={setModelId} disabled={busy} />
      </div>
      {!measured.ok ? (
        <p className="mt-3 rounded-card border-2 border-miami-red p-3 font-bold text-miami-red">
          Your files are over the repository limits. Go back to the files step and remove or
          untick some.
        </p>
      ) : null}
      {busy ? (
        <p role="status" className="mt-2 text-dark-tan">
          Reading your files and writing the page. This takes up to a minute.
        </p>
      ) : null}
      <StepNav
        {...nav}
        canContinue={measured.ok}
        busy={busy}
        nextLabel="Generate the page"
        onNext={() => void generate()}
      />
    </section>
  );
}
