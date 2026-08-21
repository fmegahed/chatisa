"use client";

import { useRef, useState } from "react";
import type { ModelOption } from "@/lib/config/models";
import { ModelChooser } from "@/components/ModelChooser";
import { FilePick } from "@/components/scout/FilePick";
import { resizePhoto } from "@/lib/portfolio/image";
import { pushable, toRoutePayloadFile } from "@/lib/portfolio/intake";
import { CAREER_REPO, careerFileSet, measure } from "@/lib/portfolio/files";
import { renderCareer } from "@/lib/portfolio/html";
import { careerContentSchema } from "@/lib/portfolio/content";
import { normalizeUrl } from "@/lib/portfolio/links";
import type { StepProps } from "@/lib/portfolio/draft";
import { SizeMeter } from "../SizeMeter";
import { StepNav } from "../StepNav";

/**
 * The last input step of the career wizard, and the one that generates. The
 * photo is resized in the browser so the original never leaves the device,
 * the model returns content JSON, and the HTML is rendered here rather than
 * on the server. The GitHub login is unknown until publish time, so project
 * folder links are filled in then.
 */

const MAX_LINKS = 4;
/** The route's caps, mirrored here so a long field is a message, not a 400. */
const MAX_NAME = 80;
const MAX_COURSES = 30;
const MAX_PROJECT_TITLE = 80;

export function DetailsStep({
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

  async function onPhoto(file: File | null) {
    if (!file) return patch({ photo: null });
    try {
      const r = await resizePhoto(file);
      patch({ photo: { base64: r.base64, bytes: r.bytes } });
    } catch (e) {
      fail((e as Error).message);
    }
  }
  const setLink = (i: number, p: Partial<{ label: string; url: string }>) =>
    patch({ links: draft.links.map((l, j) => (j === i ? { ...l, ...p } : l)) });

  // What the repository will weigh once the photo is in. The rendered page and
  // the resume PDF are added at publish time and are small next to the files.
  const fileSet = careerFileSet({
    html: "",
    photoBase64: draft.photo?.base64 ?? null,
    resumeBase64: null,
    projects: draft.projects,
  });
  const measured = measure(fileSet);

  async function generate() {
    setError(null);
    // The route caps these; saying so here keeps a long name from coming back
    // as "the request was malformed".
    if (draft.name.trim().length > MAX_NAME) {
      return fail(`Your name is longer than ${MAX_NAME} characters. Shorten it and try again.`);
    }
    if (draft.courses.length > MAX_COURSES) {
      return fail(`Pick at most ${MAX_COURSES} classes. Go back and remove a few.`);
    }
    const typed = draft.links.filter((l) => l.label.trim() && l.url.trim());
    // A student types "linkedin.com/in/ada"; that is a link, so add the
    // scheme rather than failing the whole generation on it.
    const links: { label: string; url: string }[] = [];
    for (const [i, l] of typed.entries()) {
      const url = normalizeUrl(l.url);
      if (!url) return fail(`Link ${i + 1} is not a web address. Check it and try again.`);
      links.push({ label: l.label.trim(), url });
    }
    const projects: { slug: string; title: string; externalUrl: string | null }[] = [];
    for (const [i, p] of draft.projects.entries()) {
      if (p.title.trim().length > MAX_PROJECT_TITLE) {
        return fail(`Project ${i + 1} has a title longer than ${MAX_PROJECT_TITLE} characters. Go back and shorten it.`);
      }
      const externalUrl = p.externalUrl.trim() ? normalizeUrl(p.externalUrl) : null;
      if (p.externalUrl.trim() && !externalUrl) {
        return fail(`The link on project ${i + 1} is not a web address. Go back and check it.`);
      }
      projects.push({ slug: p.slug, title: p.title.trim(), externalUrl });
    }
    setBusy(true);
    try {
      const payload = {
        student: { name: draft.name.trim(), links },
        courses: draft.courses,
        projects: projects.map((p, i) => ({
          ...p,
          files: draft.projects[i].files.map(toRoutePayloadFile),
        })),
      };
      const form = new FormData();
      form.append("modelId", modelId);
      form.append("mode", "career");
      form.append("payload", JSON.stringify(payload));
      if (draft.resume) form.append("resume", draft.resume);
      const res = await fetch("/api/portfolio/generate", { method: "POST", body: form });
      const body = await res.json();
      if (!res.ok) return fail(body.error ?? "The site did not generate. Try again.");
      const content = careerContentSchema.parse(body.content);
      const html = renderCareer(content, {
        name: draft.name.trim(), links, hasPhoto: !!draft.photo,
        resumeLink: draft.resumeLink, login: null, repoName: CAREER_REPO,
        folders: draft.projects.filter((p) => p.files.some(pushable)).map((p) => p.slug),
      });
      patch({
        content: { kind: "career", content }, html, links, step: "review",
        // The normalized links go back into the draft so the publish pushes
        // the repaired addresses, not the bare ones.
        projects: draft.projects.map((p, i) => ({ ...p, externalUrl: projects[i].externalUrl ?? "" })),
      });
    } catch {
      fail("The site did not generate. Try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="rounded-card border border-medium-tan bg-paper p-5">
      <h2 className="text-2xl">Details</h2>
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
      <label className="mt-4 block font-bold">
        Your name
        <input
          value={draft.name}
          onChange={(e) => patch({ name: e.target.value })}
          className="mt-1 w-full rounded-card border border-medium-tan p-2 font-normal"
        />
      </label>
      <h3 className="mt-4 font-bold">Photo (optional)</h3>
      <p className="text-dark-tan">
        Resized to 512 px in your browser; the original never leaves this device.
      </p>
      <div className="mt-2">
        <FilePick
          label={draft.photo ? "Choose a different photo" : "Choose a photo"}
          accept="image/jpeg,image/png"
          fileName={draft.photo ? "photo ready" : null}
          onChange={(f) => void onPhoto(f)}
        />
      </div>
      {draft.photo ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={`data:image/jpeg;base64,${draft.photo.base64}`}
          alt="Your photo, resized"
          className="mt-2 h-24 w-24 rounded-full object-cover"
        />
      ) : null}
      <h3 className="mt-4 font-bold">Links (up to {MAX_LINKS})</h3>
      {draft.links.map((l, i) => (
        <div key={i} className="mt-2 grid gap-2 md:grid-cols-2">
          <input
            aria-label={`Link ${i + 1} label`}
            placeholder="LinkedIn"
            value={l.label}
            onChange={(e) => setLink(i, { label: e.target.value })}
            className="rounded-card border border-medium-tan p-2"
          />
          <input
            aria-label={`Link ${i + 1} URL`}
            type="url"
            placeholder="https://"
            value={l.url}
            onChange={(e) => setLink(i, { url: e.target.value })}
            className="rounded-card border border-medium-tan p-2"
          />
        </div>
      ))}
      {draft.links.length < MAX_LINKS ? (
        <button
          type="button"
          className="mt-2 underline"
          onClick={() => patch({ links: [...draft.links, { label: "", url: "" }] })}
        >
          Add a link
        </button>
      ) : null}
      <label className="mt-4 flex items-start gap-2">
        <input
          type="checkbox"
          checked={draft.resumeLink}
          onChange={(e) => patch({ resumeLink: e.target.checked })}
        />
        <span>
          Include a resume download link. Your resume PDF becomes public, including any phone
          number or address on it.
        </span>
      </label>
      <div className="mt-4">
        <ModelChooser options={models} value={modelId} onChange={setModelId} disabled={busy} />
      </div>
      {!measured.ok ? (
        <>
          <p className="mt-3 rounded-card border-2 border-miami-red p-3 font-bold text-miami-red">
            Your files are over the repository limits. Go back to the projects step to remove or
            untick some, or choose a smaller photo.
          </p>
          {/* The same numbers the projects step shows, so the gate says how far over. */}
          <SizeMeter files={fileSet} />
        </>
      ) : null}
      {busy ? (
        <p role="status" className="mt-2 text-dark-tan">
          Reading your material and writing the page. This takes up to a minute.
        </p>
      ) : null}
      <StepNav
        {...nav}
        canContinue={draft.name.trim().length > 0 && measured.ok}
        busy={busy}
        nextLabel="Generate my site"
        onNext={() => void generate()}
      />
    </section>
  );
}
