"use client";

import { useEffect, useRef, useState } from "react";
import { ModelChooser } from "@/components/ModelChooser";
import { ResumePicker } from "@/components/jobs/ResumePicker";
import { DeviceResumeOffer } from "@/components/scout/DeviceResumeOffer";
import {
  CoverLetterEditor,
  GroundingBanner,
  ResumeEditor,
} from "@/components/jobs/DocumentEditor";
import type { ModelOption } from "@/lib/config/models";
import type { CoverLetterContent, ResumeContent } from "@/lib/documents/schema";

interface FlaggedClaim {
  text: string;
  sourceLine: string | null;
  verdict: string;
  note: string | null;
}

interface DocumentState {
  id: string;
  kind: "resume" | "cover_letter";
  content: ResumeContent | CoverLetterContent;
  flagged: FlaggedClaim[];
  message: string | null;
}

/**
 * JobApp Assistant, stage one: tailoring an application.
 *
 * The order is deliberate. Describe the job, give us your real resume, then get
 * a tailored draft you edit before it leaves the app. Nothing is generated
 * without the student's own resume to build on, because without it the model
 * would be inventing a career rather than presenting one.
 */
export function JobAppAssistant(props: {
  models: ModelOption[];
  defaultModelId: string;
  studentName: string;
  studentEmail: string;
  /** A Job Scout handoff: seeds the job fields so nothing is retyped.
   * Absent for every other visit; all behaviour is unchanged then. */
  initialJob?: {
    company: string;
    positionTitle: string;
    applyUrl: string;
    postingText: string;
  };
}) {
  const [step, setStep] = useState<"setup" | "documents">("setup");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [modelId, setModelId] = useState(props.defaultModelId);
  const [company, setCompany] = useState(props.initialJob?.company ?? "");
  const [positionTitle, setPositionTitle] = useState(
    props.initialJob?.positionTitle ?? "",
  );
  const [jobUrl, setJobUrl] = useState(props.initialJob?.applyUrl ?? "");
  const [postingText, setPostingText] = useState(
    props.initialJob?.postingText ?? "",
  );
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [template, setTemplate] = useState<1 | 2 | 3>(1);

  const [fullName, setFullName] = useState(props.studentName);
  const [phone, setPhone] = useState("");
  const [linkedin, setLinkedin] = useState("");
  const [recipientName, setRecipientName] = useState("");

  const [applicationId, setApplicationId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentState[]>([]);
  const [dirty, setDirty] = useState<Record<string, boolean>>({});

  const errorRef = useRef<HTMLParagraphElement>(null);
  useEffect(() => {
    if (error) errorRef.current?.focus();
  }, [error]);

  async function call(url: string, init?: RequestInit) {
    const res = await fetch(url, init);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error ?? "Something went wrong.");
    return body;
  }

  async function createApplication() {
    if (company.trim() === "" || positionTitle.trim().length < 2) {
      setError("Enter the company and the position title.");
      return;
    }
    if (!resumeFile) {
      setError(
        "Add your current resume. Everything here is built from what it already says.",
      );
      return;
    }
    setError(null);
    setNotice(null);
    setBusy("Reading your resume and the posting.");
    try {
      const form = new FormData();
      form.append("company", company.trim());
      form.append("positionTitle", positionTitle.trim());
      form.append("jobUrl", jobUrl.trim());
      form.append("postingText", postingText.trim());
      // Provenance for a Job Scout handoff, but only while the posting is
      // still the one Job Scout supplied; an edited posting is "pasted".
      if (
        props.initialJob &&
        postingText.trim() === props.initialJob.postingText.trim()
      ) {
        form.append("postingSource", "job_scout");
      }
      form.append("resume", resumeFile);

      const created = await call("/api/applications", {
        method: "POST",
        body: form,
      });
      setApplicationId(created.applicationId);
      // Says plainly where the posting came from, or why it could not be read.
      const notes = [created.postingMessage, created.resumeNote].filter(Boolean);
      setNotice(notes.length > 0 ? notes.join(" ") : null);
      if (created.postingText) setPostingText(created.postingText);
      setStep("documents");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function generate(kind: "resume" | "cover_letter") {
    if (!applicationId) return;
    setError(null);
    setBusy(
      kind === "resume"
        ? "Tailoring your resume to this job."
        : "Writing your cover letter.",
    );
    try {
      const body = await call(`/api/applications/${applicationId}/documents`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          kind,
          modelId,
          template,
          studentName: fullName.trim() || props.studentName,
          email: props.studentEmail,
          phone: phone.trim() || null,
          linkedin: linkedin.trim() || null,
          recipientName: recipientName.trim() || null,
        }),
      });
      setDocuments((current) => [
        ...current.filter((d) => d.kind !== kind),
        {
          id: body.documentId,
          kind,
          content: body.content,
          flagged: body.flagged ?? [],
          message: body.groundingMessage,
        },
      ]);
      setDirty((d) => ({ ...d, [body.documentId]: false }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function save(doc: DocumentState) {
    setBusy("Saving your edits.");
    try {
      const body = await call(`/api/documents/${doc.id}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ content: doc.content, markReviewed: true }),
      });
      // Warnings are re-checked on save, so fixing a line clears its warning
      // instead of it lingering and teaching the student to ignore warnings.
      setDocuments((current) =>
        current.map((d) =>
          d.id === doc.id ? { ...d, flagged: body.flagged ?? [] } : d,
        ),
      );
      setDirty((d) => ({ ...d, [doc.id]: false }));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy("");
    }
  }

  function updateDocument(id: string, content: ResumeContent | CoverLetterContent) {
    setDocuments((current) =>
      current.map((d) => (d.id === id ? { ...d, content } : d)),
    );
    setDirty((d) => ({ ...d, [id]: true }));
  }

  return (
    <div>
      {error ? (
        <p
          ref={errorRef}
          tabIndex={-1}
          role="alert"
          className="mb-5 rounded-card border-2 border-miami-red bg-paper p-4 font-bold text-miami-red"
        >
          {error}
        </p>
      ) : null}

      {notice ? (
        <p role="status" className="mb-5 rounded-card bg-light-tan p-4">
          {notice}
        </p>
      ) : null}

      {busy ? (
        <p role="status" className="mb-5 text-sm font-bold">
          {busy}
        </p>
      ) : null}

      {step === "setup" ? (
        <div className="flex flex-col gap-5">
          <section className="rounded-card border border-medium-tan bg-paper p-5">
            <h2 className="text-xl">1. The job</h2>

            {props.initialJob ? (
              <p role="status" className="mt-2 rounded-card bg-light-tan p-3">
                Loaded from Job Scout: {props.initialJob.positionTitle} at{" "}
                {props.initialJob.company}. Edit anything before you continue.
              </p>
            ) : null}

            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <div>
                <label htmlFor="company" className="block text-sm font-bold">
                  Company
                </label>
                <input
                  id="company"
                  value={company}
                  onChange={(e) => setCompany(e.target.value)}
                  className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
                />
              </div>
              <div>
                <label htmlFor="position" className="block text-sm font-bold">
                  Position title
                </label>
                <input
                  id="position"
                  value={positionTitle}
                  onChange={(e) => setPositionTitle(e.target.value)}
                  className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
                />
              </div>
            </div>

            <div className="mt-4">
              <label htmlFor="job-url" className="block text-sm font-bold">
                Link to the posting (optional)
              </label>
              <input
                id="job-url"
                type="url"
                value={jobUrl}
                onChange={(e) => setJobUrl(e.target.value)}
                placeholder="https://..."
                className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
              />
              <p className="mt-1 text-sm text-dark-tan">
                We will try to read it, including Workday, Greenhouse, Lever and
                most company career sites. A few boards like LinkedIn and Indeed
                block us, so for those paste the description below instead.
              </p>
            </div>

            <div className="mt-4">
              <label htmlFor="posting-text" className="block text-sm font-bold">
                Or paste the job description
              </label>
              <textarea
                id="posting-text"
                value={postingText}
                onChange={(e) => setPostingText(e.target.value)}
                rows={5}
                className="mt-1 w-full rounded-card border border-medium-tan bg-paper p-3"
              />
            </div>
          </section>

          <section className="rounded-card border border-medium-tan bg-paper p-5">
            <h2 className="text-xl">2. Your current resume</h2>
            <p className="mt-1 text-sm">
              We tailor what your resume already lists to this job. We do not add
              courses, projects, or experience you have not put on it, so upload
              a complete resume, not a bare one. Scanned resumes are fine.
            </p>
            <p className="mt-2 rounded-card bg-light-tan p-3 text-sm">
              Building or filling out your resume first? Use the Farmer School&apos;s{" "}
              <a
                href="https://miamioh.edu/fsb/student-resources/career-development/templates-and-materials.html"
                target="_blank"
                rel="noopener noreferrer"
                className="font-bold underline"
              >
                resume templates and materials
              </a>{" "}
              to add your courses, projects, and experience. The better your
              starting resume, the better we can tailor it.
            </p>

            <p className="mt-3 block text-sm font-bold">Resume PDF</p>
            <DeviceResumeOffer
              currentFile={resumeFile}
              disabled={busy !== ""}
              onUse={setResumeFile}
            />
            <ResumePicker file={resumeFile} onChoose={setResumeFile} />

            <div className="mt-4 grid gap-4 sm:grid-cols-2">
              <div>
                <label htmlFor="full-name" className="block text-sm font-bold">
                  Your name, as it should appear
                </label>
                <input
                  id="full-name"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
                />
              </div>
              <div>
                <label htmlFor="phone" className="block text-sm font-bold">
                  Phone
                </label>
                <input
                  id="phone"
                  value={phone}
                  onChange={(e) => setPhone(e.target.value)}
                  className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
                />
              </div>
            </div>

            <div className="mt-4">
              <label htmlFor="linkedin" className="block text-sm font-bold">
                LinkedIn (optional)
              </label>
              <input
                id="linkedin"
                value={linkedin}
                onChange={(e) => setLinkedin(e.target.value)}
                className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
              />
            </div>
          </section>

          <section className="rounded-card border border-medium-tan bg-paper p-5">
            <h2 className="text-xl">3. Format</h2>
            <fieldset className="mt-3">
              <legend className="text-sm font-bold">
                Farmer School resume template
              </legend>
              <div className="mt-2 flex flex-col gap-2">
                {[
                  { id: 1 as const, label: "Standard 1", note: "Arial headings, serif body. No school line." },
                  { id: 2 as const, label: "Standard 2", note: "All serif, with Farmer School of Business named." },
                  { id: 3 as const, label: "Standard 3", note: "All Arial, job title first." },
                ].map((t) => (
                  <label key={t.id} className="flex items-start gap-2">
                    <input
                      type="radio"
                      name="template"
                      checked={template === t.id}
                      onChange={() => setTemplate(t.id)}
                      className="mt-1"
                    />
                    <span>
                      <strong>{t.label}</strong>
                      <span className="block text-sm text-dark-tan">{t.note}</span>
                    </span>
                  </label>
                ))}
              </div>
            </fieldset>

            <div className="mt-5">
              <ModelChooser
                options={props.models}
                value={modelId}
                onChange={setModelId}
              />
            </div>
          </section>

          <div>
            <button
              type="button"
              onClick={createApplication}
              disabled={busy !== ""}
              className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
            >
              Continue
            </button>
          </div>
        </div>
      ) : null}

      {step === "documents" ? (
        <div className="flex flex-col gap-5">
          <section className="rounded-card border border-medium-tan bg-paper p-5">
            <h2 className="text-xl">
              {positionTitle} at {company}
            </h2>
            <div className="mt-3 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => generate("resume")}
                disabled={busy !== ""}
                className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
              >
                {documents.some((d) => d.kind === "resume")
                  ? "Rewrite the resume"
                  : "Tailor my resume"}
              </button>
              <button
                type="button"
                onClick={() => generate("cover_letter")}
                disabled={busy !== ""}
                className="rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan"
              >
                {documents.some((d) => d.kind === "cover_letter")
                  ? "Rewrite the cover letter"
                  : "Write a cover letter"}
              </button>
              <div className="mt-3">
                <label htmlFor="recipient" className="block text-sm font-bold">
                  Hiring manager name, if you know it
                </label>
                <input
                  id="recipient"
                  value={recipientName}
                  onChange={(e) => setRecipientName(e.target.value)}
                  placeholder="Ms. Cooper"
                  className="mt-1 rounded-card border border-medium-tan bg-paper px-3 py-2"
                />
              </div>
            </div>
          </section>

          {documents.map((doc) => (
            <section key={doc.id}>
              <h2 className="text-2xl">
                {doc.kind === "resume" ? "Your tailored resume" : "Your cover letter"}
              </h2>
              <p className="mt-1 mb-4 text-sm">
                Edit anything. Nothing leaves this page until you download it.
              </p>

              <GroundingBanner flagged={doc.flagged} message={doc.message} />

              {doc.kind === "resume" ? (
                <ResumeEditor
                  content={doc.content as ResumeContent}
                  flagged={doc.flagged}
                  disabled={busy !== ""}
                  onChange={(c) => updateDocument(doc.id, c)}
                />
              ) : (
                <CoverLetterEditor
                  content={doc.content as CoverLetterContent}
                  flagged={doc.flagged}
                  disabled={busy !== ""}
                  onChange={(c) => updateDocument(doc.id, c)}
                />
              )}

              <div className="mt-4 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  onClick={() => save(doc)}
                  disabled={busy !== ""}
                  className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
                >
                  Save my edits
                </button>
                <a
                  href={`/api/documents/${doc.id}/export`}
                  className="rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan"
                >
                  Download as Word
                </a>
                {dirty[doc.id] ? (
                  <p role="status" className="text-sm font-bold">
                    You have unsaved edits. Save before downloading.
                  </p>
                ) : null}
              </div>
            </section>
          ))}

          {documents.length > 0 && applicationId ? (
            // Closes the loop the schema always intended: the interview
            // starts already knowing this job (handoff, 2026-07-28).
            <p className="rounded-card border border-medium-tan bg-light-tan p-4">
              Documents ready?{" "}
              <a
                href={`/interview-mentor?application=${applicationId}`}
                className="font-bold underline"
              >
                Practice the interview for this job
              </a>{" "}
              without retyping anything.
            </p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
