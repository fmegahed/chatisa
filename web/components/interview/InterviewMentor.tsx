"use client";

import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import { SpeechAnswer } from "@/components/interview/SpeechAnswer";
import { QuestionAudio } from "@/components/interview/QuestionAudio";
import { INTERVIEW_TYPES } from "@/lib/prompts/interview-mentor";
import { ModelChooser } from "@/components/ModelChooser";
import {
  HandsFreeInterview,
  handsFreeAvailable,
} from "@/components/interview/HandsFree";
import { usePauseSetting } from "@/components/interview/PauseDial";
import { ResumePicker } from "@/components/jobs/ResumePicker";
import type { ModelOption } from "@/lib/config/models";

interface PublicTurn {
  ordinal: number;
  question: string;
  topic: string | null;
  answerText: string | null;
  answerSource: string | null;
  answered: boolean;
  criteria?: { id: string; label: string; verdict: string }[];
  band?: string;
  strength?: string | null;
  improvement?: string | null;
}

interface Rollup {
  answeredCount: number;
  skippedCount: number;
  overallBand: string | null;
  byCriterion: {
    id: string;
    label: string;
    met: number;
    partly: number;
    notMet: number;
    band: string;
  }[];
  weakest: { id: string; label: string; band: string }[];
}

interface PublicInterview {
  id: string;
  status: string;
  jobTitle: string;
  interviewType: string;
  plannedQuestions: number;
  askedCount: number;
  answeredCount: number;
  hasBrief: boolean;
  turns: PublicTurn[];
  results?: {
    rollup: Rollup;
    didWell: string[];
    workOn: string[];
    overall: string;
  };
}

interface InterviewSummaryRow {
  id: string;
  jobTitle: string;
  status: string;
  plannedQuestions: number;
  askedCount: number;
  createdAt: string;
}

/** Microphone support does not change within a session. */
function subscribeNever(): () => void {
  return () => {};
}

const VERDICT_LABEL: Record<string, string> = {
  met: "Met",
  partly: "Partly",
  not_met: "Not met",
};

export function InterviewMentor(props: {
  models: ModelOption[];
  defaultModelId: string;
}) {
  const [phase, setPhase] = useState<"setup" | "running" | "results">("setup");
  const [interview, setInterview] = useState<PublicInterview | null>(null);
  const [resumable, setResumable] = useState<InterviewSummaryRow[]>([]);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  // Setup fields
  const [modelId, setModelId] = useState(props.defaultModelId);
  const [company, setCompany] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [jobUrl, setJobUrl] = useState("");
  const [interviewType, setInterviewType] = useState("mixed");
  const [questionCount, setQuestionCount] = useState(5);
  const [gradeLevel, setGradeLevel] = useState("");
  const [major, setMajor] = useState("");
  const [postingText, setPostingText] = useState("");
  const [resumeFile, setResumeFile] = useState<File | null>(null);

  // Answering
  const [answer, setAnswer] = useState("");
  const [spoken, setSpoken] = useState(false);
  // Hands-free is the default where the browser can support it, because a
  // conversation is what an interview is. It is switched off, not on, so a
  // student who cannot use it is never stuck (user decision, 2026-07-21).
  const [handsFree, setHandsFree] = useState(true);
  const [pauseMs, setPauseMs] = usePauseSetting();
  // Set when a question is shown, never during render: Date.now() is impure.
  const startedAtRef = useRef<number | null>(null);

  // Microphone support is a client-only fact. useSyncExternalStore gives the
  // server a definite false and the client the real answer, so hydration
  // matches without setting state from an effect.
  const canHandsFree = useSyncExternalStore(
    subscribeNever,
    handsFreeAvailable,
    () => false,
  );

  const headingRef = useRef<HTMLHeadingElement>(null);
  const errorRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    void (async () => {
      try {
        const res = await fetch("/api/interview");
        if (!res.ok) return;
        const body = await res.json();
        setResumable(
          (body.interviews ?? []).filter(
            (row: InterviewSummaryRow) => row.status === "in_progress",
          ),
        );
      } catch {
        // A failed history lookup must not block starting a new interview.
      }
    })();
  }, []);

  // Focus moves to the question when a new one arrives, so a screen reader
  // reads the question rather than leaving focus on the button just pressed.
  useEffect(() => {
    if (phase !== "running") return;
    headingRef.current?.focus();
    // Timing starts when the question is actually on screen, which is also the
    // only honest moment to start counting how long an answer took.
    startedAtRef.current = Date.now();
  }, [phase, interview?.askedCount]);

  useEffect(() => {
    if (error) errorRef.current?.focus();
  }, [error]);


  async function call(url: string, init?: RequestInit) {
    const res = await fetch(url, init);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error ?? "Something went wrong.");
    return body;
  }

  function applyInterview(next: PublicInterview) {
    setInterview(next);
    setAnswer("");
    setSpoken(false);
    setPhase(next.status === "completed" ? "results" : "running");
  }

  async function start() {
    // Job and resume are required now (user decision, 2026-07-21): the
    // interview is built around a real posting and a real resume.
    if (company.trim() === "" || jobTitle.trim().length < 2) {
      setError("Enter the company and the job title.");
      return;
    }
    if (!resumeFile) {
      setError("Upload your resume as a PDF. The interview is built around your real background.");
      return;
    }
    if (jobUrl.trim() === "" && postingText.trim() === "") {
      setError("Add the job description, by link or by pasting it.");
      return;
    }
    setError(null);
    setNotice(null);
    setBusy("Reading your resume and the posting, then preparing your interview.");
    try {
      const form = new FormData();
      form.append("modelId", modelId);
      form.append("interviewType", interviewType);
      form.append("company", company.trim());
      form.append("jobTitle", jobTitle.trim());
      form.append("jobUrl", jobUrl.trim());
      form.append("postingText", postingText.trim());
      form.append("questionCount", String(questionCount));
      form.append("gradeLevel", gradeLevel.trim());
      form.append("major", major.trim());
      form.append("resume", resumeFile);

      const created = await call("/api/interview", { method: "POST", body: form });
      if (created.briefRequested && !created.briefUsed) {
        setNotice(
          "Your background could not be summarised, so the questions will be general rather than tailored.",
        );
      }
      const body = await call(`/api/interview/${created.interviewId}`);
      applyInterview(body.interview);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function resume(id: string) {
    setError(null);
    setBusy("Loading your interview.");
    try {
      const body = await call(`/api/interview/${id}`);
      applyInterview(body.interview);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy("");
    }
  }

  async function discard(id: string) {
    setError(null);
    try {
      await call(`/api/interview/${id}`, { method: "DELETE" });
      // Drop it from the list immediately; nothing else references it.
      setResumable((current) => current.filter((row) => row.id !== id));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function submit(source: "typed" | "spoken" | "skipped") {
    if (!interview) return;
    setError(null);
    setBusy(
      interview.askedCount >= interview.plannedQuestions
        ? "Writing your feedback."
        : "Thinking about your next question.",
    );
    try {
      const body = await call(`/api/interview/${interview.id}/answer`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          answerText: source === "skipped" ? "" : answer,
          answerSource: source,
          answerSeconds:
            startedAtRef.current === null
              ? null
              : Math.round((Date.now() - startedAtRef.current) / 1000),
        }),
      });
      applyInterview(body.interview);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusy("");
    }
  }

  const currentTurn = interview?.turns.find((t) => !t.answered) ?? null;

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

      {phase === "setup" ? (
        <SetupPanel
          models={props.models}
          modelId={modelId}
          setModelId={setModelId}
          jobTitle={jobTitle}
          setJobTitle={setJobTitle}
          interviewType={interviewType}
          setInterviewType={setInterviewType}
          questionCount={questionCount}
          setQuestionCount={setQuestionCount}
          gradeLevel={gradeLevel}
          setGradeLevel={setGradeLevel}
          major={major}
          setMajor={setMajor}
          company={company}
          setCompany={setCompany}
          jobUrl={jobUrl}
          setJobUrl={setJobUrl}
          postingText={postingText}
          setPostingText={setPostingText}
          resumeFile={resumeFile}
          setResumeFile={setResumeFile}
          resumable={resumable}
          onResume={resume}
          onDiscard={discard}
          onStart={start}
          busy={busy !== ""}
        />
      ) : null}

      {phase === "running" && interview && currentTurn ? (
        <section className="rounded-card border border-medium-tan bg-paper p-5">
          <p role="status" className="text-sm font-bold text-dark-tan">
            Question {currentTurn.ordinal} of {interview.plannedQuestions}
          </p>

          <h2
            ref={headingRef}
            tabIndex={-1}
            className="mt-2 text-2xl leading-snug"
          >
            {currentTurn.question}
          </h2>

          {/* Keyed per question so each one gets a fresh player and last
              question's audio can never still be loaded. */}
          {handsFree && canHandsFree ? (
            // Keyed per question so each one gets a fresh player and a fresh
            // microphone session rather than resetting state in an effect.
            <div className="mt-4">
              <HandsFreeInterview
                key={`hf-${interview.id}-${currentTurn.ordinal}`}
                questionText={currentTurn.question}
                answer={answer}
                onAnswerChange={setAnswer}
                onSubmit={() => {
                  setSpoken(true);
                  void submit("spoken");
                }}
                onDisable={() => setHandsFree(false)}
                disabled={busy !== ""}
                pauseMs={pauseMs}
                onPauseChange={setPauseMs}
              />
            </div>
          ) : (
            <>
              <QuestionAudio
                key={`${interview.id}-${currentTurn.ordinal}`}
                text={currentTurn.question}
              />
              {canHandsFree ? (
                <button
                  type="button"
                  onClick={() => setHandsFree(true)}
                  className="mt-3 rounded-card border border-medium-tan px-3 py-1.5 text-sm font-bold hover:bg-light-tan"
                >
                  Switch to voice mode
                </button>
              ) : null}
            </>
          )}

          {/*
            The answer box is always present, even in voice mode: dictation
            writes into it and the student can correct anything it mis-heard
            before it is submitted.
          */}
          <div className="mt-5">
            <SpeechAnswer
              textareaId="interview-answer"
              value={answer}
              onChange={setAnswer}
              onSpokenChange={setSpoken}
              disabled={busy !== ""}
            />
          </div>

          <div className="mt-5 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => submit(spoken ? "spoken" : "typed")}
              disabled={busy !== "" || answer.trim() === ""}
              className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
            >
              Submit answer
            </button>
            <button
              type="button"
              onClick={() => submit("skipped")}
              disabled={busy !== ""}
              className="rounded-card border border-medium-tan px-4 py-2 font-bold hover:bg-light-tan"
            >
              Skip this question
            </button>
          </div>
          <p className="mt-3 text-sm text-dark-tan">
            Feedback comes at the end, so this stays close to a real interview.
            A skipped question is recorded as skipped, not marked wrong.
          </p>
        </section>
      ) : null}

      {phase === "results" && interview?.results ? (
        <Results interview={interview} onAgain={() => setPhase("setup")} />
      ) : null}
    </div>
  );
}

function SetupPanel(props: {
  models: ModelOption[];
  modelId: string;
  setModelId: (v: string) => void;
  company: string;
  setCompany: (v: string) => void;
  jobTitle: string;
  setJobTitle: (v: string) => void;
  jobUrl: string;
  setJobUrl: (v: string) => void;
  interviewType: string;
  setInterviewType: (v: string) => void;
  questionCount: number;
  setQuestionCount: (v: number) => void;
  gradeLevel: string;
  setGradeLevel: (v: string) => void;
  major: string;
  setMajor: (v: string) => void;
  postingText: string;
  setPostingText: (v: string) => void;
  resumeFile: File | null;
  setResumeFile: (f: File | null) => void;
  resumable: InterviewSummaryRow[];
  onResume: (id: string) => void;
  onDiscard: (id: string) => void;
  onStart: () => void;
  busy: boolean;
}) {
  return (
    <div className="flex flex-col gap-5">
      {props.resumable.length > 0 ? (
        <section className="rounded-card border border-medium-tan bg-light-tan p-5">
          <h2 className="text-xl">Pick up where you left off</h2>
          <ul className="mt-3 flex flex-col gap-2">
            {props.resumable.map((row) => (
              <li key={row.id} className="flex flex-wrap items-center gap-3">
                <span>
                  <strong>{row.jobTitle}</strong>, question {row.askedCount} of{" "}
                  {row.plannedQuestions}
                </span>
                <button
                  type="button"
                  onClick={() => props.onResume(row.id)}
                  className="rounded-card bg-miami-red px-3 py-1.5 text-sm font-bold text-paper hover:bg-accent-red"
                >
                  Continue
                </button>
                <button
                  type="button"
                  onClick={() => props.onDiscard(row.id)}
                  className="rounded-card border border-medium-tan px-3 py-1.5 text-sm font-bold hover:bg-paper"
                >
                  Discard
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      <section className="rounded-card border border-medium-tan bg-paper p-5">
        <h2 className="text-xl">1. What are you practising for?</h2>

        <div className="mt-4 grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="company" className="block text-sm font-bold">
              Company
            </label>
            <input
              id="company"
              type="text"
              value={props.company}
              onChange={(e) => props.setCompany(e.target.value)}
              placeholder="Northwind Analytics"
              className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
            />
          </div>
          <div>
            <label htmlFor="job-title" className="block text-sm font-bold">
              Job title
            </label>
            <input
              id="job-title"
              type="text"
              value={props.jobTitle}
              onChange={(e) => props.setJobTitle(e.target.value)}
              placeholder="Business Analytics Intern"
              className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
            />
          </div>
        </div>

        <fieldset className="mt-5">
          <legend className="text-sm font-bold">Interview type</legend>
          <div className="mt-2 flex flex-col gap-2">
            {INTERVIEW_TYPES.map((type) => (
              <label key={type.id} className="flex items-start gap-2">
                <input
                  type="radio"
                  name="interview-type"
                  value={type.id}
                  checked={props.interviewType === type.id}
                  onChange={(e) => props.setInterviewType(e.target.value)}
                  className="mt-1"
                />
                <span>
                  <strong>{type.label}</strong>
                  <span className="block text-sm text-dark-tan">
                    {type.description}
                  </span>
                </span>
              </label>
            ))}
          </div>
        </fieldset>

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="question-count" className="block text-sm font-bold">
              Number of questions
            </label>
            <select
              id="question-count"
              value={props.questionCount}
              onChange={(e) => props.setQuestionCount(Number(e.target.value))}
              className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
            >
              {[3, 4, 5, 6, 8, 10].map((n) => (
                <option key={n} value={n}>
                  {n} questions
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="mt-5">
          <ModelChooser
            options={props.models}
            value={props.modelId}
            onChange={props.setModelId}
            help="This model asks the questions and writes your feedback. Speech is handled separately."
          />
        </div>
      </section>

      <section className="rounded-card border border-medium-tan bg-paper p-5">
        <h2 className="text-xl">2. The job and your resume</h2>
        <p className="mt-1 text-sm">
          Both are required: the questions are built around the real posting and
          your real background. Only short summaries are kept, not the documents
          themselves.
        </p>

        <div className="mt-4">
          <label htmlFor="job-url" className="block text-sm font-bold">
            Link to the posting
          </label>
          <input
            id="job-url"
            type="url"
            value={props.jobUrl}
            onChange={(e) => props.setJobUrl(e.target.value)}
            placeholder="https://..."
            className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
          />
          <p className="mt-1 text-sm text-dark-tan">
            We will try to read it, including Workday, Greenhouse, Lever and most
            company career sites. For boards that block us, like LinkedIn and
            Indeed, paste the description below instead.
          </p>
        </div>

        <div className="mt-4">
          <label htmlFor="posting-text" className="block text-sm font-bold">
            Or paste the job description
          </label>
          <textarea
            id="posting-text"
            value={props.postingText}
            onChange={(e) => props.setPostingText(e.target.value)}
            rows={4}
            className="mt-1 w-full rounded-card border border-medium-tan bg-paper p-3"
            placeholder="Paste the posting here."
          />
        </div>

        <p className="mt-4 block text-sm font-bold">Your resume (PDF)</p>
        <ResumePicker file={props.resumeFile} onChoose={props.setResumeFile} />

        <div className="mt-5 grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="grade-level" className="block text-sm font-bold">
              Year (optional)
            </label>
            <input
              id="grade-level"
              type="text"
              value={props.gradeLevel}
              onChange={(e) => props.setGradeLevel(e.target.value)}
              placeholder="Junior"
              className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
            />
          </div>
          <div>
            <label htmlFor="major" className="block text-sm font-bold">
              Major
            </label>
            <input
              id="major"
              type="text"
              value={props.major}
              onChange={(e) => props.setMajor(e.target.value)}
              placeholder="Business Analytics"
              className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
            />
          </div>
        </div>

      </section>

      <div>
        <button
          type="button"
          onClick={props.onStart}
          disabled={props.busy}
          className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
        >
          Start interview
        </button>
      </div>
    </div>
  );
}

function Results(props: { interview: PublicInterview; onAgain: () => void }) {
  const results = props.interview.results!;
  const { rollup } = results;

  return (
    <div className="flex flex-col gap-5">
      <section className="rounded-card border border-medium-tan bg-paper p-5">
        <h2 className="text-2xl">How that went</h2>
        <p role="status" className="mt-2">
          You answered {rollup.answeredCount} of{" "}
          {props.interview.plannedQuestions} questions
          {rollup.skippedCount > 0 ? `, and skipped ${rollup.skippedCount}` : ""}
          .{" "}
          {rollup.overallBand ? (
            <>
              Overall, your answers were <strong>{rollup.overallBand}</strong>.
            </>
          ) : null}
        </p>
        {results.overall ? <p className="mt-3">{results.overall}</p> : null}
        {/* No percentage anywhere: ADR-016. A band is what this can honestly
            report about judged answers. */}
        <p className="mt-3 text-sm text-dark-tan">
          There is no overall number here on purpose. Automated feedback can
          tell you whether an answer was specific and well structured; it cannot
          tell you whether you would get the job.
        </p>
      </section>

      <section className="rounded-card border border-medium-tan bg-paper p-5">
        <h3 className="text-xl">Across the whole interview</h3>
        <ul className="mt-3 flex flex-col gap-2">
          {rollup.byCriterion.map((c) => (
            <li key={c.id} className="flex flex-wrap items-baseline gap-2">
              <strong>{c.label}:</strong>
              <span>{c.band}</span>
              <span className="text-sm text-dark-tan">
                ({c.met} met, {c.partly} partly, {c.notMet} not met)
              </span>
            </li>
          ))}
        </ul>
      </section>

      <div className="grid gap-5 sm:grid-cols-2">
        <section className="rounded-card border border-medium-tan bg-paper p-5">
          <h3 className="text-xl">What you did well</h3>
          <ul className="mt-3 list-disc pl-5">
            {results.didWell.map((item, i) => (
              <li key={i} className="mt-1">
                {item}
              </li>
            ))}
          </ul>
        </section>
        <section className="rounded-card border border-medium-tan bg-paper p-5">
          <h3 className="text-xl">What to work on</h3>
          <ul className="mt-3 list-disc pl-5">
            {results.workOn.map((item, i) => (
              <li key={i} className="mt-1">
                {item}
              </li>
            ))}
          </ul>
        </section>
      </div>

      <section className="rounded-card border border-medium-tan bg-paper p-5">
        <h3 className="text-xl">Question by question</h3>
        <ol className="mt-3 flex flex-col gap-5">
          {props.interview.turns.map((turn) => (
            <li key={turn.ordinal}>
              <p className="font-bold">
                {turn.ordinal}. {turn.question}
              </p>
              {turn.answered && turn.answerText ? (
                <p className="mt-1 rounded-card bg-light-tan p-3 text-sm">
                  {turn.answerText}
                </p>
              ) : (
                <p className="mt-1 text-sm italic text-dark-tan">
                  You skipped this one.
                </p>
              )}
              {turn.band ? (
                <p className="mt-2 text-sm">
                  <strong>This answer:</strong> {turn.band}
                </p>
              ) : null}
              {turn.criteria ? (
                <ul className="mt-1 text-sm text-dark-tan">
                  {turn.criteria.map((c) => (
                    <li key={c.id}>
                      {c.label}: {VERDICT_LABEL[c.verdict] ?? c.verdict}
                    </li>
                  ))}
                </ul>
              ) : null}
              {turn.strength ? (
                <p className="mt-2 text-sm">
                  <strong>Strength:</strong> {turn.strength}
                </p>
              ) : null}
              {turn.improvement ? (
                <p className="mt-1 text-sm">
                  <strong>Next time:</strong> {turn.improvement}
                </p>
              ) : null}
            </li>
          ))}
        </ol>
      </section>

      <div>
        <button
          type="button"
          onClick={props.onAgain}
          className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red"
        >
          Practise again
        </button>
      </div>
    </div>
  );
}
