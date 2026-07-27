"use client";

import { useEffect, useRef, useState } from "react";
import { PdfPicker } from "@/components/exam/PdfPicker";
import { ModelChooser } from "@/components/ModelChooser";
import type { ModelOption } from "@/lib/config/models";

/**
 * Exam Ally, student side.
 *
 * Accessibility choices worth knowing: options use a native fieldset, legend
 * and radio group, so arrow-key navigation and high contrast work without any
 * custom code. Advancing moves focus to the question heading rather than the
 * first option, so a screen reader reads the question instead of implying a
 * pre-selection. Feedback is announced politely because it is an expected
 * result; only errors interrupt.
 */


interface Question {
  id: string;
  position: number;
  type: string;
  stem: string;
  options: string[] | null;
  topic: string;
}

interface Feedback {
  isCorrect: boolean | null;
  band: string | null;
  pointsAwarded: number | null;
  pointsPossible: number | null;
  criteria: { criterion: string; met: string; justification: string }[];
  feedback: string;
  explanation: string;
  modelAnswer: string;
  correctIndex: number | null;
  sourcePage: number;
  sourceQuote: string;
  gradedBy: string;
}

type Phase = "setup" | "generating" | "quiz" | "results";

const QUESTION_TYPES = [
  { value: "multiple_choice", label: "Multiple choice" },
  { value: "short_answer", label: "Short answer" },
  { value: "code_understanding", label: "Code understanding" },
  { value: "data_analysis", label: "Data analysis" },
];

export function ExamAlly({
  models,
  defaultModelId,
}: {
  models: ModelOption[];
  defaultModelId: string;
}) {
  const [phase, setPhase] = useState<Phase>("setup");
  const [busyMessage, setBusyMessage] = useState("");
  const [error, setError] = useState<string | null>(null);

  const [documentInfo, setDocumentInfo] = useState<{
    documentId: string;
    filename: string;
    pageCount: number;
    visionPageCount: number;
    charCount: number;
  } | null>(null);

  const [modelId, setModelId] = useState(defaultModelId);
  const [questionType, setQuestionType] = useState("multiple_choice");
  const [count, setCount] = useState(5);
  const [examMode, setExamMode] = useState<"practice" | "exam">("practice");

  const [examId, setExamId] = useState<string | null>(null);
  const [coverage, setCoverage] = useState("");
  const [shortfall, setShortfall] = useState<string | null>(null);
  const [questions, setQuestions] = useState<Question[]>([]);
  // Questions are revealed one at a time, so the total comes from the exam
  // itself rather than from how many have been handed over so far.
  const [total, setTotal] = useState(0);
  const [index, setIndex] = useState(0);
  const [choice, setChoice] = useState<number | null>(null);
  const [written, setWritten] = useState("");
  const [confidence, setConfidence] = useState("");
  const [feedback, setFeedback] = useState<Feedback | null>(null);
  const [results, setResults] = useState<ExamResults | null>(null);
  const [resumable, setResumable] = useState<
    { id: string; deliveredCount: number; currentPosition: number }[]
  >([]);

  const headingRef = useRef<HTMLHeadingElement>(null);
  const errorRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (error) errorRef.current?.focus();
  }, [error]);

  // Unfinished exams are offered back rather than quietly abandoned.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        const res = await fetch("/api/exam-prep/exams");
        if (!res.ok) return;
        const body = await res.json();
        if (cancelled) return;
        setResumable(
          body.exams.filter(
            (e: { status: string }) =>
              e.status === "ready" || e.status === "in_progress",
          ),
        );
      } catch {
        // A missing list is not worth interrupting the student for.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (phase === "quiz" && !feedback) headingRef.current?.focus();
  }, [index, phase, feedback]);

  const current = questions[index];
  const isMcq = current?.type === "multiple_choice";

  async function call(url: string, init?: RequestInit) {
    const res = await fetch(url, init);
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(body.error ?? "Something went wrong.");
    return body;
  }

  async function upload(file: File) {
    setError(null);
    setBusyMessage("Reading your document. Scanned pages take a little longer.");
    try {
      const form = new FormData();
      form.append("file", file);
      const body = await call("/api/exam-prep/documents", {
        method: "POST",
        body: form,
      });
      setDocumentInfo(body);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusyMessage("");
    }
  }

  async function generate() {
    if (!documentInfo) return;
    setError(null);
    setPhase("generating");
    setBusyMessage("Writing your questions and checking each one against your document.");
    try {
      const created = await call("/api/exam-prep/exams", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          documentId: documentInfo.documentId,
          modelId,
          questionType,
          count,
          examMode,
        }),
      });
      const exam = await call(`/api/exam-prep/exams/${created.examId}`);
      setExamId(created.examId);
      setCoverage(created.coverage);
      setShortfall(created.shortfall);
      setQuestions(exam.questions);
      setTotal(exam.deliveredCount);
      setIndex(0);
      setPhase("quiz");
    } catch (err) {
      setError((err as Error).message);
      setPhase("setup");
    } finally {
      setBusyMessage("");
    }
  }

  async function submitAnswer() {
    if (!examId || !current) return;
    setError(null);
    setBusyMessage("Checking your answer.");
    try {
      const body = await call(`/api/exam-prep/exams/${examId}/answers`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          questionId: current.id,
          selectedIndex: isMcq ? choice : null,
          responseText: isMcq ? null : written,
          confidence: confidence || null,
        }),
      });
      if (examMode === "practice") setFeedback(body as Feedback);
      else await advance(body.complete);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusyMessage("");
    }
  }

  async function advance(complete?: boolean) {
    setFeedback(null);
    setChoice(null);
    setWritten("");
    setConfidence("");
    const last = index >= total - 1;
    if (complete || last) {
      await finish();
      return;
    }
    setIndex((i) => i + 1);
    // Refresh so the next question, withheld until now, is available.
    if (examId) {
      const exam = await call(`/api/exam-prep/exams/${examId}`);
      setQuestions(exam.questions);
    }
  }

  async function finish() {
    if (!examId) return;
    setBusyMessage("Putting your results together.");
    try {
      const body = await call(`/api/exam-prep/exams/${examId}/results`, {
        method: "POST",
      });
      setResults(body);
      setPhase("results");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusyMessage("");
    }
  }

  async function resume(id: string) {
    setError(null);
    setBusyMessage("Picking up where you left off.");
    try {
      const exam = await call(`/api/exam-prep/exams/${id}`);
      setExamId(id);
      setQuestions(exam.questions);
      setTotal(exam.deliveredCount);
      setIndex(Math.min(exam.currentPosition, exam.deliveredCount - 1));
      setExamMode(exam.examMode);
      setCoverage("");
      setShortfall(null);
      setFeedback(null);
      setPhase("quiz");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusyMessage("");
    }
  }

  async function discard(id: string) {
    setError(null);
    try {
      await call(`/api/exam-prep/exams/${id}`, { method: "DELETE" });
      setResumable((current) => current.filter((e) => e.id !== id));
    } catch (err) {
      setError((err as Error).message);
    }
  }

  async function retryTopics(topics: string[]) {
    if (!examId) return;
    setError(null);
    setBusyMessage("Writing new questions on those topics.");
    try {
      const created = await call(`/api/exam-prep/exams/${examId}/retry`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ topics }),
      });
      const exam = await call(`/api/exam-prep/exams/${created.examId}`);
      setExamId(created.examId);
      setCoverage(created.coverage);
      setShortfall(created.shortfall);
      setQuestions(exam.questions);
      setTotal(exam.deliveredCount);
      setIndex(0);
      setResults(null);
      setFeedback(null);
      setPhase("quiz");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusyMessage("");
    }
  }

  function restart() {
    setPhase("setup");
    setExamId(null);
    setQuestions([]);
    setTotal(0);
    setResults(null);
    setFeedback(null);
    setIndex(0);
    setDocumentInfo(null);
  }

  return (
    <div className="flex flex-col gap-6">
      {error ? (
        <div
          ref={errorRef}
          tabIndex={-1}
          role="alert"
          className="rounded-card border-2 border-miami-red bg-paper p-4"
        >
          <h2 className="font-bold text-miami-red">That didn&apos;t work</h2>
          <p className="mt-1">{error}</p>
        </div>
      ) : null}

      <p role="status" className="text-sm text-dark-tan">
        {busyMessage}
      </p>

      {phase === "setup" && resumable.length > 0 ? (
        <section
          aria-labelledby="resume-heading"
          className="rounded-card border border-medium-tan bg-light-tan p-5"
        >
          <h2 id="resume-heading" className="text-xl">
            Pick up where you left off
          </h2>
          <ul className="mt-2 space-y-2">
            {resumable.map((e) => (
              <li key={e.id} className="flex flex-wrap items-center gap-3">
                <span>
                  An exam of {e.deliveredCount} questions, up to question{" "}
                  {Math.min(e.currentPosition + 1, e.deliveredCount)}.
                </span>
                <button
                  type="button"
                  onClick={() => void resume(e.id)}
                  className="rounded-card bg-miami-red px-3 py-1.5 text-sm font-bold text-paper hover:bg-accent-red"
                >
                  Continue this exam
                </button>
                <button
                  type="button"
                  onClick={() => void discard(e.id)}
                  className="rounded-card border border-medium-tan px-3 py-1.5 text-sm font-bold hover:bg-paper"
                >
                  Discard
                </button>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {phase === "setup" || phase === "generating" ? (
        <SetupPanel
          documentInfo={documentInfo}
          onUpload={upload}
          models={models}
          modelId={modelId}
          setModelId={setModelId}
          questionType={questionType}
          setQuestionType={setQuestionType}
          count={count}
          setCount={setCount}
          examMode={examMode}
          setExamMode={setExamMode}
          onGenerate={generate}
          busy={busyMessage.length > 0}
        />
      ) : null}

      {phase === "quiz" && current ? (
        <section aria-labelledby="question-heading">
          <ProgressBar index={index} total={total} />
          {coverage ? (
            <p className="mt-2 text-sm text-dark-tan">{coverage}</p>
          ) : null}
          {shortfall ? (
            <p role="status" className="mt-1 text-sm text-dark-tan">
              {shortfall}
            </p>
          ) : null}

          <h2
            id="question-heading"
            ref={headingRef}
            tabIndex={-1}
            className="mt-4 text-2xl"
          >
            {current.stem}
          </h2>
          <p className="mt-1 text-sm text-dark-tan">Topic: {current.topic}</p>

          {feedback ? (
            <FeedbackPanel feedback={feedback} question={current} />
          ) : (
            <AnswerForm
              question={current}
              choice={choice}
              setChoice={setChoice}
              written={written}
              setWritten={setWritten}
              confidence={confidence}
              setConfidence={setConfidence}
              onSubmit={submitAnswer}
              busy={busyMessage.length > 0}
            />
          )}

          {feedback ? (
            <button
              type="button"
              onClick={() => void advance()}
              className="mt-5 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red"
            >
              {index >= total - 1 ? "See results" : "Next question"}
            </button>
          ) : null}
        </section>
      ) : null}

      {phase === "results" && results ? (
        <ResultsPanel
          results={results}
          onRestart={restart}
          onRetryTopics={retryTopics}
        />
      ) : null}
    </div>
  );
}

function ProgressBar({ index, total }: { index: number; total: number }) {
  return (
    <div>
      <p className="text-sm font-bold">
        Question {index + 1} of {total}
      </p>
      <progress
        value={index}
        max={total}
        className="mt-1 h-2 w-full"
        aria-label={`Question ${index + 1} of ${total}`}
      />
    </div>
  );
}

function SetupPanel(props: {
  documentInfo: {
    documentId: string;
    filename: string;
    pageCount: number;
    visionPageCount: number;
    charCount: number;
  } | null;
  onUpload: (file: File) => void;
  models: ModelOption[];
  modelId: string;
  setModelId: (v: string) => void;
  questionType: string;
  setQuestionType: (v: string) => void;
  count: number;
  setCount: (v: number) => void;
  examMode: "practice" | "exam";
  setExamMode: (v: "practice" | "exam") => void;
  onGenerate: () => void;
  busy: boolean;
}) {
  return (
    <div className="flex flex-col gap-5">
      <div className="rounded-card border border-medium-tan bg-paper p-5">
        <h2 className="text-xl">1. Upload your course material</h2>
        <p className="mt-1 text-sm">
          A PDF of your notes, slides or textbook chapter. Scanned pages are
          read as images, which takes a little longer.
        </p>
        <PdfPicker
          hasDocument={props.documentInfo !== null}
          onUpload={props.onUpload}
        />
        {props.documentInfo ? (
          <p role="status" className="mt-3">
            Read <strong>{props.documentInfo.filename}</strong>:{" "}
            {props.documentInfo.pageCount} pages,{" "}
            {props.documentInfo.charCount.toLocaleString()} characters.
            {props.documentInfo.visionPageCount > 0
              ? ` ${props.documentInfo.visionPageCount} page(s) were read as images, so that text may be less exact.`
              : ""}
          </p>
        ) : null}
      </div>

      <div className="rounded-card border border-medium-tan bg-paper p-5">
        <h2 className="text-xl">2. Choose your practice</h2>
        <div className="mt-3 grid gap-4 sm:grid-cols-2">
          <div>
            <label htmlFor="exam-type" className="block text-sm font-bold">
              Question type
            </label>
            <select
              id="exam-type"
              value={props.questionType}
              onChange={(e) => props.setQuestionType(e.target.value)}
              className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
            >
              {QUESTION_TYPES.map((t) => (
                <option key={t.value} value={t.value}>
                  {t.label}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label htmlFor="exam-count" className="block text-sm font-bold">
              How many questions
            </label>
            <input
              id="exam-count"
              type="number"
              min={1}
              max={20}
              value={props.count}
              onChange={(e) => props.setCount(Number(e.target.value))}
              className="mt-1 w-full rounded-card border border-medium-tan bg-paper px-3 py-2"
            />
          </div>
          <div className="sm:col-span-2">
            <ModelChooser
              options={props.models}
              value={props.modelId}
              onChange={props.setModelId}
              help="Only models that can build a structured exam are listed here."
            />
          </div>
          <fieldset>
            <legend className="text-sm font-bold">Feedback</legend>
            <label className="mt-1 flex items-center gap-2">
              <input
                type="radio"
                name="exam-mode"
                value="practice"
                checked={props.examMode === "practice"}
                onChange={() => props.setExamMode("practice")}
              />
              After each question
            </label>
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="exam-mode"
                value="exam"
                checked={props.examMode === "exam"}
                onChange={() => props.setExamMode("exam")}
              />
              At the end, like a real exam
            </label>
          </fieldset>
        </div>

        <button
          type="button"
          onClick={props.onGenerate}
          disabled={!props.documentInfo || props.busy}
          className="mt-5 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
        >
          Build my practice exam
        </button>
      </div>
    </div>
  );
}

function AnswerForm(props: {
  question: Question;
  choice: number | null;
  setChoice: (v: number) => void;
  written: string;
  setWritten: (v: string) => void;
  confidence: string;
  setConfidence: (v: string) => void;
  onSubmit: () => void;
  busy: boolean;
}) {
  const isMcq = props.question.type === "multiple_choice";
  const ready = isMcq ? props.choice !== null : props.written.trim().length > 0;

  return (
    <form
      className="mt-4 flex flex-col gap-4"
      onSubmit={(e) => {
        e.preventDefault();
        props.onSubmit();
      }}
    >
      {isMcq && props.question.options ? (
        <fieldset className="rounded-card border border-medium-tan p-4">
          <legend className="px-1 text-sm font-bold">Choose one answer</legend>
          {props.question.options.map((option, i) => (
            <label key={option} className="mt-2 flex items-start gap-2">
              <input
                type="radio"
                name="answer"
                value={i}
                checked={props.choice === i}
                onChange={() => props.setChoice(i)}
                className="mt-1"
              />
              <span>{option}</span>
            </label>
          ))}
        </fieldset>
      ) : (
        <div>
          <label htmlFor="written-answer" className="block text-sm font-bold">
            Your answer
          </label>
          <textarea
            id="written-answer"
            rows={6}
            value={props.written}
            onChange={(e) => props.setWritten(e.target.value)}
            className="mt-1 w-full rounded-card border border-medium-tan bg-paper p-3"
          />
        </div>
      )}

      <fieldset>
        <legend className="text-sm font-bold">
          How sure are you? (optional)
        </legend>
        <div className="mt-1 flex flex-wrap gap-4">
          {[
            { value: "guessing", label: "Guessing" },
            { value: "fairly_sure", label: "Fairly sure" },
            { value: "confident", label: "Confident" },
          ].map((c) => (
            <label key={c.value} className="flex items-center gap-2">
              <input
                type="radio"
                name="confidence"
                value={c.value}
                checked={props.confidence === c.value}
                onChange={() => props.setConfidence(c.value)}
              />
              {c.label}
            </label>
          ))}
        </div>
      </fieldset>

      <div>
        <button
          type="submit"
          disabled={!ready || props.busy}
          className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
        >
          Submit answer
        </button>
      </div>
    </form>
  );
}

function FeedbackPanel({
  feedback,
  question,
}: {
  feedback: Feedback;
  question: Question;
}) {
  const correct = feedback.isCorrect === true;
  const heading =
    feedback.gradedBy === "failed"
      ? "We couldn't grade this one"
      : feedback.isCorrect === null
        ? `Your answer looks ${feedback.band}`
        : correct
          ? "Correct"
          : "Not quite";

  return (
    <div
      role="status"
      className="mt-4 rounded-card border border-medium-tan bg-light-tan p-4"
    >
      {/* Text, never colour alone, carries the result. */}
      <h3 className="text-lg font-bold">
        <span aria-hidden="true">{correct ? "✓ " : "• "}</span>
        {heading}
      </h3>

      {feedback.gradedBy === "failed" ? (
        <p className="mt-2">
          Your answer was saved, and this question will not count towards your
          results. Here is a model answer to compare against.
        </p>
      ) : null}

      {question.options && feedback.correctIndex !== null ? (
        <p className="mt-2">
          The correct answer is:{" "}
          <strong>{question.options[feedback.correctIndex]}</strong>
        </p>
      ) : null}

      {feedback.feedback ? <p className="mt-2">{feedback.feedback}</p> : null}

      {feedback.criteria.length > 0 ? (
        <ul className="mt-3 space-y-1">
          {feedback.criteria.map((c) => (
            <li key={c.criterion}>
              <strong>
                {c.met === "yes" ? "Covered" : c.met === "partial" ? "Partly covered" : "Missing"}:
              </strong>{" "}
              {c.criterion}
            </li>
          ))}
        </ul>
      ) : null}

      <p className="mt-3">{feedback.explanation}</p>

      <p className="mt-3 text-sm text-dark-tan">
        From page {feedback.sourcePage} of your document:{" "}
        <q>{feedback.sourceQuote}</q>
      </p>
    </div>
  );
}

interface ExamResults {
  exactScore: { pointsAwarded: number; pointsPossible: number } | null;
  overallBand: string | null;
  ungradedCount: number;
  topics: { topic: string; band: string }[];
  studyPlan: { topic: string; band: string; pages: number[] }[];
  questions: {
    questionId: string;
    stem: string;
    topic: string;
    isCorrect: boolean | null;
    band: string | null;
    explanation: string;
    sourcePage: number;
  }[];
}

function ResultsPanel({
  results,
  onRestart,
  onRetryTopics,
}: {
  results: ExamResults;
  onRestart: () => void;
  onRetryTopics: (topics: string[]) => void;
}) {
  const missed = results.questions.filter(
    (q) => q.isCorrect === false || (q.band && q.band !== "strong"),
  );

  return (
    <section aria-labelledby="results-heading" className="flex flex-col gap-5">
      <div>
        <h2 id="results-heading" className="text-3xl">
          Your results
        </h2>
        {results.exactScore ? (
          <p className="mt-2 text-lg">
            You answered{" "}
            <strong>
              {results.exactScore.pointsAwarded / 10} of{" "}
              {results.exactScore.pointsPossible / 10}
            </strong>{" "}
            questions correctly.
          </p>
        ) : (
          <p className="mt-2 text-lg">
            Overall, your written answers look{" "}
            <strong>{results.overallBand ?? "unscored"}</strong>. Written
            answers are described rather than scored, because automated marking
            of prose is not precise enough to put a number on.
          </p>
        )}
        {results.ungradedCount > 0 ? (
          <p className="mt-1">
            {results.ungradedCount} question(s) could not be graded and are not
            counted.
          </p>
        ) : null}
      </div>

      {results.topics.length > 0 ? (
        <div>
          <h3 className="text-xl">How you did by topic</h3>
          <table className="mt-2 w-full border-collapse text-sm">
            <caption className="sr-only">Performance by topic</caption>
            <thead>
              <tr>
                <th scope="col" className="border border-medium-tan bg-light-tan p-2 text-left">
                  Topic
                </th>
                <th scope="col" className="border border-medium-tan bg-light-tan p-2 text-left">
                  How it went
                </th>
              </tr>
            </thead>
            <tbody>
              {results.topics.map((t) => (
                <tr key={t.topic}>
                  <td className="border border-medium-tan p-2">{t.topic}</td>
                  <td className="border border-medium-tan p-2">{t.band}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}

      {results.studyPlan.length > 0 ? (
        <div>
          <h3 className="text-xl">What to review next</h3>
          <ul className="mt-2 list-disc space-y-1 pl-5">
            {results.studyPlan.map((s) => (
              <li key={s.topic}>
                <strong>{s.topic}</strong>: revisit page
                {s.pages.length === 1 ? " " : "s "}
                {s.pages.join(", ")} of your document.
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {missed.length > 0 ? (
        <div>
          <h3 className="text-xl">Questions worth another look</h3>
          {missed.map((q) => (
            <article key={q.questionId} className="mt-3 rounded-card border border-medium-tan bg-paper p-4">
              <h4 className="font-bold">{q.stem}</h4>
              <p className="mt-2">{q.explanation}</p>
              <p className="mt-2 text-sm text-dark-tan">
                See page {q.sourcePage} of your document.
              </p>
            </article>
          ))}
        </div>
      ) : null}

      <div className="flex flex-wrap gap-3">
        {results.studyPlan.length > 0 ? (
          <button
            type="button"
            onClick={() => onRetryTopics(results.studyPlan.map((s) => s.topic))}
            className="rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red"
          >
            Practise these topics again
          </button>
        ) : null}
        <button
          type="button"
          onClick={onRestart}
          className="rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red"
        >
          Start something else
        </button>
      </div>
    </section>
  );
}
