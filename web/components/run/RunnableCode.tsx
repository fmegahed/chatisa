"use client";

import {
  useEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import {
  isRunSupported,
  prepareCode,
  runCode,
  type RunOutcome,
  type RunResult,
} from "@/lib/run/manager";
import type { RunnableLanguage } from "@/lib/run/languages";
import { CopyButton } from "@/components/chat/CopyButton";
import { CodeEditor } from "@/components/run/CodeEditor";
import { loadPythonIndex, loadRIndex } from "@/lib/sandbox/availability";
import {
  assessRunnability,
  baseRIndex,
  type RunnabilityVerdict,
} from "@/lib/sandbox/runnable";
import { requirementsFor } from "@/lib/sandbox/requirements";

// Stable no-op subscribe for useSyncExternalStore: browser run support does not
// change during a session, so there is nothing to subscribe to.
const subscribeNoop = () => () => {};

/** Nothing to install and nothing to look up. */
const READY: RunnabilityVerdict = {
  status: "ready",
  willInstall: [],
  impossible: [],
  message: null,
};

/**
 * Works out whether this snippet's packages can exist in the browser.
 *
 * Returns null while undecided, and null renders exactly as before: Run is
 * offered. That ordering is deliberate. A button that appears and then vanishes
 * is worse than one that appears slightly late, so the pessimistic state is never
 * the initial one.
 *
 * Two paths, and most blocks take the cheap one. The synchronous answer covers
 * SQL, snippets that import nothing, and R snippets whose every package is in the
 * shipped bundle, which is most of them: dplyr, ggplot2 and readr all qualify.
 * Only a snippet naming something outside that set pays for the availability
 * manifest, which is 253 KB for R and 113 KB for Python.
 */
function useRunnability(
  language: RunnableLanguage,
  code: string,
): RunnabilityVerdict | null {
  // Decided without a network call, when it can be. A pure derivation rather
  // than an effect that sets state: setting state synchronously from an effect
  // is both an extra render and a lint error (react-hooks/set-state-in-effect).
  const local = useMemo<RunnabilityVerdict | null>(() => {
    if (language.id === "sql") return READY;
    const required = requirementsFor(language.id, code);
    if (required.length === 0) return READY;
    if (language.id === "r") {
      const bundled = baseRIndex().mirrored;
      if (required.every((name) => bundled.has(name))) return READY;
    }
    return null;
  }, [language.id, code]);

  // Keyed by the code it was computed for, so an edit cannot briefly show the
  // previous snippet's verdict while the new one is being worked out.
  const [fetched, setFetched] = useState<
    { code: string; verdict: RunnabilityVerdict } | null
  >(null);

  useEffect(() => {
    if (local) return;
    let cancelled = false;
    void (async () => {
      const [python, r] = await Promise.all([
        language.id === "python" ? loadPythonIndex() : Promise.resolve(null),
        language.id === "r" ? loadRIndex() : Promise.resolve(null),
      ]);
      if (cancelled) return;
      setFetched({
        code,
        verdict: assessRunnability(language.id, code, {
          python,
          r: r ?? baseRIndex(),
        }),
      });
    })();
    return () => {
      cancelled = true;
    };
  }, [language.id, code, local]);

  if (local) return local;
  return fetched?.code === code ? fetched.verdict : null;
}

/**
 * Wraps a runnable code block with Run and Customize buttons next to Copy code,
 * and an output panel below the code.
 *
 * Customize turns the block into an editable copy so a student can tweak the
 * snippet and run their own version, which is the point: experimenting is where
 * the learning is. Everything runs in the student's own browser tab, so there
 * is no server surface and a package a student installs vanishes on reload. The
 * runtime loads only on the first Run, never at mount, so the initial page
 * stays light and a code block that is still streaming is never executed.
 */
export function RunnableCode(props: {
  language: RunnableLanguage;
  code: string;
  /** The rendered <pre> block, unchanged. */
  children: React.ReactNode;
}) {
  const [phase, setPhase] = useState<"idle" | "loading" | "running" | "done">(
    "idle",
  );
  const [outcome, setOutcome] = useState<RunOutcome | null>(null);
  const [editing, setEditing] = useState(false);
  // The code that actually runs. Seeded from the model's snippet; edits replace
  // it. Kept even when the editor is collapsed, so Run and Copy stay in sync
  // with what the student sees.
  const [draft, setDraft] = useState(props.code);
  // isRunSupported() reads browser-only capabilities, so it must not decide the
  // rendered output during SSR: the server would render the unsupported branch
  // (no Run button) and the client the Run button, a hydration mismatch. This
  // surfaced once the coach session began server-rendering persisted assistant
  // transcripts, which contain code blocks. useSyncExternalStore renders the
  // SSR-safe snapshot (false) during hydration and switches to the real value
  // after commit, so server and first client render agree without a
  // set-state-in-effect.
  const supported = useSyncExternalStore(subscribeNoop, isRunSupported, () => false);

  // Whether the packages this code needs can exist here. Assessed against the
  // draft, so a student who edits an impossible snippet into a possible one gets
  // the Run button back.
  const runnability = useRunnability(props.language, draft);
  const blocked = runnability?.status === "blocked";

  // Prewarm at most once per block, on the first sign the student intends to run
  // it. Never on mount: see prepareCode in lib/run/manager for why.
  const preparedRef = useRef(false);
  function prepare() {
    if (preparedRef.current || !supported || blocked) return;
    preparedRef.current = true;
    void prepareCode(
      props.language,
      draft,
      requirementsFor(props.language.id, draft),
    );
  }

  const modified = draft !== props.code;

  async function run() {
    setOutcome(null);
    // First run downloads the runtime; say so, then switch to "running".
    setPhase("loading");
    // A tick so the loading label paints before the worker blocks on init.
    await new Promise((r) => setTimeout(r, 0));
    setPhase("running");
    try {
      const result = await runCode(props.language, draft);
      setOutcome(result);
    } catch {
      setOutcome({
        ok: false,
        error: "Something went wrong starting the runtime. Try again.",
      });
    } finally {
      setPhase("done");
    }
  }

  const busy = phase === "loading" || phase === "running";

  const buttonClass =
    "rounded-card border border-medium-tan bg-paper px-2 py-1 text-xs font-bold text-ink hover:border-miami-red hover:text-miami-red disabled:cursor-not-allowed disabled:text-medium-gray";

  return (
    <figure
      className="my-3"
      // Hover or keyboard focus is the earliest honest signal that this block is
      // the one about to be run, so the packages start downloading then.
      onPointerEnter={prepare}
      onFocus={prepare}
    >
      <figcaption className="mb-1 flex flex-wrap items-center justify-end gap-2">
        {supported && !blocked ? (
          <button
            type="button"
            onClick={run}
            disabled={busy}
            className={buttonClass}
          >
            {busy ? "Running..." : `Run ${props.language.label}`}
          </button>
        ) : null}
        {supported ? (
          <button
            type="button"
            onClick={() => setEditing((e) => !e)}
            aria-expanded={editing}
            className={buttonClass}
          >
            {editing ? "Done editing" : "Customize"}
          </button>
        ) : null}
        {editing && modified ? (
          <button
            type="button"
            onClick={() => {
              setDraft(props.code);
              setOutcome(null);
            }}
            className={buttonClass}
          >
            Reset
          </button>
        ) : null}
        <CopyButton text={draft} />
      </figcaption>

      {editing ? (
        <>
          <CodeEditor
            value={draft}
            onChange={setDraft}
            languageId={props.language.id}
            label={`Editable ${props.language.label} code`}
          />
          <p className="mt-1 text-xs text-dark-tan">
            Edit freely, then Run. Your changes run only in your browser. Indent
            with spaces.
          </p>
        </>
      ) : modified ? (
        // Collapsed after editing: show the edited code, not the original, so
        // the view matches what Run and Copy use.
        <pre
          tabIndex={0}
          role="region"
          aria-label="Edited code"
          className="overflow-x-auto rounded-card border border-medium-tan bg-light-tan p-3 text-sm"
        >
          <code className="font-mono">{draft}</code>
        </pre>
      ) : (
        <pre
          tabIndex={0}
          role="region"
          aria-label="Code sample"
          className="overflow-x-auto rounded-card border border-medium-tan bg-light-tan p-3 text-sm"
        >
          {props.children}
        </pre>
      )}

      {/* Why there is no Run button. This replaces the button rather than
          sitting beside a disabled one, because the code is still perfectly
          good: it just cannot run HERE. So it says where it can, and Copy code
          is still right there. role="status" not "alert": nothing has gone
          wrong, this is the state of the world. */}
      {supported && blocked && runnability?.message ? (
        <p role="status" className="mt-1 text-xs text-dark-tan">
          {runnability.message}
        </p>
      ) : null}

      {/* A package will be fetched on the first run, so the wait is expected
          rather than a hang. Only before the first run, and only when nothing is
          blocked. */}
      {supported && !blocked && phase === "idle" && !editing && runnability?.message ? (
        <p className="mt-1 text-xs text-dark-tan">{runnability.message}</p>
      ) : null}

      {/* A one-time nudge, before the first run, on bringing your own data.
          It disappears once the block has been run, or while editing (the
          editor carries its own guidance), so it does not linger. */}
      {supported && !blocked && phase === "idle" && !editing ? (
        <p className="mt-1 text-xs text-dark-tan">{props.language.dataHint}</p>
      ) : null}

      {/* Run state is text, never colour alone. */}
      {busy ? (
        <p role="status" className="mt-1 text-xs text-dark-tan">
          {phase === "loading"
            ? `${props.language.loadingLabel}. The first run downloads it.`
            : "Running your code."}
        </p>
      ) : null}

      {!supported ? (
        <p role="status" className="mt-1 text-xs text-dark-tan">
          This browser cannot run code here. You can still copy it and run it
          elsewhere.
        </p>
      ) : null}

      {outcome ? <RunOutputPanel outcome={outcome} /> : null}
    </figure>
  );
}

function RunOutputPanel({ outcome }: { outcome: RunOutcome }) {
  if (!outcome.ok) {
    return (
      <div
        role="alert"
        className="mt-2 rounded-card border border-miami-red bg-paper p-3"
      >
        <p className="text-xs font-bold text-miami-red">Error</p>
        <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-sm text-ink">
          {outcome.error}
        </pre>
      </div>
    );
  }

  const result: RunResult = outcome.result ?? {};
  return (
    <div className="mt-2 rounded-card border border-medium-tan bg-paper p-3">
      <p className="text-xs font-bold text-dark-tan">Output</p>

      {result.table ? <ResultTable table={result.table} /> : null}

      {result.text ? (
        // Streamed program output. Polite so a screen reader is not interrupted.
        <pre
          aria-live="polite"
          className="mt-1 overflow-x-auto whitespace-pre-wrap text-sm text-ink"
        >
          {result.text}
        </pre>
      ) : null}

      {result.imageDataUrl ? (
        // A runtime-generated data URL (matplotlib or R plot); next/image
        // cannot optimise a client-produced data URL, so a plain img is right.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={result.imageDataUrl}
          alt="Plot produced by the code"
          className="mt-2 max-w-full rounded-card border border-medium-tan"
        />
      ) : null}

      {!result.table && !result.text && !result.imageDataUrl ? (
        <p className="mt-1 text-sm text-dark-tan">
          It ran, and produced no output.
        </p>
      ) : null}
    </div>
  );
}

function ResultTable({
  table,
}: {
  table: { columns: string[]; rows: Record<string, unknown>[] };
}) {
  return (
    <div
      tabIndex={0}
      role="region"
      aria-label="Query result"
      className="mt-1 overflow-x-auto"
    >
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr>
            {table.columns.map((col) => (
              <th
                key={col}
                className="border border-medium-tan bg-light-tan p-2 text-left"
              >
                {col}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {table.rows.map((row, i) => (
            <tr key={i}>
              {table.columns.map((col) => (
                <td key={col} className="border border-medium-tan p-2">
                  {formatCell(row[col])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(value: unknown): string {
  if (value === null || value === undefined) return "NULL";
  return String(value);
}
