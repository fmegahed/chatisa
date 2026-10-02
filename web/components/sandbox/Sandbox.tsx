"use client";

import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import { ScrollRegion } from "@/components/a11y/ScrollRegion";
import { rovingKeyDown, useOverflows } from "@/components/sandbox/useOverflows";
import {
  describePlot,
  plotAltText,
  plotDetails,
  type PlotInfo,
} from "@/lib/run/plot-alt";
import { Group, Panel, Separator, type Layout } from "react-resizable-panels";
import { CodeEditor } from "@/components/run/CodeEditor";
import { RUNNABLE_LANGUAGES } from "@/lib/run/languages";
import type { RunnableLanguage } from "@/lib/run/languages";
import { cursorInMasked, type LanguageId } from "@/lib/sandbox/lang-structure";
import {
  createSession,
  isRunSupported,
  type RunOutcome,
  type RunSession,
  type SessionVariable,
} from "@/lib/run/manager";
import {
  getSandboxTheme,
  getServerSandboxTheme,
  setSandboxTheme,
  subscribeSandboxTheme,
  type SandboxTheme,
} from "@/lib/sandbox/theme";
import { buildSandboxContext } from "@/lib/sandbox/context";
import {
  resolveDoc,
  referenceHome,
  buildDocRequest,
  truncateDocText,
  type DocEntry,
  type DocText,
  type HelpRequest,
} from "@/lib/sandbox/help-docs";
import {
  extractVisualiseQuery,
  hasVisualise,
  renderSpecToImage,
  stripVisualise,
} from "@/lib/sandbox/ggsql-plot";
import { SandboxChat } from "@/components/sandbox/SandboxChat";
import { DataView, type GetData } from "@/components/sandbox/DataView";
import { ExportMenu } from "@/components/sandbox/ExportMenu";
import {
  downloadBytes,
  downloadText,
  exportFilename,
  exportWorkspaceFilename,
  mimeFor,
  workspaceMimeFor,
  type ExportFormat,
  type WorkspaceLanguage,
} from "@/lib/sandbox/export";
import { LimitationsNotice } from "@/components/sandbox/LimitationsNotice";
import { ImportDialog, type UploadFile } from "@/components/sandbox/ImportDialog";
import { ShortcutsDialog } from "@/components/sandbox/ShortcutsDialog";
import {
  acceptFor,
  formatFromName,
  nameFromFile,
  supportedFormats,
} from "@/lib/sandbox/upload";
import type { ModelOption } from "@/lib/config/models";
import type { CompletionSource } from "@/lib/sandbox/inline-completion";

/** localStorage key for the per-language scripts, so work survives a reload, a
 * closed tab, or a language switch. Versioned so the shape can change later. */
const SB_DRAFTS_KEY = "sb-drafts-v1";

/** localStorage key for the last language a student used, so they return to it. */
const SB_LANG_KEY = "sb-language";

/** Starter comments: what each language has available and how to add packages.
 * The runnable sample is kept separate (EXAMPLES) and dropped in on demand with
 * the "Insert example" button, so a student starts with guidance, not a
 * wall of code. */
const STARTERS: Record<string, string> = {
  python: `# Python runs in your browser. No setup needed.
# Bundled: pandas, numpy, matplotlib, scikit-learn, statsmodels, statsforecast, pyarrow, polars, seaborn, openpyxl.
# To add a pure-Python package, install it at runtime:
#   import micropip
#   await micropip.install("package-name")
# Packages that need compiling (for example pyreadr) cannot be installed here.
# Click "Insert example" above to load a sample chart.
`,
  r: `# R runs in your browser. No setup needed.
# tidyverse, readxl and janitor are pre-installed and load instantly.
# To add more packages, install them the usual way, for example:
#   install.packages("tidymodels")   # a larger download; needs a connection
# Packages without a WebAssembly build cannot be installed here.
# Click "Insert example" above to load a sample chart.
`,
  sql: `-- SQL runs in your browser. No setup needed.
-- Starts with an empty database. Build tables with CREATE TABLE and INSERT.
-- The plot uses ggsql (experimental, alpha): https://ggsql.org
-- Verified with ggsql-wasm 0.4.1 on 2026-07-22. The VISUALISE and DRAW notation may change in later versions.
-- Click "Insert example" above to load a sample chart.
`,
};

/** A runnable sample per language: the same ISA 401 vs ISA 444 dumbbell, drawn
 * with each language's tools. Inserted by the "Insert example" button. */
const EXAMPLES: Record<string, string> = {
  python: `import pandas as pd
import matplotlib.pyplot as plt

grades = pd.DataFrame({
    "student": ["Amanda", "Bill", "Cara", "Dan"],
    "ISA 401": [91, 88, 82, 76],
    "ISA 444": [85, 79, 90, 71],
})

fig, ax = plt.subplots(figsize=(7, 4))
for _, r in grades.iterrows():
    ax.plot([r["ISA 401"], r["ISA 444"]], [r["student"], r["student"]],
            color="#54585A", zorder=1)
ax.scatter(grades["ISA 401"], grades["student"], color="#C3142D", s=140, label="ISA 401", zorder=2)
ax.scatter(grades["ISA 444"], grades["student"], color="#F0B323", s=140, label="ISA 444", zorder=2)
ax.set(xlabel="Grade", ylabel="Student", title="ISA grades by student and course")
ax.legend(title="Course")
plt.tight_layout()
plt.show()`,
  r: `library(tidyverse)

grades <- tibble(
  student = c("Amanda", "Amanda", "Bill", "Bill", "Cara", "Cara", "Dan", "Dan"),
  course  = c("ISA 401", "ISA 444", "ISA 401", "ISA 444", "ISA 401", "ISA 444", "ISA 401", "ISA 444"),
  grade   = c(91, 85, 88, 79, 82, 90, 76, 71)
)

grades |>
  summarise(mean_grade = mean(grade), .by = course)

ggplot(grades, aes(x = grade, y = student)) +
  geom_line(aes(group = student), color = "gray30", linewidth = 1) +
  geom_point(aes(color = course), size = 5) +
  scale_color_manual(values = c("ISA 401" = "#C3142D", "ISA 444" = "#F0B323")) +
  labs(title = "ISA grades by student and course", x = "Grade", y = "Student", color = "Course") +
  theme_bw() +
  theme(legend.position = "bottom")`,
  sql: `CREATE TABLE grades(student TEXT, course TEXT, grade REAL);
INSERT INTO grades VALUES
  ('Amanda', 'ISA 401', 91), ('Amanda', 'ISA 444', 85),
  ('Bill',   'ISA 401', 88), ('Bill',   'ISA 444', 79),
  ('Cara',   'ISA 401', 82), ('Cara',   'ISA 444', 90),
  ('Dan',    'ISA 401', 76), ('Dan',    'ISA 444', 71);

WITH wide AS (
  SELECT student,
    MAX(CASE WHEN course = 'ISA 401' THEN grade END) AS g401,
    MAX(CASE WHEN course = 'ISA 444' THEN grade END) AS g444
  FROM grades
  GROUP BY student
)
SELECT * FROM wide
VISUALISE student AS y
DRAW segment MAPPING g401 AS x, g444 AS xend, student AS yend
DRAW point MAPPING g401 AS x, 'ISA 401' AS fill
DRAW point MAPPING g444 AS x, 'ISA 444' AS fill
LABEL x => 'Grade', y => 'Student', fill => 'Course'`,
};

interface ConsoleEntry {
  code: string;
  outcome: RunOutcome;
  /** Source runs are silent: show a note and errors only, not the echo/output. */
  silent?: boolean;
  label?: string;
  /** The run's number, for its "Output of run N" heading (#22). Notes from
   * exports and imports have none. */
  run?: number;
}

/** One drawn plot and what is known about it, for its alt text (#23). */
interface PlotEntry {
  url: string;
  info: PlotInfo;
}

/**
 * Below this width (about 125% zoom on a 1280px screen, and down to 320 CSS px
 * at 400%) the four panes stack in one column with no resizing, so the page
 * only ever scrolls vertically (WCAG 1.4.10 Reflow, #15).
 */
const STACK_QUERY = "(max-width: 1023.98px)";

function subscribeStacked(onChange: () => void) {
  const media = window.matchMedia(STACK_QUERY);
  media.addEventListener("change", onChange);
  return () => media.removeEventListener("change", onChange);
}
const getStacked = () => window.matchMedia(STACK_QUERY).matches;
const getServerStacked = () => false;

/**
 * Remembers a panel group's sizes across reloads. The Sandbox renders
 * client-only, so localStorage is available at render. Shared across languages
 * (the same key regardless of language), so a student's layout sticks.
 */
function usePersistedLayout(key: string) {
  const defaultLayout = useMemo<Layout | undefined>(() => {
    if (typeof window === "undefined") return undefined;
    try {
      const raw = window.localStorage.getItem(key);
      const parsed = raw ? JSON.parse(raw) : null;
      return parsed && typeof parsed === "object"
        ? (parsed as Layout)
        : undefined;
    } catch {
      return undefined;
    }
  }, [key]);

  const onLayoutChanged = useCallback(
    (layout: Layout) => {
      try {
        window.localStorage.setItem(key, JSON.stringify(layout));
      } catch {
        // Persisting the layout is best-effort.
      }
    },
    [key],
  );

  return { defaultLayout, onLayoutChanged };
}

/**
 * The Sandbox shell: it owns only what survives a language switch (the code
 * drafts and the theme). The actual session lives in a Workspace that is keyed
 * by language, so switching language cleanly tears down one session and starts
 * another, and the console, plots and variables reset with it, with no
 * state-resetting inside an effect.
 */
export function Sandbox(props: {
  models: ModelOption[];
  defaultModelId: string;
  userEmail: string;
}) {
  const supported = isRunSupported();
  // The last language used is remembered, so a returning student lands where they
  // left off rather than always on Python.
  const [languageId, setLanguageId] = useState<string>(() => {
    try {
      const saved = window.localStorage.getItem(SB_LANG_KEY);
      if (saved && RUNNABLE_LANGUAGES.some((l) => l.id === saved)) return saved;
    } catch {
      // best-effort
    }
    return "python";
  });
  useEffect(() => {
    try {
      window.localStorage.setItem(SB_LANG_KEY, languageId);
    } catch {
      // best-effort
    }
  }, [languageId]);
  // Each language's script is kept on the student's own device (localStorage, not
  // cookies: it is never sent to the server, so nothing leaves the browser, and it
  // holds far more than a cookie could). So closing the tab, reloading, or switching
  // language never loses their code; a returning student sees exactly what they left.
  // Saved scripts win; STARTERS fill in a language not saved yet (and any new one).
  const [drafts, setDrafts] = useState<Record<string, string>>(() => {
    const seeded = { ...STARTERS };
    try {
      const raw = window.localStorage.getItem(SB_DRAFTS_KEY);
      if (raw) return { ...seeded, ...(JSON.parse(raw) as Record<string, string>) };
    } catch {
      // best-effort; fall back to the starters
    }
    return seeded;
  });
  // Persist the scripts whenever they change (debounced by React's batching).
  useEffect(() => {
    try {
      window.localStorage.setItem(SB_DRAFTS_KEY, JSON.stringify(drafts));
    } catch {
      // best-effort (private mode, quota); the in-memory drafts still work this session
    }
  }, [drafts]);
  const theme = useSyncExternalStore(
    subscribeSandboxTheme,
    getSandboxTheme,
    getServerSandboxTheme,
  );
  const toggleTheme = useCallback(
    () => setSandboxTheme(theme === "dark" ? "light" : "dark"),
    [theme],
  );
  // The assistant's open/closed state survives language switches and reloads.
  // Reading localStorage at render is safe: the Sandbox renders client-only.
  const [chatOpen, setChatOpen] = useState(() => {
    try {
      return window.localStorage.getItem("sb-chat-open") === "1";
    } catch {
      return false;
    }
  });
  const toggleChat = useCallback(() => {
    setChatOpen((open) => {
      const next = !open;
      try {
        window.localStorage.setItem("sb-chat-open", next ? "1" : "0");
      } catch {
        // best-effort
      }
      return next;
    });
  }, []);
  // Inline AI completions, on by default, remembered across reloads.
  const [completionsOn, setCompletionsOn] = useState(() => {
    try {
      return window.localStorage.getItem("sb-completions") !== "0";
    } catch {
      return true;
    }
  });
  const toggleCompletions = useCallback(() => {
    setCompletionsOn((on) => {
      const next = !on;
      try {
        window.localStorage.setItem("sb-completions", next ? "1" : "0");
      } catch {
        // best-effort
      }
      return next;
    });
  }, []);

  const stacked = useSyncExternalStore(
    subscribeStacked,
    getStacked,
    getServerStacked,
  );
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  const language =
    RUNNABLE_LANGUAGES.find((l) => l.id === languageId) ?? RUNNABLE_LANGUAGES[0];

  if (!supported) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-16">
        <h1 className="text-3xl">Coding Studio</h1>
        <p className="mt-4 rounded-card border border-medium-tan bg-paper p-4">
          This browser cannot run code here. The Sandbox needs Web Workers and
          WebAssembly, which this browser does not provide.
        </p>
      </div>
    );
  }

  // The toolbar and notice live here, outside the language-keyed Workspace, so
  // choosing a language never unmounts the radio the student just pressed and
  // focus stays on it (#17). Text accents use --sb-accent-text (globals.css),
  // which stays readable on the dark skin.
  return (
    <div
      className={`sb-root flex flex-col bg-[var(--sb-bg)] text-[var(--sb-text)] ${stacked ? "" : "h-[calc(100vh-8.5rem)] min-h-[32rem]"}`}
      data-sb-theme={theme}
    >
      <Toolbar
        languageId={languageId}
        onLanguage={setLanguageId}
        onShortcuts={() => setShortcutsOpen(true)}
        theme={theme}
        onToggleTheme={toggleTheme}
        chatOpen={chatOpen}
        onToggleChat={toggleChat}
      />

      <LimitationsNotice languageId={languageId} />

      {shortcutsOpen ? (
        <ShortcutsDialog
          dark={theme === "dark"}
          onClose={() => setShortcutsOpen(false)}
        />
      ) : null}

      <Workspace
        key={languageId}
        language={language}
        draft={drafts[languageId] ?? ""}
        onDraft={(next) => setDrafts((d) => ({ ...d, [languageId]: next }))}
        theme={theme}
        stacked={stacked}
        models={props.models}
        defaultModelId={props.defaultModelId}
        chatOpen={chatOpen}
        onToggleChat={toggleChat}
        completionsOn={completionsOn}
        onToggleCompletions={toggleCompletions}
        userEmail={props.userEmail}
      />
    </div>
  );
}

/** One language's live session and the four-quadrant workspace around it. */
function Workspace(props: {
  language: RunnableLanguage;
  draft: string;
  onDraft: (next: string) => void;
  theme: SandboxTheme;
  /** One column, no resizing: narrow viewports and high zoom (#15). */
  stacked: boolean;
  models: ModelOption[];
  defaultModelId: string;
  chatOpen: boolean;
  onToggleChat: () => void;
  completionsOn: boolean;
  onToggleCompletions: () => void;
  userEmail: string;
}) {
  const { language, draft, theme } = props;
  const [entries, setEntries] = useState<ConsoleEntry[]>([]);
  const [plots, setPlots] = useState<PlotEntry[]>([]);
  // Counts runs in this session for the per-run output headings (#22).
  const runCountRef = useRef(0);
  const [plotIndex, setPlotIndex] = useState(0);
  // The single HELP tab's current target, and which of Plots/Help is showing in
  // the bottom-right pane. `helpTarget` holds the resolved doc entry (or the
  // reference-home fallback) plus the symbol; one slot, so there is one HELP tab.
  const [helpTarget, setHelpTarget] = useState<{
    symbol: string;
    entry: DocEntry;
    req: HelpRequest;
  } | null>(null);
  const [rightLowerTab, setRightLowerTab] = useState<"plots" | "help">("plots");
  // The runtime doc-text fetch for the current HELP target. `status` drives the pane:
  // "loading" shows a spinner note, "loaded" shows the doc region, "none" falls back
  // to the blurb + link. A monotonically increasing token discards stale replies when
  // a newer click supersedes an in-flight fetch.
  const [helpDoc, setHelpDoc] = useState<{
    status: "idle" | "loading" | "loaded" | "none";
    text?: string;
    signature?: string;
    truncated?: boolean;
  }>({ status: "idle" });
  const helpTokenRef = useRef(0);
  const [variables, setVariables] = useState<SessionVariable[]>([]);
  const [running, setRunning] = useState(false);
  // True while the runtime and its bundled packages load in the background. The
  // component is keyed by language, so this initial value is right per tab.
  const [preparing, setPreparing] = useState(
    () => language.id === "r" || language.id === "python",
  );
  // Open data-viewer tabs (variable/table names) and which tab is showing.
  const [dataTabs, setDataTabs] = useState<string[]>([]);
  const [activeTab, setActiveTab] = useState<string>("script");
  const sessionRef = useRef<RunSession | null>(null);
  // The busy guard for runs. A ref, not the `running` state, so a Ctrl+Enter
  // arriving right after a run starts is checked against the current value and
  // cannot slip past a stale closure to dispatch a second run into the busy
  // worker (which would time out and surface a spurious error).
  const runningRef = useRef(false);
  const columns = usePersistedLayout("sb-layout-columns");
  const left = usePersistedLayout("sb-layout-left");
  const right = usePersistedLayout("sb-layout-right");
  // The latest draft, so Run reads it without being re-created on each keystroke.
  const draftRef = useRef(draft);
  useEffect(() => {
    draftRef.current = draft;
  }, [draft]);

  // The session is created on mount and disposed on unmount. Because this
  // component is keyed by language, a language switch remounts it, which is how
  // the session and its panes reset, no effect ever sets state to do it.
  useEffect(() => {
    const session = createSession(language);
    sessionRef.current = session;
    // Warm the runtime and preload bundled packages in the background as soon as
    // the tab opens, so the first Run does not stall on a cold install (R) or
    // interpreter and package load (Python). SQL is fast enough to skip. The
    // "preparing" hint clears when this settles. The `cancelled` guard matters
    // under Strict Mode: a discarded first session's prewarm resolves instantly
    // when its worker is torn down, and must not clear the hint for the real one.
    let cancelled = false;
    if (language.id === "r" || language.id === "python") {
      session.prewarm().finally(() => {
        if (!cancelled) setPreparing(false);
      });
    }
    return () => {
      cancelled = true;
      session.dispose();
      sessionRef.current = null;
    };
  }, [language]);

  // Runs code in the session and updates the panes. `silent` (used by Source)
  // keeps the console uncluttered: it shows a one-line note plus any error, but
  // not the echoed code and output, while variables and plots still update.
  const execute = useCallback(
    async (code: string, opts: { silent?: boolean; label?: string } = {}) => {
      const session = sessionRef.current;
      if (!session || runningRef.current || !code.trim()) return;
      runningRef.current = true;
      setRunning(true);
      try {
        // A SQL snippet with a ggsql VISUALISE clause runs its plain-SQL part in
        // SQLite (building tables and a result table, which is also the fallback
        // if the plot fails), then the whole query is rendered by the plot engine.
        const sqlPlot = language.id === "sql" && hasVisualise(code);
        const runCode = sqlPlot ? stripVisualise(code) : code;
        const outcome = runCode.trim()
          ? await session.run(runCode)
          : { ok: true as const, result: {} };
        const run = ++runCountRef.current;
        setEntries((prev) => [
          ...prev,
          { code, outcome, silent: opts.silent, label: opts.label, run },
        ]);
        if (outcome.ok && outcome.result?.imageDataUrl) {
          const plot: PlotEntry = {
            url: outcome.result.imageDataUrl,
            info: describePlot(code, language.id, outcome.result.plotInfo),
          };
          setPlots((prev) => {
            setPlotIndex(prev.length);
            return [...prev, plot];
          });
        }
        if (outcome.ok && outcome.result?.variables) {
          setVariables(outcome.result.variables);
        }
        if (sqlPlot && outcome.ok) {
          // Only the VISUALISE statement goes to ggsql; its tables are handed
          // over separately (dumped to CSV), so the CREATE/INSERT must not.
          const plot = await session.plot(extractVisualiseQuery(code));
          if (plot.ok && plot.spec) {
            try {
              const url = await renderSpecToImage(plot.spec as Record<string, unknown>, {
                dark: theme === "dark",
              });
              const entry: PlotEntry = { url, info: describePlot(code, "sql") };
              setPlots((prev) => {
                setPlotIndex(prev.length);
                return [...prev, entry];
              });
            } catch {
              // Rendering failed: the result table above stands as the fallback.
            }
          }
        }
      } finally {
        runningRef.current = false;
        setRunning(false);
      }
    },
    [language.id, theme],
  );

  const runAll = useCallback(() => execute(draftRef.current), [execute]);
  const runLine = useCallback((code: string) => execute(code), [execute]);

  // Appends the language's sample below whatever is in the editor, so a student
  // who started with the comments gets the comments plus a runnable example.
  const insertExample = useCallback(() => {
    const example = EXAMPLES[language.id];
    if (!example) return;
    const current = draftRef.current.replace(/\s*$/, "");
    props.onDraft(current ? `${current}\n\n${example}\n` : `${example}\n`);
  }, [language.id, props]);

  // Upload Dataset: a hidden file input opens the import dialog for the chosen
  // file. Roughly 25 MB is as much as a browser tab handles comfortably.
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploadFile, setUploadFile] = useState<UploadFile | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

  const onFileChosen = useCallback(async (file: File) => {
    setUploadError(null);
    if (file.size > MAX_UPLOAD_BYTES) {
      setUploadError(
        `That file is ${(file.size / 1048576).toFixed(0)} MB. Please upload one under 25 MB.`,
      );
      return;
    }
    const format = formatFromName(file.name);
    if (!format || !supportedFormats(language.id).includes(format)) {
      setUploadError(
        `${language.label} cannot read that file type. Supported: ${supportedFormats(language.id).join(", ")}.`,
      );
      return;
    }
    const bytes = new Uint8Array(await file.arrayBuffer());
    setUploadFile({ filename: file.name, name: nameFromFile(file.name), format, bytes });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [language.id, language.label]);

  const onImported = useCallback((outcome: RunOutcome) => {
    if (outcome.result?.variables) setVariables(outcome.result.variables);
    if (outcome.result?.text) {
      setEntries((prev) => [
        ...prev,
        { code: "", outcome, silent: true, label: outcome.result?.text },
      ]);
    }
    setUploadFile(null);
  }, []);
  const source = useCallback(
    () =>
      execute(draftRef.current, {
        silent: true,
        label: "Sourced the script.",
      }),
    [execute],
  );

  const deletePlot = useCallback((index: number) => {
    setPlots((prev) => prev.filter((_, i) => i !== index));
    setPlotIndex((i) => (index < i ? i - 1 : i));
  }, []);

  const clearPlots = useCallback(() => {
    setPlots([]);
    setPlotIndex(0);
  }, []);

  // Clears only the console output. Variables, plots, the database, and the
  // session are untouched: this sets React state, never the worker. For SQL it
  // removes displayed results, messages, and errors without disconnecting.
  const clearConsole = useCallback(() => setEntries([]), []);

  // Ctrl/Cmd+Click or F1 in the editor lands here. Resolve the clicked symbol to a
  // doc entry (or the reference home if not curated), show it in the one HELP tab, and
  // select that tab. Then fetch the runtime documentation text (docstring for Python,
  // help page for R; SQL has none) without blocking the editor or stealing focus. A
  // request token guards against a stale reply when a newer click arrives first.
  const onHelp = useCallback((req: HelpRequest) => {
    const entry = resolveDoc(req) ?? referenceHome(req.language);
    const symbol = req.qualifier ? `${req.qualifier}.${req.name}` : req.name;
    setHelpTarget({ symbol, entry, req });
    setRightLowerTab("help");
    setHelpDoc({ status: "loading" });

    const token = ++helpTokenRef.current;
    const session = sessionRef.current;
    if (!session) {
      setHelpDoc({ status: "none" });
      return;
    }
    void session
      .fetchDoc(buildDocRequest(req, entry))
      .then((doc: DocText) => {
        if (helpTokenRef.current !== token) return; // superseded by a newer click
        if (doc.found && doc.text) {
          const { text, truncated } = truncateDocText(doc.text);
          setHelpDoc({ status: "loaded", text, signature: doc.signature, truncated });
        } else {
          setHelpDoc({ status: "none" });
        }
      })
      .catch(() => {
        if (helpTokenRef.current === token) setHelpDoc({ status: "none" });
      });
  }, []);

  const download = useCallback(() => {
    const ext = language.id === "python" ? "py" : language.id === "r" ? "R" : "sql";
    // Name the file after the student and the moment, e.g. megahefm-20260727-1430.py.
    const who = (props.userEmail.split("@")[0] || "sandbox").replace(
      /[^a-zA-Z0-9._-]/g,
      "",
    );
    const d = new Date();
    const p = (n: number) => String(n).padStart(2, "0");
    const stamp = `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
    const blob = new Blob([draftRef.current], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${who}-${stamp}.${ext}`;
    a.click();
    URL.revokeObjectURL(url);
  }, [language.id, props.userEmail]);

  const restart = useCallback(() => {
    sessionRef.current?.restart();
    setEntries([]);
    setPlots([]);
    setVariables([]);
    setDataTabs([]);
    setActiveTab("script");
  }, []);

  const getData = useCallback<GetData>(
    (name, offset, limit) =>
      sessionRef.current?.getData(name, offset, limit) ??
      Promise.resolve({ ok: false, error: "The session is not ready." }),
    [],
  );

  // Exports one tabular object as CSV/TSV, downloaded in the browser. Read-only:
  // it asks the worker to serialize a named object and never resets the session.
  // A failure is reported in the console without altering anything.
  const exportObject = useCallback(
    async (name: string, format: ExportFormat) => {
      const session = sessionRef.current;
      if (!session) return;
      const res = await session.exportObject(name, format);
      if (res.ok && res.text != null) {
        downloadText(
          res.text,
          exportFilename({ userEmail: props.userEmail, name, format }),
          mimeFor(format),
        );
      } else {
        setEntries((prev) => [
          ...prev,
          {
            code: "",
            outcome: { ok: false, error: res.error ?? `Could not export ${name}.` },
            silent: true,
            label: `Export failed for ${name}.`,
          },
        ]);
      }
    },
    [props.userEmail],
  );

  // Exports the whole session as one file (R .RData, SQL .sqlite, Python .zip), downloaded in
  // the browser. Read-only: it asks the worker to serialize the environment and never resets
  // the session. An empty environment shows a friendly note and downloads nothing; when Python
  // leaves out non-tabular objects, the console says which ones (they are in MANIFEST.txt too).
  const exportWorkspace = useCallback(async (names?: string[]) => {
    const session = sessionRef.current;
    if (!session) return;
    const lang = language.id as WorkspaceLanguage;
    const res = await session.exportWorkspace(names);
    if (res.ok && res.empty) {
      const message =
        lang === "sql"
          ? "Your database has no tables yet. Create a table, then export."
          : lang === "r"
            ? "Your R environment is empty. Define a variable, then export the workspace."
            : "Your Python environment has no data to export yet. Create a data frame, then export.";
      setEntries((prev) => [
        ...prev,
        { code: "", outcome: { ok: true, result: {} }, silent: true, label: message },
      ]);
      return;
    }
    if (res.ok && res.bytes) {
      downloadBytes(
        res.bytes,
        exportWorkspaceFilename({ lang }),
        workspaceMimeFor(lang),
      );
      const notes: string[] = [];
      if (lang === "python") {
        // Spec F: a pickle runs code when it is loaded, so warn on the way out.
        notes.push(
          "Saved your workspace as a pickle (.pkl) of every value that could be serialized. A pickle can run code when it is opened, so only load one from a source you trust.",
        );
      }
      if (res.skipped && res.skipped.length > 0) {
        notes.push(
          `These objects could not be saved and were left out: ${res.skipped.join(", ")}.`,
        );
      }
      if (notes.length > 0) {
        setEntries((prev) => [
          ...prev,
          {
            code: "",
            outcome: { ok: true, result: {} },
            silent: true,
            label: notes.join(" "),
          },
        ]);
      }
      return;
    }
    setEntries((prev) => [
      ...prev,
      {
        code: "",
        outcome: { ok: false, error: res.error ?? "Could not export the workspace." },
        silent: true,
        label: "Workspace export failed.",
      },
    ]);
  }, [language.id]);

  const openDataView = useCallback((name: string) => {
    setDataTabs((tabs) => (tabs.includes(name) ? tabs : [...tabs, name]));
    setActiveTab(name);
  }, []);

  const closeDataView = useCallback(
    (name: string) => {
      setDataTabs((tabs) => tabs.filter((t) => t !== name));
      setActiveTab((current) => (current === name ? "script" : current));
    },
    [],
  );

  // Inline completions ask the server, scoped to the current language.
  const completionSource = useCallback<CompletionSource>(
    async ({ prefix, suffix, signal }) => {
      // No suggestions inside a comment or string: the cursor is writing prose or
      // literal text, not code, so code completions would be noise.
      if (cursorInMasked(prefix, language.id as LanguageId)) return "";
      try {
        const res = await fetch("/api/complete", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ language: language.id, prefix, suffix }),
          signal,
        });
        if (!res.ok) return "";
        const data = (await res.json()) as { completion?: unknown };
        return typeof data.completion === "string" ? data.completion : "";
      } catch {
        return "";
      }
    },
    [language.id],
  );

  // Runtime autocomplete popup: asks the live session (its worker) for the real
  // members and names in scope, so `df.` lists the frame's actual methods.
  const completeSource = useCallback(
    (prefix: string) => {
      // The runtime popup, too, stays quiet inside a comment or string.
      if (cursorInMasked(prefix, language.id as LanguageId))
        return Promise.resolve(null);
      return sessionRef.current?.complete(prefix) ?? Promise.resolve(null);
    },
    [language.id],
  );

  // Gathered at send time by the assistant: current script, last run, variables.
  const getContext = useCallback(
    () =>
      buildSandboxContext({
        languageLabel: language.label,
        script: draftRef.current,
        lastRun: entries.length > 0 ? entries[entries.length - 1] : undefined,
        variables,
      }),
    [language.label, entries, variables],
  );

  const tablesTitle = language.id === "sql" ? "Tables" : "Environment";

  // Each pane is built once and placed in either layout below.
  const editorPane = (
    <EditorPane
      scriptTitle={`${language.label} script`}
      dataTabs={dataTabs}
      activeTab={activeTab}
      onSelectTab={setActiveTab}
      onCloseTab={closeDataView}
      onRun={() => void runAll()}
      onSource={() => void source()}
      onInsertExample={insertExample}
      onDownload={download}
      running={running}
      completionsOn={props.completionsOn}
      onToggleCompletions={props.onToggleCompletions}
    >
      {/* The editor stays mounted (just hidden) when a data tab is
          active, so its cursor and state survive tab switches. */}
      <div className={activeTab === "script" ? "h-full p-2" : "hidden"}>
        <CodeEditor
          value={draft}
          onChange={props.onDraft}
          languageId={language.id}
          label={`${language.label} code`}
          dark={theme === "dark"}
          fillHeight
          completionSource={props.completionsOn ? completionSource : undefined}
          completeSource={props.completionsOn ? completeSource : undefined}
          onRunLine={runLine}
          onRunAll={runAll}
          onSource={source}
          onHelp={onHelp}
        />
      </div>
      {activeTab !== "script" ? (
        <DataView getData={getData} name={activeTab} onExport={exportObject} />
      ) : null}
    </EditorPane>
  );
  const consolePane = (
    <ConsolePane
      entries={entries}
      running={running}
      preparing={preparing}
      label={language.label}
      banner={language.banner}
      onRun={(code) => void execute(code)}
      onClear={clearConsole}
    />
  );
  const variablesPane = (
    <VariablesPane
      variables={variables}
      language={language}
      onView={openDataView}
      onExport={exportObject}
      onExportWorkspace={exportWorkspace}
      onUpload={() => fileInputRef.current?.click()}
      onRestart={restart}
    />
  );
  const plotsPane = (
    <PlotsHelpPane
      tab={rightLowerTab}
      onTab={setRightLowerTab}
      help={helpTarget}
      helpDoc={helpDoc}
      plots={plots}
      language={language.id}
      index={plotIndex}
      onIndex={setPlotIndex}
      onDelete={deletePlot}
      onClear={clearPlots}
    />
  );
  const chat = props.chatOpen ? (
    <SandboxChat
      models={props.models}
      defaultModelId={props.defaultModelId}
      getContext={getContext}
      onClose={props.onToggleChat}
    />
  ) : null;

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept={acceptFor(language.id)}
        className="hidden"
        aria-hidden="true"
        tabIndex={-1}
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) void onFileChosen(f);
          e.target.value = ""; // allow re-picking the same file
        }}
      />
      {uploadError ? (
        <p
          role="alert"
          className="border-b border-[var(--sb-border)] bg-[var(--sb-panel)] px-3 py-1.5 text-xs text-[var(--sb-accent-text)]"
        >
          {uploadError}
        </p>
      ) : null}
      {uploadFile ? (
        <ImportDialog
          file={uploadFile}
          dark={theme === "dark"}
          previewFile={(req) =>
            sessionRef.current?.previewFile(req) ?? Promise.resolve(null)
          }
          importFile={(req) =>
            sessionRef.current?.importFile(req) ??
            Promise.resolve({ ok: false, error: "No session." })
          }
          onImported={onImported}
          onClose={() => setUploadFile(null)}
        />
      ) : null}

      {props.stacked ? (
        // One column at narrow widths and high zoom: each pane gets a fixed,
        // readable height and the page scrolls vertically only (#15).
        <div className="flex flex-col gap-2 p-2">
          <div className="h-[max(18rem,min(36rem,75vh))]">{editorPane}</div>
          <div className="h-[max(16rem,min(28rem,60vh))]">{consolePane}</div>
          <div className="h-[max(14rem,min(24rem,50vh))]">{variablesPane}</div>
          <div className="h-[max(16rem,min(28rem,60vh))]">{plotsPane}</div>
          {chat ? <div className="flex h-[max(20rem,min(36rem,75vh))]">{chat}</div> : null}
        </div>
      ) : (
        <div className="flex min-h-0 flex-1">
          <div className="min-h-0 flex-1 p-2">
            <Group
              orientation="horizontal"
              id="sb-columns"
              defaultLayout={columns.defaultLayout}
              onLayoutChanged={columns.onLayoutChanged}
              className="h-full"
            >
              <Panel id="left-col" defaultSize="55" minSize="30" className="min-h-0">
                <Group
                  orientation="vertical"
                  id="sb-left"
                  defaultLayout={left.defaultLayout}
                  onLayoutChanged={left.onLayoutChanged}
                  className="h-full"
                >
                  <Panel id="editor" defaultSize="62" minSize="20" className="min-h-0">
                    {editorPane}
                  </Panel>
                  <RowHandle label="Resize Script and Console" />
                  <Panel id="console" defaultSize="38" minSize="12" className="min-h-0">
                    {consolePane}
                  </Panel>
                </Group>
              </Panel>

              <ColHandle label="Resize the left and right columns" />

              <Panel id="right-col" defaultSize="45" minSize="22" className="min-h-0">
                <Group
                  orientation="vertical"
                  id="sb-right"
                  defaultLayout={right.defaultLayout}
                  onLayoutChanged={right.onLayoutChanged}
                  className="h-full"
                >
                  <Panel id="variables" defaultSize="50" minSize="12" className="min-h-0">
                    {variablesPane}
                  </Panel>
                  <RowHandle label={`Resize ${tablesTitle} and Plots`} />
                  <Panel id="plots" defaultSize="50" minSize="12" className="min-h-0">
                    {plotsPane}
                  </Panel>
                </Group>
              </Panel>
            </Group>
          </div>

          {chat ? (
            <div className="flex w-[360px] min-w-[300px] max-w-[42%] shrink-0 p-2 pl-0">
              {chat}
            </div>
          ) : null}
        </div>
      )}
    </>
  );
}

function Toolbar(props: {
  languageId: string;
  onLanguage: (id: string) => void;
  onShortcuts: () => void;
  theme: SandboxTheme;
  onToggleTheme: () => void;
  chatOpen: boolean;
  onToggleChat: () => void;
}) {
  const radioIds = RUNNABLE_LANGUAGES.map((l) => `sb-lang-${l.id}`);
  const checked = Math.max(
    0,
    RUNNABLE_LANGUAGES.findIndex((l) => l.id === props.languageId),
  );
  return (
    <div className="flex flex-wrap items-center gap-3 border-b border-[var(--sb-border)] bg-[var(--sb-header)] px-3 py-2">
      <h1 className="text-lg font-bold">Coding Studio</h1>

      {/* APG radio group (#18): only the checked radio is a tab stop, arrows
          move and select, Home/End jump to the ends. Focus stays here; the
          workspace below re-keys on the new language without touching it. */}
      <div
        role="radiogroup"
        aria-label="Language"
        className="flex gap-1"
        onKeyDown={(e) =>
          rovingKeyDown(e, radioIds, checked, (i) =>
            props.onLanguage(RUNNABLE_LANGUAGES[i].id),
          )
        }
      >
        {RUNNABLE_LANGUAGES.map((l, i) => {
          const active = l.id === props.languageId;
          return (
            <button
              key={l.id}
              id={radioIds[i]}
              type="button"
              role="radio"
              aria-checked={active}
              tabIndex={active ? 0 : -1}
              onClick={() => props.onLanguage(l.id)}
              className={`rounded-card px-3 py-1 text-sm font-bold ${active ? "bg-[var(--sb-accent)] text-white" : "border border-[var(--sb-border)] bg-[var(--sb-panel)] text-[var(--sb-text)] hover:border-[var(--sb-accent)]"}`}
            >
              {l.label}
            </button>
          );
        })}
      </div>

      {/* Global controls only: the assistant panel, a keyboard-shortcuts
          reference, and appearance. Everything that acts on the script or the
          session lives in the pane it affects (editor, environment, console). */}
      <div className="ml-auto flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={props.onToggleChat}
          aria-pressed={props.chatOpen}
          className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-3 py-1 text-sm font-bold hover:border-[var(--sb-accent)]"
        >
          {props.chatOpen ? "Hide assistant" : "Ask AI"}
        </button>
        <button
          type="button"
          onClick={props.onShortcuts}
          aria-label="Keyboard shortcuts"
          title="See every keyboard shortcut"
          className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] p-1.5 hover:border-[var(--sb-accent)]"
        >
          <KeyboardIcon />
        </button>
        <button
          type="button"
          onClick={props.onToggleTheme}
          aria-pressed={props.theme === "dark"}
          className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-3 py-1 text-sm font-bold hover:border-[var(--sb-accent)]"
        >
          {props.theme === "dark" ? "Light theme" : "Dark theme"}
        </button>
      </div>
    </div>
  );
}

/**
 * The top-left quadrant: a tab bar (the script plus any open data views) over
 * the editor / data grid. Like RStudio, opening a data frame adds a closable
 * tab here rather than replacing the script.
 */
function EditorPane(props: {
  scriptTitle: string;
  dataTabs: string[];
  activeTab: string;
  onSelectTab: (tab: string) => void;
  onCloseTab: (name: string) => void;
  onRun: () => void;
  onSource: () => void;
  onInsertExample: () => void;
  onDownload: () => void;
  running: boolean;
  completionsOn: boolean;
  onToggleCompletions: () => void;
  children: React.ReactNode;
}) {
  // APG tabs (#20): the tablist holds only tabs, the selected tab is the one
  // tab stop, arrows and Home/End move and select. A close button cannot live
  // inside a tablist, so the active data tab's close button sits just after
  // it; Delete on a data tab closes it too.
  const base = useId();
  const tabs = ["script", ...props.dataTabs];
  const tabIds = tabs.map((_, i) => `${base}-tab-${i}`);
  const panelId = `${base}-panel`;
  const current = Math.max(0, tabs.indexOf(props.activeTab));
  const closeTab = (name: string) => {
    props.onCloseTab(name);
    // The closed tab's controls vanish; land on the script tab, not the body.
    document.getElementById(tabIds[0])?.focus();
  };
  const tabClass = (active: boolean, mono: boolean) =>
    `border-r border-[var(--sb-border)] px-3 py-1 text-xs font-bold ${mono ? "font-mono" : ""} ${active ? "bg-[var(--sb-panel)] text-[var(--sb-text)]" : "text-[var(--sb-muted)] hover:text-[var(--sb-text)]"}`;

  return (
    <section
      aria-labelledby={tabIds[0]}
      className="flex h-full flex-col overflow-hidden rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)]"
    >
      <div className="flex flex-wrap items-stretch border-b border-[var(--sb-border)] bg-[var(--sb-header)]">
        <div
          role="tablist"
          aria-label="Editor tabs"
          className="flex flex-wrap items-stretch"
          onKeyDown={(e) => {
            if (rovingKeyDown(e, tabIds, current, (i) => props.onSelectTab(tabs[i]))) return;
            if (e.key === "Delete" && current > 0) {
              e.preventDefault();
              closeTab(tabs[current]);
            }
          }}
        >
          {tabs.map((name, i) => {
            const active = i === current;
            return (
              <button
                key={name}
                id={tabIds[i]}
                type="button"
                role="tab"
                aria-selected={active}
                aria-controls={active ? panelId : undefined}
                tabIndex={active ? 0 : -1}
                title={i > 0 ? "Press Delete to close this tab" : undefined}
                onClick={() => props.onSelectTab(name)}
                className={tabClass(active, i > 0)}
              >
                {i === 0 ? props.scriptTitle : name}
              </button>
            );
          })}
        </div>
        {current > 0 ? (
          <button
            type="button"
            aria-label={`Close ${tabs[current]}`}
            title={`Close ${tabs[current]}`}
            onClick={() => closeTab(tabs[current])}
            className="rounded px-2 text-[var(--sb-muted)] hover:text-[var(--sb-accent-text)]"
          >
            &times;
          </button>
        ) : null}
      </div>
      {/* Source toolbar: the actions that act on the script, next to the script
          itself (RStudio keeps Run/Source in the editor, not a global bar). Shown
          only for the script tab; a data view has nothing to run or source. */}
      {props.activeTab === "script" ? (
        <EditorToolbar
          onRun={props.onRun}
          onSource={props.onSource}
          onInsertExample={props.onInsertExample}
          onDownload={props.onDownload}
          running={props.running}
          completionsOn={props.completionsOn}
          onToggleCompletions={props.onToggleCompletions}
        />
      ) : null}
      <div
        id={panelId}
        role="tabpanel"
        aria-labelledby={tabIds[current]}
        className="min-h-0 flex-1 overflow-hidden"
      >
        {props.children}
      </div>
    </section>
  );
}

/** The editor's source toolbar: Insert example, Download, Suggestions toggle,
 *  Source, and the primary Run. Every control here acts on the script. */
function EditorToolbar(props: {
  onRun: () => void;
  onSource: () => void;
  onInsertExample: () => void;
  onDownload: () => void;
  running: boolean;
  completionsOn: boolean;
  onToggleCompletions: () => void;
}) {
  const mod =
    typeof navigator !== "undefined" && /Mac|iP(hone|ad)/.test(navigator.platform)
      ? "Cmd"
      : "Ctrl";
  const iconBtn =
    "rounded border border-[var(--sb-border)] p-1 text-[var(--sb-muted)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-accent-text)]";
  return (
    <div className="flex flex-wrap items-center gap-1.5 border-b border-[var(--sb-border)] bg-[var(--sb-header)] px-2 py-1">
      <button
        type="button"
        onClick={props.onDownload}
        aria-label="Download script"
        title="Download your script as a file you can save or hand in"
        className={iconBtn}
      >
        <DownloadIcon />
      </button>
      <button
        type="button"
        onClick={props.onToggleCompletions}
        aria-pressed={props.completionsOn}
        title="AI suggestions appear as you type; press Tab to accept, Escape to dismiss."
        className="rounded border border-[var(--sb-border)] px-2 py-0.5 text-xs font-bold text-[var(--sb-muted)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-text)]"
      >
        Suggestions: {props.completionsOn ? "On" : "Off"}
      </button>
      <div className="ml-auto flex items-center gap-1.5">
        <span className="hidden text-xs text-[var(--sb-muted)] xl:inline">
          {mod}+Enter: statement
        </span>
        <button
          type="button"
          onClick={props.onInsertExample}
          title="Insert a runnable sample into the editor"
          className="rounded border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2.5 py-0.5 text-xs font-bold hover:border-[var(--sb-accent)]"
        >
          Insert example
        </button>
        <button
          type="button"
          // aria-disabled, not disabled: a disabled button drops keyboard
          // focus to the page body mid-run. execute() ignores clicks while busy.
          onClick={props.running ? undefined : props.onSource}
          aria-disabled={props.running || undefined}
          title={`Source the whole script silently: updates the environment without printing to the console (${mod}+Shift+S).`}
          className="rounded border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2.5 py-0.5 text-xs font-bold hover:border-[var(--sb-accent)] aria-disabled:cursor-not-allowed aria-disabled:opacity-60"
        >
          Source
        </button>
        <button
          type="button"
          onClick={props.running ? undefined : props.onRun}
          aria-disabled={props.running || undefined}
          title={`Run the whole script, output shown in the console (${mod}+Shift+Enter). In the editor, ${mod}+Enter runs the current statement.`}
          className="rounded bg-[var(--sb-accent)] px-3 py-0.5 text-xs font-bold text-white aria-disabled:cursor-not-allowed aria-disabled:opacity-60"
        >
          {props.running ? "Running..." : "Run"}
        </button>
      </div>
    </div>
  );
}

/** A titled, scrollable quadrant. */
function Pane({
  title,
  actions,
  children,
}: {
  title: string;
  actions?: React.ReactNode;
  children: React.ReactNode;
}) {
  const headingId = useId();
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const overflows = useOverflows(bodyRef);
  return (
    // A region named by its heading, so screen reader users can jump between
    // panes (#11).
    <section
      aria-labelledby={headingId}
      className="flex h-full flex-col overflow-hidden rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)]"
    >
      <header className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1 border-b border-[var(--sb-border)] bg-[var(--sb-header)] px-3 py-1.5">
        <h2
          id={headingId}
          className="text-xs font-bold uppercase tracking-wide text-[var(--sb-muted)]"
        >
          {title}
        </h2>
        {actions}
      </header>
      {/* A tab stop only while the content overflows, so a keyboard user can
          scroll it without an empty stop otherwise (WCAG 2.1.1, #19). */}
      <div
        ref={bodyRef}
        tabIndex={overflows ? 0 : undefined}
        role={overflows ? "group" : undefined}
        aria-labelledby={overflows ? headingId : undefined}
        className="min-h-0 flex-1 overflow-auto"
      >
        {children}
      </div>
    </section>
  );
}

function ConsolePane({
  entries,
  running,
  preparing,
  label,
  banner,
  onRun,
  onClear,
}: {
  entries: ConsoleEntry[];
  running: boolean;
  preparing: boolean;
  label: string;
  /** REPL-style version header for the current language. */
  banner: string;
  onRun: (code: string) => void;
  onClear: () => void;
}) {
  const endRef = useRef<HTMLDivElement | null>(null);
  const logRef = useRef<HTMLDivElement | null>(null);
  const overflows = useOverflows(logRef);
  const headingId = useId();
  const [input, setInput] = useState("");
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [entries, running]);

  function submit() {
    const code = input.trim();
    if (!code || running) return;
    onRun(code);
    setInput("");
  }

  return (
    <section
      aria-labelledby={headingId}
      className="flex h-full flex-col overflow-hidden rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)]"
    >
      <header className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1 border-b border-[var(--sb-border)] bg-[var(--sb-header)] px-3 py-1.5">
        <h2
          id={headingId}
          className="text-xs font-bold uppercase tracking-wide text-[var(--sb-muted)]"
        >
          Console
        </h2>
        <div className="flex items-center gap-2">
          {running ? (
            <span
              role="status"
              className="flex items-center gap-1.5 text-xs font-bold text-[var(--sb-accent-text)]"
            >
              <Spinner />
              Running
            </span>
          ) : preparing ? (
            <span
              role="status"
              className="flex items-center gap-1.5 text-xs font-bold text-[var(--sb-muted)]"
            >
              <Spinner />
              Preparing {label}
            </span>
          ) : null}
          <button
            type="button"
            onClick={onClear}
            aria-label="Clear console"
            title="Clear the console output. Your variables, plots, database, and session stay as they are."
            className="rounded border border-[var(--sb-border)] px-1.5 py-0.5 text-xs font-bold text-[var(--sb-muted)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-accent-text)]"
          >
            Clear
          </button>
        </div>
      </header>

      {/* A log (#12): new runs are appended and read out politely. A tab
          stop only while it overflows (#19). */}
      <div
        ref={logRef}
        role="log"
        aria-label="Console output"
        tabIndex={overflows ? 0 : undefined}
        className="min-h-0 flex-1 overflow-auto p-3 font-mono text-sm"
      >
        {/* The version header stays put like a real REPL banner, surviving
            Clear; runs append below it. */}
        <pre className="mb-3 whitespace-pre-wrap text-[var(--sb-muted)]">
          {banner}
        </pre>
        {entries.length === 0 && !running && preparing ? (
          <p className="mb-3 text-[var(--sb-muted)]">
            Preparing the {label} runtime and its packages in the background.
            You can start typing now.
          </p>
        ) : null}
        {entries.length === 0 && !running && !preparing ? (
          <p className="mb-3 text-[var(--sb-muted)]">
            Output appears here. Run the script, or type {label} below to try
            things without changing your script.
          </p>
        ) : null}
        {entries.map((entry, i) => (
          <div key={i} className="mb-3">
            {entry.silent ? (
              // Source: a one-line note, not the echoed code and output.
              <pre className="whitespace-pre-wrap text-[var(--sb-muted)]">
                {`> ${entry.label ?? "Sourced the script."}`}
              </pre>
            ) : (
              <pre className="whitespace-pre-wrap text-[var(--sb-muted)]">
                {entry.code
                  .split("\n")
                  .map((line) => `> ${line}`)
                  .join("\n")}
              </pre>
            )}
            {/* Each run's result sits under its own heading, so screen
                reader users can jump from output to output (#22). h3: the
                pane headings are h2. Visually the console stays as it was. */}
            {!entry.outcome.ok ? (
              <>
                <h3 className="sr-only">
                  {entry.run ? `Error in run ${entry.run}` : "Error"}
                </h3>
                <pre
                  role="alert"
                  className="mt-1 whitespace-pre-wrap text-[var(--sb-accent-text)]"
                >
                  {entry.outcome.error}
                </pre>
              </>
            ) : entry.silent ? null : (
              <>
                <h3 className="sr-only">
                  {entry.run ? `Output of run ${entry.run}` : "Output"}
                </h3>
                <ResultBody outcome={entry.outcome} run={entry.run} />
              </>
            )}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      {/* An interactive prompt: try things here without touching the script. */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
        className="flex items-center gap-2 border-t border-[var(--sb-border)] px-3 py-1.5 font-mono text-sm"
      >
        <span aria-hidden="true" className="text-[var(--sb-accent-text)]">
          &gt;
        </span>
        <label htmlFor="sb-console-input" className="sr-only">
          Console input
        </label>
        <input
          id="sb-console-input"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          // readOnly, not disabled, while a run is going: disabling the
          // focused input would drop keyboard focus to the page body.
          readOnly={running}
          aria-busy={running || undefined}
          spellCheck={false}
          autoComplete="off"
          placeholder={`Try ${label} here`}
          className="min-w-0 flex-1 bg-transparent text-[var(--sb-text)] placeholder:text-[var(--sb-muted)]"
        />
      </form>
    </section>
  );
}

function ResultBody({ outcome, run }: { outcome: RunOutcome; run?: number }) {
  const result = outcome.result;
  if (!result) return null;
  return (
    <div className="mt-1">
      {result.table ? (
        <ScrollRegion
          label={run ? `Result table of run ${run}` : "Result table"}
          className="overflow-x-auto"
        >
          <table className="border-collapse text-xs">
            <thead>
              <tr>
                {result.table.columns.map((c) => (
                  <th
                    key={c}
                    className="border border-[var(--sb-border)] px-2 py-1 text-left"
                  >
                    {c}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.table.rows.map((row, i) => (
                <tr key={i}>
                  {result.table!.columns.map((c) => (
                    <td
                      key={c}
                      className="border border-[var(--sb-border)] px-2 py-1"
                    >
                      {row[c] === null || row[c] === undefined
                        ? "NULL"
                        : String(row[c])}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollRegion>
      ) : null}
      {result.text ? (
        <pre className="whitespace-pre-wrap text-[var(--sb-text)]">
          {result.text}
        </pre>
      ) : null}
      {result.imageDataUrl ? (
        <p className="text-[var(--sb-muted)]">(plot shown in the Plots pane)</p>
      ) : null}
    </div>
  );
}

function VariablesPane({
  variables,
  language,
  onView,
  onExport,
  onExportWorkspace,
  onUpload,
  onRestart,
}: {
  variables: SessionVariable[];
  language: RunnableLanguage;
  onView: (name: string) => void;
  onExport: (name: string, format: ExportFormat) => void;
  onExportWorkspace: (names?: string[]) => void;
  onUpload: () => void;
  onRestart: () => void;
}) {
  const title = language.id === "sql" ? "Tables" : "Environment";
  const cleared = language.id === "sql" ? "tables" : "variables";
  // Row selection for "export selected" (5e). A name that no longer exists (after a
  // Restart or a re-run) is ignored, so the selection self-heals.
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const selectedExisting = variables
    .map((v) => v.name)
    .filter((n) => selected.has(n));
  const toggle = (name: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  // SQL exports the whole database as one .sqlite; R an .RData image; Python a zip of CSVs.
  const wsLabel = language.id === "sql" ? "Export database" : "Export workspace";
  const wsTitle =
    language.id === "sql"
      ? "Download the whole database as one .sqlite file you can re-open later"
      : language.id === "r"
        ? "Download your whole R environment as one .RData file you can load later"
        : "Download your serializable values as one .pkl (pickle) file you can load later. Only open pickle files you trust.";
  return (
    <Pane
      title={title}
      actions={
        <span className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={onUpload}
            title="Upload a data file (CSV, JSON, and more) into your session"
            className="rounded border border-[var(--sb-border)] px-2 py-0.5 text-xs font-bold text-[var(--sb-text)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-accent-text)]"
          >
            Upload Dataset
          </button>
          {selectedExisting.length > 0 ? (
            <button
              type="button"
              onClick={() => onExportWorkspace(selectedExisting)}
              title={`Download only the ${selectedExisting.length} selected item${selectedExisting.length === 1 ? "" : "s"} as one file`}
              className="rounded border border-[var(--sb-accent)] px-2 py-0.5 text-xs font-bold text-[var(--sb-accent-text)] hover:bg-[var(--sb-accent)] hover:text-white"
            >
              Export selected ({selectedExisting.length})
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => onExportWorkspace()}
            aria-label={wsLabel}
            title={wsTitle}
            className="rounded border border-[var(--sb-border)] px-2 py-0.5 text-xs font-bold text-[var(--sb-text)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-accent-text)]"
          >
            {wsLabel}
          </button>
          <button
            type="button"
            onClick={onRestart}
            title={`Restart the session: clears all ${cleared} and starts a fresh runtime`}
            className="rounded border border-[var(--sb-border)] px-2 py-0.5 text-xs font-bold text-[var(--sb-muted)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-accent-text)]"
          >
            Restart
          </button>
        </span>
      }
    >
      <div className="p-2">
        {variables.length === 0 ? (
          <p className="p-1 text-sm text-[var(--sb-muted)]">
            {language.id === "sql"
              ? "Tables you create appear here."
              : "Variables you define appear here."}
          </p>
        ) : (
          <table className="w-full border-collapse text-sm">
            <thead>
              <tr className="text-left text-xs uppercase text-[var(--sb-muted)]">
                <th className="px-1 py-1">
                  <span className="sr-only">Select</span>
                </th>
                <th className="px-2 py-1">Name</th>
                <th className="px-2 py-1">Type</th>
                <th className="px-2 py-1">Value</th>
                <th className="px-2 py-1" />
              </tr>
            </thead>
            <tbody>
              {variables.map((v) => {
                const viewable = (v.columns?.length ?? 0) > 0;
                return (
                  <tr
                    key={v.name}
                    className="border-t border-[var(--sb-border)]"
                  >
                    <td className="px-1 py-1">
                      <input
                        type="checkbox"
                        checked={selected.has(v.name)}
                        onChange={() => toggle(v.name)}
                        aria-label={`Select ${v.name} to export`}
                      />
                    </td>
                    <td className="px-2 py-1 font-mono font-bold">{v.name}</td>
                    <td className="px-2 py-1 font-mono text-[var(--sb-muted)]">
                      {v.type}
                    </td>
                    <td className="px-2 py-1 font-mono">{v.info}</td>
                    <td className="px-1 py-1 text-right">
                      <span className="inline-flex items-center gap-1">
                        {viewable ? (
                          <button
                            type="button"
                            onClick={() => onView(v.name)}
                            aria-label={`View ${v.name} in a table`}
                            title={`View ${v.name}`}
                            className="rounded border border-[var(--sb-border)] px-1 text-[var(--sb-muted)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-accent-text)]"
                          >
                            <TableIcon />
                          </button>
                        ) : null}
                        {viewable ? (
                          <ExportMenu
                            name={v.name}
                            onExport={(format) => onExport(v.name, format)}
                          />
                        ) : null}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>
    </Pane>
  );
}

/** A small spinning ring; reduced-motion users see a static ring. */
function Spinner() {
  return (
    <svg
      viewBox="0 0 16 16"
      width="12"
      height="12"
      fill="none"
      className="motion-safe:animate-spin"
      aria-hidden="true"
    >
      <circle cx="8" cy="8" r="6" stroke="currentColor" strokeOpacity="0.25" strokeWidth="2" />
      <path d="M14 8a6 6 0 0 0-6-6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}

/** A small grid glyph for the "view data" affordance. */
function TableIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      aria-hidden="true"
    >
      <rect x="1.5" y="2.5" width="13" height="11" rx="1" />
      <line x1="1.5" y1="6" x2="14.5" y2="6" />
      <line x1="6" y1="6" x2="6" y2="13.5" />
      <line x1="10" y1="6" x2="10" y2="13.5" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      width="14"
      height="14"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <line x1="8" y1="2.5" x2="8" y2="10" />
      <polyline points="4.5,6.5 8,10 11.5,6.5" />
      <line x1="3" y1="13" x2="13" y2="13" />
    </svg>
  );
}

function KeyboardIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      width="15"
      height="15"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.3"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <rect x="1.5" y="4" width="13" height="8" rx="1.5" />
      <line x1="4" y1="6.5" x2="4" y2="6.5" />
      <line x1="6.5" y1="6.5" x2="6.5" y2="6.5" />
      <line x1="9" y1="6.5" x2="9" y2="6.5" />
      <line x1="11.5" y1="6.5" x2="11.5" y2="6.5" />
      <line x1="5" y1="9.5" x2="11" y2="9.5" />
    </svg>
  );
}

function PlotsHelpPane({
  tab,
  onTab,
  help,
  helpDoc,
  plots,
  language,
  index,
  onIndex,
  onDelete,
  onClear,
}: {
  tab: "plots" | "help";
  onTab: (tab: "plots" | "help") => void;
  help: { symbol: string; entry: DocEntry; req: HelpRequest } | null;
  helpDoc: { status: "idle" | "loading" | "loaded" | "none"; text?: string; signature?: string; truncated?: boolean };
  plots: PlotEntry[];
  language: string;
  index: number;
  onIndex: (i: number) => void;
  onDelete: (index: number) => void;
  onClear: () => void;
}) {
  const has = plots.length > 0;
  const safeIndex = Math.min(index, plots.length - 1);
  const plot = has ? plots[safeIndex] : null;
  const btn =
    "rounded border border-[var(--sb-border)] px-1.5 py-0.5 font-bold hover:border-[var(--sb-accent)] aria-disabled:opacity-40";
  const tabClass = (active: boolean) =>
    `px-3 py-1.5 text-xs font-bold uppercase tracking-wide ${active ? "text-[var(--sb-text)]" : "text-[var(--sb-muted)] hover:text-[var(--sb-text)]"}`;

  // APG tabs, as in the editor (#20): one tab stop, arrows move and select.
  const base = useId();
  const tabs: ("plots" | "help")[] = ["plots", "help"];
  const tabIds = [`${base}-plots`, `${base}-help`];
  const panelId = `${base}-panel`;
  const current = tabs.indexOf(tab);
  const panelRef = useRef<HTMLDivElement | null>(null);
  const overflows = useOverflows(panelRef);

  // Deleting the last plot, or clearing them all, removes the button that had
  // focus; put focus on the Plots tab rather than losing it to the page.
  const focusPlotsTab = () => document.getElementById(tabIds[0])?.focus();

  function exportCurrent() {
    if (!plot) return;
    const a = document.createElement("a");
    a.href = plot.url;
    a.download = `plot-${safeIndex + 1}.png`;
    a.click();
  }

  return (
    <section
      aria-labelledby={tabIds[current]}
      className="flex h-full flex-col overflow-hidden rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)]"
    >
      <header className="flex flex-wrap items-center justify-between gap-y-1 border-b border-[var(--sb-border)] bg-[var(--sb-header)]">
        <div
          role="tablist"
          aria-label="Plots and Help"
          className="flex items-stretch"
          onKeyDown={(e) => rovingKeyDown(e, tabIds, current, (i) => onTab(tabs[i]))}
        >
          {tabs.map((t, i) => (
            <button
              key={t}
              id={tabIds[i]}
              type="button"
              role="tab"
              aria-selected={tab === t}
              aria-controls={tab === t ? panelId : undefined}
              tabIndex={tab === t ? 0 : -1}
              onClick={() => onTab(t)}
              className={tabClass(tab === t)}
            >
              {t === "plots" ? "Plots" : "Help"}
            </button>
          ))}
        </div>
        {tab === "plots" && has ? (
          <span className="flex flex-wrap items-center gap-1.5 px-2 text-xs text-[var(--sb-muted)]">
            {/* aria-disabled, not disabled: reaching the first or last plot
                must not drop focus from the button just pressed. */}
            <button
              type="button"
              onClick={() => {
                if (safeIndex > 0) onIndex(safeIndex - 1);
              }}
              aria-disabled={safeIndex <= 0 || undefined}
              className={btn}
              aria-label="Previous plot"
            >
              &#8249;
            </button>
            <span className="tabular-nums" aria-hidden="true">
              {safeIndex + 1} / {plots.length}
            </span>
            <span className="sr-only">
              Plot {safeIndex + 1} of {plots.length}
            </span>
            <button
              type="button"
              onClick={() => {
                if (safeIndex < plots.length - 1) onIndex(safeIndex + 1);
              }}
              aria-disabled={safeIndex >= plots.length - 1 || undefined}
              className={btn}
              aria-label="Next plot"
            >
              &#8250;
            </button>
            <button
              type="button"
              onClick={exportCurrent}
              aria-label="Export plot as PNG"
              className={btn}
            >
              Export
            </button>
            <button
              type="button"
              onClick={() => {
                if (plots.length === 1) focusPlotsTab();
                onDelete(safeIndex);
              }}
              aria-label="Delete this plot"
              className={btn}
            >
              Delete
            </button>
            <button
              type="button"
              onClick={() => {
                focusPlotsTab();
                onClear();
              }}
              aria-label="Clear all plots"
              className={btn}
            >
              Clear all
            </button>
          </span>
        ) : null}
      </header>

      {/* Named by the selected tab. A tab stop only while its content
          overflows; otherwise the tab itself is the stop (#19). */}
      <div
        ref={panelRef}
        id={panelId}
        role="tabpanel"
        aria-labelledby={tabIds[current]}
        tabIndex={overflows ? 0 : undefined}
        className="min-h-0 flex-1 overflow-auto"
      >
        {tab === "plots" ? (
          plot ? (
            <figure className="flex h-full flex-col items-center justify-center gap-2 p-2">
              {/* A runtime-generated data URL; next/image cannot optimise it. */}
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={plot.url}
                // Described from the figure and the code (#23).
                alt={plotAltText({
                  language,
                  info: plot.info,
                  index: safeIndex,
                  total: plots.length,
                })}
                className="min-h-0 max-w-full flex-1 rounded bg-white object-contain"
              />
              {/* A text alternative listing what is known about the chart. */}
              <details className="w-full max-w-md text-xs text-[var(--sb-text)]">
                <summary className="cursor-pointer font-bold text-[var(--sb-muted)]">
                  Describe this plot
                </summary>
                <dl className="mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
                  {plotDetails(plot.info, language).map((row) => (
                    <div key={row.label} className="contents">
                      <dt className="font-bold">{row.label}</dt>
                      <dd>{row.value}</dd>
                    </div>
                  ))}
                </dl>
              </details>
            </figure>
          ) : (
            <div className="flex h-full items-center justify-center p-2">
              <p className="text-sm text-[var(--sb-muted)]">
                Charts you draw appear here.
              </p>
            </div>
          )
        ) : (
          <HelpBody help={help} helpDoc={helpDoc} />
        )}
      </div>
    </section>
  );
}

/**
 * The HELP tab body. It renders the documentation text the language runtime carries
 * locally (a Python docstring, an R help page), fetched with no network. While the
 * fetch is out it shows a small loading status; when the runtime has no local doc for
 * the symbol (SQLite always, or an unknown name) it falls back to the curated blurb.
 * The "Open full documentation" link is always present as the path to the full,
 * formatted, canonical page, which cannot be embedded here because /coding-studio is
 * cross-origin isolated (COEP require-corp blocks an iframe and a cross-origin fetch).
 */
function HelpBody({
  help,
  helpDoc,
}: {
  help: { symbol: string; entry: DocEntry; req: HelpRequest } | null;
  helpDoc: { status: "idle" | "loading" | "loaded" | "none"; text?: string; signature?: string; truncated?: boolean };
}) {
  if (!help) {
    return (
      <div className="p-3 text-sm text-[var(--sb-muted)]">
        <p>
          Ctrl or Cmd click a function or keyword in your script, or put the cursor on
          it and press F1, to see its documentation here.
        </p>
      </div>
    );
  }
  const { symbol, entry } = help;
  const loading = helpDoc.status === "loading";
  const loaded = helpDoc.status === "loaded" && !!helpDoc.text;
  return (
    <div className="flex flex-col gap-2 p-3 text-sm">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="font-mono text-base font-bold text-[var(--sb-text)]">
          {symbol}
        </span>
        <span className="rounded border border-[var(--sb-border)] px-1.5 py-0.5 text-xs font-bold uppercase tracking-wide text-[var(--sb-muted)]">
          {entry.source}
        </span>
      </div>

      {helpDoc.signature ? (
        <p className="font-mono text-xs text-[var(--sb-muted)]">
          {symbol}
          {helpDoc.signature}
        </p>
      ) : null}

      {loading ? (
        <p
          role="status"
          aria-live="polite"
          className="flex items-center gap-1.5 text-xs font-bold text-[var(--sb-muted)]"
        >
          Loading documentation
        </p>
      ) : null}

      {loaded ? (
        <>
          {/* Labelled, and keyboard-scrollable while it overflows, so a
              keyboard user can read a long doc (WCAG 2.1.1, #19); the runtime
              produced this text with no network. */}
          <ScrollRegion
            label={`Documentation for ${symbol}`}
            className="max-h-[22rem] overflow-auto rounded-card border border-[var(--sb-border)] bg-[var(--sb-header)] p-2"
          >
            <pre className="whitespace-pre-wrap font-mono text-xs text-[var(--sb-text)]">
              {helpDoc.text}
            </pre>
          </ScrollRegion>
          {helpDoc.truncated ? (
            <p className="text-xs text-[var(--sb-muted)]">
              Showing the first part of the documentation. The full page is one click
              away below.
            </p>
          ) : null}
        </>
      ) : null}

      {helpDoc.status === "none" ? (
        <>
          {entry.blurb ? (
            <p className="text-[var(--sb-text)]">{entry.blurb}</p>
          ) : null}
          <p className="text-xs text-[var(--sb-muted)]">
            No documentation text is available for {symbol} in this runtime. Open the
            full documentation below.
          </p>
        </>
      ) : null}

      {entry.note ? (
        <p className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-header)] px-2 py-1.5 text-xs text-[var(--sb-muted)]">
          {entry.note}
        </p>
      ) : null}

      <a
        href={entry.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex w-fit items-center gap-1 rounded-card bg-[var(--sb-accent)] px-3 py-1 text-sm font-bold text-white hover:opacity-90"
      >
        Open full documentation
      </a>
      <p className="text-xs text-[var(--sb-muted)]">
        Opens the official {entry.source} page in a new tab.
      </p>
    </div>
  );
}

/*
 * Pane dividers are full window splitters (#16), not removed from the tab
 * order: react-resizable-panels already gives each one role="separator",
 * aria-orientation, aria-valuenow/min/max, aria-controls, and the keyboard
 * (arrow keys resize by 5%, Home/End go to the limits, F6 jumps to the next
 * divider). What was missing was a name saying which panes it resizes, and a
 * visible state on keyboard focus.
 */
const handleFocus =
  "data-[separator=focus]:bg-[var(--sb-accent)] focus-visible:bg-[var(--sb-accent)]";

/** A draggable divider between two side-by-side columns (a thin vertical bar). */
function ColHandle({ label }: { label: string }) {
  return (
    <Separator
      aria-label={label}
      className={`group mx-0.5 flex w-1.5 items-stretch justify-center bg-transparent hover:bg-[var(--sb-accent)] ${handleFocus}`}
    >
      <div className="w-px bg-[var(--sb-border)] group-hover:bg-transparent" />
    </Separator>
  );
}

/** A draggable divider between two stacked panes (a thin horizontal bar). */
function RowHandle({ label }: { label: string }) {
  return (
    <Separator
      aria-label={label}
      className={`group my-0.5 flex h-1.5 items-center justify-stretch bg-transparent hover:bg-[var(--sb-accent)] ${handleFocus}`}
    >
      <div className="h-px w-full bg-[var(--sb-border)] group-hover:bg-transparent" />
    </Separator>
  );
}
