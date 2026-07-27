/**
 * Which fenced code languages can be run in the browser, and the worker that
 * runs each. This is the single gate deciding whether the Run button appears.
 *
 * Phase 1 shipped SQL, phase 2 Python, phase 3 R. Adding a language is one
 * entry plus its worker file.
 */

export interface RunnableLanguage {
  /** Canonical id used in the UI and telemetry. */
  id: "sql" | "python" | "r";
  /** Fence tokens that map to this language, lowercased. */
  aliases: string[];
  label: string;
  /** Static worker URL, served from public/. */
  workerUrl: string;
  /** Shown while the runtime downloads on first run. */
  loadingLabel: string;
  /** One-line nudge, shown before the first run, on bringing your own data.
   * Each run is a fresh sandbox, so data has to live in the snippet itself. */
  dataHint: string;
  /** Optional per-language cap on a single run before the worker is stopped.
   * Larger for R, whose runtime is far bigger to load and whose packages
   * download while code runs. Falls back to the manager's default. */
  runTimeoutMs?: number;
  /** REPL-style header shown at the top of the Coding Studio console: the
   * bundled version and the honest word on first-run slowness. Versions are
   * pinned by what setup-runtimes.mjs actually bundles; update together. */
  banner: string;
}

export const RUNNABLE_LANGUAGES: RunnableLanguage[] = [
  {
    id: "sql",
    aliases: ["sql", "sqlite"],
    label: "SQL",
    workerUrl: "/workers/sqlite-worker.mjs",
    loadingLabel: "Loading SQLite",
    banner:
      "SQLite 3.53.0 (WebAssembly) running in your browser.\nTables you create stay until you Restart the session. The first run loads the engine.",
    dataHint:
      "Starts with an empty database. To use your own data, add it with CREATE TABLE and INSERT in the snippet.",
  },
  {
    id: "python",
    aliases: ["python", "py"],
    label: "Python",
    workerUrl: "/workers/pyodide-worker.mjs",
    loadingLabel: "Loading Python",
    banner:
      "Python 3.14.0 (Pyodide, WebAssembly) running in your browser.\nThe first run loads the interpreter, and the first import of a bundled package (pandas, matplotlib, ...) fetches it, so early runs take longer. After that they are fast.",
    dataHint:
      "Starts fresh each run. To use your own data, paste it into the snippet, for example a CSV read with io.StringIO.",
    // The first run downloads the interpreter and any packages the snippet
    // imports (pandas and matplotlib are several MB), so a single run gets more
    // room before it counts as a runaway.
    runTimeoutMs: 60_000,
  },
  {
    id: "r",
    aliases: ["r"],
    label: "R",
    workerUrl: "/workers/webr-worker.mjs",
    loadingLabel: "Loading R",
    banner:
      "R version 4.6.0 (WebR 0.6.0, WebAssembly) running in your browser.\nThe first run loads the R runtime and install.packages() fetches from this site, so early runs take noticeably longer. After that they are fast.",
    dataHint:
      'Starts fresh each run. Put your data in the snippet, and install packages with install.packages("name").',
    // R's runtime is tens of MB and its packages download while code runs, so
    // a single run is given far more room before it counts as runaway.
    runTimeoutMs: 120_000,
  },
];

/**
 * Reads the language of a fenced block. react-markdown puts it on the inner
 * <code> element as `language-xxx`; callers pass that className.
 */
export function languageFromClassName(
  className: string | undefined,
): string | null {
  if (!className) return null;
  const match = /language-([a-z0-9]+)/i.exec(className);
  return match ? match[1].toLowerCase() : null;
}

export function runnerFor(language: string | null): RunnableLanguage | null {
  if (!language) return null;
  const lang = language.toLowerCase();
  return (
    RUNNABLE_LANGUAGES.find((r) => r.aliases.includes(lang)) ?? null
  );
}
