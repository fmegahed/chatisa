/**
 * Wording and output-shape logic for the Ask Anything tool cards, split out of
 * the component so it can be unit-tested (the test runner has no DOM) and so
 * the download surface has one definition rather than one per renderer.
 *
 * Client-safe by construction: no "server-only", no credentials, no fetching.
 */

export const TOOL_LABELS: Record<string, string> = {
  run_python: "Python",
  run_r: "R",
  run_sql: "SQL",
};

export const RUNNING_TEXT: Record<string, string> = {
  run_r: "Preparing R (the first R run in a chat takes a moment)...",
  search_papers: "Searching arXiv, Semantic Scholar, and OpenAlex...",
  get_paper: "Looking up the paper...",
  read_url: "Reading the page...",
  get_miami_style: "Fetching Miami's style assets...",
  code_interpreter: "Running code on OpenAI's servers...",
  code_execution: "Running code on Anthropic's servers...",
};

const DONE_TEXT: Record<string, string> = {
  get_paper: "Looked up a paper",
  get_miami_style: "Fetched Miami's style",
  code_interpreter: "Ran on OpenAI's servers",
  code_execution: "Ran on Anthropic's servers",
};

/**
 * Whole sentences, written out. These used to be derived by taking the first
 * word of the success label, which produced "Looked failed", "Searched failed",
 * and "Ran failed" at the exact moment a student needed to understand what
 * broke (found 2026-07-25).
 */
const FAILED_TEXT: Record<string, string> = {
  search_papers: "The literature search failed",
  get_paper: "The paper lookup failed",
  read_url: "The page could not be read",
  get_miami_style: "Miami's style assets could not be fetched",
  code_interpreter: "The run on OpenAI's servers failed",
  code_execution: "The run on Anthropic's servers failed",
};

/** Anthropic and OpenAI id formats, mirrored from lib/ask/hosted-files so the
 * card never renders a link the download route would reject. */
const ANTHROPIC_FILE_ID = /^file[_-][A-Za-z0-9_-]{4,64}$/;
const OPENAI_CONTAINER_ID = /^cntr[_-][A-Za-z0-9_-]{4,64}$/;

export function hostOf(url: unknown): string {
  try {
    return new URL(String(url)).hostname.replace(/^www\./, "");
  } catch {
    return "page";
  }
}

/** The one-line card summary for every tool and state. */
export function toolSummary({
  toolName,
  running = false,
  failed = false,
  input,
  output,
}: {
  toolName: string;
  running?: boolean;
  failed?: boolean;
  input?: Record<string, unknown>;
  output?: Record<string, unknown>;
}): string {
  const label = TOOL_LABELS[toolName] ?? toolName;
  if (running) return RUNNING_TEXT[toolName] ?? `Running ${label}...`;
  if (failed) return FAILED_TEXT[toolName] ?? `The ${label} run failed`;
  if (toolName === "search_papers") {
    const papers = output?.papers;
    return `Searched the literature${
      Array.isArray(papers) ? ` (${papers.length} papers)` : ""
    }`;
  }
  if (toolName === "read_url") return `Read ${hostOf(input?.url)}`;
  const done = DONE_TEXT[toolName];
  if (done) return done;
  const ms = output?.ms;
  return `Ran ${label}${
    typeof ms === "number" ? ` in ${(ms / 1000).toFixed(1)}s` : ""
  }`;
}

/**
 * Provider file ids for the files a hosted run CREATED. Every result shape of
 * Anthropic's code execution tool (plain, encrypted, and bash) reports created
 * files the same way, as content[].file_id, so the result type is not inspected.
 */
export function createdFileIds(
  toolName: string,
  output: Record<string, unknown> | undefined,
): string[] {
  if (toolName !== "code_execution") return [];
  const content = output?.content;
  if (!Array.isArray(content)) return [];
  return content
    .map((entry) => (entry as { file_id?: unknown } | null)?.file_id)
    .filter(
      (id): id is string =>
        typeof id === "string" && ANTHROPIC_FILE_ID.test(id),
    );
}

/** The interpreter container to list files from; OpenAI streams the id but not
 * the file list, so the card fetches that lazily. */
export function openaiContainerId(
  toolName: string,
  input: Record<string, unknown> | undefined,
): string | null {
  if (toolName !== "code_interpreter") return null;
  const id = input?.containerId;
  return typeof id === "string" && OPENAI_CONTAINER_ID.test(id) ? id : null;
}
