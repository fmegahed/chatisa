/**
 * Ask Anything's browser-executed tools (design 2026-07-24). The server route
 * declares these WITHOUT execute, so tool calls stream to the student's browser,
 * which runs them on the Coding Studio WASM runtimes and returns the results.
 * The descriptions carry the parts of the runtime contract the model needs at
 * the moment it picks a tool; the system prompt carries the full routing rules.
 */

import { tool } from "ai";
import { z } from "zod";
import { proxyCapText } from "@/lib/net/proxy-limits";
import {
  classifyPythonPackage,
  type PyodideIndex,
} from "@/lib/sandbox/packages";

export const ASK_TOOL_NAMES = ["run_python", "run_r", "run_sql"] as const;
export type AskToolName = (typeof ASK_TOOL_NAMES)[number];

/**
 * Request body for POST /api/ask-anything. Unlike the shared chatRequestSchema
 * (whose part schema strips unknown keys), parts here pass through intact:
 * a continuation request carries tool parts whose toolCallId/state/input/output
 * the conversion to model messages needs verbatim. Bounds still apply.
 */
export const askRequestSchema = z.object({
  module: z.string().min(1).max(64).optional(),
  modelId: z.string().min(1).max(128),
  messages: z
    .array(
      z.object({
        id: z.string().optional(),
        role: z.enum(["user", "assistant", "system"]),
        parts: z
          .array(z.looseObject({ type: z.string() }))
          .max(200)
          .optional(),
      }),
    )
    .min(1)
    .max(200),
});

const codeInput = (what: string) =>
  z.object({
    code: z
      .string()
      .min(1)
      .max(20_000)
      .describe(`The ${what} to run, complete and self-contained for this step.`),
  });

/**
 * The tool set for streamText. No execute functions: the browser runs the code
 * (private, free) and posts the results back; the loop continues on the next
 * request. Descriptions are the per-tool contract, em-dash-free.
 */
export function askToolDefs() {
  return {
    run_python: tool({
      description:
        `Run Python in the student's browser (Pyodide). Preloaded: numpy, pandas, matplotlib, scipy, scikit-learn, statsmodels, statsforecast (with utilsforecast and coreforecast), pyarrow, polars, seaborn, openpyxl, beautifulsoup4, lxml, requests. Only Pyodide-built or pure-Python packages can be added; compiled packages (pyreadr, tensorflow) can never install. requests can reach the web: GET and POST are routed through a built-in guarded proxy (${proxyCapText()} response cap, private hosts blocked), so requests.get plus BeautifulSoup works on ordinary websites. A response body starting with 'ChatISA proxy:' explains why a fetch was refused. Variables persist across calls in this chat. Plots (matplotlib) are captured automatically and shown to the student.`,
      inputSchema: codeInput("Python code"),
    }),
    run_r: tool({
      description:
        "Run R in the student's browser (WebR). Bundled: tidyverse, readxl, janitor, httr2, rvest; most of CRAN installable with install.packages (tidymodels, fpp3 and the like download on demand). R can reach the web through a built-in proxy, so rvest::read_html(url) works even on sites without CORS. The first R call in a chat takes about 30 seconds to prepare; prefer Python for ordinary data work. Variables persist across calls in this chat. Plots are captured automatically.",
      inputSchema: codeInput("R code"),
    }),
    run_sql: tool({
      description:
        "Run SQL in the student's browser on an in-memory SQLite database (SQLite dialect only; no DATE_TRUNC and friends). Tables persist across calls in this chat.",
      inputSchema: codeInput("SQL"),
    }),
  };
}

/** Cap on tool output returned to the model, so a huge print loop cannot blow
 * the context. The student's tool card shows the same capped text with a note. */
export const TOOL_OUTPUT_MAX = 8000;

export function truncateToolOutput(text: string): {
  text: string;
  truncated: boolean;
} {
  if (text.length <= TOOL_OUTPUT_MAX) return { text, truncated: false };
  return {
    text: `${text.slice(0, TOOL_OUTPUT_MAX)}\n[output truncated at ${TOOL_OUTPUT_MAX} characters]`,
    truncated: true,
  };
}

/** The module name from a Python import failure, or null for other errors. */
export function missingModuleFrom(errorText: string): string | null {
  const m = /No module named ['"]([A-Za-z0-9_.-]+)['"]/.exec(errorText);
  return m ? m[1] : null;
}

/**
 * Appends the package checker's verdict to a Python import failure, so the
 * model learns in one step whether the package is installable (and how) or can
 * never work here, instead of thrashing on a bare traceback.
 */
export function enrichPythonError(
  errorText: string,
  index: PyodideIndex | null,
): string {
  const moduleName = missingModuleFrom(errorText);
  if (!moduleName || !index) return errorText;
  const verdict = classifyPythonPackage(moduleName, index);
  if (!verdict) return errorText;
  return `${errorText}\n[package check] ${verdict.message}`;
}
