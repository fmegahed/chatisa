import type { RunOutcome, SessionVariable } from "@/lib/run/manager";

/**
 * Builds the plain-text context the Sandbox side chat sends with a question:
 * the current script, the last run's code and result (or error), and the
 * variables in scope with their types and, for data frames, columns. Only
 * shapes and column types travel, never the data's actual values. Every part is
 * bounded so the payload stays small.
 */
export function buildSandboxContext(input: {
  languageLabel: string;
  script: string;
  lastRun?: { code: string; outcome: RunOutcome };
  variables: SessionVariable[];
}): string {
  const parts: string[] = [`Language: ${input.languageLabel}`];

  if (input.script.trim()) {
    parts.push(`Current script:\n${fence(input.script)}`);
  }

  if (input.lastRun) {
    parts.push(`Last run:\n${fence(input.lastRun.code)}`);
    const outcome = input.lastRun.outcome;
    if (outcome.ok) {
      const text = outcome.result?.text?.trim();
      const plot = outcome.result?.imageDataUrl ? "\n(a plot was produced)" : "";
      parts.push(
        `Last result:\n${text ? truncate(text, 2000) : "(ran, no text output)"}${plot}`,
      );
    } else {
      parts.push(`Last run error:\n${truncate(outcome.error ?? "", 2000)}`);
    }
  }

  if (input.variables.length > 0) {
    parts.push(
      `Variables in scope:\n${input.variables.map(formatVariable).join("\n")}`,
    );
  }

  return parts.join("\n\n");
}

function formatVariable(v: SessionVariable): string {
  const columns = v.columns?.length
    ? `; columns: ${v.columns.map((c) => `${c.name} (${c.type})`).join(", ")}`
    : "";
  return `- ${v.name}: ${v.type} [${v.info}]${columns}`;
}

function fence(code: string): string {
  return "```\n" + truncate(code, 4000) + "\n```";
}

function truncate(text: string, max: number): string {
  return text.length <= max ? text : `${text.slice(0, max)}\n... (truncated)`;
}
