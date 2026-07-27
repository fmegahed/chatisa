import { parser as pyParser } from "@lezer/python";
import { SQLite } from "@codemirror/lang-sql";
import { maskStringsAndComments } from "./mask";
import type { LanguageId } from "./types";

export interface LintProblem {
  from: number;
  to: number;
  severity: "error" | "warning";
  message: string;
}

const OPEN: Record<string, string> = { ")": "(", "]": "[", "}": "{" };

/** Unbalanced brackets and unterminated strings in R (no grammar available). Reads
 *  the Slice 2 mask so brackets/quotes inside strings and comments are ignored. */
export function rBalanceProblems(text: string): LintProblem[] {
  const mask = maskStringsAndComments(text, "r");
  const problems: LintProblem[] = [];
  const stack: { ch: string; pos: number }[] = [];
  for (let i = 0; i < mask.length; i++) {
    const c = mask[i];
    if (c === "(" || c === "[" || c === "{") stack.push({ ch: c, pos: i });
    else if (c === ")" || c === "]" || c === "}") {
      const top = stack.pop();
      if (!top || top.ch !== OPEN[c]) {
        problems.push({ from: i, to: i + 1, severity: "error", message: `Unmatched ${c}` });
      }
    }
  }
  for (const open of stack) {
    problems.push({
      from: open.pos,
      to: open.pos + 1,
      severity: "error",
      message: `Unclosed ${open.ch}`,
    });
  }
  problems.push(...unterminatedStrings(text));
  return problems;
}

/** Quote runs that reach end of file without closing (R/Python quoting). */
function unterminatedStrings(text: string): LintProblem[] {
  const out: LintProblem[] = [];
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === "#") {
      while (i < text.length && text[i] !== "\n") i++;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === "`") {
      const quote = ch;
      const start = i;
      i++;
      let closed = false;
      while (i < text.length) {
        if (text[i] === "\\" && quote !== "`") {
          i += 2;
          continue;
        }
        if (text[i] === quote) {
          i++;
          closed = true;
          break;
        }
        i++;
      }
      if (!closed) {
        out.push({ from: start, to: text.length, severity: "error", message: "Unterminated string" });
      }
      continue;
    }
    i++;
  }
  return out;
}

/** Parser error nodes for a language with a Lezer grammar (Python or SQL). */
export function treeErrorProblems(text: string, lang: "python" | "sql"): LintProblem[] {
  const tree = lang === "python" ? pyParser.parse(text) : SQLite.language.parser.parse(text);
  const out: LintProblem[] = [];
  const cursor = tree.cursor();
  do {
    if (cursor.type.isError) {
      const from = cursor.from;
      const to = Math.max(cursor.to, from + 1);
      // Collapse a run of adjacent error nodes into one marker.
      const last = out[out.length - 1];
      if (last && from <= last.to) last.to = Math.max(last.to, to);
      else out.push({ from, to, severity: "error", message: "Syntax error" });
    }
  } while (cursor.next());
  return out;
}

/** Lines whose leading whitespace mixes tabs and spaces (Python correctness). */
export function pyTabSpaceProblems(text: string): LintProblem[] {
  const out: LintProblem[] = [];
  let from = 0;
  for (const line of text.split("\n")) {
    const lead = /^[ \t]*/.exec(line)![0];
    if (lead.includes("\t") && lead.includes(" ")) {
      out.push({
        from,
        to: from + lead.length,
        severity: "warning",
        message: "Mixed tabs and spaces in indentation",
      });
    }
    from += line.length + 1; // + the newline
  }
  return out;
}

/** The unobtrusive lint problems for a document, by language. */
export function lintProblems(text: string, languageId: LanguageId): LintProblem[] {
  if (languageId === "r") return rBalanceProblems(text);
  if (languageId === "python") {
    return [...treeErrorProblems(text, "python"), ...pyTabSpaceProblems(text)];
  }
  return treeErrorProblems(text, "sql");
}
