import type { LanguageId, StatementRange } from "./types";
import { rStatementRange } from "./r-scan";
import { pyStatementRange } from "./py";
import { sqlStatementRange } from "./sql";

export type { LanguageId, StatementRange } from "./types";
export { rStatementRange, scanR } from "./r-scan";
export { pyStatementRange, topLevelStatementAt } from "./py";
export { sqlStatementRange, statementNodeAt } from "./sql";
export { maskStringsAndComments, cursorInMasked } from "./mask";
export {
  rIndentColumns,
  sqlIndentColumns,
  R_INDENT_UNIT,
  SQL_INDENT_UNIT,
} from "./indent";
export { lintProblems } from "./lint";
export type { LintProblem } from "./lint";

/** The next executable position after `to`: the start of the next non-blank
 *  line, or the document end. */
export function nextStatementPos(text: string, to: number): number {
  let i = to;
  while (i < text.length && (text[i] === "\n" || /\s/.test(text[i]))) {
    if (text[i] === "\n") {
      // Snap to the first non-whitespace on/after the next line.
      let j = i + 1;
      while (j < text.length && text[j] !== "\n" && /\s/.test(text[j])) j++;
      if (j < text.length && text[j] !== "\n") return j;
    }
    i++;
  }
  return text.length;
}

/**
 * The complete logical statement containing `pos`, for the given language.
 * `nextPos` is where the caret advances after running. The per-language finders
 * fall back to the physical line, so this never returns less than a line.
 */
export function statementRangeAt(
  text: string,
  pos: number,
  languageId: LanguageId,
): StatementRange {
  const clamped = Math.max(0, Math.min(pos, text.length));
  const { from, to } =
    languageId === "python"
      ? pyStatementRange(text, clamped)
      : languageId === "sql"
        ? sqlStatementRange(text, clamped)
        : rStatementRange(text, clamped);
  return { from, to, nextPos: nextStatementPos(text, to) };
}
