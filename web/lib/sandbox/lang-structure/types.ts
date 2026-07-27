/** The three runnable Coding Studio languages. */
export type LanguageId = "r" | "python" | "sql";

/**
 * A resolved logical-statement range in a document.
 * `from`/`to` bound the statement (to excludes the trailing newline); `nextPos`
 * is where the caret advances after running (start of the next executable
 * statement, or the document end).
 */
export interface StatementRange {
  from: number;
  to: number;
  nextPos: number;
}
