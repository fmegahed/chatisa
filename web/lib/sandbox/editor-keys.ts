/** The native R pipe, inserted with surrounding spaces (RStudio's insert-pipe
 *  behaviour). Deliberately the native `|>`, never the magrittr `%>%`. */
export const PIPE_TOKEN = " |> ";

/**
 * Pure description of the edit that inserts the native pipe. Given the current
 * main selection range, it replaces that range (empty or not) with ` |> ` and
 * reports where the caret should land: immediately after the inserted token.
 * DOM-free so it can be unit tested; Task 3 turns this into a CodeMirror
 * transaction via `view.dispatch`.
 */
export function buildPipeInsertion(sel: { from: number; to: number }): {
  from: number;
  to: number;
  insert: string;
  anchor: number;
} {
  return {
    from: sel.from,
    to: sel.to,
    insert: PIPE_TOKEN,
    anchor: sel.from + PIPE_TOKEN.length,
  };
}

/** The CodeMirror binding that reads the caret position aloud (#21). Alt+L is
 *  CodeMirror's select-line and Ctrl+Alt is AltGr on many layouts, so Shift is
 *  added; on macOS Alt is the Option key. */
export const CARET_POSITION_KEY = "Alt-Shift-l";

/** How the caret-position shortcut is written for the current platform. */
export function caretPositionKeysLabel(isMac: boolean): string {
  return isMac ? "Option+Shift+L" : "Alt+Shift+L";
}

/** A caret position: 1-based line and column, plus the document's line count. */
export interface CaretPosition {
  line: number;
  column: number;
  lines: number;
}

/** The caret position for an offset in a document, from its line table. */
export function caretPositionAt(
  lineAt: (pos: number) => { number: number; from: number },
  lines: number,
  head: number,
): CaretPosition {
  const line = lineAt(head);
  return { line: line.number, column: head - line.from + 1, lines };
}

/** The compact, VS Code style status text ("Ln 3, Col 5"). */
export function caretStatusText(pos: CaretPosition): string {
  return `Ln ${pos.line}, Col ${pos.column}`;
}

/** What the shortcut announces: the position and any problems on that line. */
export function caretAnnouncement(
  pos: CaretPosition,
  problems: readonly string[] = [],
): string {
  const where = `Line ${pos.line} of ${pos.lines}, column ${pos.column}.`;
  return problems.length ? `${where} ${problems.join(" ")}` : where;
}
