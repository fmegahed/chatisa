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
