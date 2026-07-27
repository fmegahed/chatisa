import type { DocEntry, HelpRequest } from "./types";

/**
 * The read-only documentation request sent to a language worker. `name` is the
 * clicked symbol, `qualifier` the receiver it hangs off when known (Python
 * `df.groupby` -> `df`; R `dplyr::summarise` -> `dplyr`), and `source` is the
 * resolver's label (pandas, NumPy, Python, dplyr, ggplot2, base R, SQLite), used
 * as a hint so the worker can find the topic even when no live object exists yet.
 */
export interface DocRequest {
  name: string;
  qualifier?: string;
  source?: string;
}

/** The runtime documentation for one symbol, as returned by the worker. */
export interface DocText {
  /** True when the runtime produced help/docstring text for the symbol. */
  found: boolean;
  /** The plain-text documentation (docstring or rendered help page). */
  text?: string;
  /** The call signature, when the runtime can produce one (Python). */
  signature?: string;
  /** True when the text was capped; the pane says the full docs are one click away. */
  truncated?: boolean;
}

/** Main-thread caps. The worker also hard-caps to bound the postMessage size; these
 *  are the display caps and the source of the "truncated" note in the pane. */
export const DOC_MAX_CHARS = 8000;
export const DOC_MAX_LINES = 160;

/** Builds the worker doc request from a resolved click. The resolver already ran on
 *  the main thread, so its `source` rides along as a topic hint. */
export function buildDocRequest(req: HelpRequest, entry: DocEntry): DocRequest {
  return {
    name: req.name,
    qualifier: req.qualifier,
    source: entry.source,
  };
}

/**
 * Removes terminal overstrike formatting from help text. R's `Rd2txt` (like man
 * pages and groff) renders bold and underline as overstrike: a character, a
 * backspace (0x08), then the character that overwrites it. A terminal shows that as
 * bold or underline, but a plain `<pre>` shows literal "_B" junk. Collapsing each
 * "<char><backspace>" pair keeps the overwriting character and drops the styling.
 * A no-op for text without backspaces (Python docstrings, SQL), so it is safe for all.
 */
export function stripOverstrike(raw: string): string {
  let text = raw;
  let prev: string;
  // Loop so double overstrike (bold + underline, "_\b X \b X") fully collapses.
  do {
    prev = text;
    text = text.replace(/[^\n]\x08/g, "");
  } while (text !== prev && text.includes("\x08"));
  return text.replace(/\x08/g, ""); // any stray backspace with nothing before it
}

/**
 * Caps long documentation text by line and character count, reporting whether it
 * was shortened so the pane can note that the full documentation is one click away.
 * Pure; the actual doc text is produced in the worker runtime.
 */
export function truncateDocText(
  raw: string,
  opts: { maxChars?: number; maxLines?: number } = {},
): { text: string; truncated: boolean } {
  const maxChars = opts.maxChars ?? DOC_MAX_CHARS;
  const maxLines = opts.maxLines ?? DOC_MAX_LINES;
  let truncated = false;
  let text = stripOverstrike(raw);

  const lines = text.split("\n");
  if (lines.length > maxLines) {
    text = lines.slice(0, maxLines).join("\n");
    truncated = true;
  }
  if (text.length > maxChars) {
    text = text.slice(0, maxChars);
    truncated = true;
  }
  return { text, truncated };
}
