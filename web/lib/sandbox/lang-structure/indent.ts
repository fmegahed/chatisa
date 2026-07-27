import { scanR } from "./r-scan";
import { maskStringsAndComments } from "./mask";

export const R_INDENT_UNIT = 2;
export const SQL_INDENT_UNIT = 2;

const OPENERS = new Set(["(", "[", "{"]);
const CLOSERS = new Set([")", "]", "}"]);
// Pipe, magrittr-style %...%, ggplot/arithmetic +, and other binary operators
// that cannot end an R expression. Read from the mask, so string/comment content
// never matches.
const R_TRAILING_CONT = /(\|>|%[^%\s]*%|[-+*/^~:?<>=&|!])$/;

function leadingWidth(text: string, from: number, to: number): number {
  let n = 0;
  for (let i = from; i < to; i++) {
    if (text[i] === " ") n++;
    else if (text[i] === "\t") n++;
    else break;
  }
  return n;
}

/**
 * Columns of indentation for a new line created by pressing Enter at `pos` in R.
 * Reuses Slice 2's `scanR` line table (bracket depth + continuation flags) and the
 * string/comment mask, so a bracket or operator inside a string or comment can
 * never influence the result.
 */
export function rIndentColumns(text: string, pos: number): number {
  const clamped = Math.max(0, Math.min(pos, text.length));
  const { lines } = scanR(text);
  if (lines.length === 0) return 0;
  const mask = maskStringsAndComments(text, "r");

  let cur = lines.findIndex((l) => clamped >= l.from && clamped <= l.to);
  if (cur < 0) cur = lines.length - 1;

  // Open-bracket depth at the cursor.
  let depth = lines[cur].depthAtStart;
  for (let i = lines[cur].from; i < clamped && i < lines[cur].to; i++) {
    const c = mask[i];
    if (OPENERS.has(c)) depth++;
    else if (CLOSERS.has(c)) depth = Math.max(0, depth - 1);
  }

  // Base indent: leading whitespace of the statement's first line. Walk back over
  // the same continuation rule the scanner uses.
  let start = cur;
  while (start > 0) {
    const L = lines[start];
    let p = start - 1;
    while (p >= 0 && lines[p].blank) p--;
    const continues =
      L.depthAtStart > 0 ||
      L.startsWithContinuation ||
      (p >= 0 && lines[p].endsWithContinuation);
    if (!continues || p < 0) break;
    start = p;
  }
  const base = leadingWidth(text, lines[start].from, lines[start].to);

  // Trailing continuation operator on the current line up to the cursor.
  const curSlice = mask.slice(lines[cur].from, clamped).replace(/\s+$/, "");
  const trailingCont = R_TRAILING_CONT.test(curSlice);

  let indent = base + depth * R_INDENT_UNIT;
  if (depth === 0 && trailingCont) indent += R_INDENT_UNIT;

  // Dedent when the new line will begin with an existing closing bracket.
  let q = clamped;
  while (q < text.length && (text[q] === " " || text[q] === "\t")) q++;
  if (q < text.length && CLOSERS.has(text[q])) {
    indent = Math.max(0, indent - R_INDENT_UNIT);
  }

  return indent;
}

const SQL_CLAUSE =
  /^(select|from|where|having|group\s+by|order\s+by|limit|on|union(\s+all)?|values|inner\s+join|left\s+join|right\s+join|join|with)\b/i;
const SQL_BODY_CLAUSE = /^(select|where|having|on|group\s+by|order\s+by)\b/i;

function lineStart(text: string, p: number): number {
  let s = Math.max(0, Math.min(p, text.length));
  while (s > 0 && text[s - 1] !== "\n") s--;
  return s;
}

function lineEnd(text: string, p: number): number {
  let e = Math.max(0, Math.min(p, text.length));
  while (e < text.length && text[e] !== "\n") e++;
  return e;
}

/**
 * Columns of indentation for a new line created by pressing Enter at `pos` in SQL.
 * Reuses Slice 2's mask so a `;`, keyword, or bracket inside a string or comment is
 * ignored. A pragmatic keyword-and-bracket approximation, not a full formatter.
 */
export function sqlIndentColumns(text: string, pos: number): number {
  const clamped = Math.max(0, Math.min(pos, text.length));
  const mask = maskStringsAndComments(text, "sql");

  // Bracket depth at the cursor.
  let depth = 0;
  for (let i = 0; i < clamped; i++) {
    const c = mask[i];
    if (OPENERS.has(c)) depth++;
    else if (CLOSERS.has(c)) depth = Math.max(0, depth - 1);
  }

  // Statement start: one past the previous top-level ";".
  let stmtStart = 0;
  let d = 0;
  for (let i = clamped - 1; i >= 0; i--) {
    const c = mask[i];
    if (CLOSERS.has(c)) d++;
    else if (OPENERS.has(c)) d = Math.max(0, d - 1);
    else if (c === ";" && d === 0) {
      stmtStart = i + 1;
      break;
    }
  }

  // Base indent = leading whitespace of the statement's first non-blank line.
  let s = stmtStart;
  while (s < text.length && /\s/.test(text[s])) s++;
  const base = leadingWidth(text, lineStart(text, s), s);

  // Governing clause: last clause keyword seen from the statement start through the
  // current line.
  let governingIsBody = false;
  const curStart = lineStart(text, clamped);
  let i = lineStart(text, stmtStart);
  while (i <= curStart && i < text.length) {
    const le = lineEnd(text, i);
    const first = mask.slice(i, le).replace(/^\s+/, "");
    if (SQL_CLAUSE.test(first)) governingIsBody = SQL_BODY_CLAUSE.test(first);
    if (le >= curStart) break;
    i = le + 1;
  }

  return base + depth * SQL_INDENT_UNIT + (governingIsBody ? SQL_INDENT_UNIT : 0);
}
