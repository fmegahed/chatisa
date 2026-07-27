import { maskStringsAndComments } from "./mask";

export interface RLine {
  from: number;
  to: number;
  blank: boolean;
  depthAtStart: number;
  endsWithContinuation: boolean;
  startsWithContinuation: boolean;
}

const TRAILING_OP = /(\|>|%[^%\s]*%|<<-|<-|->>|->|[-+*/^~:?<>=&|!,])$/;
const LEADING_OP = /^(\|>|%[^%\s]*%|\+)/;

/** Lexes R into a line table over the string/comment mask. Reused by Slice 3. */
export function scanR(text: string): { lines: RLine[]; mask: string } {
  const mask = maskStringsAndComments(text, "r");
  const lines: RLine[] = [];
  let depth = 0;
  let from = 0;
  for (let i = 0; i <= mask.length; i++) {
    if (i === mask.length || mask[i] === "\n") {
      const slice = mask.slice(from, i);
      const depthAtStart = depth;
      for (const ch of slice) {
        if (ch === "(" || ch === "[" || ch === "{") depth++;
        else if (ch === ")" || ch === "]" || ch === "}") depth = Math.max(0, depth - 1);
      }
      const trimmed = slice.trim();
      lines.push({
        from,
        to: i,
        blank: trimmed === "",
        depthAtStart,
        endsWithContinuation: TRAILING_OP.test(slice.replace(/\s+$/, "")),
        startsWithContinuation: LEADING_OP.test(slice.replace(/^\s+/, "")),
      });
      from = i + 1;
    }
  }
  return { lines, mask };
}

function startsNew(lines: RLine[], idx: number): boolean {
  const L = lines[idx];
  if (L.blank || L.depthAtStart > 0 || L.startsWithContinuation) return false;
  for (let p = idx - 1; p >= 0; p--) {
    if (lines[p].blank) continue;
    return !lines[p].endsWithContinuation;
  }
  return true;
}

/** The range of the complete R logical statement containing `pos`. */
export function rStatementRange(
  text: string,
  pos: number,
): { from: number; to: number } {
  const { lines, mask } = scanR(text);
  if (lines.length === 0) return { from: 0, to: 0 };

  let cur = lines.findIndex((l) => pos >= l.from && pos <= l.to);
  if (cur < 0) cur = lines.length - 1;

  // Blank line: snap to the previous non-blank, else the next non-blank.
  if (lines[cur].blank) {
    let up = cur - 1;
    while (up >= 0 && lines[up].blank) up--;
    if (up >= 0) cur = up;
    else {
      let down = cur + 1;
      while (down < lines.length && lines[down].blank) down++;
      if (down < lines.length) cur = down;
      else return physicalLine(text, pos);
    }
  }

  // Walk back to the statement start.
  let start = cur;
  while (start > 0 && !startsNew(lines, start)) {
    let p = start - 1;
    while (p >= 0 && lines[p].blank) p--;
    if (p < 0) break;
    start = p;
  }

  // Walk forward while the next non-blank line does not start a new statement.
  let end = start;
  let j = end + 1;
  while (j < lines.length) {
    if (lines[j].blank) {
      j++;
      continue;
    }
    if (startsNew(lines, j)) break;
    end = j;
    j++;
  }

  // Trim leading indentation on the start line.
  let f = lines[start].from;
  while (f < lines[start].to && /\s/.test(text[f])) f++;
  const range = { from: f, to: lines[end].to };

  // Single-line statement: honor top-level semicolons at the cursor.
  if (start === end) {
    const seg = splitTopLevelSemicolons(mask, range.from, range.to, pos);
    if (seg) return seg;
  }
  return range;
}

function splitTopLevelSemicolons(
  mask: string,
  from: number,
  to: number,
  pos: number,
): { from: number; to: number } | null {
  const stops: number[] = [];
  let depth = 0;
  for (let i = from; i < to; i++) {
    const ch = mask[i];
    if (ch === "(" || ch === "[" || ch === "{") depth++;
    else if (ch === ")" || ch === "]" || ch === "}") depth = Math.max(0, depth - 1);
    else if (ch === ";" && depth === 0) stops.push(i);
  }
  if (stops.length === 0) return null;
  const bounds = [from, ...stops, to];
  for (let k = 0; k < bounds.length - 1; k++) {
    const segFrom = k === 0 ? bounds[0] : bounds[k] + 1;
    const segTo = bounds[k + 1];
    if (pos >= segFrom && pos <= segTo) {
      let f = segFrom;
      while (f < segTo && /\s/.test(mask[f])) f++;
      return { from: f, to: segTo };
    }
  }
  return null;
}

function physicalLine(text: string, pos: number): { from: number; to: number } {
  let from = pos;
  while (from > 0 && text[from - 1] !== "\n") from--;
  let to = pos;
  while (to < text.length && text[to] !== "\n") to++;
  return { from, to };
}
