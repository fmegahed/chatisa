import type { LanguageId } from "./types";

const SPACE = " ";
const STRING = "x";

/**
 * Returns a same-length "code view" of `text` where comment characters are
 * spaces and string characters are the placeholder `x`. Offsets are preserved,
 * so a scan of the result indexes back into `text` at the same positions. This
 * lets bracket, operator, and semicolon detection ignore anything inside a
 * string or comment for R (and, for Slice 3, SQL indentation).
 *
 * String rules: R and Python use `"`, `'`, and backtick with backslash escapes
 * (backtick in R has no escape but closes on the next backtick, which is fine
 * here). SQL uses `'` (doubled `''` escape) and `"`/backtick as quoted
 * identifiers. Comment rules: `#` to end of line for R and Python; `--` to end
 * of line and block comments for SQL. Python triple-quoted strings are covered
 * because the same delimiter opens and the scanner consumes until it recurs; the
 * Python and SQL statement finders use the Lezer tree and do not rely on this
 * mask, so triple-quote subtleties never affect them. The mask is primarily for
 * R.
 */
export function maskStringsAndComments(
  text: string,
  languageId: LanguageId,
): string {
  const out = new Array<string>(text.length);
  const lineComment = languageId === "sql" ? "--" : "#";
  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    // Line comment.
    if (text.startsWith(lineComment, i)) {
      while (i < text.length && text[i] !== "\n") out[i++] = SPACE;
      continue;
    }
    // SQL block comment.
    if (languageId === "sql" && text.startsWith("/*", i)) {
      while (i < text.length && !text.startsWith("*/", i)) {
        out[i] = text[i] === "\n" ? "\n" : SPACE;
        i++;
      }
      if (i < text.length) {
        out[i++] = SPACE;
        if (i < text.length) out[i++] = SPACE; // the closing "/"
      }
      continue;
    }
    // String / quoted identifier.
    if (ch === '"' || ch === "'" || ch === "`") {
      const quote = ch;
      const sqlDoubled = languageId === "sql" && quote === "'";
      out[i++] = STRING; // opening quote
      while (i < text.length) {
        const c = text[i];
        if (c === "\\" && languageId !== "sql") {
          out[i] = STRING;
          if (i + 1 < text.length) out[i + 1] = STRING;
          i += 2;
          continue;
        }
        if (c === quote) {
          if (sqlDoubled && text[i + 1] === "'") {
            out[i] = STRING;
            out[i + 1] = STRING;
            i += 2;
            continue;
          }
          out[i++] = STRING; // closing quote
          break;
        }
        out[i] = c === "\n" ? "\n" : STRING;
        i++;
      }
      continue;
    }
    out[i] = ch;
    i++;
  }
  return out.join("");
}

/**
 * True when the cursor, sitting at the END of `prefix`, is inside a string or a
 * comment rather than in code. A one-character probe is appended and masked: if it
 * comes back changed, an open comment or string reaches the cursor. Used to
 * suppress autocomplete inside comments and strings, where code suggestions are noise.
 */
export function cursorInMasked(prefix: string, languageId: LanguageId): boolean {
  const PROBE = "Z"; // survives masking in code; becomes space/x inside a comment/string
  const mask = maskStringsAndComments(prefix + PROBE, languageId);
  return mask[mask.length - 1] !== PROBE;
}
