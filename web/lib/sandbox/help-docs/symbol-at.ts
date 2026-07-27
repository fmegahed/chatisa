import { parser as pyParser } from "@lezer/python";
import { SQLite } from "@codemirror/lang-sql";
import type { SyntaxNode } from "@lezer/common";
import { maskStringsAndComments } from "@/lib/sandbox/lang-structure/mask";
import type { HelpLanguage, HelpRequest } from "./types";

const sqlParser = SQLite.language.parser;

/**
 * The documentation request for the token at `pos`, or null when there is no
 * resolvable identifier there (whitespace, punctuation, or a click inside a
 * string or comment). Python and SQL use their Lezer grammars; R uses a word
 * scan over the string/comment mask.
 */
export function symbolAt(
  text: string,
  pos: number,
  language: HelpLanguage,
): HelpRequest | null {
  if (language === "python") return pythonSymbol(text, pos);
  if (language === "sql") return sqlSymbol(text, pos);
  return rSymbol(text, pos);
}

// --- R -------------------------------------------------------------------
// R identifiers: a letter or dot, then letters, digits, dots, underscores.
const R_WORD = /[A-Za-z0-9._]/;

function rSymbol(text: string, pos: number): HelpRequest | null {
  const mask = maskStringsAndComments(text, "r");
  // A click inside a string ('x' placeholder) or comment (space) is not a symbol.
  if (pos < 0 || pos >= mask.length) return null;
  if (mask[pos] !== text[pos]) return null; // masked away (string/comment)
  if (!R_WORD.test(text[pos])) return null;
  let from = pos;
  while (from > 0 && R_WORD.test(text[from - 1]) && mask[from - 1] === text[from - 1]) from--;
  let to = pos;
  while (to < text.length && R_WORD.test(text[to]) && mask[to] === text[to]) to++;
  const name = text.slice(from, to);
  if (!name) return null;
  // Qualified call: pkg::name or pkg:::name just before the word.
  let qualifier: string | undefined;
  let q = from;
  if (text.slice(Math.max(0, from - 3), from).endsWith("::")) {
    let colon = from;
    while (colon > 0 && text[colon - 1] === ":") colon--;
    let pk = colon;
    while (pk > 0 && R_WORD.test(text[pk - 1])) pk--;
    if (pk < colon) qualifier = text.slice(pk, colon);
    q = pk;
  }
  void q;
  return { name, qualifier, kind: "function", language: "r" };
}

// --- Python --------------------------------------------------------------
function pythonSymbol(text: string, pos: number): HelpRequest | null {
  const tree = pyParser.parse(text);
  let node: SyntaxNode | null = tree.resolveInner(pos, -1);
  if (node && node.name !== "VariableName" && node.name !== "PropertyName") {
    node = tree.resolveInner(pos, 1);
  }
  if (!node || (node.name !== "VariableName" && node.name !== "PropertyName")) {
    return null;
  }
  const name = text.slice(node.from, node.to);
  // A PropertyName (`df.groupby`) hangs off a MemberExpression; the receiver is
  // its previous sibling. Read the receiver's trailing identifier as qualifier.
  let qualifier: string | undefined;
  if (node.name === "PropertyName") {
    const object = node.parent?.firstChild ?? null;
    if (object && object.to <= node.from) {
      const objText = text.slice(object.from, object.to);
      const m = objText.match(/([A-Za-z_][A-Za-z0-9_]*)\s*$/);
      qualifier = m ? m[1] : objText;
    }
  }
  return { name, qualifier, kind: "function", language: "python" };
}

// --- SQL (SQLite) --------------------------------------------------------
const SQL_WORD = /[A-Za-z0-9_]/;

function sqlSymbol(text: string, pos: number): HelpRequest | null {
  // Guard: a click inside a string or comment is not a symbol.
  const mask = maskStringsAndComments(text, "sql");
  if (pos < 0 || pos >= text.length) return null;
  if (mask[pos] !== text[pos]) return null;
  if (!SQL_WORD.test(text[pos])) return null;

  // Token boundaries. The Lezer node gives the token; fall back to a word scan.
  const tree = sqlParser.parse(text);
  const node = tree.resolveInner(pos, -1);
  let from = node.from;
  let to = node.to;
  if (!/^[A-Za-z0-9_]+$/.test(text.slice(from, to))) {
    from = pos;
    while (from > 0 && SQL_WORD.test(text[from - 1])) from--;
    to = pos;
    while (to < text.length && SQL_WORD.test(text[to])) to++;
  }
  let name = text.slice(from, to);
  if (!name) return null;
  const kind: HelpRequest["kind"] = node.name === "Keyword" ? "keyword" : "function";

  // Combine two-word clauses: GROUP BY, ORDER BY.
  const upper = name.toUpperCase();
  if (upper === "GROUP" || upper === "ORDER") {
    const after = text.slice(to).match(/^\s+([A-Za-z]+)/);
    if (after && after[1].toUpperCase() === "BY") {
      name = `${upper} BY`;
      return { name, kind: "keyword", language: "sql" };
    }
  }
  return { name, kind, language: "sql" };
}
