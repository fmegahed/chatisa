import { SQLite } from "@codemirror/lang-sql";
import type { SyntaxNode, Tree } from "@lezer/common";

const sqlParser = SQLite.language.parser;

/**
 * The `Statement` node containing `pos`, or (for a cursor in a gap between
 * statements or in a comment) the nearest `Statement`. Returns null only for an
 * empty document.
 */
export function statementNodeAt(tree: Tree, pos: number): SyntaxNode | null {
  // Bias left so a cursor at a statement's end stays in the statement it
  // terminates; if that lands in the gap between statements (e.g. the cursor is
  // exactly at the start of the next statement), bias right instead.
  const left = climbToStatement(tree.resolveInner(pos, -1));
  if (left) return left;
  const right = climbToStatement(tree.resolveInner(pos, 1));
  if (right) return right;
  // Cursor sits between statements (blank line/comment): pick the nearest one.
  const before = tree.topNode.childBefore(pos);
  if (before && before.name === "Statement") return before;
  const after = tree.topNode.childAfter(pos);
  if (after && after.name === "Statement") return after;
  return null;
}

function climbToStatement(start: SyntaxNode): SyntaxNode | null {
  let node: SyntaxNode | null = start;
  while (node && node.name !== "Statement" && node.parent) node = node.parent;
  return node && node.name === "Statement" ? node : null;
}

/**
 * The range of the complete SQL statement containing `pos` (through its `;` or
 * the end of the script). Falls back to the physical line when nothing resolves.
 */
export function sqlStatementRange(
  text: string,
  pos: number,
): { from: number; to: number } {
  const tree = sqlParser.parse(text);
  const node = statementNodeAt(tree, pos);
  if (node) return { from: node.from, to: node.to };
  let from = pos;
  while (from > 0 && text[from - 1] !== "\n") from--;
  let to = pos;
  while (to < text.length && text[to] !== "\n") to++;
  return { from, to };
}
