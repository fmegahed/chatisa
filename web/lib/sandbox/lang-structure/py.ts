import { parser } from "@lezer/python";
import type { SyntaxNode, Tree } from "@lezer/common";

/**
 * The top-level statement node containing `pos`: the ancestor whose parent is
 * the `Script` root. Returns null only for an empty document or a cursor in
 * leading whitespace with no statement to attach to.
 */
export function topLevelStatementAt(tree: Tree, pos: number): SyntaxNode | null {
  // Bias left so a cursor at a line end stays in the statement it terminates;
  // if that lands on the Script root (blank line / gap), try biasing right, then
  // the nearest child before/after the position.
  let node: SyntaxNode | null = tree.resolveInner(pos, -1);
  if (node.name === "Script") node = tree.resolveInner(pos, 1);
  if (node.name === "Script") {
    node = tree.topNode.childBefore(pos) ?? tree.topNode.childAfter(pos);
  }
  if (!node) return null;
  while (node.parent && node.parent.name !== "Script") node = node.parent;
  return node.parent ? node : null; // must be a direct child of Script
}

/**
 * The range of the complete Python logical statement/block containing `pos`.
 * Falls back to the physical line when no statement resolves, so the caller
 * never runs less than a line.
 */
export function pyStatementRange(
  text: string,
  pos: number,
): { from: number; to: number } {
  const tree = parser.parse(text);
  const node = topLevelStatementAt(tree, pos);
  if (node) {
    // A compound block's Body node extends through its terminating newline in
    // this @lezer/python build; StatementRange excludes the trailing newline, so
    // trim trailing whitespace off the end.
    let to = node.to;
    while (to > node.from && /\s/.test(text[to - 1])) to--;
    return { from: node.from, to };
  }
  return physicalLine(text, pos);
}

function physicalLine(text: string, pos: number): { from: number; to: number } {
  let from = pos;
  while (from > 0 && text[from - 1] !== "\n") from--;
  let to = pos;
  while (to < text.length && text[to] !== "\n") to++;
  return { from, to };
}
