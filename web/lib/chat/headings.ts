/**
 * Keep model-written headings below the message heading that contains them
 * (WCAG 1.3.1, issue #4). A reply sits under "ChatISA" (an h2 in the chat
 * log), so its own top heading must be an h3, whatever markdown level the
 * model chose. The plugin shifts the shallowest heading to `parentLevel + 1`
 * and closes gaps (a reply using only ## and #### becomes h3 and h4), capped
 * at h6. Each heading keeps an `md-hN` class for its original markdown level
 * so replies look the same as before.
 */
interface Node {
  type: string;
  depth?: number;
  children?: Node[];
  data?: { hProperties?: Record<string, unknown> };
}

function collect(node: Node, out: Node[]) {
  if (node.type === "heading") out.push(node);
  for (const child of node.children ?? []) collect(child, out);
}

export function demoteHeadings(tree: Node, parentLevel: number) {
  const headings: Node[] = [];
  collect(tree, headings);
  const levels = [...new Set(headings.map((h) => h.depth ?? 1))].sort((a, b) => a - b);
  for (const h of headings) {
    const original = h.depth ?? 1;
    const rank = levels.indexOf(original);
    h.depth = Math.min(6, parentLevel + 1 + rank);
    h.data = {
      ...h.data,
      hProperties: { ...h.data?.hProperties, className: `md-h${original}` },
    };
  }
}

/** remark plugin form: `[remarkDemoteHeadings, { parentLevel: 2 }]`. */
export function remarkDemoteHeadings(options: { parentLevel?: number } = {}) {
  const parentLevel = options.parentLevel ?? 2;
  return (tree: Node) => demoteHeadings(tree, parentLevel);
}
