import { describe, it, expect } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import { Markdown } from "@/components/chat/Markdown";
import { demoteHeadings } from "@/lib/chat/headings";

const heading = (depth: number) => ({ type: "heading", depth, children: [] });

describe("demoteHeadings (#4)", () => {
  it("puts the shallowest heading one level below the message heading", () => {
    const tree = { type: "root", children: [heading(1), heading(2), heading(3)] };
    demoteHeadings(tree, 2);
    expect(tree.children.map((h) => h.depth)).toEqual([3, 4, 5]);
  });

  it("closes gaps between the levels a reply uses", () => {
    const tree = { type: "root", children: [heading(2), heading(4), heading(2)] };
    demoteHeadings(tree, 2);
    expect(tree.children.map((h) => h.depth)).toEqual([3, 4, 3]);
  });

  it("never goes past h6", () => {
    const tree = { type: "root", children: [1, 2, 3, 4, 5, 6].map(heading) };
    demoteHeadings(tree, 2);
    expect(tree.children.map((h) => h.depth)).toEqual([3, 4, 5, 6, 6, 6]);
  });
});

// Markdown has no hooks of its own, so it can be called directly; this keeps
// the test a .ts file (no JSX) without passing children as a prop.
const render = (text: string, headingLevel?: number) =>
  renderToStaticMarkup(Markdown({ children: text, headingLevel }));

describe("Markdown headings", () => {
  it("renders a reply's ## as an h3 that keeps the h2 look", () => {
    const html = render("## Summary\n\ntext\n\n### Detail");
    expect(html).toContain('<h3 class="md-h2">Summary</h3>');
    expect(html).toContain('<h4 class="md-h3">Detail</h4>');
    expect(html).not.toMatch(/<h[12][ >]/);
  });

  it("starts below a deeper message heading when told to", () => {
    expect(render("# Title", 3)).toContain('<h4 class="md-h1">Title</h4>');
  });
});
