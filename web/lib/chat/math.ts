/**
 * Math delimiter normalization for model output. remark-math parses TeX only
 * between dollar delimiters ($...$, $$...$$), but models routinely emit the
 * LaTeX bracket forms \( ... \) and \[ ... \]. Those are converted here,
 * OUTSIDE code (fenced blocks and inline spans stay byte-identical: a student
 * pasting regex or LaTeX source into a code block must see it untouched).
 *
 * Dollar amounts are safe: remark-math's own rules ignore "$5 and $10"
 * because a closing dollar cannot be followed by a digit.
 */

/** Fenced code blocks and inline code spans, kept verbatim. */
const CODE_SEGMENTS = /(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`)/g;

function convertOutsideCode(segment: string): string {
  return segment
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, body: string) => `\n$$\n${body}\n$$\n`)
    .replace(/\\\((.*?)\\\)/g, (_, body: string) => `$${body}$`);
}

export function normalizeMathDelimiters(markdown: string): string {
  if (!markdown.includes("\\(") && !markdown.includes("\\[")) return markdown;
  return markdown
    .split(CODE_SEGMENTS)
    .map((piece, index) =>
      // Odd indices are the captured code segments.
      index % 2 === 1 ? piece : convertOutsideCode(piece),
    )
    .join("");
}
