/**
 * Builds tiny valid PDFs in memory for tests, with no external dependency.
 *
 * Pages are either text-bearing or drawing-only. A drawing-only page yields no
 * extractable text, which is how a scanned page behaves, so these fixtures
 * exercise the automatic routing without committing binary files.
 */

interface PageSpec {
  /** Omit for a drawing-only page (stands in for a scanned page). */
  text?: string;
}

function contentStream(spec: PageSpec): string {
  if (spec.text === undefined) {
    // A filled rectangle: visible content, zero text.
    return "0.9 0.9 0.9 rg\n72 600 200 120 re\nf\n";
  }
  // Escape the few characters that are special inside a PDF string literal.
  const escaped = spec.text
    .replace(/\\/g, "\\\\")
    .replace(/\(/g, "\\(")
    .replace(/\)/g, "\\)");
  return `BT\n/F1 12 Tf\n72 720 Td\n14 TL\n(${escaped}) Tj\nET\n`;
}

/**
 * Assembles a minimal PDF with a correct cross-reference table.
 * Object layout: 1 catalog, 2 page tree, 3 font, then per page a page object
 * followed by its content stream.
 */
export function makePdf(pages: PageSpec[]): Uint8Array {
  const objects: string[] = [];
  const pageObjectNumbers: number[] = [];

  // Reserve 1..3 for catalog, page tree and font.
  const firstPageObject = 4;
  pages.forEach((spec, index) => {
    const pageNum = firstPageObject + index * 2;
    const contentNum = pageNum + 1;
    pageObjectNumbers.push(pageNum);
    const stream = contentStream(spec);
    objects[pageNum] =
      `${pageNum} 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] ` +
      `/Resources << /Font << /F1 3 0 R >> >> /Contents ${contentNum} 0 R >>\nendobj\n`;
    objects[contentNum] =
      `${contentNum} 0 obj\n<< /Length ${stream.length} >>\nstream\n${stream}endstream\nendobj\n`;
  });

  objects[1] = "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n";
  objects[2] =
    `2 0 obj\n<< /Type /Pages /Count ${pages.length} ` +
    `/Kids [${pageObjectNumbers.map((n) => `${n} 0 R`).join(" ")}] >>\nendobj\n`;
  objects[3] =
    "3 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n";

  const header = "%PDF-1.4\n";
  let body = "";
  const offsets: number[] = [];
  const highest = 3 + pages.length * 2;

  for (let i = 1; i <= highest; i += 1) {
    offsets[i] = header.length + body.length;
    body += objects[i] ?? `${i} 0 obj\n<< >>\nendobj\n`;
  }

  const xrefOffset = header.length + body.length;
  let xref = `xref\n0 ${highest + 1}\n0000000000 65535 f \n`;
  for (let i = 1; i <= highest; i += 1) {
    xref += `${String(offsets[i]).padStart(10, "0")} 00000 n \n`;
  }
  const trailer =
    `trailer\n<< /Size ${highest + 1} /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF\n`;

  return new TextEncoder().encode(header + body + xref + trailer);
}

/** A document whose pages all carry selectable text. */
export function makeTextPdf(texts: string[]): Uint8Array {
  return makePdf(texts.map((text) => ({ text })));
}

/** A document with no selectable text, standing in for a scanned upload. */
export function makeScannedPdf(pageCount: number): Uint8Array {
  return makePdf(Array.from({ length: pageCount }, () => ({})));
}
