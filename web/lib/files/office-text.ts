/**
 * Text extraction for Word and PowerPoint attachments (slice C). Office files
 * are zip archives of XML; neither roster provider reads them in messages
 * (Anthropic's docs say to convert to plain text), so the text is pulled out
 * client-side and rides as an attachment block.
 *
 * The XML-to-text functions are pure string processing (no DOMParser), so they
 * run identically in the browser and in unit tests. That is deliberate: OOXML
 * text lives in leaf elements (`w:t`, `a:t`) whose extraction needs no real
 * XML tree, and a parser would add nothing but an environment dependency.
 */

/** Decodes the five XML entities OOXML uses in text runs. */
function decodeEntities(s: string): string {
  return s
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/&amp;/g, "&");
}

/** All text inside `<tag ...>...</tag>` occurrences, in document order. */
function leafTexts(xml: string, tag: string): string[] {
  const re = new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)</${tag}>`, "g");
  const out: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(xml)) !== null) out.push(decodeEntities(m[1]));
  return out;
}

/**
 * word/document.xml to plain text: one line per paragraph (`w:p`), text runs
 * (`w:t`) joined, tabs and line breaks preserved. Empty paragraphs collapse.
 */
export function docxXmlToText(xml: string): string {
  const paragraphs = xml.split(/<\/w:p>/);
  const lines: string[] = [];
  for (const p of paragraphs) {
    // Tabs and breaks are empty elements BETWEEN text runs; turning them into
    // synthetic runs keeps them in document order when the runs are joined.
    const withBreaks = p
      .replace(/<w:tab\/>/g, "<w:t>\t</w:t>")
      .replace(/<w:br\/>/g, "<w:t>\n</w:t>");
    const text = leafTexts(withBreaks, "w:t").join("");
    if (text.trim().length > 0) lines.push(text);
  }
  return lines.join("\n");
}

/** One slide's XML (`ppt/slides/slideN.xml`) to text: paragraphs (`a:p`) to
 * lines, runs (`a:t`) joined. */
export function pptxSlideXmlToText(xml: string): string {
  const paragraphs = xml.split(/<\/a:p>/);
  const lines: string[] = [];
  for (const p of paragraphs) {
    const text = leafTexts(p, "a:t").join("");
    if (text.trim().length > 0) lines.push(text);
  }
  return lines.join("\n");
}

/** Orders slide file paths numerically (slide2 before slide10). */
export function sortSlidePaths(paths: string[]): string[] {
  const num = (p: string) => Number(/slide(\d+)\.xml$/.exec(p)?.[1] ?? 0);
  return [...paths]
    .filter((p) => /ppt\/slides\/slide\d+\.xml$/.test(p))
    .sort((a, b) => num(a) - num(b));
}

/**
 * Browser-side entry: opens the zip (jszip, lazily imported so the chunk
 * loads only when someone attaches an Office file) and extracts the text.
 * Throws with a student-readable message when the file is not what its
 * extension claims.
 */
export async function officeTextFromFile(
  file: File,
  kind: "docx" | "pptx",
): Promise<string> {
  const { default: JSZip } = await import("jszip");
  let zip: InstanceType<typeof JSZip>;
  try {
    zip = await JSZip.loadAsync(await file.arrayBuffer());
  } catch {
    throw new Error(`"${file.name}" could not be opened. Is it a real ${kind} file?`);
  }

  if (kind === "docx") {
    const doc = zip.file("word/document.xml");
    if (!doc) throw new Error(`"${file.name}" has no Word document inside.`);
    return docxXmlToText(await doc.async("string"));
  }

  const slidePaths = sortSlidePaths(Object.keys(zip.files));
  if (slidePaths.length === 0)
    throw new Error(`"${file.name}" has no slides inside.`);
  const parts: string[] = [];
  for (const [i, path] of slidePaths.entries()) {
    const xml = await zip.file(path)!.async("string");
    const text = pptxSlideXmlToText(xml);
    if (text.trim().length > 0) parts.push(`[Slide ${i + 1}]\n${text}`);
  }
  return parts.join("\n\n");
}
