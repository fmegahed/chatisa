/**
 * Jupyter notebook (.ipynb) extraction (v6.1.1). A notebook is JSON whose
 * bulk is usually base64 plot blobs; neither roster provider reads the format
 * natively, so the readable cells are pulled out client-side (the office-text
 * pattern) and the plots are handed back separately for callers that want to
 * attach them as native image parts. Pure string/JSON processing, no browser
 * APIs, so it runs identically in the browser and in unit tests.
 */

export interface NotebookImage {
  mediaType: "image/png" | "image/jpeg";
  /** Raw base64 payload as stored in the notebook (no data: prefix). */
  base64: string;
  /** 0-based index of the cell the plot came from. */
  cellIndex: number;
}

export interface NotebookText {
  /** Markdown + fenced code cells + capped text outputs, in cell order. */
  text: string;
  /** Plot outputs, at most `maxImages`, in document order. */
  images: NotebookImage[];
  cellCount: number;
  /** Kernel language ("python", "r", ...); drives the code fences. */
  language: string;
}

/** Per-output text cap: enough for a head() or a traceback, not a data dump. */
const DEFAULT_OUTPUT_CHARS = 2_000;

/** nbformat stores source and text either as one string or a line array. */
function joinLines(value: unknown): string {
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return value.filter((v) => typeof v === "string").join("");
  return "";
}

/** Terminal color codes from rich tracebacks (IPython) read as garbage.
 * Built via fromCharCode so no control character lives in the source. */
const ANSI_RE = new RegExp(`${String.fromCharCode(27)}\\[[0-9;]*[A-Za-z]`, "g");

function stripAnsi(text: string): string {
  return text.replace(ANSI_RE, "");
}

function capOutput(text: string, max: number): string {
  const clean = stripAnsi(text).trimEnd();
  if (clean.length <= max) return clean;
  return `${clean.slice(0, max)}\n[output truncated]`;
}

/**
 * Extracts the readable content of an nbformat-4 notebook. Returns null when
 * the JSON does not parse or has no cells array, so callers can fall back to
 * treating the file as plain text; a notebook never hard-fails an upload.
 */
export function notebookToText(
  raw: string,
  opts?: { maxOutputChars?: number; maxImages?: number },
): NotebookText | null {
  const maxOutputChars = opts?.maxOutputChars ?? DEFAULT_OUTPUT_CHARS;
  const maxImages = opts?.maxImages ?? 4;

  let nb: unknown;
  try {
    nb = JSON.parse(raw);
  } catch {
    return null;
  }
  if (typeof nb !== "object" || nb === null) return null;
  const cells = (nb as { cells?: unknown }).cells;
  if (!Array.isArray(cells)) return null;

  const metadata = (nb as { metadata?: Record<string, unknown> }).metadata ?? {};
  const kernelspec = metadata.kernelspec as { language?: unknown } | undefined;
  const languageInfo = metadata.language_info as { name?: unknown } | undefined;
  const language =
    (typeof kernelspec?.language === "string" && kernelspec.language) ||
    (typeof languageInfo?.name === "string" && languageInfo.name) ||
    "python";

  const blocks: string[] = [];
  const images: NotebookImage[] = [];

  cells.forEach((cell: unknown, cellIndex: number) => {
    if (typeof cell !== "object" || cell === null) return;
    const c = cell as {
      cell_type?: unknown;
      source?: unknown;
      outputs?: unknown;
    };
    const source = joinLines(c.source).trimEnd();

    if (c.cell_type === "markdown" || c.cell_type === "raw") {
      if (source.length > 0) blocks.push(source);
      return;
    }
    if (c.cell_type !== "code") return;

    blocks.push(`\`\`\`${language.toLowerCase()}\n${source}\n\`\`\``);

    const outputs = Array.isArray(c.outputs) ? c.outputs : [];
    const outputLines: string[] = [];
    for (const output of outputs) {
      if (typeof output !== "object" || output === null) continue;
      const o = output as {
        output_type?: unknown;
        text?: unknown;
        data?: Record<string, unknown>;
        ename?: unknown;
        evalue?: unknown;
        traceback?: unknown;
      };
      if (o.output_type === "stream") {
        const text = joinLines(o.text);
        if (text.trim()) outputLines.push(capOutput(text, maxOutputChars));
        continue;
      }
      if (o.output_type === "error") {
        const trace = joinLines(
          Array.isArray(o.traceback) ? o.traceback.join("\n") : o.traceback,
        );
        const head = [o.ename, o.evalue].filter((v) => typeof v === "string").join(": ");
        const text = [head, trace].filter(Boolean).join("\n");
        if (text.trim()) outputLines.push(capOutput(text, maxOutputChars));
        continue;
      }
      if (o.output_type === "execute_result" || o.output_type === "display_data") {
        const data = o.data ?? {};
        const png = joinLines(data["image/png"]);
        const jpeg = joinLines(data["image/jpeg"]);
        const imagePayload = png || jpeg;
        if (imagePayload) {
          if (images.length < maxImages) {
            images.push({
              mediaType: png ? "image/png" : "image/jpeg",
              base64: imagePayload.replace(/\s/g, ""),
              cellIndex,
            });
            outputLines.push(`[plot ${images.length} from cell ${cellIndex + 1}]`);
          } else {
            outputLines.push("[plot output omitted]");
          }
          continue;
        }
        const text = joinLines(data["text/plain"]);
        if (text.trim()) outputLines.push(capOutput(text, maxOutputChars));
      }
    }
    if (outputLines.length > 0) blocks.push(`Output:\n${outputLines.join("\n")}`);
  });

  return {
    text: blocks.join("\n\n"),
    images,
    cellCount: cells.length,
    language: language.toLowerCase(),
  };
}
