/**
 * Pure helpers for Ask Anything attachments (slice C, plan 2026-07-24).
 * Classification, caps, and the shapes of the parts a file becomes. Everything
 * here is side-effect free so it is unit-testable; the browser-only work
 * (canvas, IndexedDB, the Pyodide session) lives elsewhere.
 *
 * Routing summary: images and PDFs ride NATIVELY as file parts (both roster
 * providers render them); Word/PowerPoint become extracted text; csv/xlsx are
 * loaded into the chat's Python session as a DataFrame and only an
 * announcement rides in the message; json/txt/md and code files (py, R, Rmd,
 * qmd, html) become capped text; Jupyter notebooks become extracted cells
 * plus up to four plot outputs riding natively as images (v6.1.1).
 */

export type AttachmentKind =
  | "image"
  | "pdf"
  | "dataset"
  | "office"
  | "notebook"
  | "text"
  | "unsupported";

/** Per file AND per message, matching the Coding Studio convention and staying
 * inside Anthropic's 32 MB request limit with headroom for the JSON around it. */
export const MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024;

/** Cap on extracted text per file. Keeps a whole chapter from consuming the
 * context; the note tells the model (and the student) it is looking at a cut. */
export const ATTACH_TEXT_MAX = 60_000;

/** Anthropic's per-document page limit; OpenAI's is the same order. Used by the
 * client-side heuristic to warn before a provider rejects the request. */
export const PDF_PAGE_SOFT_CAP = 100;

/** Prefix for persisted attachment references: the localStorage chat stores
 * `aa-file:<id>` in place of the data URL; bytes live in IndexedDB. */
export const FILE_REF_PREFIX = "aa-file:";

export interface Classified {
  kind: AttachmentKind;
  /** For datasets: how the runtime should parse it. */
  format?: "csv" | "xlsx";
  /** For office files: which XML layout to extract. */
  office?: "docx" | "pptx";
  /** Media type to use on a native file part (image/pdf). */
  mediaType?: string;
}

/** Formats read as plain text. Code files (.py, .R, .Rmd, .qmd) are listed by
 * extension because Windows browsers report an EMPTY MIME type for them, so
 * the text/* fallback below never fires (v6.1.1). */
const TEXT_EXTENSIONS = new Set([
  "txt",
  "md",
  "json",
  "py",
  "r",
  "rmd",
  "qmd",
  "html",
  "htm",
]);

const IMAGE_TYPES: Record<string, string> = {
  png: "image/png",
  jpg: "image/jpeg",
  jpeg: "image/jpeg",
  webp: "image/webp",
  gif: "image/gif",
};

/**
 * Classifies by extension first (students rename files; browsers report
 * unreliable MIME types for Office formats), falling back to the reported
 * MIME type for extensionless files.
 */
export function classifyFile(name: string, mimeType?: string): Classified {
  const ext = (name.split(".").pop() ?? "").toLowerCase();
  if (ext in IMAGE_TYPES) return { kind: "image", mediaType: IMAGE_TYPES[ext] };
  if (ext === "pdf") return { kind: "pdf", mediaType: "application/pdf" };
  if (ext === "csv" || ext === "tsv") return { kind: "dataset", format: "csv" };
  if (ext === "xlsx") return { kind: "dataset", format: "xlsx" };
  if (ext === "docx") return { kind: "office", office: "docx" };
  if (ext === "pptx") return { kind: "office", office: "pptx" };
  if (ext === "ipynb") return { kind: "notebook" };
  if (TEXT_EXTENSIONS.has(ext)) return { kind: "text" };
  if (mimeType?.startsWith("image/")) return { kind: "image", mediaType: mimeType };
  if (mimeType === "application/pdf")
    return { kind: "pdf", mediaType: "application/pdf" };
  if (mimeType?.startsWith("text/")) return { kind: "text" };
  return { kind: "unsupported" };
}

/** Student-readable reason for a refused file, or null when it is accepted. */
export function rejectionReason(
  name: string,
  sizeBytes: number,
  mimeType?: string,
): string | null {
  if (classifyFile(name, mimeType).kind === "unsupported") {
    return `"${name}" isn't a supported file type. Attach images, PDF, Word, PowerPoint, Excel, code and notebook files (.py, .R, .Rmd, .qmd, .ipynb), csv, json, html, md, or txt.`;
  }
  if (sizeBytes > MAX_ATTACHMENT_BYTES) {
    return `"${name}" is larger than 25 MB. Attach a smaller file, or the part of it you need.`;
  }
  if (sizeBytes === 0) return `"${name}" is empty.`;
  return null;
}

export function truncateAttachmentText(text: string): {
  text: string;
  truncated: boolean;
} {
  if (text.length <= ATTACH_TEXT_MAX) return { text, truncated: false };
  return {
    text: `${text.slice(0, ATTACH_TEXT_MAX)}\n[file truncated at ${ATTACH_TEXT_MAX} characters]`,
    truncated: true,
  };
}

/**
 * Best-effort page count from raw PDF bytes, WITHOUT a PDF library: counts
 * `/Type /Page` object markers (excluding the `/Pages` tree nodes). Modern
 * PDFs often keep page objects inside compressed object streams where this
 * finds nothing, so 0 means "unknown", never "zero pages". Good enough for a
 * pre-send warning; the provider gives the authoritative error.
 */
export function estimatePdfPages(bytes: Uint8Array): number | null {
  let text = "";
  // Latin-1 view of the raw bytes; chunked to avoid call-stack limits.
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    text += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  const matches = text.match(/\/Type\s*\/Page[^s]/g);
  return matches && matches.length > 0 ? matches.length : null;
}

/** The data carried by a data-attachment UI part. */
export interface AttachmentData {
  kind: "dataset" | "office" | "notebook" | "text";
  name: string;
  /** Short human label for the chip ("loaded as sales, 120 rows x 5 columns"). */
  detail: string;
  /** What the model reads (extracted text, or the dataset announcement). */
  text: string;
  truncated?: boolean;
}

/** The UI message part for a non-native attachment. The `data-` prefix is the
 * AI SDK's convention for custom parts; the server converts it via
 * convertToModelMessages' convertDataPart, other consumers ignore it. */
export function attachmentPart(data: AttachmentData): {
  type: "data-attachment";
  data: AttachmentData;
} {
  return { type: "data-attachment", data };
}

/**
 * The announcement the model receives for a dataset that was loaded into the
 * chat's Python session. It carries everything needed to start analyzing
 * without re-reading the file: the variable, the shape, and the columns.
 */
export function datasetAnnouncement(params: {
  fileName: string;
  varName: string;
  columns: string[];
  rowCount?: number;
}): string {
  const cols = params.columns.slice(0, 60);
  const colNote =
    params.columns.length > cols.length
      ? ` (first ${cols.length} of ${params.columns.length})`
      : "";
  const rows =
    params.rowCount != null ? `${params.rowCount} rows, ` : "";
  return [
    `The student attached "${params.fileName}". It is loaded in the run_python session as the pandas DataFrame \`${params.varName}\` (${rows}${cols.length} columns${colNote}).`,
    `Columns: ${cols.join(", ")}`,
    `Analyze it with run_python (for example \`${params.varName}.describe()\`). Do not ask the student to re-upload it. If \`${params.varName}\` is not defined (the session resets on reload), tell the student to re-attach the file.`,
  ].join("\n");
}

/**
 * Server-side rendering of a data-attachment part into the text block the
 * model actually reads. Bounded here as well: the client caps honestly, but
 * the server cannot trust request bodies.
 */
export function attachmentBlockText(data: AttachmentData): string {
  const capped = truncateAttachmentText(String(data.text ?? ""));
  const label =
    data.kind === "dataset"
      ? `[Attached dataset: ${data.name}]`
      : data.kind === "notebook"
        ? `[Attached notebook: ${data.name}]`
        : `[Attached file: ${data.name}]`;
  return `${label}\n${capped.text}`;
}
