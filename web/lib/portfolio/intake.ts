/**
 * Browser-side file intake for the Portfolio Builder. Uploads never leave the
 * browser as bytes: this turns a picked File into a PreparedFile that carries
 * text for the model prompt and/or base64 for the GitHub push, with a size cap
 * that keeps a single file inside what the contents API will accept.
 */
import { notebookToText } from "@/lib/files/notebook-text";
import { officeTextFromFile } from "@/lib/files/office-text";
import type { FileRole, PreparedFile } from "./files";

const TEXT_EXT = /\.(py|r|ipynb|sql|md|txt|csv|tsv|qmd|rmd|js|ts|json|yml|yaml|html)$/i;
const OFFICE_EXT = /\.(docx|pptx)$/i;
const MAX_TEXT_BYTES = 400_000;
const MAX_NOTEBOOK_BYTES = 5_000_000;

export async function fileToBase64(file: File): Promise<string> {
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export async function prepareFile(file: File, role: FileRole): Promise<PreparedFile> {
  const base = { name: file.name, role, bytes: file.size, publish: role !== "data" };
  const isNotebook = /\.ipynb$/i.test(file.name);
  if (TEXT_EXT.test(file.name) && file.size <= (isNotebook ? MAX_NOTEBOOK_BYTES : MAX_TEXT_BYTES)) {
    const raw = await file.text();
    if (isNotebook) {
      const parsed = notebookToText(raw, { maxImages: 0 });
      if (parsed) {
        // The model sees the stripped cells; the push keeps the original
        // notebook (plots included) whenever it fits under the per-file cap.
        const base64 = file.size <= MAX_TEXT_BYTES ? await fileToBase64(file) : "";
        return { ...base, publish: base.publish && base64.length > 0, text: parsed.text, base64 };
      }
      if (file.size > MAX_TEXT_BYTES) return { ...base, publish: false, text: null, base64: "" };
    }
    return { ...base, text: raw, base64: null };
  }
  if (file.size > MAX_TEXT_BYTES) {
    return { ...base, publish: false, text: null, base64: "" };
  }
  if (OFFICE_EXT.test(file.name)) {
    let text: string | null = null;
    try {
      text = await officeTextFromFile(file, /\.docx$/i.test(file.name) ? "docx" : "pptx");
    } catch {
      text = null;
    }
    return { ...base, text, base64: await fileToBase64(file) };
  }
  return { ...base, text: null, base64: await fileToBase64(file) };
}

export function toRoutePayloadFile(f: PreparedFile):
  | { kind: "text"; name: string; content: string }
  | { kind: "binary"; name: string; sizeBytes: number } {
  return f.text !== null
    ? { kind: "text", name: f.name, content: f.text }
    : { kind: "binary", name: f.name, sizeBytes: f.bytes };
}

export function pushable(f: PreparedFile): boolean {
  // An empty base64 means "the bytes were not held", not "an empty file": an
  // oversize notebook keeps its stripped cell text for the prompt but has no
  // bytes to push, and pushing the stripped text under the .ipynb name would
  // publish a broken notebook. So base64 decides whenever it is present, and
  // text only stands in for files that were read as text all along.
  return f.base64 === null ? f.text !== null : f.base64.length > 0;
}
