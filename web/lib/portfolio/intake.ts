/**
 * Browser-side file intake for the Portfolio Builder. Uploads never leave the
 * browser as bytes: this turns a picked File into a PreparedFile that carries
 * text for the model prompt and/or base64 for the GitHub push, with a size cap
 * that keeps a single file inside what the contents API will accept.
 */
import { notebookToText } from "@/lib/files/notebook-text";
import { officeTextFromFile } from "@/lib/files/office-text";
import { MAX_CHARS_PER_FILE, type FileRole, type PreparedFile } from "./files";
import { PUSH_LIMITS } from "@/lib/scout/github";

const TEXT_EXT = /\.(py|r|ipynb|sql|md|txt|csv|tsv|qmd|rmd|js|ts|json|yml|yaml|html)$/i;
const OFFICE_EXT = /\.(docx|pptx)$/i;
/** Text read for the model prompt; larger text files are held as bytes only. */
export const MAX_TEXT_BYTES = 400_000;
export const MAX_NOTEBOOK_BYTES = 5_000_000;
/** Bytes held in the browser for the push; must match PUSH_LIMITS.fileBytes. */
const MAX_HELD_BYTES = PUSH_LIMITS.fileBytes;

/**
 * Base64 for the push. In the browser FileReader does the encoding natively,
 * which matters at the 25 MB per-file cap: building the string by hand holds
 * the bytes, a binary string, and the base64 all at once, roughly four times
 * the file, and that is what tips a small laptop or a phone into a silent tab
 * crash. The manual path remains for runtimes without FileReader (tests).
 */
export async function fileToBase64(file: File): Promise<string> {
  if (typeof FileReader !== "undefined") {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const url = String(reader.result ?? "");
        resolve(url.slice(url.indexOf(",") + 1));
      };
      reader.onerror = () => reject(reader.error ?? new Error("That file could not be read."));
      reader.readAsDataURL(file);
    });
  }
  const bytes = new Uint8Array(await file.arrayBuffer());
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

export function base64ToBytes(base64: string): Uint8Array<ArrayBuffer> {
  const binary = atob(base64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

export async function prepareFile(file: File, role: FileRole): Promise<PreparedFile> {
  const base = { name: file.name, role, bytes: file.size, publish: role !== "data" };
  if (file.size > MAX_HELD_BYTES) {
    // Too large to push: nothing is held, and the model only hears its size.
    return { ...base, publish: false, text: null, base64: "" };
  }
  const isNotebook = /\.ipynb$/i.test(file.name);
  if (TEXT_EXT.test(file.name) && file.size <= (isNotebook ? MAX_NOTEBOOK_BYTES : MAX_TEXT_BYTES)) {
    const raw = await file.text();
    if (isNotebook) {
      const parsed = notebookToText(raw, { maxImages: 0 });
      // The model sees the stripped cells; the push keeps the original
      // notebook, plots included.
      if (parsed) return { ...base, text: parsed.text, base64: await fileToBase64(file) };
    }
    return { ...base, text: raw, base64: null };
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
    ? { kind: "text", name: f.name, content: f.text.slice(0, MAX_CHARS_PER_FILE) }
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
