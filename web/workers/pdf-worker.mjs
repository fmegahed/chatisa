/**
 * PDF worker: all CPU-heavy PDF work happens here, off the request thread.
 *
 * Parsing and rasterization block whichever thread they run on. Doing them on
 * the main thread stalls every other student's request, including streaming
 * chat responses. This process keeps that cost isolated.
 *
 * Run as a CHILD PROCESS rather than a worker thread, deliberately. Node's
 * module loader hooks are inherited by worker threads, and the framework's
 * dev and production servers register a bundler resolver that cannot resolve
 * the optional native canvas binding. A separate process gets clean Node
 * module resolution, and gives true CPU isolation as a bonus.
 *
 * Plain ESM on purpose: it is launched by absolute path and never goes
 * through the application bundler.
 */
import { extractText, getDocumentProxy, renderPageAsImage } from "unpdf";
import { classifyDocument, classifyPage } from "../lib/exam/pdf-core.mjs";

const MAX_PAGES = 1500;
const MAX_TOTAL_CHARS = 4_000_000;
const RENDER_SCALE = 2;

const canvasImport = () => import("@napi-rs/canvas");

async function processDocument({ bytes, maxVisionPages, deadlineMs }) {
  const deadline = Date.now() + (deadlineMs ?? 30_000);
  const warnings = [];

  let pdf;
  try {
    pdf = await getDocumentProxy(bytes);
  } catch (err) {
    const name = err?.name ?? "";
    if (name === "PasswordException") {
      const e = new Error(
        "That PDF is password protected. Upload a copy that opens without a password.",
      );
      e.code = "ENCRYPTED_PDF";
      throw e;
    }
    const e = new Error("That PDF could not be read.");
    e.code = "UNREADABLE_PDF";
    throw e;
  }

  const pageCount = pdf.numPages;
  if (pageCount > MAX_PAGES) {
    const e = new Error(
      `That PDF has ${pageCount.toLocaleString()} pages. The limit is ${MAX_PAGES.toLocaleString()}.`,
    );
    e.code = "TOO_MANY_PAGES";
    throw e;
  }

  const { text } = await extractText(pdf, { mergePages: false });
  const rawPages = Array.isArray(text) ? text : [text];

  const pages = [];
  let totalChars = 0;
  let cidPages = 0;

  for (let i = 0; i < rawPages.length; i += 1) {
    if (Date.now() > deadline) {
      warnings.push("DEADLINE_REACHED");
      break;
    }
    if (totalChars >= MAX_TOTAL_CHARS) {
      warnings.push("TRUNCATED_CHARS");
      break;
    }
    const raw = rawPages[i] ?? "";
    if (raw.includes("(cid:")) cidPages += 1;
    const page = classifyPage(i + 1, raw);
    totalChars += page.charCount;
    pages.push(page);
  }

  if (pages.length < rawPages.length && !warnings.includes("DEADLINE_REACHED")) {
    warnings.push("TRUNCATED_PAGES");
  }
  if (pages.length > 0 && cidPages / pages.length > 0.05) {
    warnings.push("NONSTANDARD_ENCODING");
  }

  // Render only the pages that need visual transcription, reusing the parsed
  // document so the file is not re-parsed once per page.
  const needVision = pages
    .filter((p) => p.source === "needs_vision")
    .map((p) => p.pageNumber);
  const cap = maxVisionPages ?? 40;
  const toRender = needVision.slice(0, cap);
  const skippedVisionPages = needVision.slice(cap);

  const images = [];
  for (const pageNumber of toRender) {
    if (Date.now() > deadline) {
      warnings.push("DEADLINE_REACHED");
      break;
    }
    const buffer = await renderPageAsImage(pdf, pageNumber, {
      scale: RENDER_SCALE,
      canvasImport,
    });
    images.push({ pageNumber, png: new Uint8Array(buffer) });
  }

  return {
    pageCount,
    pages,
    classification: classifyDocument(pages),
    warnings: [...new Set(warnings)],
    images,
    skippedVisionPages,
  };
}

process.on("message", async (message) => {
  const { id, payload } = message;
  try {
    const result = await processDocument(payload);
    process.send({ id, ok: true, result });
  } catch (err) {
    process.send({
      id,
      ok: false,
      code: err?.code ?? "UNREADABLE_PDF",
      message: err?.message ?? "That PDF could not be read.",
    });
  }
});

// Tell the parent we are ready to accept work.
process.send?.({ ready: true });
