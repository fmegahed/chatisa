import { afterAll, describe, expect, it } from "vitest";
import {
  MIN_CHARS_FOR_TEXT_PAGE,
  PdfError,
  looksLikePdf,
  normalizePageText,
  pagesNeedingVision,
  readPdf,
  type PdfPage,
} from "@/lib/exam/pdf";
import { shutdownPdfPool } from "@/lib/exam/pdf-pool";
import { makePdf, makeScannedPdf, makeTextPdf } from "../helpers/make-pdf";

afterAll(async () => {
  await shutdownPdfPool();
});

/** Long enough to clear the text-page threshold. */
const para = (label: string) =>
  `${label}. ` + "Normalization removes transitive dependencies in a relation. ".repeat(2);

describe("looksLikePdf", () => {
  it("accepts the PDF magic number", () => {
    expect(looksLikePdf(makeTextPdf([para("A")]))).toBe(true);
  });

  it("rejects a renamed non-PDF file", () => {
    const exe = new Uint8Array([0x4d, 0x5a, 0x90, 0x00, 0x03]);
    expect(looksLikePdf(exe)).toBe(false);
  });

  it("rejects a file too short to classify", () => {
    expect(looksLikePdf(new Uint8Array([0x25, 0x50]))).toBe(false);
  });
});

describe("normalizePageText", () => {
  it("normalizes line endings and collapses excess blank lines", () => {
    expect(normalizePageText("a\r\n\r\n\r\n\r\nb")).toBe("a\n\nb");
  });

  it("removes soft hyphens and expands common ligatures", () => {
    expect(normalizePageText("clas­sify the ﬁle and ﬂow")).toBe(
      "classify the file and flow",
    );
  });

  it("trims trailing spaces before newlines", () => {
    expect(normalizePageText("a   \nb")).toBe("a\nb");
  });
});

describe("readPdf", () => {
  it("reads a text document page by page", async () => {
    const pdf = makeTextPdf([para("Page one"), para("Page two")]);
    const result = await readPdf(pdf);

    expect(result.pageCount).toBe(2);
    expect(result.classification).toBe("text");
    expect(result.pages.map((p: PdfPage) => p.pageNumber)).toEqual([1, 2]);
    expect(result.pages[0].source).toBe("text");
    expect(result.pages[0].text).toContain("Page one");
    expect(result.pages[1].text).toContain("Page two");
    expect(result.pages[0].charCount).toBeGreaterThan(
      MIN_CHARS_FOR_TEXT_PAGE,
    );
    expect(pagesNeedingVision(result)).toEqual([]);
  });

  it("classifies a document with no selectable text as scanned", async () => {
    const result = await readPdf(makeScannedPdf(3));

    expect(result.classification).toBe("scanned");
    expect(result.pages.every((p: PdfPage) => p.source === "needs_vision")).toBe(true);
    expect(pagesNeedingVision(result)).toEqual([1, 2, 3]);
  });

  it("routes each page independently in a mixed document", async () => {
    // Pages 1 and 3 carry text; page 2 is drawing only, like a scanned insert.
    const pdf = makePdf([
      { text: para("Readable one") },
      {},
      { text: para("Readable three") },
    ]);
    const result = await readPdf(pdf);

    expect(result.classification).toBe("mixed");
    expect(result.pages.map((p: PdfPage) => p.source)).toEqual([
      "text",
      "needs_vision",
      "text",
    ]);
    expect(pagesNeedingVision(result)).toEqual([2]);
    // A page awaiting transcription holds no partial text.
    expect(result.pages[1].text).toBe("");
    expect(result.pages[1].charCount).toBe(0);
  });

  it("treats a page with only a scrap of text as needing vision", async () => {
    const result = await readPdf(makeTextPdf(["Fig. 2"]));
    expect(result.pages[0].source).toBe("needs_vision");
  });

  it("rejects a file that is not a PDF before parsing it", async () => {
    const exe = new Uint8Array([0x4d, 0x5a, 0x90, 0x00, 0x03, 0x00]);
    await expect(readPdf(exe)).rejects.toMatchObject({
      name: "PdfError",
      code: "NOT_A_PDF",
    });
  });

  it("reports unreadable PDFs as an error rather than empty text", async () => {
    const corrupt = new TextEncoder().encode("%PDF-1.4\nnot really a pdf");
    await expect(readPdf(corrupt)).rejects.toBeInstanceOf(PdfError);
  });

  it("never surfaces the uploaded bytes in the error message", async () => {
    const exe = new Uint8Array([0x4d, 0x5a, 0x90]);
    await expect(readPdf(exe)).rejects.toThrowError(
      /that file is not a pdf/i,
    );
  });

  it("reads a real course PDF, not just synthetic fixtures", async () => {
    // The scoping worksheet ships with the repo and is genuinely text-bearing.
    const path = new URL(
      "../../../assets/project_scoping_worksheet.pdf",
      import.meta.url,
    );
    const { readFileSync, existsSync } = await import("node:fs");
    if (!existsSync(path)) return; // asset not present; synthetic cases still cover the logic
    const result = await readPdf(new Uint8Array(readFileSync(path)));

    expect(result.pageCount).toBe(7);
    expect(result.classification).toBe("text");
    expect(result.pages[0].text).toContain("Project Scoping Worksheet");
    expect(pagesNeedingVision(result)).toEqual([]);
  });

  it("stops early and reports when the deadline passes", async () => {
    const pdf = makeTextPdf(
      Array.from({ length: 40 }, (_, i) => para(`Page ${i + 1}`)),
    );
    const result = await readPdf(pdf, { deadlineMs: 0 });
    expect(result.warnings).toContain("DEADLINE_REACHED");
    expect(result.pages.length).toBeLessThan(40);
  });
});

describe("page rendering for the vision path", () => {
  it("returns a rendered PNG for every page needing transcription", async () => {
    const result = await readPdf(makeScannedPdf(2));
    expect(result.images.map((i) => i.pageNumber)).toEqual([1, 2]);
    const png = result.images[0].png;
    // PNG signature: the vision path needs a real image, not a stub.
    expect(Array.from(png.slice(0, 4))).toEqual([0x89, 0x50, 0x4e, 0x47]);
    expect(png.byteLength).toBeGreaterThan(500);
  }, 60_000);

  it("does not render pages that already have text", async () => {
    const result = await readPdf(makeTextPdf([para("A"), para("B")]));
    expect(result.images).toEqual([]);
  }, 60_000);

  it("caps how many pages it renders for one document", async () => {
    const result = await readPdf(makeScannedPdf(5), { maxVisionPages: 2 });
    expect(result.images).toHaveLength(2);
    expect(result.skippedVisionPages).toEqual([3, 4, 5]);
  }, 60_000);
});
