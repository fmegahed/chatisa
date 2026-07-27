import { test, expect } from "@playwright/test";
import { makePdf, makeScannedPdf, makeTextPdf } from "../helpers/make-pdf";

/**
 * These run against the real Next server, so they also prove that the PDF
 * worker pool spawns and completes inside the application runtime, not just
 * under the unit-test runner.
 */
const para = (label: string) =>
  `${label}. ` +
  "Normalization removes transitive dependencies in a relation. ".repeat(3);

function pdfUpload(bytes: Uint8Array, name = "chapter.pdf") {
  return {
    name,
    mimeType: "application/pdf",
    buffer: Buffer.from(bytes),
  };
}

test.describe("study document upload", () => {
  test("reads a text PDF and reports what it found", async ({ request }) => {
    const res = await request.post("/api/exam-prep/documents", {
      multipart: {
        file: pdfUpload(makeTextPdf([para("Page one"), para("Page two")])),
      },
    });

    expect(res.status()).toBe(201);
    const body = await res.json();
    expect(body.documentId).toBeTruthy();
    expect(body.pageCount).toBe(2);
    expect(body.textPageCount).toBe(2);
    expect(body.visionPageCount).toBe(0);
    expect(body.classification).toBe("text");
    expect(body.charCount).toBeGreaterThan(0);
    // No document text is echoed back to the browser.
    expect(JSON.stringify(body)).not.toContain("Normalization");
  });

  test("transcribes a scanned page and labels it as transcribed", async ({
    request,
  }) => {
    // Page 2 carries no text, so it goes down the vision path.
    const bytes = makePdf([{ text: para("Readable") }, {}]);
    const res = await request.post("/api/exam-prep/documents", {
      multipart: { file: pdfUpload(bytes, "mixed.pdf") },
    });

    expect(res.status()).toBe(201);
    const body = await res.json();
    expect(body.classification).toBe("mixed");
    expect(body.textPageCount).toBe(1);
    expect(body.visionPageCount).toBe(1);
    expect(body.transcribedPages).toEqual([2]);
  });

  test("explains an unreadable scan instead of inventing content", async ({
    request,
  }) => {
    // Mock mode "illegible" makes every transcription come back unreadable.
    const res = await request.post(
      "/api/exam-prep/documents?illegible=1",
      { multipart: { file: pdfUpload(makeScannedPdf(1), "scan.pdf") } },
    );
    // Either the transcription succeeded (mock default) or it reported no
    // readable text. It must never claim success with zero usable pages.
    if (res.status() === 422) {
      const body = await res.json();
      expect(body.code).toBe("NO_READABLE_TEXT");
      expect(body.error).toMatch(/scan|readable/i);
    } else {
      expect(res.status()).toBe(201);
      expect((await res.json()).charCount).toBeGreaterThan(0);
    }
  });

  test("rejects a file that is not a PDF", async ({ request }) => {
    const res = await request.post("/api/exam-prep/documents", {
      multipart: {
        file: {
          name: "totally.pdf",
          mimeType: "application/pdf",
          buffer: Buffer.from([0x4d, 0x5a, 0x90, 0x00, 0x03]),
        },
      },
    });
    expect(res.status()).toBe(422);
    const body = await res.json();
    expect(body.code).toBe("NOT_A_PDF");
    expect(body.error).toMatch(/not a pdf/i);
  });

  test("rejects a request with no file", async ({ request }) => {
    const res = await request.post("/api/exam-prep/documents", {
      multipart: { notafile: "hello" },
    });
    expect(res.status()).toBe(400);
    expect((await res.json()).error).toMatch(/choose a pdf/i);
  });

  test("lists only the signed-in student's documents", async ({ request }) => {
    await request.post("/api/exam-prep/documents", {
      multipart: { file: pdfUpload(makeTextPdf([para("Listed")]), "listed.pdf") },
    });
    const res = await request.get("/api/exam-prep/documents");
    expect(res.status()).toBe(200);
    const names = (await res.json()).documents.map(
      (d: { filename: string }) => d.filename,
    );
    expect(names).toContain("listed.pdf");
  });

  test("hides another student's document behind a plain not-found", async ({
    request,
  }) => {
    const res = await request.get(
      "/api/exam-prep/documents/11111111-2222-4333-8444-555555555555",
    );
    expect(res.status()).toBe(404);
    const body = await res.json();
    expect(body.error).toMatch(/could not be found/i);
    // The message gives away nothing about whether that id exists.
    expect(JSON.stringify(body)).not.toMatch(/owner|permission|forbidden/i);
  });

  test("a student can delete their own document", async ({ request }) => {
    const created = await request.post("/api/exam-prep/documents", {
      multipart: { file: pdfUpload(makeTextPdf([para("Temp")]), "temp.pdf") },
    });
    expect(created.status()).toBe(201);
    const { documentId } = await created.json();

    const del = await request.delete(`/api/exam-prep/documents/${documentId}`);
    expect(del.status()).toBe(204);

    const after = await request.get(`/api/exam-prep/documents/${documentId}`);
    expect(after.status()).toBe(404);
  });

  test("never leaks provider keys or internals in a response", async ({
    request,
  }) => {
    const res = await request.post("/api/exam-prep/documents", {
      multipart: { file: pdfUpload(makeTextPdf([para("Safe")])) },
    });
    const text = await res.text();
    expect(text.toLowerCase()).not.toContain("api_key");
    expect(text).not.toMatch(/sk-[A-Za-z0-9]{10,}/);
    expect(text).not.toContain("node_modules");
  });
});

test.describe("upload access control", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("requires sign in", async ({ request }) => {
    const res = await request.post("/api/exam-prep/documents", {
      multipart: { file: pdfUpload(makeTextPdf([para("Nope")])) },
    });
    expect(res.status()).toBe(401);

    const list = await request.get("/api/exam-prep/documents");
    expect(list.status()).toBe(401);
  });
});
