import { afterAll, describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import {
  PdfBusyError,
  pdfPoolStats,
  processPdfInWorker,
  shutdownPdfPool,
} from "@/lib/exam/pdf-pool";
import { readPdf } from "@/lib/exam/pdf";
import { makeScannedPdf } from "../helpers/make-pdf";

afterAll(async () => {
  await shutdownPdfPool();
});

/**
 * Tracks the worst gap between timer ticks. While the event loop is blocked,
 * every other student's request is waiting, so this is the number that decides
 * whether the app survives a lab section rather than a demo.
 */
function startLagMonitor() {
  let max = 0;
  let last = performance.now();
  const timer = setInterval(() => {
    const now = performance.now();
    max = Math.max(max, now - last - 5);
    last = now;
  }, 5);
  return {
    stop() {
      clearInterval(timer);
      return max;
    },
  };
}

describe("PDF work stays off the request thread", () => {
  it("keeps the event loop responsive while rasterizing many pages", async () => {
    // Rasterization is the most expensive path: it previously blocked the
    // request thread for roughly its whole duration.
    const pdf = makeScannedPdf(6);

    const monitor = startLagMonitor();
    const started = performance.now();
    const result = await readPdf(pdf, { maxVisionPages: 6 });
    const elapsed = performance.now() - started;
    const lag = monitor.stop();

    expect(result.images).toHaveLength(6);
    // The work must actually have taken meaningful time, or the assertion
    // below proves nothing.
    expect(elapsed).toBeGreaterThan(100);
    // The request thread stayed free: lag is a small fraction of the work.
    expect(lag).toBeLessThan(elapsed / 2);
  }, 120_000);

  it("handles several documents at once without serializing on the main thread", async () => {
    const docs = [3, 3, 3, 3].map((n) => makeScannedPdf(n));

    const monitor = startLagMonitor();
    const started = performance.now();
    const results = await Promise.all(
      docs.map((bytes) => readPdf(bytes, { maxVisionPages: 3 })),
    );
    const elapsed = performance.now() - started;
    const lag = monitor.stop();

    expect(results).toHaveLength(4);
    for (const r of results) expect(r.images).toHaveLength(3);
    expect(lag).toBeLessThan(elapsed / 2);
  }, 180_000);

  it("reads a real course PDF through the pool", async () => {
    const path = new URL(
      "../../../assets/project_scoping_worksheet.pdf",
      import.meta.url,
    );
    const result = await readPdf(new Uint8Array(readFileSync(path)));
    expect(result.pageCount).toBe(7);
    expect(result.classification).toBe("text");
  }, 60_000);
});

describe("pool limits", () => {
  it("never grows beyond its configured size", async () => {
    const docs = Array.from({ length: 6 }, () => makeScannedPdf(1));
    await Promise.all(docs.map((b) => readPdf(b, { maxVisionPages: 1 })));
    const stats = pdfPoolStats();
    expect(stats.size).toBeLessThanOrEqual(stats.maxSize);
  }, 180_000);

  it("refuses politely instead of queueing without limit", async () => {
    const stats = pdfPoolStats();
    const flood = Array.from(
      { length: stats.maxQueue + stats.maxSize + 5 },
      () =>
        processPdfInWorker({ bytes: makeScannedPdf(1), maxVisionPages: 1 }).catch(
          (err) => err,
        ),
    );
    const settled = await Promise.all(flood);
    const refused = settled.filter((r) => r instanceof PdfBusyError);
    expect(refused.length).toBeGreaterThan(0);
    // The refusal tells the student what to do, without internal detail.
    expect((refused[0] as Error).message).toMatch(/try again/i);
  }, 180_000);
});
