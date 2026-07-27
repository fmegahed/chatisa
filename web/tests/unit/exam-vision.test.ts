import { afterAll, afterEach, beforeEach, describe, expect, it } from "vitest";
import { readPdf } from "@/lib/exam/pdf";
import { shutdownPdfPool } from "@/lib/exam/pdf-pool";
import {
  MAX_VISION_PAGES,
  mergeTranscriptions,
  pickVisionModel,
  transcribeScannedPages,
} from "@/lib/exam/vision";
import { MODELS } from "@/lib/config/models";
import { makePdf, makeScannedPdf } from "../helpers/make-pdf";

const para = (label: string) =>
  `${label}. ` + "Normalization removes transitive dependencies. ".repeat(3);

beforeEach(() => {
  process.env.CHATISA_MOCK_LLM = "1";
});

afterEach(() => {
  delete process.env.CHATISA_MOCK_LLM;
  delete process.env.CHATISA_MOCK_LLM_MODE;
});

afterAll(async () => {
  await shutdownPdfPool();
});

describe("pickVisionModel", () => {
  it("chooses a model that actually supports vision", () => {
    const chosen = pickVisionModel();
    expect(chosen).not.toBeNull();
    expect(MODELS[chosen as string].supportsVision).toBe(true);
  });

  it("never chooses the realtime speech model", () => {
    expect(MODELS[pickVisionModel() as string].realtimeOnly ?? false).toBe(false);
  });
});

describe("transcribeScannedPages", () => {
  it("transcribes the pages that carry no selectable text", async () => {
    const bytes = makeScannedPdf(2);
    const extracted = await readPdf(bytes);

    const result = await transcribeScannedPages({
      extracted,
      pageNumbers: [1, 2],
    });

    expect(result.pages.map((p) => p.pageNumber)).toEqual([1, 2]);
    expect(result.pages[0].text).toContain("Normalization");
    expect(result.pages[0].legible).toBe(true);
    expect(result.skippedPages).toEqual([]);
    expect(result.illegiblePages).toEqual([]);
  }, 60_000);

  it("caps how many pages one document may transcribe", async () => {
    const bytes = makeScannedPdf(3);
    const extracted = await readPdf(bytes);

    const result = await transcribeScannedPages({
      extracted,
      pageNumbers: [1, 2, 3],
      maxPages: 2,
    });

    expect(result.pages).toHaveLength(2);
    expect(result.skippedPages).toEqual([3]);
  }, 60_000);

  it("records an unreadable page rather than inventing text for it", async () => {
    process.env.CHATISA_MOCK_LLM_MODE = "illegible";
    const bytes = makeScannedPdf(1);
    const extracted = await readPdf(bytes);

    const result = await transcribeScannedPages({
      extracted,
      pageNumbers: [1],
    });

    expect(result.pages[0].legible).toBe(false);
    expect(result.pages[0].text).toBe("");
    expect(result.illegiblePages).toEqual([1]);
  }, 60_000);

  it("does nothing when no page needs transcription", async () => {
    const bytes = makeScannedPdf(1);
    const extracted = await readPdf(bytes);
    const result = await transcribeScannedPages({
      extracted,
      pageNumbers: [],
    });
    expect(result.pages).toEqual([]);
    expect(result.modelId).toBeNull();
  });

  it("has a sane default cap", () => {
    expect(MAX_VISION_PAGES).toBeGreaterThan(0);
    expect(MAX_VISION_PAGES).toBeLessThanOrEqual(100);
  });
});

describe("mergeTranscriptions", () => {
  it("fills in transcribed pages and marks their provenance", async () => {
    // Page 2 is drawing-only, so it needs transcription; 1 and 3 do not.
    const bytes = makePdf([{ text: para("One") }, {}, { text: para("Three") }]);
    const extracted = await readPdf(bytes);
    expect(extracted.pages[1].source).toBe("needs_vision");

    const transcription = await transcribeScannedPages({
      extracted,
      pageNumbers: [2],
    });
    const merged = mergeTranscriptions(extracted, transcription);

    expect(merged.pages.map((p) => p.source)).toEqual([
      "text",
      "vision",
      "text",
    ]);
    expect(merged.pages[1].text).toContain("Normalization");
    expect(merged.pages[1].charCount).toBeGreaterThan(0);
    // Directly extracted pages are untouched.
    expect(merged.pages[0].text).toContain("One");
  }, 60_000);

  it("leaves a page needing vision alone when transcription produced nothing", async () => {
    process.env.CHATISA_MOCK_LLM_MODE = "illegible";
    const bytes = makeScannedPdf(1);
    const extracted = await readPdf(bytes);
    const transcription = await transcribeScannedPages({
      extracted,
      pageNumbers: [1],
    });
    const merged = mergeTranscriptions(extracted, transcription);

    expect(merged.pages[0].source).toBe("needs_vision");
    expect(merged.pages[0].text).toBe("");
  }, 60_000);
});
