import { afterEach, describe, expect, it } from "vitest";
import {
  anthropicTemplateMessage,
  templateFileId,
  wantsGeneratedFile,
} from "@/lib/ask/hosted";
import {
  ANTHROPIC_FILE_ID,
  OPENAI_CONTAINER_ID,
  OPENAI_FILE_ID,
  mediaTypeForName,
  mockZipBytes,
  safeDownloadName,
} from "@/lib/ask/hosted-files";

afterEach(() => {
  delete process.env.CHATISA_MOCK_LLM;
});

describe("generated-file intent gate", () => {
  it("matches deck and document requests", () => {
    expect(wantsGeneratedFile("Please make me a PowerPoint about widgets")).toBe(true);
    expect(wantsGeneratedFile("build a slide deck for my pitch")).toBe(true);
    expect(wantsGeneratedFile("export this as an excel workbook")).toBe(true);
    expect(wantsGeneratedFile("write me a word document summary")).toBe(true);
  });
  it("stays quiet for ordinary questions", () => {
    expect(wantsGeneratedFile("what is a p-value?")).toBe(false);
    expect(wantsGeneratedFile("plot revenue by month")).toBe(false);
    // "presentation" as a topic word is accepted noise; but plain analysis is not.
    expect(wantsGeneratedFile("summarize my attached pdf")).toBe(false);
  });
});

describe("anthropic template injection message", () => {
  it("carries the file reference with the containerUpload option", () => {
    const msg = anthropicTemplateMessage("file_tpl123");
    expect(msg.role).toBe("user");
    const parts = msg.content as Array<Record<string, unknown>>;
    const filePart = parts.find((p) => p.type === "file")!;
    expect(filePart.data).toEqual({
      type: "reference",
      reference: { anthropic: "file_tpl123" },
    });
    expect(filePart.providerOptions).toEqual({
      anthropic: { containerUpload: true },
    });
    const textPart = parts.find((p) => p.type === "text")!;
    expect(String(textPart.text)).toContain("miami_template_by_fadel_megahed.pptx");
  });

  it("mock mode resolves a template id without touching a provider", async () => {
    process.env.CHATISA_MOCK_LLM = "1";
    expect(await templateFileId("anthropic")).toBe("mock-template-anthropic");
    expect(await templateFileId("openai")).toBe("mock-template-openai");
  });
});

describe("hosted file retrieval guards", () => {
  it("validates provider id formats strictly", () => {
    expect(ANTHROPIC_FILE_ID.test("file_011CNha8iCJcU1wXNR6q4V8w")).toBe(true);
    expect(ANTHROPIC_FILE_ID.test("file_mockdeck1")).toBe(true);
    expect(ANTHROPIC_FILE_ID.test("../etc/passwd")).toBe(false);
    expect(ANTHROPIC_FILE_ID.test("file_ab/c")).toBe(false);
    expect(OPENAI_CONTAINER_ID.test("cntr_abc123DEF")).toBe(true);
    expect(OPENAI_CONTAINER_ID.test("file_abc123")).toBe(false);
    expect(OPENAI_FILE_ID.test("cfile_xyz789")).toBe(true);
    expect(OPENAI_FILE_ID.test("cfile_!")).toBe(false);
  });

  it("sanitizes download filenames and maps media types", () => {
    expect(safeDownloadName("/mnt/data/miami deck.pptx", "f")).toBe(
      "miami deck.pptx",
    );
    expect(safeDownloadName('a"b\u0000c.pptx', "f")).toBe("abc.pptx");
    expect(safeDownloadName("", "fallback")).toBe("fallback");
    expect(mediaTypeForName("deck.pptx")).toContain("presentationml");
    expect(mediaTypeForName("report.docx")).toContain("wordprocessingml");
    expect(mediaTypeForName("weird.bin")).toBe("application/octet-stream");
  });

  it("emits a structurally valid empty zip for mock downloads", () => {
    const bytes = mockZipBytes();
    // End-of-central-directory signature: PK\x05\x06
    expect([...bytes.slice(0, 4)]).toEqual([0x50, 0x4b, 0x05, 0x06]);
    expect(bytes.length).toBe(22);
  });
});
