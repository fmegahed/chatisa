import { describe, expect, it } from "vitest";
import "fake-indexeddb/auto";
import {
  ATTACH_TEXT_MAX,
  MAX_ATTACHMENT_BYTES,
  attachmentBlockText,
  attachmentPart,
  classifyFile,
  datasetAnnouncement,
  estimatePdfPages,
  rejectionReason,
  truncateAttachmentText,
} from "@/lib/files/attachments";
import {
  docxXmlToText,
  pptxSlideXmlToText,
  sortSlidePaths,
} from "@/lib/files/office-text";
import { keepOriginal, targetDimensions } from "@/lib/files/image";
import {
  deleteFilesForChat,
  fileRef,
  getFile,
  idFromRef,
  isFileRef,
  putFile,
} from "@/lib/ask/file-store";

describe("attachment classification", () => {
  it("routes each supported type", () => {
    expect(classifyFile("plot.PNG").kind).toBe("image");
    expect(classifyFile("chapter.pdf").kind).toBe("pdf");
    expect(classifyFile("sales.csv")).toMatchObject({ kind: "dataset", format: "csv" });
    expect(classifyFile("sales.tsv").format).toBe("csv");
    expect(classifyFile("book.xlsx").format).toBe("xlsx");
    expect(classifyFile("notes.docx").office).toBe("docx");
    expect(classifyFile("deck.pptx").office).toBe("pptx");
    expect(classifyFile("readme.md").kind).toBe("text");
    expect(classifyFile("data.json").kind).toBe("text");
    expect(classifyFile("song.mp3").kind).toBe("unsupported");
  });

  it("accepts code and notebook files despite the empty MIME type Windows reports", () => {
    // Chrome on Windows sends "" for these, so the text/* fallback never fires.
    expect(classifyFile("clean.py", "").kind).toBe("text");
    expect(classifyFile("model.R", "").kind).toBe("text");
    expect(classifyFile("report.Rmd", "").kind).toBe("text");
    expect(classifyFile("slides.qmd", "").kind).toBe("text");
    expect(classifyFile("index.html", "").kind).toBe("text");
    expect(classifyFile("page.htm", "").kind).toBe("text");
    expect(classifyFile("analysis.ipynb", "").kind).toBe("notebook");
    expect(rejectionReason("analysis.ipynb", 1234, "")).toBeNull();
    expect(rejectionReason("model.R", 1234, "")).toBeNull();
  });

  it("extension wins over the browser MIME type, MIME fills the gap", () => {
    // Browsers report Office files with odd MIME types; the extension decides.
    expect(classifyFile("deck.pptx", "application/zip").office).toBe("pptx");
    // Extensionless camera upload: the MIME type is all there is.
    expect(classifyFile("IMG0001", "image/jpeg").kind).toBe("image");
  });

  it("rejects oversized, empty, and unsupported files with readable reasons", () => {
    expect(rejectionReason("song.mp3", 10)).toMatch(/isn't a supported file type/);
    expect(rejectionReason("big.pdf", MAX_ATTACHMENT_BYTES + 1)).toMatch(/25 MB/);
    expect(rejectionReason("empty.csv", 0)).toMatch(/empty/);
    expect(rejectionReason("ok.pdf", 1234)).toBeNull();
  });
});

describe("attachment text handling", () => {
  it("passes short text through and truncates long text with a note", () => {
    expect(truncateAttachmentText("hello")).toEqual({
      text: "hello",
      truncated: false,
    });
    const long = "x".repeat(ATTACH_TEXT_MAX + 100);
    const cut = truncateAttachmentText(long);
    expect(cut.truncated).toBe(true);
    expect(cut.text).toContain("[file truncated at");
    expect(cut.text.length).toBeLessThan(long.length);
  });

  it("renders dataset and file blocks the model reads, re-capped server-side", () => {
    const part = attachmentPart({
      kind: "text",
      name: "notes.txt",
      detail: "text file",
      text: "y".repeat(ATTACH_TEXT_MAX * 2), // a lying client
    });
    const block = attachmentBlockText(part.data);
    expect(block).toContain("[Attached file: notes.txt]");
    expect(block.length).toBeLessThan(ATTACH_TEXT_MAX + 200);
  });

  it("labels notebook attachments so the model knows what it is reading", () => {
    const part = attachmentPart({
      kind: "notebook",
      name: "analysis.ipynb",
      detail: "3 cells, python",
      text: "```python\ndf.head()\n```",
    });
    expect(attachmentBlockText(part.data)).toContain(
      "[Attached notebook: analysis.ipynb]",
    );
  });

  it("announces a loaded dataset with variable, shape, and columns", () => {
    const text = datasetAnnouncement({
      fileName: "2024 sales.csv",
      varName: "x2024_sales",
      columns: ["region", "month", "revenue"],
      rowCount: 120,
    });
    expect(text).toContain("`x2024_sales`");
    expect(text).toContain("120 rows");
    expect(text).toContain("region, month, revenue");
    expect(text).toContain("re-attach");
  });
});

describe("PDF page estimate", () => {
  const bytes = (s: string) => new TextEncoder().encode(s);

  it("counts /Type /Page objects, not the /Pages tree", () => {
    const pdf = "%PDF-1.4\n1 0 obj<</Type /Pages /Kids[]>>\n2 0 obj<</Type /Page >>\n3 0 obj<</Type/Page >>";
    expect(estimatePdfPages(bytes(pdf))).toBe(2);
  });

  it("returns null when markers are hidden (compressed streams)", () => {
    expect(estimatePdfPages(bytes("%PDF-1.7 stream...gibberish"))).toBeNull();
  });
});

describe("office XML extraction", () => {
  it("extracts Word paragraphs with runs, tabs, breaks, and entities", () => {
    const xml =
      "<w:document><w:p><w:r><w:t>Hello</w:t></w:r><w:tab/><w:r><w:t xml:space=\"preserve\">R &amp; Python</w:t></w:r></w:p>" +
      "<w:p></w:p>" +
      "<w:p><w:r><w:t>Second</w:t></w:r></w:p></w:document>";
    expect(docxXmlToText(xml)).toBe("Hello\tR & Python\nSecond");
  });

  it("extracts slide text by paragraph", () => {
    const xml =
      "<p:sld><a:p><a:r><a:t>Title here</a:t></a:r></a:p><a:p><a:r><a:t>Point one</a:t></a:r><a:r><a:t> continued</a:t></a:r></a:p></p:sld>";
    expect(pptxSlideXmlToText(xml)).toBe("Title here\nPoint one continued");
  });

  it("orders slides numerically and ignores non-slide entries", () => {
    expect(
      sortSlidePaths([
        "ppt/slides/slide10.xml",
        "ppt/slides/slide2.xml",
        "ppt/slides/_rels/slide2.xml.rels",
        "ppt/slides/slide1.xml",
      ]),
    ).toEqual([
      "ppt/slides/slide1.xml",
      "ppt/slides/slide2.xml",
      "ppt/slides/slide10.xml",
    ]);
  });
});

describe("image sizing decision", () => {
  it("keeps small in-range images and scales large ones proportionally", () => {
    expect(targetDimensions(800, 600)).toEqual({ width: 800, height: 600, scaled: false });
    const scaled = targetDimensions(3200, 1600);
    expect(scaled).toEqual({ width: 1600, height: 800, scaled: true });
    expect(keepOriginal(100_000, 800, 600)).toBe(true);
    expect(keepOriginal(5_000_000, 800, 600)).toBe(false);
    expect(keepOriginal(100_000, 4000, 100)).toBe(false);
  });
});

describe("file store (IndexedDB)", () => {
  it("round-trips a payload and resolves references", async () => {
    const stored = {
      id: "f1",
      chatId: "c1",
      name: "chapter.pdf",
      mediaType: "application/pdf",
      dataUrl: "data:application/pdf;base64,AAAA",
    };
    expect(await putFile(stored)).toBe(true);
    expect(await getFile("f1")).toEqual(stored);
    const ref = fileRef("f1");
    expect(isFileRef(ref)).toBe(true);
    expect(isFileRef("data:image/png;base64,x")).toBe(false);
    expect(idFromRef(ref)).toBe("f1");
  });

  it("deletes a chat's payloads and leaves other chats alone", async () => {
    await putFile({ id: "a", chatId: "chat-A", name: "a.pdf", mediaType: "application/pdf", dataUrl: "data:x" });
    await putFile({ id: "b", chatId: "chat-A", name: "b.png", mediaType: "image/png", dataUrl: "data:y" });
    await putFile({ id: "c", chatId: "chat-B", name: "c.png", mediaType: "image/png", dataUrl: "data:z" });
    await deleteFilesForChat("chat-A");
    expect(await getFile("a")).toBeNull();
    expect(await getFile("b")).toBeNull();
    expect((await getFile("c"))?.chatId).toBe("chat-B");
  });
});
