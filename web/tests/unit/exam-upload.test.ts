import { describe, expect, it } from "vitest";
import {
  MAX_UPLOAD_BYTES,
  formatBytes,
  safeFilename,
} from "@/lib/exam/upload";

describe("safeFilename", () => {
  it("keeps an ordinary name intact, spaces and all", () => {
    expect(safeFilename("ISA 401 chapter 3.pdf")).toBe(
      "ISA 401 chapter 3.pdf",
    );
  });

  it("keeps only the final segment of a path", () => {
    expect(safeFilename("C:\\Users\\me\\notes.pdf")).toBe("notes.pdf");
    expect(safeFilename("/var/tmp/notes.pdf")).toBe("notes.pdf");
  });

  it("defuses a traversal attempt", () => {
    expect(safeFilename("../../etc/passwd")).toBe("passwd");
    expect(safeFilename("..\\..\\windows\\system32")).toBe("system32");
  });

  it("strips control characters", () => {
    expect(safeFilename("no\u0000tes\u001f.pdf")).toBe("notes.pdf");
  });

  it("falls back when nothing usable remains", () => {
    expect(safeFilename("")).toBe("document.pdf");
    expect(safeFilename("   ")).toBe("document.pdf");
    expect(safeFilename("/")).toBe("document.pdf");
  });

  it("limits absurdly long names", () => {
    expect(safeFilename("a".repeat(500)).length).toBeLessThanOrEqual(120);
  });
});

describe("upload limits", () => {
  it("caps uploads at a size a student can understand", () => {
    expect(MAX_UPLOAD_BYTES).toBe(25 * 1024 * 1024);
    expect(formatBytes(MAX_UPLOAD_BYTES)).toBe("25 MB");
  });
});
