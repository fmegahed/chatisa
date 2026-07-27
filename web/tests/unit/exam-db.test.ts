import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

/** Each run gets its own database file, so tests never touch dev data. */
const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-exam-db-"));
process.env.CHATISA_DATA_DIR = dataDir;

const {
  closeDb,
  createExamDocument,
  deleteExamDocument,
  getDocumentPages,
  getOwnedExamDocument,
  listExamDocuments,
  purgeDocumentText,
  purgeExpiredDocumentText,
  upsertUser,
} = await import("@/lib/db");

const OWNER = "owner@miamioh.edu";
const OTHER = "someone.else@miamioh.edu";

function makeDoc(userEmail: string, filename = "chapter-3.pdf") {
  return createExamDocument({
    userEmail,
    filename,
    sizeBytes: 2048,
    pageCount: 3,
    textPageCount: 2,
    visionPageCount: 1,
    charCount: 300,
    classification: "mixed",
    warnings: [],
    pages: [
      { pageNumber: 1, text: "Page one text", charCount: 13, source: "text" },
      { pageNumber: 2, text: "Page two text", charCount: 13, source: "text" },
      {
        pageNumber: 3,
        text: "Transcribed page three",
        charCount: 22,
        source: "vision",
      },
    ],
  });
}

beforeAll(() => {
  upsertUser(OWNER, "Owner");
  upsertUser(OTHER, "Other");
});

afterAll(() => {
  // Release the SQLite handle before removing the directory.
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
});

describe("exam document ownership", () => {
  it("returns the document to its owner", () => {
    const id = makeDoc(OWNER);
    const doc = getOwnedExamDocument(id, OWNER);
    expect(doc?.filename).toBe("chapter-3.pdf");
    expect(doc?.classification).toBe("mixed");
    expect(doc?.visionPageCount).toBe(1);
  });

  it("hides another user's document exactly as if it did not exist", () => {
    const id = makeDoc(OWNER);
    expect(getOwnedExamDocument(id, OTHER)).toBeUndefined();
    expect(getOwnedExamDocument("11111111-2222-4333-8444-555555555555", OTHER))
      .toBeUndefined();
  });

  it("is case-insensitive about the owner's address", () => {
    const id = makeDoc(OWNER);
    expect(getOwnedExamDocument(id, "Owner@MiamiOH.edu")).toBeDefined();
  });

  it("lists only the requesting user's documents", () => {
    makeDoc(OWNER, "mine.pdf");
    makeDoc(OTHER, "theirs.pdf");
    const names = listExamDocuments(OTHER).map((d) => d.filename);
    expect(names).toContain("theirs.pdf");
    expect(names).not.toContain("mine.pdf");
  });

  it("refuses to delete a document belonging to someone else", () => {
    const id = makeDoc(OWNER);
    expect(deleteExamDocument(id, OTHER)).toBe(false);
    expect(getOwnedExamDocument(id, OWNER)).toBeDefined();
    expect(deleteExamDocument(id, OWNER)).toBe(true);
    expect(getOwnedExamDocument(id, OWNER)).toBeUndefined();
  });
});

describe("page storage and scoping", () => {
  it("stores page text with its provenance", () => {
    const id = makeDoc(OWNER);
    const pages = getDocumentPages(id);
    expect(pages).toHaveLength(3);
    expect(pages[0].pageNumber).toBe(1);
    expect(pages[2].source).toBe("vision");
  });

  it("returns only the requested page range", () => {
    const id = makeDoc(OWNER);
    const pages = getDocumentPages(id, 2, 3);
    expect(pages.map((p) => p.pageNumber)).toEqual([2, 3]);
  });
});

describe("retention (ADR-015: no standing copy of a textbook)", () => {
  it("purges page text while keeping the document's metadata", () => {
    const id = makeDoc(OWNER);
    expect(getDocumentPages(id)).toHaveLength(3);

    purgeDocumentText(id);

    expect(getDocumentPages(id)).toHaveLength(0);
    const doc = getOwnedExamDocument(id, OWNER);
    expect(doc).toBeDefined();
    expect(doc?.pageCount).toBe(3);
    expect(doc?.textPurgedAt).not.toBeNull();
  });

  it("sweeps text that was never purged, and leaves fresh uploads alone", () => {
    const fresh = makeDoc(OWNER);
    // Nothing is old enough yet.
    expect(purgeExpiredDocumentText(24)).toBe(0);
    expect(getDocumentPages(fresh).length).toBeGreaterThan(0);

    // A negative age puts the cutoff in the future, so every existing document
    // counts as stale. A zero cutoff would be racy: a document created in the
    // same millisecond is not strictly older than "now".
    const swept = purgeExpiredDocumentText(-1);
    expect(swept).toBeGreaterThan(0);
    expect(getDocumentPages(fresh)).toHaveLength(0);
  });

  it("does not count already-purged documents twice", () => {
    makeDoc(OWNER);
    expect(purgeExpiredDocumentText(-1)).toBeGreaterThan(0);
    expect(purgeExpiredDocumentText(-1)).toBe(0);
  });
});
