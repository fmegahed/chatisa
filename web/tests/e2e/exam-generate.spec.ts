import { test, expect } from "@playwright/test";
import { makeTextPdf } from "../helpers/make-pdf";

const TOPICS = [
  "normalization removes transitive dependencies",
  "primary keys uniquely identify rows",
  "foreign keys enforce referential integrity",
  "indexes trade write cost for read speed",
  "joins combine rows across related tables",
  "aggregation summarises groups of rows",
];

function coursePages(count: number): Uint8Array {
  return makeTextPdf(
    Array.from({ length: count }, (_, i) => {
      const topic = TOPICS[i % TOPICS.length];
      return (
        `Page ${i + 1}. Section on ${topic}. ` +
        `In practice, ${topic}, which matters when designing schemas. `.repeat(4)
      );
    }),
  );
}

async function uploadCourseDoc(
  request: import("@playwright/test").APIRequestContext,
  pages = 6,
) {
  const res = await request.post("/api/exam-prep/documents", {
    multipart: {
      file: {
        name: "course.pdf",
        mimeType: "application/pdf",
        buffer: Buffer.from(coursePages(pages)),
      },
    },
  });
  expect(res.status()).toBe(201);
  return (await res.json()).documentId as string;
}

const MODEL = "gpt-5.6-terra";

test.describe("exam generation", () => {
  test("builds a grounded exam and reports coverage", async ({ request }) => {
    const documentId = await uploadCourseDoc(request);

    const res = await request.post("/api/exam-prep/exams", {
      data: {
        documentId,
        modelId: MODEL,
        questionType: "short_answer",
        count: 4,
        examMode: "practice",
      },
    });

    expect(res.status()).toBe(201);
    const body = await res.json();
    expect(body.examId).toBeTruthy();
    expect(body.deliveredCount).toBeGreaterThan(0);
    expect(body.coverage).toMatch(/pages/i);
  });

  test("keeps the answer key out of the browser until the exam is finished", async ({
    request,
  }) => {
    const documentId = await uploadCourseDoc(request);
    const created = await request.post("/api/exam-prep/exams", {
      data: {
        documentId,
        modelId: MODEL,
        questionType: "multiple_choice",
        count: 3,
        examMode: "practice",
      },
    });
    expect(created.status()).toBe(201);
    const { examId } = await created.json();

    const res = await request.get(`/api/exam-prep/exams/${examId}`);
    expect(res.status()).toBe(200);
    const text = await res.text();
    const body = JSON.parse(text);

    // A question is present to answer...
    expect(body.questions.length).toBeGreaterThan(0);
    // ...but nothing that gives the answer away.
    for (const q of body.questions) {
      expect(q).not.toHaveProperty("correctIndex");
      expect(q).not.toHaveProperty("modelAnswer");
      expect(q).not.toHaveProperty("explanation");
      expect(q).not.toHaveProperty("sourceQuote");
      expect(q.stem).toBeTruthy();
    }
    expect(text).not.toContain("correctIndex");
    expect(text).not.toContain("modelAnswer");
    expect(text).not.toContain("rubric");
  });

  test("refuses a model that is not allowed for this module", async ({
    request,
  }) => {
    const documentId = await uploadCourseDoc(request);
    const res = await request.post("/api/exam-prep/exams", {
      data: {
        documentId,
        modelId: "gpt-4o-realtime-preview-2025-06-03",
        questionType: "short_answer",
        count: 3,
        examMode: "practice",
      },
    });
    expect(res.status()).toBe(400);
    expect((await res.json()).error).toContain("isn't available");
  });

  test("rejects a malformed request without leaking internals", async ({
    request,
  }) => {
    const res = await request.post("/api/exam-prep/exams", {
      data: { documentId: "not-a-uuid", count: 999 },
    });
    expect(res.status()).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("That request wasn't valid.");
    expect(JSON.stringify(body)).not.toContain("node_modules");
  });

  test("treats another student's document id as not found", async ({
    request,
  }) => {
    const res = await request.post("/api/exam-prep/exams", {
      data: {
        documentId: "11111111-2222-4333-8444-555555555555",
        modelId: MODEL,
        questionType: "short_answer",
        count: 3,
        examMode: "practice",
      },
    });
    expect(res.status()).toBe(404);
  });

  test("keeps the document usable until the exam is finished, then releases it", async ({
    request,
  }) => {
    const documentId = await uploadCourseDoc(request);
    const first = await request.post("/api/exam-prep/exams", {
      data: {
        documentId,
        modelId: MODEL,
        questionType: "short_answer",
        count: 3,
        examMode: "practice",
      },
    });
    expect(first.status()).toBe(201);
    const { examId } = await first.json();

    // Still usable while the student is working: this is what makes
    // "practise these topics again" possible.
    const second = await request.post("/api/exam-prep/exams", {
      data: {
        documentId,
        modelId: MODEL,
        questionType: "short_answer",
        count: 3,
        examMode: "practice",
      },
    });
    expect(second.status()).toBe(201);

    // Finishing releases the material (ADR-015): no standing copy is kept.
    expect(
      (await request.post(`/api/exam-prep/exams/${examId}/results`)).status(),
    ).toBe(200);

    const third = await request.post("/api/exam-prep/exams", {
      data: {
        documentId,
        modelId: MODEL,
        questionType: "short_answer",
        count: 3,
        examMode: "practice",
      },
    });
    expect(third.status()).toBe(410);
    const body = await third.json();
    expect(body.code).toBe("TEXT_PURGED");
    expect(body.error).toMatch(/upload the file again/i);
  });

  test("lists and deletes the student's own exams", async ({ request }) => {
    const documentId = await uploadCourseDoc(request);
    const created = await request.post("/api/exam-prep/exams", {
      data: {
        documentId,
        modelId: MODEL,
        questionType: "short_answer",
        count: 3,
        examMode: "practice",
      },
    });
    const { examId } = await created.json();

    const list = await request.get("/api/exam-prep/exams");
    expect(list.status()).toBe(200);
    const ids = (await list.json()).exams.map((e: { id: string }) => e.id);
    expect(ids).toContain(examId);

    expect((await request.delete(`/api/exam-prep/exams/${examId}`)).status()).toBe(
      204,
    );
    expect((await request.get(`/api/exam-prep/exams/${examId}`)).status()).toBe(
      404,
    );
  });
});

test.describe("exam generation access control", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("requires sign in", async ({ request }) => {
    expect((await request.get("/api/exam-prep/exams")).status()).toBe(401);
    expect(
      (
        await request.post("/api/exam-prep/exams", {
          data: { documentId: "11111111-2222-4333-8444-555555555555" },
        })
      ).status(),
    ).toBe(401);
  });
});
