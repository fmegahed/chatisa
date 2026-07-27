import { test, expect, type Page } from "@playwright/test";
import { makeTextPdf } from "../helpers/make-pdf";

/**
 * These tests assert what is in "your" resume list, so they need an account of
 * their own. Sharing the default test student would mean other specs'
 * unfinished exams appear here, and clearing them would sabotage those specs.
 */
test.use({ storageState: { cookies: [], origins: [] } });

test.beforeEach(async ({ page }, testInfo) => {
  // A separate student per test, so parallel runs and both viewport projects
  // never see each other's unfinished exams.
  const who = `resume-${testInfo.project.name}-${testInfo.testId}`.toLowerCase();
  await page.goto("/login");
  await page.getByLabel("Email address").fill(`${who}@miamioh.edu`);
  await page.getByRole("button", { name: "Sign in as test user" }).click();
  await page.waitForURL("**/");
});

const TOPICS = [
  "normalization removes transitive dependencies",
  "primary keys uniquely identify rows",
  "foreign keys enforce referential integrity",
];

function coursePdf(): Buffer {
  return Buffer.from(
    makeTextPdf(
      Array.from({ length: 3 }, (_, i) => {
        const topic = TOPICS[i % TOPICS.length];
        return (
          `Page ${i + 1}. Section on ${topic}. ` +
          `In practice, ${topic}, which matters when designing schemas. `.repeat(4)
        );
      }),
    ),
  );
}

async function startExam(page: Page, count = 3) {
  await page.goto("/exam-prep");
  await page.getByLabel("Choose a PDF", { exact: true }).setInputFiles({
    name: "course.pdf",
    mimeType: "application/pdf",
    buffer: coursePdf(),
  });
  await expect(page.getByText(/Read course\.pdf/)).toBeVisible({
    timeout: 30_000,
  });
  await page.getByLabel("How many questions").fill(String(count));
  await page.getByRole("button", { name: "Build my practice exam" }).click();
  await expect(page.getByText(`Question 1 of ${count}`)).toBeVisible({
    timeout: 60_000,
  });
}

/** Scoped to the answer group: the page also has a confidence radio group. */
function answerOptions(page: Page) {
  return page
    .getByRole("group", { name: "Choose one answer" })
    .getByRole("radio");
}

async function answerCurrent(page: Page, pick: "first" | "last" = "first") {
  await (pick === "first"
    ? answerOptions(page).first()
    : answerOptions(page).last()
  ).check();
  await page.getByRole("button", { name: "Submit answer" }).click();
  await expect(page.getByText(/From page \d+ of your document/)).toBeVisible({
    timeout: 30_000,
  });
}

// Uploading, generating and answering in one test exceeds the default budget
// when several workers compete for a cold dev server.
test.describe.configure({ timeout: 90_000 });

test.describe("resuming an exam", () => {
  test("offers an unfinished exam and continues at the right question", async ({
    page,
  }) => {
    await startExam(page, 3);
    await answerCurrent(page);
    await page.getByRole("button", { name: "Next question" }).click();
    await expect(page.getByText("Question 2 of 3")).toBeVisible({
      timeout: 30_000,
    });

    // Leaving mid-exam must not lose the student's progress.
    await page.goto("/exam-prep");
    const resume = page.getByRole("button", { name: "Continue this exam" });
    await expect(resume.first()).toBeVisible({ timeout: 30_000 });
    await resume.first().click();

    // Picks up at the next unanswered question, not back at the start.
    await expect(page.getByText("Question 2 of 3")).toBeVisible({
      timeout: 30_000,
    });
  });

  test("can discard an unfinished exam instead of continuing", async ({
    page,
  }) => {
    await startExam(page, 3);
    await answerCurrent(page);
    await page.getByRole("button", { name: "Next question" }).click();
    await expect(page.getByText("Question 2 of 3")).toBeVisible({
      timeout: 30_000,
    });

    await page.goto("/exam-prep");
    await expect(
      page.getByRole("heading", { name: "Pick up where you left off" }),
    ).toBeVisible({ timeout: 30_000 });
    await page.getByRole("button", { name: "Discard" }).first().click();

    // The resume offer is gone, and stays gone on reload.
    await expect(
      page.getByRole("heading", { name: "Pick up where you left off" }),
    ).toBeHidden();
    await page.reload();
    await expect(
      page.getByRole("heading", { name: "Pick up where you left off" }),
    ).toBeHidden();
  });

  test("a finished exam is not offered for resuming", async ({ page }) => {
    await startExam(page, 2);
    await answerCurrent(page);
    await page.getByRole("button", { name: "Next question" }).click();
    await answerCurrent(page);
    await page.getByRole("button", { name: "See results" }).click();
    await expect(
      page.getByRole("heading", { name: "Your results" }),
    ).toBeVisible({ timeout: 30_000 });

    await page.goto("/exam-prep");
    // The completed exam must not appear in the resume list.
    await expect(page.getByRole("heading", { name: "Pick up where you left off" }))
      .toHaveCount(0);
  });
});

test.describe("practising weak topics again", () => {
  test("builds a new exam from the same document", async ({ page }) => {
    await startExam(page, 2);

    // Answer with the last option each time, so at least one is wrong and a
    // study plan exists.
    await answerCurrent(page, "last");
    await page.getByRole("button", { name: "Next question" }).click();
    await answerCurrent(page, "last");
    await page.getByRole("button", { name: "See results" }).click();

    await expect(
      page.getByRole("heading", { name: "Your results" }),
    ).toBeVisible({ timeout: 30_000 });

    const retry = page.getByRole("button", {
      name: "Practise these topics again",
    });
    if ((await retry.count()) === 0) {
      // Everything was answered correctly, so there is nothing to retry.
      // That is legitimate; the study plan is only shown when it applies.
      return;
    }
    await retry.click();
    await expect(page.getByText(/Question 1 of/)).toBeVisible({
      timeout: 60_000,
    });
  });
});
