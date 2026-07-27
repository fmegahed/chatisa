import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { makeTextPdf } from "../helpers/make-pdf";

const TOPICS = [
  "normalization removes transitive dependencies",
  "primary keys uniquely identify rows",
  "foreign keys enforce referential integrity",
  "indexes trade write cost for read speed",
];

function coursePdf(pages = 4): Buffer {
  return Buffer.from(
    makeTextPdf(
      Array.from({ length: pages }, (_, i) => {
        const topic = TOPICS[i % TOPICS.length];
        return (
          `Page ${i + 1}. Section on ${topic}. ` +
          `In practice, ${topic}, which matters when designing schemas. `.repeat(4)
        );
      }),
    ),
  );
}

async function setUpExam(
  page: Page,
  opts: { type?: string; count?: number; mode?: "practice" | "exam" } = {},
) {
  await page.goto("/exam-prep");
  await page
    .getByLabel("Choose a PDF", { exact: true })
    .setInputFiles({
      name: "course.pdf",
      mimeType: "application/pdf",
      buffer: coursePdf(),
    });
  await expect(page.getByText(/Read course\.pdf/)).toBeVisible({
    timeout: 30_000,
  });

  if (opts.type) {
    await page.getByLabel("Question type").selectOption(opts.type);
  }
  await page.getByLabel("How many questions").fill(String(opts.count ?? 3));
  if (opts.mode === "exam") {
    await page.getByLabel("At the end, like a real exam").check();
  }
  await page.getByRole("button", { name: "Build my practice exam" }).click();
  await expect(page.getByText(/Question 1 of/)).toBeVisible({
    timeout: 60_000,
  });
}

test.describe.configure({ timeout: 90_000 });

test.describe("Exam Ally quiz", () => {
  test("runs a multiple choice exam through to results", async ({ page }) => {
    await setUpExam(page, { type: "multiple_choice", count: 3 });

    // Progress is real, not decorative.
    await expect(page.getByText("Question 1 of 3")).toBeVisible();
    await expect(page.getByText(/Questions were drawn from/)).toBeVisible();

    for (let i = 0; i < 3; i += 1) {
      await page.getByRole("radio").first().check();
      await page.getByRole("button", { name: "Submit answer" }).click();

      // Feedback cites the page it came from.
      await expect(page.getByText(/From page \d+ of your document/)).toBeVisible({
        timeout: 30_000,
      });

      const next = page.getByRole("button", {
        name: i === 2 ? "See results" : "Next question",
      });
      await next.click();
    }

    await expect(
      page.getByRole("heading", { name: "Your results" }),
    ).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/questions correctly/)).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "How you did by topic" }),
    ).toBeVisible();
  });

  test("moves focus to the question, not the first option", async ({ page }) => {
    await setUpExam(page, { type: "multiple_choice", count: 2 });
    // A screen reader should hear the question, not a pre-selected option.
    const focused = await page.evaluate(() => document.activeElement?.id);
    expect(focused).toBe("question-heading");
  });

  test("uses a real radio group so arrow keys work", async ({ page }) => {
    await setUpExam(page, { type: "multiple_choice", count: 2 });
    const first = page.getByRole("radio").first();
    await first.check();
    await expect(first).toBeChecked();
    await page.keyboard.press("ArrowDown");
    await expect(page.getByRole("radio").nth(1)).toBeChecked();
  });

  test("written answers report a band and criteria, never a percentage", async ({
    page,
  }) => {
    await setUpExam(page, { type: "short_answer", count: 2 });
    await page
      .getByLabel("Your answer")
      .fill("Normalization removes transitive dependencies between attributes.");
    await page.getByRole("button", { name: "Submit answer" }).click();

    const panel = page.getByRole("status").filter({ hasText: "Your answer looks" });
    await expect(panel).toBeVisible({ timeout: 30_000 });
    // ADR-016: no percentage for prose.
    await expect(panel).not.toContainText("%");
  });

  test("exam mode withholds feedback until the end", async ({ page }) => {
    await setUpExam(page, { type: "multiple_choice", count: 2, mode: "exam" });

    await page.getByRole("radio").first().check();
    await page.getByRole("button", { name: "Submit answer" }).click();

    // No per-question feedback in exam mode.
    await expect(page.getByText(/From page \d+ of your document/)).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Next question" })).toHaveCount(
      0,
    );
  });

  test("no WCAG A/AA violations across setup, question and results", async ({
    page,
  }) => {
    const scan = async () => {
      const results = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
        .analyze();
      expect(results.violations).toEqual([]);
    };

    await page.goto("/exam-prep");
    await scan();

    await setUpExam(page, { type: "multiple_choice", count: 2 });
    await scan();

    await page.getByRole("radio").first().check();
    await page.getByRole("button", { name: "Submit answer" }).click();
    await expect(page.getByText(/From page \d+ of your document/)).toBeVisible({
      timeout: 30_000,
    });
    await scan();

    await page.getByRole("button", { name: "Next question" }).click();
    await page.getByRole("radio").first().check();
    await page.getByRole("button", { name: "Submit answer" }).click();
    await page.getByRole("button", { name: "See results" }).click();
    await expect(
      page.getByRole("heading", { name: "Your results" }),
    ).toBeVisible({ timeout: 30_000 });
    await scan();
  });
});
