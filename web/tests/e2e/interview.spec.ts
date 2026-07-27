import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { makeTextPdf } from "../helpers/make-pdf";

/**
 * The whole interview, driven by typing.
 *
 * Typing is the path under test on purpose: speech is an enhancement, and the
 * legacy module's worst defect was that there was no text path at all, so a
 * student without a working microphone had no way through the feature. If this
 * suite can complete an interview without ever touching the microphone, that
 * defect cannot come back unnoticed.
 */

const ANSWER =
  "On a class project our survey data had duplicate responses from the same student ID. " +
  "I wrote a short R script to flag the duplicates, checked a sample by hand to confirm they " +
  "were genuine repeats rather than siblings, and kept the most recent response. That changed " +
  "our headline average by about four points, so we reported the corrected figure.";

function resumePdf(): Buffer {
  return Buffer.from(
    makeTextPdf([
      [
        "Kaitlin Jones joneskl@MiamiOH.edu",
        "Data Analytics Intern, Acme Logistics, Summer 2025",
        "Built weekly reports in Excel and SQL for the operations team",
        "Cleaned shipment data and flagged duplicate records",
      ].join(" "),
    ]),
  );
}

async function startInterview(
  page: Page,
  questions = 3,
  jobTitle = "Business Analytics Intern",
): Promise<string> {
  await page.goto("/interview-mentor");
  await page.getByLabel("Company").fill("Northwind Analytics");
  await page.getByLabel("Job title").fill(jobTitle);
  await page.getByLabel("Number of questions").selectOption(String(questions));
  await page
    .getByLabel("Or paste the job description")
    .fill("We need an analyst comfortable with SQL and reporting.");
  await page.locator("#resume-file").setInputFiles({
    name: "resume.pdf",
    mimeType: "application/pdf",
    buffer: resumePdf(),
  });

  // The signed-in account is shared across specs and projects, so the id comes
  // from this request's own response rather than from "the newest interview",
  // which another spec running in parallel could own.
  const created = page.waitForResponse(
    (res) =>
      res.url().endsWith("/api/interview") &&
      res.request().method() === "POST" &&
      res.ok(),
  );
  await page.getByRole("button", { name: "Start interview" }).click();
  const interviewId = (await (await created).json()).interviewId as string;

  await expect(page.getByText(/Question 1 of/)).toBeVisible({ timeout: 30_000 });
  return interviewId;
}

async function answerCurrent(page: Page, text = ANSWER) {
  await page.getByLabel("Your answer").fill(text);
  await page.getByRole("button", { name: "Submit answer" }).click();
}

test.describe("Interview Mentor", () => {
  test("runs an interview to results using only the keyboard path", async ({
    page,
  }) => {
    await startInterview(page, 3);

    // The question heading, not the footer's h2s. Questions are not always
    // phrased with a question mark, so match the tabbable one instead.
    await expect(page.locator("h2[tabindex]")).toBeVisible();
    await answerCurrent(page);
    await expect(page.getByText(/Question 2 of 3/)).toBeVisible({
      timeout: 30_000,
    });
    await answerCurrent(page);
    await expect(page.getByText(/Question 3 of 3/)).toBeVisible({
      timeout: 30_000,
    });
    await answerCurrent(page);

    await expect(
      page.getByRole("heading", { name: "How that went" }),
    ).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/You answered 3 of 3 questions/)).toBeVisible();
  });

  test("never shows a percentage or a score out of 100", async ({ page }) => {
    // ADR-016. The legacy module asked the model to compute a score out of 100
    // mid-conversation and showed it to the student as their interview result.
    await startInterview(page, 3);
    for (let i = 0; i < 3; i++) {
      await answerCurrent(page);
      await page.waitForTimeout(300);
    }
    await expect(
      page.getByRole("heading", { name: "How that went" }),
    ).toBeVisible({ timeout: 30_000 });

    const body = (await page.locator("body").innerText()).toLowerCase();
    expect(body).not.toMatch(/\d+\s*%/);
    expect(body).not.toMatch(/out of 100/);
    expect(body).not.toMatch(/score:/);
  });

  test("records a skipped question as skipped, not as wrong", async ({
    page,
  }) => {
    await startInterview(page, 3);
    await page.getByRole("button", { name: "Skip this question" }).click();
    await expect(page.getByText(/Question 2 of 3/)).toBeVisible({
      timeout: 30_000,
    });
    await answerCurrent(page);
    await expect(page.getByText(/Question 3 of 3/)).toBeVisible({
      timeout: 30_000,
    });
    await answerCurrent(page);

    await expect(page.getByText(/skipped 1/)).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText("You skipped this one.")).toBeVisible();
  });

  test("does not reveal any judgement while the interview is running", async ({
    page,
  }) => {
    // Feedback mid-interview changes how a student answers the rest, so it is
    // withheld by the projection rather than merely hidden by the UI.
    const interviewId = await startInterview(page, 3);
    await answerCurrent(page);
    await expect(page.getByText(/Question 2 of 3/)).toBeVisible({
      timeout: 30_000,
    });

    const state = await page.evaluate(async (id) => {
      return (await (await fetch(`/api/interview/${id}`)).json()).interview;
    }, interviewId);

    expect(state.status).toBe("in_progress");
    for (const turn of state.turns) {
      expect(turn.criteria).toBeUndefined();
      expect(turn.band).toBeUndefined();
      expect(turn.strength).toBeUndefined();
    }
    expect(state.results).toBeUndefined();
  });

  test("moves focus to the question so it is announced", async ({ page }) => {
    await startInterview(page, 3);
    const focusedTag = await page.evaluate(
      () => document.activeElement?.tagName,
    );
    expect(focusedTag).toBe("H2");
  });

  test("offers an unfinished interview and continues at the right question", async ({
    page,
  }) => {
    await startInterview(page, 3);
    await answerCurrent(page);
    await expect(page.getByText(/Question 2 of 3/)).toBeVisible({
      timeout: 30_000,
    });

    await page.goto("/interview-mentor");
    await expect(
      page.getByRole("heading", { name: "Pick up where you left off" }),
    ).toBeVisible();
    await page
      .getByRole("listitem")
      .filter({ hasText: "question 2 of 3" })
      .first()
      .getByRole("button", { name: "Continue" })
      .click();
    await expect(page.getByText(/Question 2 of 3/)).toBeVisible({
      timeout: 30_000,
    });
  });

test("can discard an unfinished interview instead of continuing", async ({
    page,
  }) => {
    // A unique title so this test finds its own row in the shared account.
    const title = `Discardable Role ${Date.now()}`;
    await startInterview(page, 3, title);
    await answerCurrent(page);
    await expect(page.getByText(/Question 2 of 3/)).toBeVisible({
      timeout: 30_000,
    });

    await page.goto("/interview-mentor");
    const item = page.getByRole("listitem").filter({ hasText: title });
    await expect(item).toBeVisible();
    await item.getByRole("button", { name: "Discard" }).click();
    // It leaves the resume list, and is gone on reload rather than resurrected.
    await expect(item).toHaveCount(0);
    await page.reload();
    await expect(
      page.getByRole("listitem").filter({ hasText: title }),
    ).toHaveCount(0);
  });

  test("answering works with no microphone available", async ({ page }) => {
    // Simulates a machine with no recording support at all. The interview must
    // still be completable, and the UI must say so rather than going silent.
    await page.addInitScript(() => {
      // @ts-expect-error deliberately removing the API
      delete window.MediaRecorder;
    });
    await startInterview(page, 3);
    await expect(
      page.getByText(/Dictation is not available in this browser/),
    ).toBeVisible();
    await answerCurrent(page);
    await expect(page.getByText(/Question 2 of 3/)).toBeVisible({
      timeout: 30_000,
    });
  });

  test("no WCAG A/AA violations across setup, question and results", async ({
    page,
  }) => {
    await page.goto("/interview-mentor");
    const setup = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(setup.violations).toEqual([]);

    await startInterview(page, 3);
    const question = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(question.violations).toEqual([]);

    for (let i = 0; i < 3; i++) {
      await answerCurrent(page);
      await page.waitForTimeout(300);
    }
    await expect(
      page.getByRole("heading", { name: "How that went" }),
    ).toBeVisible({ timeout: 30_000 });
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(results.violations).toEqual([]);
  });
});

/**
 * What a student is told when the interviewer cannot speak.
 *
 * Added 2026-07-26. The professor reported that Interview Mentor produces no
 * voice in production, and the reason it could not be diagnosed from the report
 * is that the UI said NOTHING: a failed /api/speech/speak fell straight through
 * to opening the microphone. No voice and no explanation are indistinguishable
 * from the feature not existing.
 *
 * The failures are injected with page.route rather than by breaking the server,
 * so these run in the ordinary mock suite: free, deterministic, and they cover
 * the branch a real Deepgram outage would take.
 */
test.describe("Interview Mentor without speech", () => {
  /** The 503 body /api/speech/speak returns when DEEPGRAM_TOKEN is absent. */
  const notConfigured = {
    status: 503,
    contentType: "application/json",
    body: JSON.stringify({
      error: "Speech is not set up on this server. The question is shown as text.",
    }),
  };

  /**
   * Playwright's Chromium DOES expose getUserMedia and MediaRecorder, so
   * handsFreeAvailable() is true and the module renders the HANDS-FREE path by
   * default. An earlier version of these tests assumed the opposite and looked
   * for the manual "Hear this question" button, which never rendered. Both paths
   * are covered below, because the fix touched both components.
   */
  test("hands-free says why the interviewer is silent, and keeps going", async ({
    page,
  }) => {
    await page.route("**/api/speech/speak", (route) => route.fulfill(notConfigured));

    await startInterview(page, 3);
    // Voice mode is the default where the browser supports it, so the
    // hands-free controls are what appear.
    await expect(
      page.getByRole("button", { name: "Switch to typing" }),
    ).toBeVisible();

    // This is the defect: before the fix, a failed speak() call fell straight
    // through to opening the microphone and said nothing at all.
    await expect(
      page.getByText(/Spoken questions are not set up on this server/i),
    ).toBeVisible();

    // And the interview itself is unaffected: speech is an enhancement.
    await answerCurrent(page);
    await expect(page.getByText(/Question 2 of 3/)).toBeVisible();
  });

  test("the manual player says speech is not set up, and offers no retry", async ({
    page,
  }) => {
    await page.route("**/api/speech/speak", (route) => route.fulfill(notConfigured));

    await startInterview(page, 3);
    // Leave voice mode to reach QuestionAudio, which is also what a student on a
    // machine without a microphone sees.
    await page.getByRole("button", { name: "Switch to typing" }).click();

    const hear = page.getByRole("button", { name: "Hear this question" });
    await expect(hear).toBeVisible();
    await hear.click();

    await expect(
      page.getByText(/Spoken questions are not set up on this server/i),
    ).toBeVisible();
    // A retry cannot help when the server has no credential, so the control must
    // not invite one.
    await expect(hear).toBeDisabled();
  });

  test("the manual player invites a retry for a transient failure", async ({
    page,
  }) => {
    await page.route("**/api/speech/speak", (route) =>
      route.fulfill({
        status: 502,
        contentType: "application/json",
        body: JSON.stringify({ error: "That question could not be read aloud." }),
      }),
    );

    await startInterview(page, 3);
    await page.getByRole("button", { name: "Switch to typing" }).click();

    const hear = page.getByRole("button", { name: "Hear this question" });
    await hear.click();

    await expect(
      page.getByText(/Try again, or read the question above/i),
    ).toBeVisible();
    // 502 is worth another press, so the control stays live. This is the
    // distinction the UI used to collapse into one silent path.
    await expect(hear).toBeEnabled();
  });
});

test.describe("interview API guards", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("requires sign-in for every interview route", async ({ request }) => {
    const list = await request.get("/api/interview");
    expect(list.status()).toBe(401);

    const start = await request.post("/api/interview", {
      data: { modelId: "gpt-5.6-terra", interviewType: "mixed", jobTitle: "Analyst", questionCount: 3 },
    });
    expect(start.status()).toBe(401);

    const token = await request.post("/api/speech/token");
    expect(token.status()).toBe(401);

    const speak = await request.post("/api/speech/speak", {
      data: { text: "hello" },
    });
    expect(speak.status()).toBe(401);
  });
});
