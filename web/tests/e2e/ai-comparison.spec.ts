import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test.describe("AI Comparison setup", () => {
  test("shows the module and its two setup modes", async ({ page }) => {
    await page.goto("/ai-comparison");
    await expect(
      page.getByRole("heading", { level: 1, name: "AI Comparison" }),
    ).toBeVisible();

    // Anonymous is the default and is a real radio.
    const surprise = page.getByRole("radio", { name: /Surprise me/i });
    await expect(surprise).toBeVisible();
    await expect(surprise).toBeChecked();
    await expect(
      page.getByRole("radio", { name: /Pick the two models/i }),
    ).toBeVisible();

    // Trial count control is present and bounded.
    const trials = page.getByLabel(/How many questions/i);
    await expect(trials).toBeVisible();
    await expect(trials).toHaveValue("1");

    // No model names are shown yet in the default anonymous mode.
    await expect(page.getByText("Choose a different model")).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: "Start comparing" }),
    ).toBeVisible();
  });
});

test.describe("AI Comparison setup interactions", () => {
  test("reveals two model pickers only in pick mode and enforces distinctness", async ({
    page,
  }) => {
    await page.goto("/ai-comparison");

    // Anonymous mode: no model pickers.
    await expect(page.getByText("Choose a different model")).toHaveCount(0);

    // Switching to pick mode reveals two pickers.
    await page.getByRole("radio", { name: /Pick the two models/i }).check();
    await expect(
      page.getByRole("button", { name: "Choose a different model" }),
    ).toHaveCount(2);

    // The trial count clamps to the maximum of five.
    const trials = page.getByLabel(/How many questions/i);
    await trials.fill("9");
    await trials.blur();
    await expect(trials).toHaveValue("5");

    // Start is available with two distinct default picks.
    await expect(
      page.getByRole("button", { name: "Start comparing" }),
    ).toBeEnabled();
  });
});

test.describe("AI Comparison trial", () => {
  test("asks both models, streams two blind answers, and takes a vote", async ({
    page,
  }) => {
    await page.goto("/ai-comparison");

    // Default anonymous mode, one trial.
    await page.getByRole("button", { name: "Start comparing" }).click();

    // The prompt goes to both models.
    await page
      .getByLabel(/Your question for both models/i)
      .fill("How do I read a CSV?");
    await page.getByRole("button", { name: "Ask both models" }).click();

    // Two blind panes appear, labelled by side, never by model name.
    const left = page.getByRole("article", { name: "Answer on the left" });
    const right = page.getByRole("article", { name: "Answer on the right" });
    await expect(left).toContainText("read a CSV in both languages", {
      timeout: 15_000,
    });
    await expect(right).toContainText("read a CSV in both languages", {
      timeout: 15_000,
    });

    // The models are still hidden: no result section during the trial.
    await expect(page.getByText("Result", { exact: true })).toHaveCount(0);

    // Voting is offered once both answers are ready.
    await expect(
      page.getByText(/Both answers are ready/i),
    ).toBeVisible({ timeout: 15_000 });
    await expect(
      page.getByRole("button", { name: "Prefer the left answer" }),
    ).toBeEnabled();
    await page.getByRole("button", { name: "Prefer the left answer" }).click();
  });
});

test.describe("AI Comparison multiple trials", () => {
  test("runs one prompt per trial and advances after each vote", async ({
    page,
  }) => {
    await page.goto("/ai-comparison");

    // Two trials.
    await page.getByLabel(/How many questions/i).fill("2");
    await page.getByRole("button", { name: "Start comparing" }).click();

    // Trial 1.
    await expect(page.getByText("Trial 1 of 2")).toBeVisible();
    await page.getByLabel(/Your question for both models/i).fill("First question");
    await page.getByRole("button", { name: "Ask both models" }).click();
    await expect(page.getByText(/Both answers are ready/i)).toBeVisible({
      timeout: 15_000,
    });
    await page.getByRole("button", { name: "Prefer the left answer" }).click();

    // Trial 2 starts with a fresh, empty prompt (one prompt at a time).
    await expect(page.getByText("Trial 2 of 2")).toBeVisible();
    const secondPrompt = page.getByLabel(/Your question for both models/i);
    await expect(secondPrompt).toHaveValue("");
    await secondPrompt.fill("Second question");
    await page.getByRole("button", { name: "Ask both models" }).click();
    await expect(page.getByText(/Both answers are ready/i)).toBeVisible({
      timeout: 15_000,
    });
    await page.getByRole("button", { name: "Prefer the right answer" }).click();

    // After the last vote the trial screen is gone.
    await expect(page.getByText("Trial 2 of 2")).toHaveCount(0);
  });
});

test.describe("AI Comparison report", () => {
  test("reveals both models and highlights a winner after one trial", async ({
    page,
  }) => {
    await page.goto("/ai-comparison");
    await page.getByRole("button", { name: "Start comparing" }).click();

    await page.getByLabel(/Your question for both models/i).fill("A question");
    await page.getByRole("button", { name: "Ask both models" }).click();
    await expect(page.getByText(/Both answers are ready/i)).toBeVisible({
      timeout: 15_000,
    });
    await page.getByRole("button", { name: "Prefer the left answer" }).click();

    // Now the models are revealed.
    const result = page.getByRole("region", { name: "Comparison result" });
    await expect(result).toBeVisible();
    // Exactly one winner marker for a single-vote trial (no tie).
    await expect(result.getByText("Winner", { exact: true })).toHaveCount(1);
    // Both vote tallies are shown.
    await expect(result.getByText(/vote/)).toHaveCount(2);

    // The report is accessible.
    const axe = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);

    // Restarting returns to setup.
    await page.getByRole("button", { name: "Run another comparison" }).click();
    await expect(
      page.getByRole("radio", { name: /Surprise me/i }),
    ).toBeVisible();
  });

  test("declares a tie when votes are split across two trials", async ({
    page,
  }) => {
    await page.goto("/ai-comparison");
    await page.getByLabel(/How many questions/i).fill("2");
    await page.getByRole("button", { name: "Start comparing" }).click();

    // Sides alternate every trial (D4), so voting the same screen side both
    // times sends one vote to each model: a guaranteed tie.
    for (let trial = 0; trial < 2; trial++) {
      await page
        .getByLabel(/Your question for both models/i)
        .fill(`Question ${trial + 1}`);
      await page.getByRole("button", { name: "Ask both models" }).click();
      await expect(page.getByText(/Both answers are ready/i)).toBeVisible({
        timeout: 15_000,
      });
      await page
        .getByRole("button", { name: "Prefer the left answer" })
        .click();
    }

    const result = page.getByRole("region", { name: "Comparison result" });
    await expect(result.getByRole("heading", { name: "It is a tie" })).toBeVisible();
    await expect(result.getByText("Winner", { exact: true })).toHaveCount(0);
  });
});
