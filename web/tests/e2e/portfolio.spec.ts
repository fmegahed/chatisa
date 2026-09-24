import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { makeTextPdf } from "../helpers/make-pdf";
import { fakeGithubApi } from "./support/fake-github";

/**
 * Portfolio Builder end to end (2026-08-20). Both wizards run against the
 * mock model's canned content and the in-test GitHub API, so the assertions
 * are about the flow and the exact file set a publish pushes, never about
 * the words a real model would write.
 */

function resumePdf(): Buffer {
  return Buffer.from(
    makeTextPdf([
      [
        "Ada Lovelace",
        "lovelaa@MiamiOH.edu | (513) 555-1010",
        "Analytics Intern, Acme Logistics, Summer 2025",
        "Built weekly reports in SQL for the operations team",
      ].join(" "),
    ]),
  );
}

const NOTEBOOK = JSON.stringify({
  cells: [{ cell_type: "code", source: ["print(1)"], outputs: [] }],
  metadata: {},
  nbformat: 4,
  nbformat_minor: 5,
});

/** A minimal PNG header: enough bytes for the intake, tiny to push. */
const PNG = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

/** CHATISA_MOCK_GITHUB=1: the start route short-circuits to the callback. */
async function connectGithub(page: Page) {
  const popup = page.waitForEvent("popup");
  await page.getByRole("button", { name: "Connect GitHub" }).click();
  await popup;
  await expect(
    page.getByText("Connected to GitHub as").first(),
  ).toBeVisible({ timeout: 15_000 });
}

/** A course row in the checklist (v6.7.0): a fieldset named by its legend. */
function courseRow(page: Page, code: string) {
  return page.getByRole("group", { name: new RegExp(`^${code} `) }).first();
}

/** Course, one notebook, one figure, generate: the shortest way to review. */
async function showcaseToReview(page: Page) {
  await page.goto("/portfolio?mode=project");
  await page.getByTitle("Principles of Business Analytics").click();
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await page.getByLabel("Add project files").setInputFiles([
    { name: "analysis.ipynb", mimeType: "application/json", buffer: Buffer.from(NOTEBOOK) },
    { name: "roc.png", mimeType: "image/png", buffer: PNG },
  ]);
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await page.getByRole("button", { name: "Generate the page" }).click();
  await expect(
    page.getByRole("heading", { name: "Edit the page" }),
  ).toBeVisible({ timeout: 30_000 });
}

test.describe("Portfolio Builder", () => {
  test.beforeEach(async ({ page }) => {
    // Drafts and published records live in localStorage, the device resume in
    // IndexedDB; start clean so a run never inherits another test's site.
    await page.goto("/portfolio");
    await page.evaluate(() => {
      localStorage.clear();
      indexedDB.deleteDatabase("js-files-v1");
    });
  });

  test("career portfolio: inputs, generate, edit, publish", async ({ page }) => {
    const gh = await fakeGithubApi(page);
    await page.goto("/portfolio?mode=career");
    await page
      .locator('input[type="file"]')
      .first()
      .setInputFiles({
        name: "ada-resume.pdf",
        mimeType: "application/pdf",
        buffer: resumePdf(),
      });
    await page.getByRole("button", { name: "Next", exact: true }).click();

    await courseRow(page, "ISA 225").getByRole("radio", { name: "Done" }).check();
    await page.getByRole("button", { name: "Next", exact: true }).click();

    await page.getByRole("button", { name: "Add a project" }).click();
    await page.getByLabel("Add files to project 1").setInputFiles([
      { name: "model.R", mimeType: "text/plain", buffer: Buffer.from("fit <- lm(y ~ x)") },
      { name: "train.csv", mimeType: "text/csv", buffer: Buffer.from("a,b\n1,2") },
    ]);
    // Data is unpublished by default; code is not.
    await expect(page.getByLabel("model.R")).toBeChecked();
    await expect(page.getByLabel("train.csv")).not.toBeChecked();
    await page.getByRole("button", { name: "Next", exact: true }).click();

    await page.getByLabel("Your name").fill("Ada Lovelace");
    await page.getByRole("button", { name: "Generate my site" }).click();
    await expect(
      page.getByRole("heading", { name: "Edit the page" }),
    ).toBeVisible({ timeout: 30_000 });

    // Editing updates the preview: it renders from the same function the
    // publish pushes, so what is on screen is what goes to GitHub.
    await page.getByLabel("Headline").fill("Analytics student who ships");
    const frame = page.frameLocator('iframe[title="Site preview"]');
    await expect(frame.getByText("Analytics student who ships")).toBeVisible();

    await connectGithub(page);
    await page.getByRole("button", { name: "Publish to GitHub Pages" }).click();
    await expect(page.getByText(/Your site is live at/)).toBeVisible({ timeout: 15_000 });
    const paths = gh.trees.at(-1)!.map((t) => t.path);
    expect(paths).toContain("index.html");
    expect(paths).toContain("projects/project-1/model.R");
    expect(paths).not.toContain("projects/project-1/train.csv");
    // The resume is only published when the student ticks the download link.
    expect(paths).not.toContain("resume.pdf");
  });

  test("project showcase: roles, story, publish, then counts in Job Scout and JobApp Drafter", async ({
    page,
  }) => {
    const gh = await fakeGithubApi(page);
    await page.goto("/portfolio?mode=project");
    await page.getByTitle("Principles of Business Analytics").click();
    await page.getByPlaceholder("Ann Lee, Bo Chen").fill("Ann Lee, Bo Chen");
    await page.getByRole("button", { name: "Next", exact: true }).click();

    await page.getByLabel("Add project files").setInputFiles([
      { name: "analysis.ipynb", mimeType: "application/json", buffer: Buffer.from(NOTEBOOK) },
      { name: "roc.png", mimeType: "image/png", buffer: PNG },
      { name: "data.csv", mimeType: "text/csv", buffer: Buffer.from("a,b") },
    ]);
    // The guessed role decides the folder, and data starts unpublished.
    await expect(page.getByText("code/analysis.ipynb")).toBeVisible();
    await expect(page.getByLabel("Role for roc.png")).toHaveValue("figure");
    await expect(page.getByLabel("Publish data.csv")).not.toBeChecked();
    await page.getByRole("button", { name: "Next", exact: true }).click();

    await page.getByLabel(/What problem were you solving/).fill("Predict churn.");
    await page.getByRole("button", { name: "Generate the page" }).click();
    await expect(
      page.getByRole("heading", { name: "Edit the page" }),
    ).toBeVisible({ timeout: 30_000 });
    const frame = page.frameLocator('iframe[title="Site preview"]');
    await expect(frame.getByText(/Ann Lee, Bo Chen/)).toBeVisible();

    await connectGithub(page);
    await page.getByRole("button", { name: "Publish to GitHub Pages" }).click();
    await expect(page.getByText(/Your site is live at/)).toBeVisible({ timeout: 15_000 });
    const paths = gh.trees.at(-1)!.map((t) => t.path);
    expect(paths).toEqual(
      expect.arrayContaining([
        "index.html",
        "README.md",
        ".gitignore",
        "code/analysis.ipynb",
        "figures/roc.png",
      ]),
    );
    expect(paths).not.toContain("data/data.csv");

    // The mode step lists it.
    await page.goto("/portfolio");
    await expect(page.getByRole("heading", { name: "Your sites" })).toBeVisible();

    // JobApp Drafter offers it.
    await page.goto("/jobapp-drafter");
    await expect(page.getByText(/Include my published work \(1\)/)).toBeVisible();

    // Job Scout counts it as real work: the published page's skills join the
    // profile's strengths. ISA 225 teaches neither R nor SQL, so both rows
    // can only have come from the site that was just published.
    await page.goto("/job-scout");
    await courseRow(page, "ISA 225").getByRole("radio", { name: "Done" }).check();
    await page
      .getByRole("button", { name: "Save profile and see this week's jobs" })
      .click();
    await expect(
      page.getByRole("heading", { name: "This week's jobs" }),
    ).toBeVisible();
    await page.getByRole("tab", { name: "My Profile" }).click();
    await expect(page.getByRole("heading", { name: "Skills you are building" })).toBeVisible();
    await expect(page.getByLabel("Your level for R", { exact: true })).toBeVisible();
    await expect(page.getByLabel("Your level for SQL", { exact: true })).toBeVisible();
  });

  test("a reload mid-wizard offers the saved draft back, files included", async ({ page }) => {
    // The old error copy said "reload and try again", and a reload used to
    // wipe every upload. Now the draft autosaves and is offered on return.
    await page.goto("/portfolio?mode=project");
    await page.getByTitle("Principles of Business Analytics").click();
    await page.getByRole("button", { name: "Next", exact: true }).click();
    await page.getByLabel("Add project files").setInputFiles([
      { name: "analysis.ipynb", mimeType: "application/json", buffer: Buffer.from(NOTEBOOK) },
    ]);
    await expect(page.getByText("analysis.ipynb").first()).toBeVisible();
    // The limits are stated up front, from the same constants the code enforces.
    await expect(page.getByText(/Up to 25 MB per file and 100 MB for the whole site/)).toBeVisible();
    await page.waitForTimeout(1_000); // past the autosave debounce
    await page.reload();
    await expect(page.getByText("You have an unfinished showcase")).toBeVisible();
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(page.getByRole("heading", { name: "Project files" })).toBeVisible();
    await expect(page.getByText("analysis.ipynb").first()).toBeVisible();
    // Discarding from the front door clears it.
    await page.goto("/portfolio");
    await page.getByRole("button", { name: "Discard" }).click();
    await expect(page.getByText("You have an unfinished showcase")).toHaveCount(0);
    await page.reload();
    await expect(page.getByRole("heading", { name: "Portfolio Builder" })).toBeVisible();
    await expect(page.getByText("You have an unfinished showcase")).toHaveCount(0);
  });

  test("a Miami showcase names its repository after the course, with the course title in the header", async ({ page }) => {
    await fakeGithubApi(page);
    await showcaseToReview(page);
    const frame = page.frameLocator('iframe[title="Site preview"]');
    await expect(frame.getByText("ISA 225 - Principles of Business Analytics")).toBeVisible();
    await connectGithub(page);
    await expect(page.getByLabel("Repository name")).toHaveValue(/^isa-225-/);
  });

  for (const [choice, label] of [
    ["Self-study", "Self-study project"],
    ["A hobby or personal project", "Personal project"],
  ] as const) {
    test(`a showcase marked "${label}" needs no course and names the repository after the title`, async ({ page }) => {
      await fakeGithubApi(page);
      await page.goto("/portfolio?mode=project");
      await expect(page.getByRole("radio", { name: "A Miami course" })).toBeChecked();
      await page.getByRole("radio", { name: choice }).check();
      // No course to pick for this origin: the picker is gone and Next is open.
      await expect(page.getByLabel("Find a course")).toHaveCount(0);
      await page.getByRole("button", { name: "Next", exact: true }).click();
      await page.getByLabel("Add project files").setInputFiles([
        { name: "analysis.ipynb", mimeType: "application/json", buffer: Buffer.from(NOTEBOOK) },
      ]);
      await page.getByRole("button", { name: "Next", exact: true }).click();
      await page.getByRole("button", { name: "Generate the page" }).click();
      await expect(page.getByRole("heading", { name: "Edit the page" })).toBeVisible({ timeout: 30_000 });
      const frame = page.frameLocator('iframe[title="Site preview"]');
      await expect(frame.getByText(label, { exact: true })).toBeVisible();
      await connectGithub(page);
      const repo = page.getByLabel("Repository name");
      await expect(repo).not.toHaveValue(/^isa-/);
      await expect(repo).not.toHaveValue("");
    });
  }

  test("the origin step switches between a Miami course and a typed course without leaking text", async ({ page }) => {
    await page.goto("/portfolio?mode=project");
    const next = page.getByRole("button", { name: "Next", exact: true });
    await expect(next).toBeDisabled();
    await page.getByRole("radio", { name: "A course at another school" }).check();
    await expect(next).toBeDisabled();
    await page.getByLabel("Course and school").fill("STAT 4520, Ohio State");
    await expect(next).toBeEnabled();
    // Back to Miami: the typed text must not count as a Miami course.
    await page.getByRole("radio", { name: "A Miami course" }).check();
    await expect(next).toBeDisabled();
    const scan = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    expect(scan.violations).toEqual([]);
  });

  test("the origin step is a heading screen readers can jump to", async ({ page }) => {
    await page.goto("/portfolio?mode=project");
    await expect(page.getByRole("heading", { level: 2, name: "Where did this project come from?" })).toBeVisible();
  });

  test("arrowing through the origins keeps what was already chosen for each", async ({ page }) => {
    // Review fix: arrow keys move a radio group's selection, and each move
    // used to wipe the picked course, so a keyboard user browsing the
    // options lost their pick.
    await page.goto("/portfolio?mode=project");
    const next = page.getByRole("button", { name: "Next", exact: true });
    await page.getByTitle("Principles of Business Analytics").click();
    await expect(next).toBeEnabled();
    await page.getByRole("radio", { name: "A Miami course" }).focus();
    await page.keyboard.press("ArrowDown");
    await expect(page.getByRole("radio", { name: "A course at another school" })).toBeChecked();
    await page.getByLabel("Course and school").fill("STAT 4520, Ohio State");
    await page.getByRole("radio", { name: "A course at another school" }).focus();
    await page.keyboard.press("ArrowUp");
    await expect(page.getByRole("radio", { name: "A Miami course" })).toBeChecked();
    await expect(page.getByTitle("Principles of Business Analytics")).toHaveAttribute("aria-pressed", "true");
    await expect(next).toBeEnabled();
    await page.getByRole("radio", { name: "A Miami course" }).focus();
    await page.keyboard.press("ArrowDown");
    await expect(page.getByLabel("Course and school")).toHaveValue("STAT 4520, Ohio State");
  });

  test("meets WCAG A and AA on the mode step and the review step", async ({ page }) => {
    await page.goto("/portfolio");
    await expect(page.getByRole("heading", { name: "Portfolio Builder" })).toBeVisible();
    const modeScan = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    expect(modeScan.violations).toEqual([]);

    await showcaseToReview(page);
    const reviewScan = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    expect(reviewScan.violations).toEqual([]);
  });
});

test.describe("Portfolio Builder access control", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("unauthenticated visitors are redirected and the API answers 401", async ({
    page,
    request,
  }) => {
    await page.goto("/portfolio");
    await expect(page).toHaveURL(/\/login/);
    const res = await request.post("/api/portfolio/generate", {
      multipart: { modelId: "x", mode: "career", payload: "{}" },
    });
    expect(res.status()).toBe(401);
    const event = await request.post("/api/portfolio/event", {
      data: { kind: "career" },
    });
    expect(event.status()).toBe(401);
  });
});
