import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { makeTextPdf } from "../helpers/make-pdf";

/**
 * Portfolio Builder for guests (v6.6.0). Guests sign in through a pass, have
 * no Miami courses, and may type their own or skip; a skipped list leaves no
 * trace on the page. Runs against the mock model, which rewords one typed
 * course and invents another, so the route's filter is what these assert.
 */

const GOOD_PASS = "e2e-guest-pass-1234567890abcdef";
test.use({ storageState: { cookies: [], origins: [] } });

async function signInAsGuest(page: Page) {
  await page.goto(`/guest?pass=${GOOD_PASS}`);
  await page.getByRole("button", { name: "Enter ChatISA as a guest" }).click();
  await expect(page.getByText(/guest-1@guest\.chatisa/)).toBeVisible();
  await page.goto("/portfolio");
  await page.evaluate(() => {
    localStorage.clear();
    indexedDB.deleteDatabase("js-files-v1");
  });
}

function resumePdf(): Buffer {
  return Buffer.from(
    makeTextPdf(["Sam Rivera. Analyst intern, Columbus, 2025. Built dashboards in Tableau."]),
  );
}

/** Resume, then land on the courses step. */
async function careerToCourses(page: Page) {
  await page.goto("/portfolio?mode=career");
  await page.locator('input[type="file"]').first().setInputFiles({
    name: "resume.pdf", mimeType: "application/pdf", buffer: resumePdf(),
  });
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Relevant courses (optional)" })).toBeVisible();
}

/** One project with a file, a name, generate; returns the preview frame. */
async function finishCareer(page: Page) {
  await page.getByRole("button", { name: "Add a project" }).click();
  await page.getByLabel("Add files to project 1").setInputFiles([
    { name: "model.R", mimeType: "text/plain", buffer: Buffer.from("fit <- lm(y ~ x)") },
  ]);
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await page.getByLabel("Your name").fill("Sam Rivera");
  await page.getByRole("button", { name: "Generate my site" }).click();
  await expect(page.getByRole("heading", { name: "Edit the page" })).toBeVisible({ timeout: 30_000 });
  return page.frameLocator('iframe[title="Site preview"]');
}

test.describe("Portfolio Builder for guests", () => {
  test.beforeEach(async ({ page }) => {
    await signInAsGuest(page);
  });

  test("a guest's typed courses appear on the page in their own words", async ({ page }) => {
    await careerToCourses(page);
    // Nothing from the Miami catalog is offered to a guest.
    await expect(page.getByLabel("Find a course")).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Continue without courses" })).toBeEnabled();
    await page.getByLabel("Course 1", { exact: true }).fill("Applied Regression");
    await page.getByLabel("School 1 (optional)").fill("Ohio State");
    await page.getByRole("button", { name: "Add a course" }).click();
    await page.getByLabel("Course 2", { exact: true }).fill("Data Mining");
    const scan = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    expect(scan.violations).toEqual([]);
    await page.getByRole("button", { name: "Continue", exact: true }).click();

    const frame = await finishCareer(page);
    await expect(frame.getByRole("heading", { name: "Coursework" })).toBeVisible();
    await expect(frame.getByText("Applied Regression, Ohio State", { exact: true })).toBeVisible();
    await expect(frame.getByText("Data Mining", { exact: true })).toBeVisible();
    await expect(frame.getByText(/Underwater Basket Weaving/)).toHaveCount(0);
    // The editor lets the guest adjust them.
    await expect(page.getByRole("group", { name: "Other courses" })).toBeVisible();
  });

  test("a guest who removes every outside course in the editor can add them back", async ({ page }) => {
    // Review fix: the Other courses list used to vanish with its last row, and
    // with it any way to restore a course the guest had typed.
    await careerToCourses(page);
    await page.getByLabel("Course 1", { exact: true }).fill("Data Mining");
    await page.getByRole("button", { name: "Continue", exact: true }).click();
    const frame = await finishCareer(page);
    const group = page.getByRole("group", { name: "Other courses" });
    await group.getByRole("button", { name: "Remove" }).click();
    await expect(frame.getByRole("heading", { name: "Coursework" })).toHaveCount(0);
    await expect(group).toBeVisible();
    await group.getByRole("button", { name: "Add" }).click();
    await expect(group.getByLabel("Course", { exact: true })).toHaveValue("Data Mining");
    await expect(frame.getByText("Data Mining", { exact: true })).toBeVisible();
  });

  test("a guest who skips courses gets a page with no Coursework at all", async ({ page }) => {
    await careerToCourses(page);
    await page.getByRole("button", { name: "Continue without courses" }).click();
    const frame = await finishCareer(page);
    await expect(frame.getByRole("heading", { name: "Projects" })).toBeVisible();
    await expect(frame.getByRole("heading", { name: "Coursework" })).toHaveCount(0);
  });

  test("a guest showcase starts from a course at another school", async ({ page }) => {
    await page.goto("/portfolio?mode=project");
    await expect(page.getByRole("radio", { name: "A Miami course" })).toHaveCount(0);
    await expect(page.getByRole("radio", { name: "A course at another school" })).toBeChecked();
    const next = page.getByRole("button", { name: "Next", exact: true });
    await expect(next).toBeDisabled();
    await page.getByLabel("Course and school").fill("STAT 4520, Ohio State");
    const scan = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
    expect(scan.violations).toEqual([]);
    await next.click();
    await page.getByLabel("Add project files").setInputFiles([
      { name: "model.R", mimeType: "text/plain", buffer: Buffer.from("fit <- lm(y ~ x)") },
    ]);
    await page.getByRole("button", { name: "Next", exact: true }).click();
    await page.getByRole("button", { name: "Generate the page" }).click();
    await expect(page.getByRole("heading", { name: "Edit the page" })).toBeVisible({ timeout: 30_000 });
    const frame = page.frameLocator('iframe[title="Site preview"]');
    await expect(frame.getByText("STAT 4520, Ohio State")).toBeVisible();
  });
});
