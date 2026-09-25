import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { fakeGithubApi } from "./support/fake-github";
import { makeTextPdf } from "../helpers/make-pdf";

async function openCareerProjects(page: Page) {
  await page.goto("/portfolio?mode=career");
  // Resume step: a resume is required; Classes step: mark one course.
  await page.locator('input[type="file"]').first().setInputFiles({
    name: "ada-resume.pdf", mimeType: "application/pdf", buffer: Buffer.from(makeTextPdf(["Ada Lovelace, analytics student"])),
  });
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await page.getByRole("group", { name: /^ISA 225 / }).first().getByRole("radio", { name: "Done" }).check();
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await page.getByRole("button", { name: "Add a project" }).click();
}

test.describe("Import from GitHub", () => {
  test.beforeEach(async ({ page }) => {
    // Drafts live in this browser; start clean so no test inherits another's.
    await page.goto("/portfolio");
    // Connected before the builder opens: a reload after it autosaves would
    // offer the saved draft back instead of the step under test.
    await page.evaluate(() => {
      localStorage.clear();
      indexedDB.deleteDatabase("js-files-v1");
      localStorage.setItem("js-github-v1", JSON.stringify({ v: 1, token: "t", login: "mockstudent", connectedAt: "" }));
    });
  });

  test("a career project imports chosen files and links back to the repository", async ({ page }) => {
    await fakeGithubApi(page);
    await openCareerProjects(page);
    const toggle = page.getByRole("button", { name: "Import from GitHub into project 1" });
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-expanded", "true");
    await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).check();
    // README and code are pre-ticked; the 14 MB notebook is listed.
    await expect(page.getByRole("checkbox", { name: /README\.md/ })).toBeChecked();
    await expect(page.getByRole("checkbox", { name: /src\/model\.py/ })).toBeChecked();
    await page.getByRole("checkbox", { name: /figures\/roc\.png/ }).check();
    await page.getByRole("checkbox", { name: /tests\/model\.py/ }).check();
    // This fake file is listed but answers 404; its own test covers that.
    await page.getByRole("checkbox", { name: /src\/vanishing\.py/ }).uncheck();
    await page.getByRole("button", { name: /^Import \d+ files$/ }).click();
    // Both model.py files arrive under distinguishable names; the PNG is a figure.
    await expect(page.getByText("src_model.py")).toBeVisible();
    await expect(page.getByText("tests_model.py")).toBeVisible();
    await expect(page.getByText("roc.png", { exact: true })).toBeVisible();
    await expect(page.getByLabel("Title (optional)").first()).toHaveValue("churn-model");
    await expect(page.getByLabel("Link (repo or demo, optional)").first()).toHaveValue("https://github.com/mockstudent/churn-model");
    await expect(toggle).toBeFocused();
  });

  test("edits made while files download are kept, and a reopened panel offers nothing twice (review fix)", async ({ page }) => {
    await fakeGithubApi(page, { slowContentsMs: 700 });
    await openCareerProjects(page);
    const toggle = page.getByRole("button", { name: "Import from GitHub into project 1" });
    await toggle.click();
    await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).check();
    await page.getByRole("checkbox", { name: /src\/vanishing\.py/ }).uncheck();
    await page.getByRole("button", { name: /^Import \d+ files$/ }).click();
    // Typed during the downloads: the import must not overwrite it.
    await page.getByLabel("Title (optional)").first().fill("My churn project");
    await expect(page.getByLabel("Link (repo or demo, optional)").first()).toHaveValue("https://github.com/mockstudent/churn-model", { timeout: 30_000 });
    await expect(page.getByLabel("Title (optional)").first()).toHaveValue("My churn project");
    // The confirmation is announced, outside the closed panel.
    await expect(page.locator("p[aria-live=polite]").filter({ hasText: /^Imported \d+ files from mockstudent\/churn-model\.$/ })).toHaveCount(1);
    // Reopening offers nothing to import twice.
    await toggle.click();
    await expect(page.getByRole("button", { name: "Import 0 files" })).toBeDisabled();
  });

  test("the repository choice is locked while a file list loads (review fix)", async ({ page }) => {
    await fakeGithubApi(page, { slowTreeMs: 3_000 });
    await openCareerProjects(page);
    await page.getByRole("button", { name: "Import from GitHub into project 1" }).click();
    await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).check();
    await expect(page.getByRole("radio", { name: /mockstudent\/class-notes/ })).toBeDisabled();
    await expect(page.getByRole("checkbox", { name: /README\.md/ })).toBeVisible({ timeout: 15_000 });
    await expect(page.getByRole("radio", { name: /mockstudent\/class-notes/ })).toBeEnabled();
  });

  test("a file that fails to download is named, and the rest still import", async ({ page }) => {
    await fakeGithubApi(page);
    await openCareerProjects(page);
    await page.getByRole("button", { name: "Import from GitHub into project 1" }).click();
    await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).check();
    await page.getByRole("checkbox", { name: /src\/vanishing\.py/ }).check();
    await page.getByRole("button", { name: /^Import \d+ files$/ }).click();
    await expect(page.getByRole("alert").filter({ hasText: "src/vanishing.py could not be read" })).toBeVisible();
    await expect(page.getByText("README.md").first()).toBeVisible();
  });

  test("an expired connection stops and offers Connect GitHub", async ({ page }) => {
    await fakeGithubApi(page, { expireAfterList: true });
    await openCareerProjects(page);
    await page.getByRole("button", { name: "Import from GitHub into project 1" }).click();
    // click, not check: choosing it expires the connection and removes the
    // list, so a checked-state confirmation would wait forever.
    await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).click();
    await expect(page.getByRole("alert").filter({ hasText: "Your GitHub connection has expired. Connect again to continue." })).toBeVisible();
    await expect(page.getByRole("button", { name: "Connect GitHub" })).toBeVisible();
  });

  test("a showcase imports files and the preview links the original repository", async ({ page }) => {
    await fakeGithubApi(page);
    await page.goto("/portfolio?mode=project");
    await page.getByTitle("Principles of Business Analytics").click();
    await page.getByRole("button", { name: "Next", exact: true }).click();
    await page.getByRole("button", { name: "Import from GitHub" }).click();
    await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).check();
    await page.getByRole("button", { name: /^Import \d+ files$/ }).click();
    await expect(page.getByText("README.md").first()).toBeVisible();
    await expect(page.getByRole("link", { name: "https://github.com/mockstudent/churn-model" })).toBeVisible();
  });

  test("an upload over 25 MB is refused with the reason, and the rest are added (professor, 2026-09-24)", async ({ page }) => {
    await page.goto("/portfolio?mode=project");
    await page.getByTitle("Principles of Business Analytics").click();
    await page.getByRole("button", { name: "Next", exact: true }).click();
    await page.getByLabel("Add project files").setInputFiles([
      { name: "big.csv", mimeType: "text/csv", buffer: Buffer.alloc(26 * 1024 * 1024) },
      { name: "model.R", mimeType: "text/plain", buffer: Buffer.from("x <- 1") },
    ]);
    await expect(page.getByRole("alert").filter({ hasText: "big.csv (26.0 MB) is over the 25 MB limit for one file on a published page, so it was not added." })).toBeVisible();
    await expect(page.getByText("model.R").first()).toBeVisible();
    await expect(page.getByText("A file over 25 MB cannot be added")).toBeVisible();
  });

  for (const width of [1280, 320]) {
    test(`meets WCAG A and AA with the panel open at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await fakeGithubApi(page);
      await openCareerProjects(page);
      await page.getByRole("button", { name: "Import from GitHub into project 1" }).click();
      await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).check();
      await expect(page.getByRole("checkbox", { name: /README\.md/ })).toBeVisible();
      const scan = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
      expect(scan.violations).toEqual([]);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
    });
  }
});
