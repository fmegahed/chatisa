import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

/**
 * Project Assistant foundations, end to end.
 *
 * Auth: this project runs with the saved test-student session (storageState
 * from auth.setup.ts), so the page is already signed in. No inline login.
 *
 * Axe: the same WCAG A/AA scan the other specs run, inlined as runAxe below.
 *
 * The project name carries a per-run suffix because the desktop and mobile
 * Playwright projects run this spec in parallel against one shared data dir;
 * a fixed name would let two "My projects" links match in strict mode.
 */
async function runAxe(page: Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(results.violations).toEqual([]);
}

test.describe("Project Assistant foundations", () => {
  test("create a project, see it listed, open its workspace", async ({ page }, testInfo) => {
    const projectName = `Playwright project ${testInfo.project.name} ${Date.now()}`;

    await page.goto("/project-assistant");
    await expect(
      page.getByRole("heading", { name: "Project Assistant" }),
    ).toBeVisible();
    await runAxe(page);

    await page.getByRole("link", { name: "New project" }).click();
    await expect(
      page.getByRole("heading", { name: "New project" }),
    ).toBeVisible();
    await runAxe(page);

    await page.getByLabel("Course").selectOption("401/501");
    await page.getByLabel("Project name").fill(projectName);
    await page.getByLabel("Organization (optional)").fill("Test Org");
    await page.getByRole("button", { name: "Create project" }).click();

    // Lands on the workspace for the new project.
    await expect(
      page.getByRole("heading", { name: projectName }),
    ).toBeVisible();
    await expect(page.getByText("Test Org")).toBeVisible();
    await expect(page.getByText("(lead)")).toBeVisible();
    await runAxe(page);

    // The project now appears under My projects.
    await page.getByRole("link", { name: "Back to my projects" }).click();
    await expect(
      page.getByRole("link", { name: new RegExp(projectName) }),
    ).toBeVisible();
  });
});
