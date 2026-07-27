// tests/e2e/project-team.spec.ts
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("lead invites and removes a teammate and changes coaches", async ({ page }, testInfo) => {
  const name = `Team ${testInfo.project.name} ${Date.now()}`;
  const mate = `teammate.${Date.now()}@miamioh.edu`;

  await page.goto("/project-assistant/new");
  await page.getByLabel("Course").selectOption("496");
  await page.getByLabel("Project name").fill(name);
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible();

  // Add a teammate.
  await page.getByLabel("Add a teammate by email").fill(mate);
  await page.getByRole("button", { name: "Add teammate" }).click();
  await expect(page.getByText(mate)).toBeVisible();

  // Axe on the lead workspace with the controls present.
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(results.violations).toEqual([]);

  // Enable a coach that was not chosen at creation (Premortem), then confirm the link.
  await page.getByRole("checkbox", { name: /Premortem/ }).check();
  await page.getByRole("button", { name: "Save coaches" }).click();
  await expect(page.getByRole("link", { name: /Premortem/ })).toBeVisible();

  // Remove the teammate.
  await page.getByRole("button", { name: `Remove ${mate}` }).click();
  await expect(page.getByText(mate)).toHaveCount(0);
});

test("owner deletes a project from My Projects", async ({ page }, testInfo) => {
  const name = `Delete ${testInfo.project.name} ${Date.now()}`;

  await page.goto("/project-assistant/new");
  await page.getByLabel("Course").selectOption("496");
  await page.getByLabel("Project name").fill(name);
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible();

  await page.goto("/project-assistant");
  await expect(page.getByRole("link", { name: new RegExp(name) })).toBeVisible();

  // Trash button, then confirm.
  await page.getByRole("button", { name: `Delete ${name}` }).click();
  await page.getByRole("button", { name: "Delete", exact: true }).click();
  await expect(page.getByRole("link", { name: new RegExp(name) })).toHaveCount(0);
});
