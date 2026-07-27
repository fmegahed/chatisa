// tests/e2e/project-assistant-session.spec.ts
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

async function runAxe(page: import("@playwright/test").Page) {
  const results = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
    .analyze();
  expect(results.violations).toEqual([]);
}

test("scope a project: chat, edit the worksheet, and persist", async ({ page }, testInfo) => {
  const name = `Scoping ${testInfo.project.name} ${Date.now()}`;

  await page.goto("/project-assistant/new");
  await page.getByLabel("Course").selectOption("401/501");
  await page.getByLabel("Project name").fill(name);
  await page.getByRole("button", { name: "Create project" }).click();

  await expect(page.getByRole("heading", { name })).toBeVisible();
  await page.getByRole("link", { name: /Project Scoping/ }).click();

  // The coach session route compiles on first hit; under full parallel load
  // that first compile can exceed the default 5s, so wait explicitly longer.
  await expect(
    page.getByRole("heading", { name: "Project Scoping Coach" }),
  ).toBeVisible({ timeout: 20000 });
  await runAxe(page);

  // Chat streams a reply from the mock model.
  await page.getByLabel("Your message").fill("We want to cut stockouts at a grocery chain.");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByText("Coach", { exact: true }).first()).toBeVisible();

  // Direct edit persists across a reload.
  const org = page.getByLabel("Organization", { exact: true });
  await org.fill("Kroger");
  // The debounced save PATCHes the deliverable route, which compiles on first
  // hit; under full parallel load allow longer than the default 5s.
  await expect(page.getByText(/Last updated by/)).toBeVisible({ timeout: 20000 });
  await page.waitForTimeout(800); // let the debounced save flush
  await page.reload();
  await expect(page.getByLabel("Organization", { exact: true })).toHaveValue("Kroger");

  // The worksheet exports to a .docx.
  const downloadPromise = page.waitForEvent("download");
  await page.getByRole("link", { name: "Download Word" }).click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/\.docx$/);
});

test("a generic coach fills, edits, persists, and exports", async ({ page }, testInfo) => {
  const name = `Reflection ${testInfo.project.name} ${Date.now()}`;

  await page.goto("/project-assistant/new");
  await page.getByLabel("Course").selectOption("496");
  await page.getByLabel("Project name").fill(name);
  // Enable Reflection (the New project form lists all five coaches as checkboxes).
  await page.getByRole("checkbox", { name: /Reflection/ }).check();
  await page.getByRole("button", { name: "Create project" }).click();

  await expect(page.getByRole("heading", { name })).toBeVisible();
  await page.getByRole("link", { name: /Reflection/ }).click();

  await expect(
    page.getByRole("heading", { name: "Reflection Coach" }),
  ).toBeVisible({ timeout: 20000 });

  await page.getByLabel("Your message").fill("We struggled with scheduling but shipped on time.");
  await page.getByRole("button", { name: "Send message" }).click();
  await expect(page.getByText("Coach", { exact: true }).first()).toBeVisible();

  const challenges = page.getByLabel("Challenges", { exact: true });
  await challenges.fill("Scheduling across time zones");
  await expect(page.getByText(/Last updated by/)).toBeVisible({ timeout: 20000 });
  await page.waitForTimeout(800);
  await page.reload();
  await expect(page.getByLabel("Challenges", { exact: true })).toHaveValue("Scheduling across time zones");

  const downloadPromise = page.waitForEvent("download", { timeout: 20000 });
  await page.getByRole("link", { name: "Download Word" }).click();
  expect((await downloadPromise).suggestedFilename()).toMatch(/\.docx$/);
});

test("project export downloads a combined .docx", async ({ page }, testInfo) => {
  const name = `Export ${testInfo.project.name} ${Date.now()}`;
  await page.goto("/project-assistant/new");
  await page.getByLabel("Course").selectOption("496");
  await page.getByLabel("Project name").fill(name);
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible();

  // Start the scoping deliverable so there is something to export.
  await page.getByRole("link", { name: /Project Scoping/ }).click();
  await expect(page.getByRole("heading", { name: "Project Scoping Coach" })).toBeVisible();
  await page.getByLabel("Organization", { exact: true }).fill("Kroger");
  await expect(page.getByText(/Last updated by/)).toBeVisible({ timeout: 20000 });
  await page.waitForTimeout(800);

  await page.getByRole("link", { name: "Back to project" }).click();
  const downloadPromise = page.waitForEvent("download", { timeout: 20000 });
  await page.getByRole("link", { name: "Download all deliverables" }).click();
  expect((await downloadPromise).suggestedFilename()).toMatch(/\.docx$/);
});

test("every enabled coach opens", async ({ page }, testInfo) => {
  const name = `All coaches ${testInfo.project.name} ${Date.now()}`;
  await page.goto("/project-assistant/new");
  await page.getByLabel("Course").selectOption("496");
  await page.getByLabel("Project name").fill(name);
  for (const c of ["Premortem", "Team Structuring", "Devil's Advocate", "Reflection"]) {
    await page.getByRole("checkbox", { name: new RegExp(c) }).check();
  }
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible();

  for (const [label, heading] of [
    ["Project Scoping", "Project Scoping Coach"],
    ["Premortem", "Premortem Coach"],
    ["Team Structuring", "Team Structuring Coach"],
    ["Devil's Advocate", "Devil's Advocate Coach"],
    ["Reflection", "Reflection Coach"],
  ] as const) {
    await page.getByRole("link", { name: new RegExp(label) }).click();
    await expect(page.getByRole("heading", { name: heading })).toBeVisible();
    await page.getByRole("link", { name: "Back to project" }).click();
    await expect(page.getByRole("heading", { name })).toBeVisible();
  }
});
