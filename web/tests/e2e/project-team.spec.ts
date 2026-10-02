// tests/e2e/project-team.spec.ts
import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { auditDom } from "./support/a11y";

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
  await expect(page.getByText(mate, { exact: true })).toBeVisible();

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
  await expect(page.getByText(mate, { exact: true })).toHaveCount(0);
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

  // Trash button, then confirm. The list grows with every e2e run (the data
  // dir persists), and a click before hydration is a no-op, so retry it
  // until the confirm pair shows.
  const confirm = page.getByRole("button", { name: "Delete", exact: true });
  await expect(async () => {
    await page.getByRole("button", { name: `Delete ${name}` }).click();
    await expect(confirm).toBeVisible({ timeout: 2_000 });
  }).toPass({ timeout: 30_000 });
  // Focus follows the swap to the confirm button.
  await expect(confirm).toBeFocused();
  await confirm.click();
  await expect(page.getByRole("link", { name: new RegExp(name) })).toHaveCount(0);
});

async function newProject(page: import("@playwright/test").Page, name: string) {
  await page.goto("/project-assistant/new");
  await page.getByLabel("Course").selectOption("496");
  await page.getByLabel("Project name").fill(name);
  await page.getByRole("button", { name: "Create project" }).click();
  await expect(page.getByRole("heading", { name })).toBeVisible();
}

test.describe("project page accessibility (2026-09 audit)", () => {
  test("the new project form marks its required fields (#25)", async ({ page }) => {
    await page.goto("/project-assistant/new");
    await expect(page.getByText(/Fields marked \* are required/)).toBeVisible();
    await expect(page.getByLabel("Project name")).toHaveAttribute("required", "");
    await expect(page.getByLabel("Course")).toHaveAttribute("required", "");
    const dom = await auditDom(page);
    expect(dom.unmarkedRequired).toEqual([]);
  });

  test("the project page is titled with the project and course (#28)", async ({ page }, testInfo) => {
    const name = `Titled ${testInfo.project.name} ${Date.now()}`;
    await newProject(page, name);
    await expect(page).toHaveTitle(`${name} (ISA 496) · ChatISA`);
    await page.getByRole("link", { name: /Project Scoping/ }).click();
    await expect(page.getByRole("heading", { name: "Project Scoping Coach" })).toBeVisible({
      timeout: 20_000,
    });
    await expect(page).toHaveTitle(`Project Scoping Coach: ${name} (ISA 496) · ChatISA`);
  });

  test("the teammate field states its rule, explains errors inline, and announces the result (#26, #27)", async ({
    page,
  }, testInfo) => {
    await newProject(page, `Invite ${testInfo.project.name} ${Date.now()}`);
    const field = page.getByLabel("Add a teammate by email");
    // The rule is stated before anything is submitted.
    await expect(field).toHaveAccessibleDescription(/miamioh\.edu address they sign in with/);

    const add = page.getByRole("button", { name: "Add teammate" });
    await field.fill("someone@gmail.com");
    await add.click();
    await expect(field).toHaveAttribute("aria-invalid", "true");
    await expect(field).toHaveAccessibleDescription(/Other email addresses cannot be added/);

    const mate = `invitee.${Date.now()}@miamioh.edu`;
    await field.fill(mate);
    await expect(field).not.toHaveAttribute("aria-invalid", "true");
    await add.click();
    await expect(page.getByTestId("announcer-polite")).toHaveText(`Added ${mate} to the team.`);
    // The button never became unavailable, so focus stayed on it.
    await expect(add).toBeFocused();
    await expect(add).toBeEnabled();
    await expect(page.getByText(mate, { exact: true })).toBeVisible();
  });

  test("the teammate explanation comes right after its button, before the field (#29)", async ({
    page,
  }, testInfo) => {
    await newProject(page, `Disclosure ${testInfo.project.name} ${Date.now()}`);
    const toggle = page.getByRole("button", { name: "What does adding a teammate do?" });
    await expect(toggle).toHaveAttribute("aria-expanded", "false");
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-expanded", "true");
    const help = page.locator("#add-teammate-help");
    await expect(help).toBeVisible();
    await expect(toggle).toHaveAttribute("aria-controls", "add-teammate-help");
    // Document order: button, explanation, then the email field.
    const order = await page.evaluate(() => {
      const ids = ["add-teammate-help", "invite-email"];
      const [help, input] = ids.map((id) => document.getElementById(id)!);
      const button = document.querySelector('[aria-controls="add-teammate-help"]')!;
      const follows = (a: Node, b: Node) =>
        Boolean(a.compareDocumentPosition(b) & Node.DOCUMENT_POSITION_FOLLOWING);
      return { helpAfterButton: follows(button, help), fieldAfterHelp: follows(help, input) };
    });
    expect(order).toEqual({ helpAfterButton: true, fieldAfterHelp: true });
    // Tab from the button reaches the field next; the explanation is read in between.
    await toggle.focus();
    await page.keyboard.press("Tab");
    await expect(page.getByLabel("Add a teammate by email")).toBeFocused();
  });

  test("the lead's project page passes the DOM audit", async ({ page }, testInfo) => {
    await newProject(page, `Audit ${testInfo.project.name} ${Date.now()}`);
    await page.getByRole("button", { name: "What does adding a teammate do?" }).click();
    const dom = await auditDom(page);
    expect(dom.nonInteractiveTabStops).toEqual([]);
    expect(dom.hiddenTabStops).toEqual([]);
    expect(dom.roleNameOnGeneric).toEqual([]);
    expect(dom.unmarkedRequired).toEqual([]);
  });

  test("coaching worksheets have section headings like the Word download (#30)", async ({
    page,
  }, testInfo) => {
    const name = `Headings ${testInfo.project.name} ${Date.now()}`;
    await page.goto("/project-assistant/new");
    await page.getByLabel("Course").selectOption("496");
    await page.getByLabel("Project name").fill(name);
    await page.getByRole("checkbox", { name: /Reflection/ }).check();
    await page.getByRole("button", { name: "Create project" }).click();
    await expect(page.getByRole("heading", { name })).toBeVisible();

    await page.getByRole("link", { name: /Project Scoping/ }).click();
    await expect(page.getByRole("heading", { level: 2, name: "Scoping worksheet" })).toBeVisible({
      timeout: 20_000,
    });
    for (const section of ["Project", "Problem", "Ethics", "Experiment"]) {
      await expect(page.getByRole("heading", { level: 3, name: section, exact: true })).toBeVisible();
    }
    // The fieldsets keep their group names.
    await expect(page.getByRole("group", { name: "Problem", exact: true })).toBeVisible();

    await page.getByRole("link", { name: "Back to project" }).click();
    await page.getByRole("link", { name: /Reflection/ }).click();
    await expect(page.getByRole("heading", { level: 2, name: "Reflection worksheet" })).toBeVisible({
      timeout: 20_000,
    });
    await expect(page.getByRole("heading", { level: 3, name: "Details", exact: true })).toBeVisible();

    // No skipped levels anywhere on the page.
    const levels = await page.evaluate(() =>
      Array.from(document.querySelectorAll("main h1, main h2, main h3, main h4, main h5, main h6"))
        .filter((h) => !h.closest("[hidden]"))
        .map((h) => Number(h.tagName[1])),
    );
    for (let i = 1; i < levels.length; i++) {
      expect(levels[i] - levels[i - 1]).toBeLessThanOrEqual(1);
    }
  });
});
