import { test, expect } from "@playwright/test";
import { auditDom, axeViolations } from "./support/a11y";

/**
 * Site-wide accessibility sweep (2026-09 audit, tracking issue #41).
 * Every module's landing state must be axe-clean at WCAG 2.1 AA and free of
 * the finding classes axe does not check (see support/a11y.ts). Interaction
 * states (a sent message, a run, a saved profile) are covered in each
 * module's own spec.
 */
const ROUTES = [
  "/",
  "/ask-anything",
  "/coding-tutor",
  "/coding-studio",
  "/exam-prep",
  "/interview-mentor",
  "/jobapp-drafter",
  "/job-scout",
  "/portfolio",
  "/project-assistant",
  "/project-assistant/new",
  "/ai-comparison",
];

for (const route of ROUTES) {
  test(`${route} passes the accessibility sweep`, async ({ page }) => {
    await page.goto(route);
    await expect(page.locator("main h1").first()).toBeVisible();
    // Let client panes (editors, runtime placeholders) mount. Coding Studio
    // keeps fetching runtime files, so the network never goes idle there.
    await page.waitForLoadState("networkidle", { timeout: 10_000 }).catch(() => {});
    expect.soft(await axeViolations(page)).toEqual([]);
    const dom = await auditDom(page);
    expect.soft(dom.nonInteractiveTabStops, "non-interactive tab stops").toEqual([]);
    expect.soft(dom.hiddenTabStops, "invisible tab stops").toEqual([]);
    expect.soft(dom.roleNameOnGeneric, "aria-label on a generic element").toEqual([]);
    expect.soft(dom.placeholderContrast, "placeholder contrast").toEqual([]);
    expect.soft(dom.unmarkedRequired, "required without a visible marker").toEqual([]);
  });
}

test("every page has a distinct, descriptive title", async ({ page }) => {
  const titles = new Map<string, string>();
  for (const route of ROUTES) {
    await page.goto(route);
    titles.set(route, await page.title());
  }
  const home = titles.get("/");
  for (const [route, title] of titles) {
    if (route === "/") continue;
    expect.soft(title, route).not.toBe(home);
    expect.soft(title, route).toMatch(/ · ChatISA$/);
  }
  await page.goto("/no-such-page");
  await expect(page).toHaveTitle("Page not found · ChatISA");
});
