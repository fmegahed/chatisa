import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

/**
 * The major-first course checklist (v6.7.0) in Job Scout's profile: a
 * program choice, Done / Taking now / Not yet per row, prerequisites that
 * arrive labelled and leave for good when removed, the required-core
 * shortcut, and the next-term prompt. Set SHOT_DIR to keep screenshots.
 */

function courseRow(page: Page, code: string) {
  return page.getByRole("group", { name: new RegExp(`^${code} `) }).first();
}

async function shot(page: Page, name: string) {
  if (process.env.SHOT_DIR) await page.screenshot({ path: `${process.env.SHOT_DIR}/${name}.png`, fullPage: true });
}

test.describe("Course checklist", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/job-scout");
    await page.evaluate(() => localStorage.clear());
    await page.reload();
    await expect(page.getByRole("heading", { name: "Your courses" })).toBeVisible();
  });

  test("Taking now brings prerequisites along the major's path; a removed one never comes back", async ({ page }) => {
    await page.getByRole("checkbox", { name: "Business Analytics" }).check();
    await expect(page.getByRole("button", { name: /Business Analytics/ })).toHaveAttribute("aria-expanded", "true");

    const isa401 = courseRow(page, "ISA 401");
    await isa401.getByRole("radio", { name: "Taking now" }).check();
    // ISA 345 is the major's own option (over ISA 245 and CSE 385), then its
    // chain: ISA 235 (business core, over ISA 211), then CSE 148.
    const because = page.getByText("Added because you are taking ISA 401:").first();
    await expect(because).toBeVisible();
    await expect(courseRow(page, "ISA 345").getByRole("radio", { name: "Done" })).toBeChecked();
    await expect(courseRow(page, "ISA 235").getByRole("radio", { name: "Done" })).toBeChecked();
    await expect(courseRow(page, "ISA 345").getByText("Added automatically for ISA 401.")).toBeVisible();
    await expect(page.locator("p[aria-live=polite]", { hasText: "because you are taking ISA 401" })).toHaveCount(1);
    await shot(page, "checklist-desktop");

    await page.getByRole("button", { name: "Remove ISA 345, added because of ISA 401" }).first().click();
    // The button is gone; focus goes back to the course that caused it (review fix).
    await expect(isa401.getByRole("radio", { name: "Taking now" })).toBeFocused();
    await expect(courseRow(page, "ISA 345").getByRole("radio", { name: "Not yet" })).toBeChecked();
    // Nothing that came through ISA 345 stays either.
    await expect(courseRow(page, "ISA 235").getByRole("radio", { name: "Not yet" })).toBeChecked();

    // Changing ISA 401 to Done re-derives prerequisites; ISA 345 stays out.
    await isa401.getByRole("radio", { name: "Done" }).check();
    await expect(courseRow(page, "ISA 345").getByRole("radio", { name: "Not yet" })).toBeChecked();
  });

  test("the required-core shortcut marks the shared courses and leaves the choices", async ({ page }) => {
    await page.getByRole("button", { name: "I've finished the required core courses" }).click();
    // 18 shared courses, plus MTH 141: ISA 225 needs MTH 141 or 151, and the
    // core's own option comes first. The capstone stays the student's pick.
    await expect(page.getByRole("button", { name: /FSB business core/ })).toContainText("19 of 35 marked");
    await expect(courseRow(page, "ISA 225").getByRole("radio", { name: "Done" })).toBeChecked();
    await expect(courseRow(page, "MTH 141").getByText("Added automatically for ISA 225.")).toBeVisible();
    await expect(courseRow(page, "ISA 495").getByRole("radio", { name: "Not yet" })).toBeChecked();
    // Collapsing a section hides its rows and says so.
    await page.getByRole("button", { name: /FSB business core/ }).click();
    await expect(page.getByRole("button", { name: /FSB business core/ })).toHaveAttribute("aria-expanded", "false");
    await expect(courseRow(page, "ISA 225")).toBeHidden();
  });

  test("a Taking-now course from an earlier term prompts once for the new term", async ({ page }) => {
    await page.evaluate(() =>
      localStorage.setItem(
        "js-profile-v1",
        JSON.stringify({
          v: 2, programs: ["business-analytics"], removedPrereqs: [], extras: [], overrides: [],
          courses: [{ code: "ISA 444", status: "now", term: "Spring 2020" }],
        }),
      ),
    );
    await page.reload();
    // A returning student lands on the jobs tab; the prompt is there too.
    await expect(page.getByRole("tab", { name: "This Week's Jobs" })).toHaveAttribute("aria-selected", "true");
    const prompt = page.getByRole("region", { name: /Did you finish these\?/ });
    await expect(prompt).toBeVisible();
    await expect(prompt).toContainText("ISA 444 (Taking now since Spring 2020)");
    await shot(page, "next-term-prompt");
    await prompt.getByRole("button", { name: "Finished ISA 444" }).click();
    await expect(prompt).toBeHidden();
    await page.getByRole("tab", { name: "My Profile" }).click();
    // A section already filled in opens folded to its count; the core, empty, stays open.
    await expect(page.getByRole("button", { name: /Business Analytics/ })).toHaveAttribute("aria-expanded", "false");
    await expect(page.getByRole("button", { name: /FSB business core/ })).toHaveAttribute("aria-expanded", "true");
    await page.getByLabel("Search all FSB courses").fill("444");
    await expect(courseRow(page, "ISA 444").getByRole("radio", { name: "Done" })).toBeChecked();
  });

  test("answering the prompt on the profile tab keeps the work in progress there (deferred minor, v6.9.2)", async ({ page }) => {
    await page.evaluate(() =>
      localStorage.setItem(
        "js-profile-v1",
        JSON.stringify({
          v: 2, programs: ["business-analytics"], removedPrereqs: [], extras: [], overrides: [],
          courses: [{ code: "ISA 444", status: "now", term: "Spring 2020" }],
        }),
      ),
    );
    await page.reload();
    await page.getByRole("tab", { name: "My Profile" }).click();
    const notes = page.getByLabel("Internship, ISA 340/480/481, or independent work");
    await notes.fill("Summer analytics internship at a regional bank");
    await page.getByRole("region", { name: /Did you finish these\?/ }).getByRole("button", { name: "Finished ISA 444" }).click();
    await expect(notes).toHaveValue("Summer analytics internship at a regional bank");
    await page.getByLabel("Search all FSB courses").fill("444");
    await expect(courseRow(page, "ISA 444").getByRole("radio", { name: "Done" })).toBeChecked();
  });

  test("a row that disappears hands focus to the search box, not the page (review fix)", async ({ page }) => {
    const search = page.getByLabel("Search all FSB courses");
    await search.fill("241");
    await courseRow(page, "ISA 241").getByRole("radio", { name: "Done" }).check();
    // It now also sits under "Your other courses", above the search results.
    const other = page.getByRole("region", { name: "Your other courses" });
    await other.getByRole("radio", { name: "Not yet" }).click();
    await expect(other).toHaveCount(0);
    await expect(search).toBeFocused();
  });

  for (const width of [1280, 320]) {
    test(`meets WCAG A and AA and fits the screen at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await page.getByRole("checkbox", { name: "Business Analytics" }).check();
      await courseRow(page, "ISA 401").getByRole("radio", { name: "Taking now" }).check();
      await page.getByLabel("Search all FSB courses").fill("forecast");
      await expect(courseRow(page, "ISA 444")).toBeVisible();
      const scan = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
      expect(scan.violations).toEqual([]);
      // The page itself must not scroll sideways (the site menu scrolls
      // inside its own strip, by design). On failure, name the widest
      // elements outside that strip so the message says what to fix.
      const overflow = await page.evaluate(() => {
        const excess = document.documentElement.scrollWidth - window.innerWidth;
        if (excess <= 0) return [];
        return [...document.querySelectorAll("main *")]
          .filter((el) => el.getBoundingClientRect().right > window.innerWidth + 1)
          .slice(0, 5)
          .map((el) => `${el.tagName.toLowerCase()}.${el.className} "${(el.textContent ?? "").slice(0, 60)}"`);
      });
      expect(overflow).toEqual([]);
      await shot(page, `checklist-${width}`);
    });
  }
});
