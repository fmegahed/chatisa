import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { makeTextPdf } from "../helpers/make-pdf";
import { fakeGithubApi } from "./support/fake-github";

/**
 * Job Scout end to end, against the mock-mode fixture feed (six postings
 * seeded at boot) and the mock model's canned extraction/scaffold output.
 *
 * The flow under test is the 2026-07-29 tab redesign: My Profile (popular-
 * first course chips + live skills panel) -> My Projects (artifacts with a
 * home) -> This Week's Jobs (multi-state filter) -> Saved Jobs, plus the
 * device-resume handoff into JobApp Drafter.
 */

function resumePdf(): Buffer {
  return Buffer.from(
    makeTextPdf([
      [
        "Kaitlin Jones",
        "joneskl@MiamiOH.edu | (513) 555-5555",
        "Data Analytics Intern, Acme Logistics, Summer 2025",
        "Built weekly reports in Excel and SQL for the operations team",
        "Cleaned shipment data and flagged duplicate records across three systems",
      ].join(" "),
    ]),
  );
}


/** A course row in the checklist (v6.7.0): a fieldset named by its legend. */
function courseRow(page: Page, code: string) {
  return page.getByRole("group", { name: new RegExp(`^${code} `) }).first();
}

/** Builds a profile through the real checklist: a major, two rows, one search. */
async function setUpProfile(page: Page) {
  await page.goto("/job-scout");
  await expect(
    page.getByRole("heading", { name: "Your courses" }),
  ).toBeVisible();
  await page.getByRole("checkbox", { name: "Business Analytics" }).check();
  await courseRow(page, "ISA 225").getByRole("radio", { name: "Done" }).check();
  await courseRow(page, "ISA 401").getByRole("radio", { name: "Done" }).check();
  // ISA 241 is in no program's groups: search finds it.
  await page.getByLabel("Search all FSB courses").fill("241");
  await courseRow(page, "ISA 241").getByRole("radio", { name: "Done" }).check();
  await page
    .getByRole("button", { name: "Save profile and see this week's jobs" })
    .click();
  await expect(
    page.getByRole("heading", { name: "This week's jobs" }),
  ).toBeVisible();
}

test.describe("Job Scout", () => {
  test.beforeEach(async ({ page }) => {
    // Profiles live in localStorage and the resume in IndexedDB; start clean.
    await page.goto("/job-scout");
    await page.evaluate(() => {
      localStorage.clear();
      indexedDB.deleteDatabase("js-files-v1");
    });
  });

  test("profile setup shows earned skills live, then the matched feed", async ({
    page,
  }) => {
    await page.goto("/job-scout");
    // The portfolio work moved out of Job Scout (2026-08-20): the profile
    // points at the builder instead of carrying a Portfolio Site tab.
    await expect(
      page.getByRole("link", { name: "Build your portfolio" }),
    ).toHaveAttribute("href", "/portfolio?mode=career");
    await expect(page.getByRole("tab", { name: "Portfolio Site" })).toHaveCount(0);
    // Marking a course updates the skills panel without saving anything.
    await page.getByLabel("Search all FSB courses").fill("database for");
    await courseRow(page, "ISA 241").getByRole("radio", { name: "Done" }).check();
    const skillsPanel = page.getByRole("heading", {
      name: "Skills you are building",
    });
    await expect(skillsPanel).toBeVisible();
    await expect(page.getByText("SQL", { exact: true }).first()).toBeVisible();

    await courseRow(page, "ISA 225").getByRole("radio", { name: "Done" }).check();
    await page
      .getByRole("button", { name: "Save profile and see this week's jobs" })
      .click();

    await expect(page.getByText(/postings from employer career sites and USAJobs/)).toBeVisible();
    await expect(page.getByText(/required skills covered/).first()).toBeVisible();
    await expect(
      page.getByText(/Strong match|Good match|Stretch/).first(),
    ).toBeVisible();
  });

  test("multi-state filter narrows the feed and details disclose in place", async ({
    page,
  }) => {
    await setUpProfile(page);

    // The fixture feed has few states, so all render as one-click chips
    // (the top-by-demand chips + type-ahead pattern, 2026-07-29).
    await page.getByTitle("District of Columbia").click();
    await expect(
      page.getByRole("heading", { name: "Management Analyst" }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Data Analyst", exact: true }),
    ).not.toBeVisible();
    // A second state widens it again.
    await page.getByTitle("Ohio", { exact: true }).click();
    await expect(
      page.getByRole("heading", { name: "Data Analyst", exact: true }),
    ).toBeVisible();

    const details = page.getByRole("button", { name: "Details" }).first();
    await details.click();
    await expect(details).toHaveAttribute("aria-expanded", "true");
    await expect(
      page.getByRole("link", { name: "Apply on employer site" }).first(),
    ).toHaveAttribute("href", /careers\.example\.com/);
  });

  test("saved jobs get a home that survives filters and revisits", async ({
    page,
  }) => {
    await setUpProfile(page);
    // Save names its job ("Save Data Analyst at ...", #34).
    await page.getByRole("button", { name: /^Save / }).first().click();
    await page.getByRole("tab", { name: /Saved Jobs/ }).click();
    await expect(page.getByRole("heading", { name: "Saved jobs" })).toBeVisible();
    await expect(
      page.getByRole("link", { name: "Apply on employer site" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Unsave" }).click();
    await expect(page.getByText("Nothing saved yet")).toBeVisible();
  });

  test("saving the profile moves focus to the jobs heading (#38)", async ({
    page,
  }) => {
    await setUpProfile(page);
    await expect(
      page.getByRole("heading", { name: "This week's jobs" }),
    ).toBeFocused();
    // The tabs follow the APG pattern: only the selected tab is in the tab
    // order and points at the rendered panel.
    const jobsTab = page.getByRole("tab", { name: /This Week's Jobs/ });
    await expect(jobsTab).toHaveAttribute("aria-selected", "true");
    await expect(jobsTab).toHaveAttribute("aria-controls", "panel-jobs");
    await expect(page.getByRole("tabpanel")).toHaveAttribute("id", "panel-jobs");
  });

  test("card controls name their job and Save exposes its state (#34, #35)", async ({
    page,
  }) => {
    await setUpProfile(page);
    const job = "Data Analyst at Queen City Insurance";
    await expect(
      page.getByRole("button", { name: `Details for ${job}`, exact: true }),
    ).toBeVisible();
    await expect(
      page.getByRole("link", {
        name: `Apply on employer site for ${job} (opens in a new tab)`,
        exact: true,
      }),
    ).toBeVisible();
    await expect(
      page.getByRole("link", {
        name: `Draft my resume and cover letter for ${job}`,
        exact: true,
      }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: `Hide ${job}`, exact: true }),
    ).toBeVisible();

    const save = page.getByRole("button", { name: `Save ${job}`, exact: true });
    await expect(save).toHaveAttribute("aria-pressed", "false");
    await save.click();
    await expect(save).toHaveAttribute("aria-pressed", "true");
    await expect(page.getByTestId("announcer-polite")).toHaveText(
      `Saved ${job}.`,
    );
    await save.click();
    await expect(save).toHaveAttribute("aria-pressed", "false");
    await expect(page.getByTestId("announcer-polite")).toHaveText(
      `Removed ${job} from saved jobs.`,
    );
  });

  test("hidden jobs can be undone, listed, and unhidden (#35, #39)", async ({
    page,
  }) => {
    await setUpProfile(page);
    const job = "Data Analyst at Queen City Insurance";
    const card = page.getByRole("heading", { name: "Data Analyst", exact: true });
    const announcer = page.getByTestId("announcer-polite");

    // Hide: the card leaves, focus lands on a neighbouring card's heading.
    await page.getByRole("button", { name: `Hide ${job}`, exact: true }).click();
    await expect(card).toHaveCount(0);
    await expect(announcer).toHaveText(
      `Hidden: ${job}. Use Undo or Show hidden jobs to bring it back.`,
    );
    await expect(page.locator(":focus")).toHaveAttribute("id", /^job-title-/);

    // Undo puts it straight back and focuses it.
    await page.getByRole("button", { name: `Undo hiding ${job}`, exact: true }).click();
    await expect(card).toBeFocused();
    await expect(announcer).toHaveText(`${job} is back in the list.`);

    // Hidden state survives a reload; the list brings it back later.
    await page.getByRole("button", { name: `Hide ${job}`, exact: true }).click();
    await expect(card).toHaveCount(0);
    await page.reload();
    await expect(
      page.getByRole("heading", { name: "This week's jobs" }),
    ).toBeVisible();
    await expect(card).toHaveCount(0);
    const toggle = page.getByRole("button", { name: "Show hidden jobs (1)" });
    await expect(toggle).toHaveAttribute("aria-expanded", "false");
    await toggle.click();
    const hiddenList = page.getByRole("region", { name: "Hidden jobs" });
    await expect(hiddenList.getByText("Hidden", { exact: true })).toBeVisible();
    await hiddenList
      .getByRole("button", { name: `Unhide ${job}`, exact: true })
      .click();
    await expect(card).toBeVisible();
    await expect(card).toBeFocused();
    await expect(announcer).toHaveText(`${job} is back in the list.`);
    await expect(page.getByRole("button", { name: /hidden jobs/ })).toHaveCount(0);
  });

  test("filter changes announce the result count (#36)", async ({ page }) => {
    await setUpProfile(page);
    const announcer = page.getByTestId("announcer-polite");
    const cards = page.locator('h3[id^="job-title-"]');

    await page.getByTitle("District of Columbia").click();
    await expect(announcer).toHaveText(/^\d+ jobs? match(es)? your filters\.$/);
    expect(await announcer.textContent()).toContain(`${await cards.count()} `);

    // Rapid toggles announce once, with the final count.
    await page.getByTitle("Ohio", { exact: true }).click();
    await page.getByLabel("Remote only").check();
    await page.getByLabel("Remote only").uncheck();
    await expect(announcer).toHaveText(
      `${await cards.count()} jobs match your filters.`,
    );
    // The state pills show a focus ring through their hidden checkbox.
    await page.getByRole("checkbox", { name: /^OH, Ohio/ }).focus();
    await page.keyboard.press("Shift+Tab");
    await page.keyboard.press("Tab");
    const outline = await page
      .getByTitle("Ohio", { exact: true })
      .evaluate((el) => getComputedStyle(el).outlineStyle);
    expect(outline).toBe("solid");
  });

  test("profile checkbox and skill groups are fieldsets with legends (#37)", async ({
    page,
  }) => {
    await page.goto("/job-scout");
    for (const name of ["Majors", "Co-majors", "Minors"]) {
      await expect(page.getByRole("group", { name, exact: true })).toBeVisible();
    }
    await expect(
      page
        .getByRole("group", { name: "Majors", exact: true })
        .getByRole("checkbox", { name: "Business Analytics" }),
    ).toBeVisible();
    await page.getByLabel("Search all FSB courses").fill("database for");
    await courseRow(page, "ISA 241").getByRole("radio", { name: "Done" }).check();
    await expect(
      page
        .getByRole("group", { name: "Programming", exact: true })
        .getByRole("combobox", { name: "Your level for SQL" }),
    ).toBeVisible();
  });

  test("projects become artifacts, and a repo link marks them built", async ({
    page,
  }) => {
    await setUpProfile(page);
    await page.getByRole("tab", { name: "My Projects" }).click();

    await page.locator("select").last().selectOption({ label: "SQL" });
    await page.getByRole("button", { name: "Generate project scaffold" }).click();
    await expect(
      page.getByRole("heading", { name: "retail-demand-analytics" }).first(),
    ).toBeVisible({ timeout: 30_000 });
    // Zip downloads and the CLI disclosure were removed (2026-08-20):
    // GitHub is the only destination, so neither may reappear.
    await expect(page.getByText("Prefer the command line?")).toHaveCount(0);
    await expect(page.getByRole("button", { name: /zip/i })).toHaveCount(0);

    // The artifact card persists in "Your projects" with its actions.
    await page.getByRole("button", { name: "I pushed it to GitHub" }).click();
    await page
      .getByLabel("GitHub repository URL")
      .fill("https://github.com/student/retail-demand-analytics");
    await page.getByRole("button", { name: "Save link" }).click();
    await expect(
      page.getByText("Built. Its skills count in your profile."),
    ).toBeVisible();
  });

  test("one click pushes a scaffold to GitHub and records the repo link", async ({
    page,
  }) => {
    await setUpProfile(page);
    await fakeGithubApi(page);
    await page.getByRole("tab", { name: "My Projects" }).click();
    await page.locator("select").last().selectOption({ label: "SQL" });
    await page.getByRole("button", { name: "Generate project scaffold" }).click();
    await expect(
      page.getByRole("heading", { name: "retail-demand-analytics" }).first(),
    ).toBeVisible({ timeout: 30_000 });

    // Connect through the real popup flow (mock GitHub server-side).
    const popupPromise = page.waitForEvent("popup");
    await page.getByRole("button", { name: "Connect GitHub" }).click();
    await popupPromise;
    await expect(
      page.getByText("Connected to GitHub as").first(),
    ).toBeVisible({ timeout: 15_000 });

    // The push flips the card to built with no manual link entry.
    await page.getByRole("button", { name: "Push to GitHub", exact: true }).first().click();
    await expect(
      page.getByText("Built. Its skills count in your profile."),
    ).toBeVisible({ timeout: 15_000 });
    // The CLI disclosure went with the zip download (2026-08-20).
    await expect(page.getByText("Prefer the command line?")).toHaveCount(0);
  });

  test("an unverifiable OAuth callback is rejected with plain language", async ({
    page,
  }) => {
    // No state cookie exists, so this forged callback must not connect.
    await page.goto("/api/scout/github/callback?code=x&state=forged");
    await expect(page).toHaveURL(/\/portfolio\/github-connected/);
    // Role-scoped and filtered: the test-mode banner is also an alert.
    await expect(
      page.getByRole("alert").filter({ hasText: "could not be verified" }),
    ).toBeVisible();
    const connected = await page.evaluate(() => localStorage.getItem("js-github-v1"));
    expect(connected).toBeNull();
  });

  test("the resume saved in the profile forwards into JobApp Drafter", async ({
    page,
  }) => {
    await setUpProfile(page);
    await page.getByRole("tab", { name: "My Profile" }).click();

    await page
      .locator('input[type="file"]')
      .setInputFiles({
        name: "kaitlin-resume.pdf",
        mimeType: "application/pdf",
        buffer: resumePdf(),
      });
    await page.getByRole("button", { name: "Suggest skills from it" }).click();
    await expect(
      page.getByRole("heading", { name: "Suggested skills to confirm" }),
    ).toBeVisible({ timeout: 30_000 });
    await page.getByRole("button", { name: "Add all as suggested" }).click();

    // The handoff: job prefilled AND the device resume offered.
    await page.getByRole("tab", { name: /This Week's Jobs/ }).click();
    await page
      .getByRole("link", { name: "Draft my resume and cover letter" })
      .first()
      .click();
    await expect(page).toHaveURL(/\/jobapp-drafter\?job=/);
    await expect(page.getByText(/Loaded from Job Scout/)).toBeVisible();
    await expect(page.getByLabel("Company")).not.toHaveValue("");
    await expect(
      page.getByText(/Job Scout has your resume on this device/),
    ).toBeVisible();
    await page.getByRole("button", { name: "Use it here" }).click();
    await expect(page.getByText("kaitlin-resume.pdf")).toBeVisible();
  });

  test("meets WCAG A and AA on the profile and jobs tabs", async ({ page }) => {
    await page.goto("/job-scout");
    await expect(
      page.getByRole("heading", { name: "Your courses" }),
    ).toBeVisible();
    const profileScan = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa"])
      .analyze();
    expect(profileScan.violations).toEqual([]);

    await setUpProfile(page);
    const feedScan = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa"])
      .analyze();
    expect(feedScan.violations).toEqual([]);
  });
});

test.describe("Job Scout access control", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("unauthenticated visitors are redirected and the APIs answer 401", async ({
    page,
    request,
  }) => {
    await page.goto("/job-scout");
    await expect(page).toHaveURL(/\/login/);
    for (const path of [
      "/api/scout/feed",
      "/api/scout/feed?shape=index",
      "/api/scout/postings/x",
      "/api/scout/github/start",
      "/api/scout/github/callback?code=x&state=y",
    ]) {
      const res = await request.get(path);
      expect(res.status(), path).toBe(401);
    }
    const post = await request.post("/api/scout/project", {
      data: { modelId: "gpt-6-sol", skillIds: ["sql"] },
    });
    expect(post.status()).toBe(401);
    const refresh = await request.post("/api/scout/refresh");
    expect(refresh.status()).toBe(401);
  });
});
