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

/** Builds a profile through the real UI: two popular chips + one disclosed. */
async function setUpProfile(page: Page) {
  await page.goto("/job-scout");
  await expect(
    page.getByRole("heading", { name: "Your ISA courses" }),
  ).toBeVisible();
  // Popular chips are visible immediately.
  await page.getByTitle("Principles of Business Analytics").click();
  await page.getByTitle("Business Intelligence and Data Visualization").click();
  // ISA 241 sits behind the Foundations disclosure.
  await page.getByRole("button", { name: /Show 4 more/ }).click();
  await page.getByTitle("Database for Analytics").click();
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
    // Checking a course updates the skills panel without saving anything.
    await page.getByRole("button", { name: /Show 4 more/ }).click();
    await page.getByTitle("Database for Analytics").click();
    const skillsPanel = page.getByRole("heading", {
      name: "Skills you are building",
    });
    await expect(skillsPanel).toBeVisible();
    await expect(page.getByText("SQL", { exact: true }).first()).toBeVisible();

    await page.getByTitle("Principles of Business Analytics").click();
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
    await page.getByRole("button", { name: "Save", exact: true }).first().click();
    await page.getByRole("tab", { name: /Saved Jobs/ }).click();
    await expect(page.getByRole("heading", { name: "Saved jobs" })).toBeVisible();
    await expect(
      page.getByRole("link", { name: "Apply on employer site" }),
    ).toBeVisible();
    await page.getByRole("button", { name: "Unsave" }).click();
    await expect(page.getByText("Nothing saved yet")).toBeVisible();
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
      page.getByRole("heading", { name: "Your ISA courses" }),
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
      data: { modelId: "gpt-5.6-terra", skillIds: ["sql"] },
    });
    expect(post.status()).toBe(401);
    const refresh = await request.post("/api/scout/refresh");
    expect(refresh.status()).toBe(401);
  });
});
