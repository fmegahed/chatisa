import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { makeTextPdf } from "../helpers/make-pdf";

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

/**
 * An in-test GitHub API: the push engine runs in the browser against
 * api.github.com, so the e2e suite intercepts that origin and answers like
 * GitHub would. No test traffic ever reaches the real GitHub.
 */
async function fakeGithubApi(page: Page) {
  const repos = new Set<string>();
  await page.route("https://api.github.com/**", async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    const method = req.method();
    const reply = (status: number, body: unknown) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (method === "POST" && path === "/user/repos") {
      const name = (JSON.parse(req.postData() ?? "{}") as { name: string }).name;
      repos.add(name);
      return reply(201, { default_branch: "main" });
    }
    const repoMatch = /^\/repos\/mockstudent\/([^/]+)(\/.*)?$/.exec(path);
    if (repoMatch) {
      const [, name, rest] = repoMatch;
      if (!rest) {
        return repos.has(name)
          ? reply(200, {
              html_url: `https://github.com/mockstudent/${name}`,
              default_branch: "main",
            })
          : reply(404, {});
      }
      if (rest.startsWith("/git/ref/")) return reply(200, { object: { sha: "p" } });
      if (rest.startsWith("/git/commits/")) return reply(200, { tree: { sha: "b" } });
      if (rest === "/git/trees") return reply(201, { sha: "t" });
      if (rest === "/git/commits") return reply(201, { sha: "c" });
      if (rest.startsWith("/git/refs/")) return reply(200, {});
      if (rest === "/pages") return reply(201, {});
    }
    return reply(500, { unexpected: path });
  });
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

  test("polishing real coursework files yields a plan and an artifact", async ({
    page,
  }) => {
    await setUpProfile(page);
    await page.getByRole("tab", { name: "My Projects" }).click();
    // Polish is the default mode (user decision, 2026-07-29).
    await page.locator('input[type="file"]').setInputFiles([
      {
        name: "forecast.R",
        mimeType: "text/plain",
        buffer: Buffer.from("library(fpp3)\n# ARIMA model for weekly sales\n"),
      },
      {
        name: "sales.csv",
        mimeType: "text/csv",
        buffer: Buffer.from("week,units\n1,10\n2,12\n"),
      },
      {
        // Space in the name on purpose: the guard hyphenates repo paths
        // instead of failing the request (v6.1.1).
        name: "Final Project.ipynb",
        mimeType: "application/octet-stream",
        buffer: Buffer.from(
          JSON.stringify({
            nbformat: 4,
            nbformat_minor: 5,
            metadata: { kernelspec: { language: "python" } },
            cells: [
              {
                cell_type: "code",
                metadata: {},
                execution_count: 1,
                source: "df.groupby('week').sum()",
                outputs: [],
              },
            ],
          }),
        ),
      },
    ]);
    await page
      .getByLabel("One line about the project (optional)")
      .fill("ISA 444 forecasting project");
    await page.getByRole("button", { name: "Organize my project" }).click();

    // The name renders twice on success (result pane + the artifact card),
    // which is the feature working; assert both explicitly.
    await expect(
      page.getByRole("heading", { name: "course-project-polished" }).first(),
    ).toBeVisible({ timeout: 30_000 });
    await expect(
      page.getByRole("heading", { name: "course-project-polished" }),
    ).toHaveCount(2);
    // Their file is placed, the data file is excluded with a reason, and
    // the code is never rewritten (suggestions only).
    await expect(page.getByText("forecast.R", { exact: false }).first()).toBeVisible();
    // The notebook is placed under notebooks/ with the space hyphenated.
    await expect(
      page.getByText("notebooks/Final-Project.ipynb", { exact: false }).first(),
    ).toBeVisible();
    // Role-scoped: the README preview repeats these phrases as markdown.
    await expect(
      page.getByRole("heading", { name: "Left out on purpose" }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Suggested improvements" }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", { name: /Download course-project-polished/ }),
    ).toBeVisible();
    // It is recorded as an artifact immediately (real work counts now).
    await expect(
      page.getByText("Its skills now count in your profile", { exact: false }),
    ).toBeVisible();
  });

  test("projects become artifacts, and a repo link marks them built", async ({
    page,
  }) => {
    await setUpProfile(page);
    await page.getByRole("tab", { name: "My Projects" }).click();
    await page.getByRole("radio", { name: "Start something new" }).check();

    await page.locator("select").last().selectOption({ label: "SQL" });
    await page.getByRole("button", { name: "Generate project scaffold" }).click();
    await expect(
      page.getByRole("heading", { name: "retail-demand-analytics" }).first(),
    ).toBeVisible({ timeout: 30_000 });
    // The CLI instructions moved behind a disclosure in v6.3.0; they must
    // still be there for students who prefer that path.
    await page.getByText("Prefer the command line?").click();
    await expect(page.getByText("gh repo create")).toBeVisible();

    // The artifact card persists in "Your projects" with its actions.
    await expect(
      page.getByRole("button", { name: "Download the zip again" }),
    ).toBeVisible();
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
    await page.getByRole("radio", { name: "Start something new" }).check();
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
    // The CLI path survives for students who prefer it.
    await expect(page.getByText("Prefer the command line?")).toBeVisible();
  });

  test("an unverifiable OAuth callback is rejected with plain language", async ({
    page,
  }) => {
    // No state cookie exists, so this forged callback must not connect.
    await page.goto("/api/scout/github/callback?code=x&state=forged");
    await expect(page).toHaveURL(/\/job-scout\/github-connected/);
    // Role-scoped and filtered: the test-mode banner is also an alert.
    await expect(
      page.getByRole("alert").filter({ hasText: "could not be verified" }),
    ).toBeVisible();
    const connected = await page.evaluate(() => localStorage.getItem("js-github-v1"));
    expect(connected).toBeNull();
  });

  test("a tailored portfolio site generates from saved jobs and publishes", async ({
    page,
  }) => {
    await setUpProfile(page);
    await fakeGithubApi(page);
    await page.getByRole("button", { name: "Save", exact: true }).first().click();
    await page.getByRole("tab", { name: "Portfolio Site" }).click();
    await expect(
      page.getByRole("heading", { name: "Build your portfolio site" }),
    ).toBeVisible();

    await page.getByRole("checkbox").first().check();
    await page.getByRole("button", { name: "Generate my site" }).click();
    await expect(page.getByRole("heading", { name: "Preview" })).toBeVisible({
      timeout: 30_000,
    });
    // The tailoring evidence: private notes name the picked job.
    await expect(
      page.getByRole("heading", { name: "How this speaks to the jobs you picked" }),
    ).toBeVisible();

    const popupPromise = page.waitForEvent("popup");
    await page.getByRole("button", { name: "Connect GitHub" }).click();
    await popupPromise;
    await expect(page.getByText("Connected to GitHub as").first()).toBeVisible({
      timeout: 15_000,
    });
    await page.getByRole("button", { name: "Publish to GitHub Pages" }).click();
    await expect(
      page.getByText(/Your site is live at https:\/\/mockstudent\.github\.io\/portfolio/),
    ).toBeVisible({ timeout: 15_000 });

    const portfolioScan = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa"])
      .analyze();
    expect(portfolioScan.violations).toEqual([]);
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
    const portfolio = await request.post("/api/scout/portfolio");
    expect(portfolio.status()).toBe(401);
  });
});
