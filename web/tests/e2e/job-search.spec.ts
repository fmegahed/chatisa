import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { makeTextPdf } from "../helpers/make-pdf";

/**
 * Tailoring an application end to end.
 *
 * The assertion that matters most is that a fabricated line is shown to the
 * student with a warning and is still present: flagged, not silently deleted
 * (user decision, 2026-07-21). The mock deliberately generates one such line.
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
        "Treasurer, Business Analytics Club, 2024 to 2025",
        "Managed a budget of $4,000 across eight events",
      ].join(" "),
    ]),
  );
}

async function setUpApplication(page: Page) {
  await page.goto("/jobapp-drafter");
  await page.getByLabel("Company").fill("Northwind Analytics");
  await page.getByLabel("Position title").fill("Analytics Intern");
  await page
    .getByLabel("Or paste the job description")
    .fill("We need an analyst comfortable with SQL, reporting and clean data.");
  await page.locator("#resume-file").setInputFiles({
    name: "resume.pdf",
    mimeType: "application/pdf",
    buffer: resumePdf(),
  });
  await page.getByRole("button", { name: "Continue" }).click();
  await expect(
    page.getByRole("heading", { name: /Analytics Intern at Northwind/ }),
  ).toBeVisible({ timeout: 60_000 });
}

test.describe("JobApp Assistant", () => {
  test("tailors a resume the student can edit and download", async ({ page }) => {
    await setUpApplication(page);
    await page.getByRole("button", { name: "Tailor my resume" }).click();

    await expect(
      page.getByRole("heading", { name: "Your tailored resume" }),
    ).toBeVisible({ timeout: 60_000 });

    // Every bullet is editable: the student has to be able to change any claim
    // they will be asked to defend.
    const bullets = page.locator("textarea[id^='bullet-']");
    expect(await bullets.count()).toBeGreaterThan(0);
    await bullets.first().fill("Automated weekly operations reporting in SQL");

    await page.getByRole("button", { name: "Save my edits" }).click();
    await expect(page.getByRole("link", { name: "Download as Word" })).toBeVisible();
  });

  test("published work rides along only when the student opts in", async ({
    page,
  }) => {
    // The Portfolio Builder keeps published sites in this browser only, so
    // seeding localStorage is exactly what a real publish leaves behind.
    await page.goto("/jobapp-drafter");
    await page.evaluate(() => {
      localStorage.setItem(
        "pb-published-v1",
        JSON.stringify([
          {
            id: "s1",
            kind: "showcase",
            title: "Churn Showcase",
            summary: "A churn model with a write-up.",
            skillIds: ["sql"],
            repoUrl: "https://github.com/mockstudent/churn",
            pagesUrl: "https://mockstudent.github.io/churn/",
            publishedAt: "2026-08-20T00:00:00.000Z",
          },
        ]),
      );
    });

    let body = "";
    await page.route("**/api/applications", async (route) => {
      if (route.request().method() === "POST") {
        body = route.request().postData() ?? "";
      }
      await route.continue();
    });

    await page.goto("/jobapp-drafter");
    const toggle = page.getByRole("checkbox", {
      name: /Include my published work \(1\)/,
    });
    // Default off: nothing leaves the browser unless the student says so.
    await expect(toggle).not.toBeChecked();
    await toggle.check();

    await page.getByLabel("Company").fill("Northwind Analytics");
    await page.getByLabel("Position title").fill("Analytics Intern");
    await page
      .getByLabel("Or paste the job description")
      .fill("We need an analyst comfortable with SQL, reporting and clean data.");
    await page.locator("#resume-file").setInputFiles({
      name: "resume.pdf",
      mimeType: "application/pdf",
      buffer: resumePdf(),
    });
    await page.getByRole("button", { name: "Continue" }).click();
    await expect(
      page.getByRole("heading", { name: /Analytics Intern at Northwind/ }),
    ).toBeVisible({ timeout: 60_000 });

    expect(body).toContain("publishedWork");
    expect(body).toContain("https://mockstudent.github.io/churn/");
  });

  test("warns about a line it cannot trace, and keeps it", async ({ page }) => {
    // The mock generates "Directed a team of forty consultants across three
    // continents", which is nowhere in the resume. It must be flagged loudly
    // and must still be present, because the student decides, not us.
    await setUpApplication(page);
    await page.getByRole("button", { name: "Tailor my resume" }).click();
    await expect(
      page.getByRole("heading", { name: "Your tailored resume" }),
    ).toBeVisible({ timeout: 60_000 });

    await expect(
      page.getByRole("heading", { name: /Read these lines before you send/ }),
    ).toBeVisible();
    await expect(page.getByText(/could not be traced back to your resume/i))
      .toBeVisible();

    const values = await page.locator("textarea[id^='bullet-']").allInnerTexts();
    const inputs = await page
      .locator("textarea[id^='bullet-']")
      .evaluateAll((els) => els.map((e) => (e as HTMLTextAreaElement).value));
    expect([...values, ...inputs].join(" ")).toMatch(/forty consultants/i);

    // The student can acknowledge and move on rather than being blocked.
    await page.getByRole("button", { name: "I have read these" }).click();
    await expect(
      page.getByRole("heading", { name: /Read these lines before you send/ }),
    ).toBeHidden();
  });

  test("clears a warning once the student fixes the line", async ({ page }) => {
    // A warning that lingers after a fix trains students to ignore warnings.
    await setUpApplication(page);
    await page.getByRole("button", { name: "Tailor my resume" }).click();
    await expect(
      page.getByRole("heading", { name: "Your tailored resume" }),
    ).toBeVisible({ timeout: 60_000 });

    const flagged = page.locator("textarea.border-miami-red").first();
    await expect(flagged).toBeVisible();
    await flagged.fill("Cleaned shipment data and flagged duplicate records");
    await page.getByRole("button", { name: "Save my edits" }).click();

    await expect(page.getByText(/forty consultants/i)).toBeHidden({
      timeout: 30_000,
    });
  });

  test("writes a cover letter and reports its length against the standard", async ({
    page,
  }) => {
    await setUpApplication(page);
    await page.getByRole("button", { name: "Write a cover letter" }).click();
    await expect(
      page.getByRole("heading", { name: "Your cover letter" }),
    ).toBeVisible({ timeout: 60_000 });

    await expect(page.getByText(/The Farmer School example runs about 205/))
      .toBeVisible();
    await expect(page.locator("textarea[id^='paragraph-']").first()).toBeVisible();
  });

  test("refuses to generate without the student's own resume", async ({
    request,
  }) => {
    // Without a resume there is nothing to ground against, so generating would
    // mean inventing a career rather than presenting one.
    const created = await request.post("/api/applications", {
      multipart: {
        company: "Northwind",
        positionTitle: "Analytics Intern",
        jobUrl: "",
        postingText: "SQL and reporting.",
      },
    });
    expect(created.ok()).toBe(true);
    const { applicationId } = await created.json();

    const generated = await request.post(
      `/api/applications/${applicationId}/documents`,
      {
        data: {
          kind: "resume",
          modelId: "gpt-5.6-terra",
          template: 1,
          studentName: "Kaitlin Jones",
        },
      },
    );
    expect(generated.status()).toBe(400);
    expect((await generated.json()).error).toMatch(/upload your current resume/i);
  });

  test("no WCAG A/AA violations across setup and the editor", async ({ page }) => {
    await page.goto("/jobapp-drafter");
    const setup = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(setup.violations).toEqual([]);

    await setUpApplication(page);
    await page.getByRole("button", { name: "Tailor my resume" }).click();
    await expect(
      page.getByRole("heading", { name: "Your tailored resume" }),
    ).toBeVisible({ timeout: 60_000 });

    const editor = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(editor.violations).toEqual([]);
  });
});

test.describe("job search access control", () => {
  test.use({ storageState: { cookies: [], origins: [] } });

  test("requires sign-in", async ({ request }) => {
    expect((await request.get("/api/applications")).status()).toBe(401);
    expect((await request.get("/api/documents/anything")).status()).toBe(401);
    expect((await request.get("/api/documents/anything/export")).status()).toBe(401);
  });
});
