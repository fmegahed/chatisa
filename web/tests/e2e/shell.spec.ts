import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test.describe("app shell", () => {
  test("home page renders with correct title and landmarks", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page).toHaveTitle(/ChatISA/);
    await expect(page.getByRole("banner")).toBeVisible();
    await expect(page.getByRole("main")).toBeVisible();
    await expect(page.getByRole("contentinfo")).toBeVisible();
    await expect(
      page.getByRole("navigation", { name: "ChatISA modules" }),
    ).toBeVisible();
    await expect(
      page.getByRole("heading", { level: 1, name: /AI tools for your coursework/ }),
    ).toBeVisible();
  });

  test("skip link is first tab stop and moves focus to main", async ({
    page,
  }) => {
    await page.goto("/");
    await page.keyboard.press("Tab");
    const skip = page.getByRole("link", { name: "Skip to main content" });
    await expect(skip).toBeFocused();
    await page.keyboard.press("Enter");
    await expect(page.locator("#main")).toBeFocused();
  });

  test("deep health exercises the PDF worker, database, brand assets, and speech", async ({
    request,
  }) => {
    // The checks that only ever broke in production (2026-07-25): a real
    // child-process PDF parse, a db write with read-back, brand assets on
    // disk. The deploy pipeline gates bundles on this same endpoint.
    const res = await request.get("/api/health?deep=1");
    expect(res.status()).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("ok");
    expect(body.checks.deep.pdfWorker).toBe("ok");
    expect(body.checks.deep.dbRoundtrip).toBe("ok");
    expect(body.checks.deep.brandAssets).toBe("ok");

    // Speech joined the deep block on 2026-07-26 (Interview Mentor's voice was
    // undiagnosable remotely). Asserted per key rather than by deep equality on
    // the whole object, so adding the NEXT check does not break this test again:
    // the exact-shape assertion is what failed here, not the endpoint.
    //
    // The e2e server has no DEEPGRAM_TOKEN, so "not-configured" is the correct
    // answer AND must not make the server unhealthy, which is the whole point of
    // the asymmetry. A configured-but-refused credential returns 503 instead;
    // that path is covered in tests/unit/speech-probe.test.ts.
    expect(body.checks.deep.speech).toMatch(/^(ok|not-configured)/);
    expect(Object.keys(body.checks.deep).sort()).toEqual([
      "brandAssets",
      "dbRoundtrip",
      "pdfWorker",
      "speech",
    ]);
  });

  test("old module URLs redirect to the renamed slugs", async ({ page }) => {
    // Slugs were renamed 2026-07-24 to match display names; bookmarks and
    // shared links to the old paths must still land on the module.
    const renames: [string, RegExp][] = [
      ["/ai-sandbox", /\/coding-studio$/],
      ["/general-chat", /\/ask-anything$/],
      ["/project-coach", /\/project-assistant$/],
    ];
    for (const [oldPath, newUrl] of renames) {
      await page.goto(oldPath);
      await expect(page).toHaveURL(newUrl);
    }
  });

  test("module cards navigate and mark current nav item", async ({ page }) => {
    await page.goto("/");
    await page
      .getByRole("link", { name: /Open Coding Tutor/ })
      .click();
    await expect(page).toHaveURL(/coding-tutor/);
    await expect(
      page.getByRole("heading", { level: 1, name: "Coding Tutor" }),
    ).toBeVisible();
    const current = page.locator('nav a[aria-current="page"]');
    await expect(current).toHaveText("Coding Tutor");
  });

  test("unknown module slug returns not-found page", async ({ page }) => {
    const res = await page.goto("/does-not-exist");
    expect(res?.status()).toBe(404);
    await expect(
      page.getByRole("heading", { name: /nothing at this address/i }),
    ).toBeVisible();
  });

  test("health endpoint reports readiness without secret values", async ({
    request,
  }) => {
    const res = await request.get("/api/health");
    expect([200, 503]).toContain(res.status());
    const body = await res.json();
    expect(["ok", "degraded"]).toContain(body.status);
    expect(body.checks).toHaveProperty("missingProviderKeys");
  });
});

test.describe("accessibility (axe)", () => {
  for (const path of ["/", "/coding-tutor", "/interview-mentor"]) {
    test(`no WCAG A/AA violations on ${path}`, async ({ page }) => {
      await page.goto(path);
      const results = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
        .analyze();
      expect(results.violations).toEqual([]);
    });
  }
});
