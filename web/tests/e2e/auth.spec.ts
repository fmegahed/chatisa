import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

// These tests exercise the UNAUTHENTICATED experience.
test.use({ storageState: { cookies: [], origins: [] } });

test.describe("authentication", () => {
  test("unauthenticated visit to home redirects to login", async ({
    page,
  }) => {
    await page.goto("/");
    await expect(page).toHaveURL(/\/login/);
    await expect(
      page.getByRole("heading", { level: 1, name: "Sign in to ChatISA" }),
    ).toBeVisible();
  });

  test("unauthenticated visit to a module redirects to login", async ({
    page,
  }) => {
    await page.goto("/exam-prep");
    await expect(page).toHaveURL(/\/login/);
  });

  test("login page shows Miami logo and Google button; passes axe", async ({
    page,
  }) => {
    await page.goto("/login");
    await expect(
      page.getByRole("img", { name: "Miami University" }),
    ).toBeVisible();
    await expect(
      page.getByRole("button", {
        name: "Sign in with your Miami Google account",
      }),
    ).toBeVisible();
    await expect(page.getByText("Only @miamioh.edu accounts")).toBeVisible();
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(results.violations).toEqual([]);
  });

  test("rejects a non-Miami email with an accessible error", async ({
    page,
  }) => {
    await page.goto("/login");
    await page.getByLabel("Email address").fill("intruder@gmail.com");
    await page.getByRole("button", { name: "Sign in as test user" }).click();
    await expect(page).toHaveURL(/error=/);
    const alert = page
      .getByRole("alert")
      .filter({ hasText: "Sign-in problem" });
    await expect(alert).toBeVisible();
    await expect(alert).toContainText("@miamioh.edu");
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(results.violations).toEqual([]);
  });

  test("accepts a Miami email, shows it in the header, and signs out", async ({
    page,
  }) => {
    await page.goto("/login");
    await page.getByLabel("Email address").fill("newstudent@miamioh.edu");
    await page.getByRole("button", { name: "Sign in as test user" }).click();
    await page.waitForURL("**/");
    await expect(page.getByText("newstudent@miamioh.edu")).toBeVisible();

    await page.getByRole("button", { name: "Sign out" }).click();
    await expect(page).toHaveURL(/\/login/);

    // Session is really gone: revisiting home bounces back to login.
    await page.goto("/");
    await expect(page).toHaveURL(/\/login/);
  });

  test("health endpoint stays public", async ({ request }) => {
    const res = await request.get("/api/health");
    expect([200, 503]).toContain(res.status());
    const body = await res.json();
    expect(body.checks).toHaveProperty("authConfigured");
    expect(body.checks).toHaveProperty("db");
  });
});
