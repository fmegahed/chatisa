import { test, expect } from "@playwright/test";

/**
 * Guest magic-pass flow (2026-07-24): collaborators outside Miami open an
 * invite link, click once, and land signed in as a stable guest identity.
 * The webServer env pins one test pass; see playwright.config.ts.
 */
const GOOD_PASS = "e2e-guest-pass-1234567890abcdef";

// These tests manage their own (guest) session, so the shared student
// storage state must not leak in.
test.use({ storageState: { cookies: [], origins: [] } });

test.describe("guest magic passes", () => {
  test("a valid link signs the visitor in as a numbered guest", async ({
    page,
  }) => {
    await page.goto(`/guest?pass=${GOOD_PASS}`);
    await expect(
      page.getByRole("heading", { name: /invited to try ChatISA/ }),
    ).toBeVisible();
    await page
      .getByRole("button", { name: "Enter ChatISA as a guest" })
      .click();

    // Landed on the app, session attributed to the positional guest identity.
    await expect(
      page.getByRole("heading", { level: 1, name: /AI tools for your coursework/ }),
    ).toBeVisible();
    await expect(page.getByText("guest-1@guest.chatisa")).toBeVisible();

    // The guest can actually use a module (chat streams through the mock).
    await page.goto("/coding-tutor");
    await page.getByLabel("Your message").fill("Show me some SQL");
    await page.getByRole("button", { name: "Send message" }).click();
    await expect(page.getByRole("article", { name: "ChatISA" })).toContainText(
      "SELECT 1 AS n",
      { timeout: 15_000 },
    );
  });

  test("a wrong or truncated pass is refused with a readable message", async ({
    page,
  }) => {
    await page.goto("/guest?pass=this-is-not-a-valid-pass-at-all");
    await page
      .getByRole("button", { name: "Enter ChatISA as a guest" })
      .click();
    await expect(
      page.getByText(/This invite link didn't work/),
    ).toBeVisible();
    // Still signed out: the app redirects to login.
    await page.goto("/coding-tutor");
    await expect(page).toHaveURL(/login/);
  });

  test("a bare /guest visit explains it needs the full link", async ({
    page,
  }) => {
    await page.goto("/guest");
    await expect(
      page.getByText(/needs the full invite link/),
    ).toBeVisible();
  });

  test("knowing module paths does not bypass authentication", async ({
    page,
    request,
  }) => {
    // Direct navigation to every module: redirected to login, no content.
    for (const path of [
      "/ask-anything",
      "/coding-studio",
      "/exam-prep",
      "/project-assistant",
    ]) {
      await page.goto(path);
      await expect(page, path).toHaveURL(/login/);
    }
    // Direct API calls: JSON 401, never a redirect that could leak content.
    const post = await request.post("/api/ask-anything", {
      data: { modelId: "claude-sonnet-5", messages: [] },
    });
    expect(post.status()).toBe(401);
    const files = await request.get(
      "/api/ask-anything/files/anthropic/file_mockdeck1",
    );
    expect(files.status()).toBe(401);
    // The /guest exclusion is the exact segment, not a prefix: an adjacent
    // path stays behind the wall.
    const probe = await page.goto("/guestbook");
    await expect(page).toHaveURL(/login/);
    expect(probe?.status()).toBeLessThan(500);
  });
});
