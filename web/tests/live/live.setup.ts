import { test as setup, expect } from "@playwright/test";
import { mkdirSync } from "node:fs";

const AUTH_FILE = "tests/live/.auth/user.json";

/**
 * Signs in once for the live run, into its own storage state.
 *
 * Separate from the e2e suite's .auth file on purpose: the two suites run
 * against different servers (different ports, different data dirs), and a cookie
 * minted for one is not valid for the other. Sharing the file made a live run
 * silently start unauthenticated and redirect every navigation to /login.
 */
setup("authenticate for the live run", async ({ page, baseURL }) => {
  setup.setTimeout(300_000);
  mkdirSync("tests/live/.auth", { recursive: true });

  await page.goto("/login");

  const testLogin = page.getByRole("button", { name: "Sign in as test user" });
  await expect(
    testLogin,
    `No test-login button at ${baseURL}/login. Start the server with AUTH_TEST_MODE=1.`,
  ).toBeVisible({ timeout: 60_000 });

  await page.getByLabel("Email address").fill("student@miamioh.edu");
  await testLogin.click();
  await page.waitForURL("**/");
  await expect(
    page.getByRole("heading", { level: 1, name: /AI tools for your coursework/ }),
  ).toBeVisible();

  // The mock-mode guard belongs here too: failing at setup is a clearer signal
  // than every spec failing the same way.
  await expect(
    page.getByRole("alert").filter({ hasText: /Test mode/i }),
    "This server runs the canned mock model. A live run against it proves nothing.",
  ).toHaveCount(0);

  await page.context().storageState({ path: AUTH_FILE });
});
