import { test as setup, expect } from "@playwright/test";

const AUTH_FILE = "tests/e2e/.auth/user.json";

/**
 * Signs in once via the non-production test provider and saves the session
 * cookie for all authenticated test projects.
 */
setup("authenticate as test student", async ({ page }) => {
  // Sign-in itself is quick; the route warmups below (each capped individually)
  // can add a couple of minutes on a cold compile, so the test gets its own
  // budget instead of the default 60s.
  //
  // Raised from 300s to 600s on 2026-07-26 (professor's instruction) after a run
  // hit the 300s ceiling exactly on a cold Turbopack compile and passed at 1.5m
  // on the next attempt. This setup is a dependency of every authenticated
  // project, so when it times out NOTHING else runs and the whole suite reports
  // as broken. A generous ceiling costs nothing on a warm machine, because the
  // work here finishes when it finishes; it only decides how long a cold one is
  // allowed to take before the suite gives up.
  setup.setTimeout(600_000);
  // A dev-server failure seen five times on 2026-11-17, always on the first run
  // after a batch of source edits: Turbopack's restored on-disk dev cache kept
  // sending "restart" for a stale entry chunk list, the page reloaded itself in
  // a loop, and the suite burned the full 10 minutes in waitForURL. The test
  // server now runs without that cache (playwright.config.ts,
  // CHATISA_E2E_NO_DEV_CACHE), which fixed it in repeated trials. This guard
  // stays as a safety net: real sign-in is two or three navigations, dozens
  // mean a loop, so fail in seconds with a message that says what happened.
  let navigations = 0;
  let looping: () => void = () => {};
  const loop = new Promise<never>((_, reject) => {
    looping = () =>
      reject(
        new Error(
          "The dev server is reloading /login in a loop (seen on the first run after source edits). Re-run the suite.",
        ),
      );
  });
  const countNavigation = (frame: import("@playwright/test").Frame) => {
    if (frame === page.mainFrame() && ++navigations > 40) looping();
  };
  page.on("framenavigated", countNavigation);
  // The loop starts on the first page load, so the whole sign-in races it.
  await Promise.race([
    (async () => {
      await page.goto("/login");
      await page.getByLabel("Email address").fill("student@miamioh.edu");
      await page.getByRole("button", { name: "Sign in as test user" }).click();
      await page.waitForURL("**/");
    })(),
    loop,
  ]);
  page.off("framenavigated", countNavigation);
  await expect(
    page.getByRole("heading", { level: 1, name: /AI tools for your coursework/ }),
  ).toBeVisible();
  await page.context().storageState({ path: AUTH_FILE });

  // Warm the heaviest routes once, up front (the page is still authenticated), so
  // the parallel run never pays a cold Turbopack per-route compile under
  // contention, which is the main source of first-hit flakes. Best-effort: a
  // warmup hiccup must never fail auth for the whole suite.
  for (const route of ["/coding-studio", "/coding-tutor", "/project-assistant"]) {
    try {
      await page.goto(route, { waitUntil: "domcontentloaded", timeout: 60_000 });
      await page.waitForLoadState("networkidle", { timeout: 20_000 });
    } catch {
      // ignore; the route still compiled on the way through
    }
  }
  // The CodeMirror chunks are the heaviest lazy imports; make sure both are
  // compiled before the tests that assert on them, so `.cm-content` never waits
  // on a cold compile. The sandbox editor and the chat "Customize" editor live
  // in different chunks (different pages), so each needs its own touch: the
  // Customize path was the one residual full-suite flake until warmed here.
  try {
    await page.goto("/coding-studio", { waitUntil: "domcontentloaded", timeout: 60_000 });
    await page.locator(".cm-content").first().waitFor({ timeout: 60_000 });
  } catch {
    // ignore
  }
  try {
    await page.goto("/coding-tutor", { waitUntil: "domcontentloaded", timeout: 60_000 });
    // Capped like every other warmup: typed before hydration, the text can be
    // reset and Send stays disabled; an uncapped click then waited out the
    // whole 10-minute setup budget (seen twice with Next.js 16.3).
    await page.getByLabel("Your message").fill("Show me some SQL", { timeout: 30_000 });
    await page.getByRole("button", { name: "Send message" }).click({ timeout: 30_000 });
    const customize = page.getByRole("button", { name: "Customize" }).first();
    await customize.waitFor({ timeout: 30_000 });
    await customize.click();
    await page.locator(".cm-content").first().waitFor({ timeout: 60_000 });
  } catch {
    // ignore
  }
});
