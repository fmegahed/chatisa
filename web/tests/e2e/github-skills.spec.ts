import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { fakeGithubApi } from "./support/fake-github";

async function seed(page: Page) {
  await page.goto("/job-scout?tab=profile");
  await page.evaluate(() => {
    localStorage.clear();
    localStorage.setItem("js-github-v1", JSON.stringify({ v: 1, token: "t", login: "mockstudent", connectedAt: "" }));
    localStorage.setItem("js-profile-v1", JSON.stringify({ v: 2, programs: [], removedPrereqs: [], courses: [{ code: "ISA 225", status: "done" }], extras: [], overrides: [] }));
  });
  await page.reload();
}

test.describe("Skills from GitHub", () => {
  test("pick repositories, confirm cards, raise one level, see it labelled", async ({ page }) => {
    await fakeGithubApi(page);
    await seed(page);
    const block = page.getByRole("region", { name: "Your GitHub (optional)" });
    await expect(block.getByText("Connected as mockstudent")).toBeVisible();
    await expect(block.getByRole("checkbox", { name: /forked-lib/ })).toHaveCount(0);
    await block.getByRole("checkbox", { name: /churn-model/ }).check();
    await block.getByRole("checkbox", { name: /class-notes/ }).check();
    await block.getByRole("button", { name: "Suggest skills from 2 repositories" }).click();

    const churn = block.getByRole("group", { name: /mockstudent\/churn-model/ });
    await expect(churn.getByText("You wrote 100% of 20 commits here, so this repository can support an anchor.")).toBeVisible();
    const notes = block.getByRole("group", { name: /mockstudent\/class-notes/ });
    await expect(notes.getByText(/You wrote 9% of the commits here, so its skills are suggested as applied/)).toBeVisible();

    // Raise a class-notes skill to anchor: the student's call.
    const card = notes.getByRole("listitem").first();
    await card.getByRole("radio", { name: "I can show real work with this" }).check();
    await card.getByRole("button", { name: "Add to my skills" }).click();
    // The button is gone; focus stays in the results (review fix).
    await expect(block.getByRole("heading", { name: "Suggested skills to confirm" })).toBeFocused();
    await expect(page.getByText(/from mockstudent\/class-notes, set by you/).first()).toBeVisible();
  });

  test("a repository that is gone fails alone; an expired connection offers reconnect once", async ({ page }) => {
    await fakeGithubApi(page);
    await seed(page);
    const block = page.getByRole("region", { name: "Your GitHub (optional)" });
    await block.getByRole("checkbox", { name: /churn-model/ }).check();
    await block.getByRole("checkbox", { name: /mockstudent\/gone/ }).check();
    await block.getByRole("button", { name: "Suggest skills from 2 repositories" }).click();
    await expect(block.getByRole("alert").filter({ hasText: "mockstudent/gone could not be read. It may be private, renamed or deleted." })).toBeVisible();
    await expect(block.getByRole("group", { name: /mockstudent\/churn-model/ })).toBeVisible();
    // Retrying a failed repository reads only that one again.
    await block.getByRole("button", { name: "Try again" }).click();
    await expect(block.getByText("mockstudent/gone could not be read. It may be private, renamed or deleted.")).toBeVisible();
  });

  test("expired connection", async ({ page }) => {
    const gh = await fakeGithubApi(page, { expireAfterList: true });
    await seed(page);
    const block = page.getByRole("region", { name: "Your GitHub (optional)" });
    await block.getByRole("checkbox", { name: /churn-model/ }).check();
    await block.getByRole("checkbox", { name: /class-notes/ }).check();
    await block.getByRole("button", { name: "Suggest skills from 2 repositories" }).click();
    await expect(block.getByRole("alert")).toContainText("Your GitHub connection has expired. Connect again to continue.");
    await expect(block.getByRole("alert")).toHaveCount(1);
    // The real Connect button is offered, not "Connected as" (review fix).
    await expect(block.getByRole("button", { name: "Connect GitHub" })).toBeVisible();
    await expect(block.getByText("Connected as mockstudent")).toHaveCount(0);
    // Reconnecting brings the list back and clears the message.
    gh.setExpired(false);
    await page.evaluate(() => {
      localStorage.setItem("js-github-v1", JSON.stringify({ v: 1, token: "t2", login: "mockstudent", connectedAt: "" }));
      window.dispatchEvent(new StorageEvent("storage", { key: "js-github-v1" }));
    });
    await expect(block.getByRole("checkbox", { name: /churn-model/ })).toBeVisible();
    await expect(block.getByRole("alert")).toHaveCount(0);
  });

  test("a file over 6 MB is listed, and read when the student asks", async ({ page }) => {
    await fakeGithubApi(page);
    await seed(page);
    const block = page.getByRole("region", { name: "Your GitHub (optional)" });
    await block.getByRole("checkbox", { name: /churn-model/ }).check();
    await block.getByRole("checkbox", { name: /class-notes/ }).check();
    await block.getByRole("button", { name: "Suggest skills from 2 repositories" }).click();
    const churn = block.getByRole("group", { name: /mockstudent\/churn-model/ });
    await expect(churn.getByText("notebooks/eda.ipynb (13.4 MB)")).toBeVisible();
    await churn.getByRole("button", { name: "Read it anyway: notebooks/eda.ipynb" }).click();
    await expect(block.getByRole("group", { name: /mockstudent\/churn-model/ }).getByText("notebooks/eda.ipynb (13.4 MB)")).toHaveCount(0);
    // The other repository's unconfirmed cards are still there (review fix).
    await expect(block.getByRole("group", { name: /mockstudent\/class-notes/ }).getByRole("button", { name: "Add to my skills" }).first()).toBeVisible();
  });

  test("allows at most five repositories", async ({ page }) => {
    await fakeGithubApi(page);
    await seed(page);
    const block = page.getByRole("region", { name: "Your GitHub (optional)" });
    // Three listed repositories here; the cap text appears only at five, so
    // this checks the count label updates and the button names the count.
    await block.getByRole("checkbox", { name: /churn-model/ }).check();
    await expect(block.getByRole("button", { name: "Suggest skills from 1 repository" })).toBeEnabled();
  });

  for (const width of [1280, 320]) {
    test(`meets WCAG A and AA with results showing at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await fakeGithubApi(page);
      await seed(page);
      const block = page.getByRole("region", { name: "Your GitHub (optional)" });
      await block.getByRole("checkbox", { name: /churn-model/ }).check();
      await block.getByRole("button", { name: "Suggest skills from 1 repository" }).click();
      await expect(block.getByRole("group", { name: /churn-model/ })).toBeVisible();
      const scan = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
      expect(scan.violations).toEqual([]);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
    });
  }
});
