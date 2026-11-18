import { test, expect, type Page, type Locator } from "@playwright/test";
import { auditDom, axeViolations } from "./support/a11y";
import pin from "../../lib/run/maidr-pin.json";

/**
 * Navigable, audible plots (#40). A plot's data travels from the runtime
 * (matplotlib introspection, ggplot_build, ggsql's Vega-Lite spec) to MAIDR,
 * which turns the picture into a chart a student can explore from the
 * keyboard: Tab to it, then the arrow keys announce each value.
 *
 * The announcement is the end-to-end proof: it can only read "Course is
 * ISA 401, Average grade is 3.2" if the right numbers and labels crossed every
 * boundary. Wording measured against maidr 4.16.0 (spike, 2026-11-17).
 */

const PY_BAR = [
  "import matplotlib.pyplot as plt",
  'plt.bar(["ISA 401", "ISA 444", "ISA 491"], [3.2, 3.6, 2.9])',
  'plt.title("Average grade by course")',
  'plt.xlabel("Course")',
  'plt.ylabel("Average grade")',
].join("\n");

const R_COL = [
  "library(ggplot2)",
  'grades <- data.frame(course = c("ISA 401", "ISA 444"), avg = c(3.2, 3.6))',
  'ggplot(grades, aes(course, avg)) + geom_col() + labs(x = "Course", y = "Average grade")',
].join("\n");

async function setCode(page: Page, language: "Python" | "R" | "SQL", code?: string) {
  await page.getByRole("radio", { name: language }).click();
  const editor = page.locator(".cm-content").first();
  await expect(editor).toBeVisible({ timeout: 30_000 });
  await editor.click();
  await page.keyboard.press("ControlOrMeta+A");
  await page.keyboard.press("Delete");
  if (code) await page.keyboard.insertText(code);
  else await page.getByRole("button", { name: "Insert example" }).click();
}

/** The focusable MAIDR chart inside a container, once maidr.js wrapped it. */
function chartIn(scope: Locator) {
  return scope.locator('[data-maidr-plot] figure[id^="maidr-figure-"] [tabindex="0"]');
}

/** Focus the chart, press ArrowRight, and return what MAIDR announced. The
 * wait covers a cold Python runtime under a full parallel suite. MAIDR
 * activates about 80ms after focus (role img -> application); a key pressed
 * before that is dropped (measured in Chrome, Nov 17), so wait for it. */
async function firstAnnouncement(scope: Locator) {
  const chart = chartIn(scope);
  await expect(chart).toBeVisible({ timeout: 120_000 });
  await chart.focus();
  await expect(chart).toHaveAttribute("role", "application");
  await chart.press("ArrowRight");
  const article = scope.locator('[data-maidr-plot] article[id^="maidr-article-"]');
  return article;
}

test.describe("Navigable, audible plots (#40)", () => {
  test("a Python plot in the Coding Studio can be explored from the keyboard", async ({ page }) => {
    test.setTimeout(240_000);
    const requests: string[] = [];
    page.on("request", (r) => requests.push(r.url()));
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(m.text());
    });

    await page.goto("/coding-studio");
    await setCode(page, "Python", PY_BAR);
    await page.getByRole("button", { name: "Run", exact: true }).click();
    const panel = page.getByRole("tabpanel", { name: "Plots" });

    const article = await firstAnnouncement(panel);
    await expect(article).toContainText("Course is ISA 401, Average grade is 3.2", { timeout: 15_000 });
    await page.keyboard.press("ArrowRight");
    await expect(article).toContainText("Course is ISA 444, Average grade is 3.6");
    // The description (#23) is still the image's alt text, and the hint is
    // there for screen readers (on screen only with keyboard focus; see the
    // click test below).
    await expect(panel.locator("img").first()).toHaveAttribute("alt", /bar chart titled "Average grade by course"/);
    await expect(panel.getByText(/Explore this chart from the keyboard/)).toBeAttached();

    // maidr.js came from our own origin, pinned, and nothing else left it.
    const origin = new URL(page.url()).origin;
    expect(requests).toContain(`${origin}/runtimes/maidr/${pin.version}/maidr.js`);
    const foreign = requests.filter((u) => !u.startsWith(origin) && !/^(data|blob):/.test(u));
    expect(foreign, "third-party requests").toEqual([]);

    // MAIDR's single-letter keys stay inside the chart: typing in the editor
    // after the chart loaded types normally.
    const editor = page.locator(".cm-content").first();
    await editor.click();
    await page.keyboard.press("ControlOrMeta+End");
    await page.keyboard.insertText("\n# bts");
    await page.keyboard.type("/r?");
    await expect(editor).toContainText("# bts/r?");

    // Re-running adds a second plot; only the shown one is a MAIDR chart, and
    // paging back remounts cleanly.
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(page.getByText("Plot 2 of 2", { exact: true })).toBeAttached({ timeout: 120_000 });
    await expect(chartIn(panel)).toHaveCount(1, { timeout: 30_000 });
    await page.getByRole("button", { name: /previous plot/i }).click();
    await expect(chartIn(panel)).toHaveCount(1, { timeout: 30_000 });
    expect(await page.locator('article[id^="maidr-article-"]').count()).toBe(1);
    expect(errors.filter((e) => /maidr|removeChild|NotFoundError/i.test(e))).toEqual([]);
  });

  test("the Coding Studio's Python example (a dumbbell) reads student names and fits the pane", async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto("/coding-studio");
    await setCode(page, "Python");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    const panel = page.getByRole("tabpanel", { name: "Plots" });
    const article = await firstAnnouncement(panel);
    // Category names, not their positions on the axis ("Student is 0" before
    // Nov 17), the connector lines are not a layer of their own, and the
    // legend title names the groups (MAIDR's default was "Level is ISA 401"):
    // the same reading as the R and SQL versions of this example.
    await expect(article).toContainText("Student is Amanda, Grade is 91, Course is ISA 401", { timeout: 15_000 });
    // MAIDR's wrappers must not stop the image shrinking to the pane: it was
    // drawn at full size and clipped (429px of plot in a 242px pane).
    const host = await panel.locator("[data-maidr-plot]").boundingBox();
    const img = await panel.locator("[data-maidr-plot] img").boundingBox();
    expect(img!.height).toBeLessThanOrEqual(host!.height + 1);
    expect(img!.width).toBeLessThanOrEqual(host!.width + 1);

    // No keyboard trap (WCAG 2.1.2): MAIDR takes the arrows and letters, but
    // Tab and Shift+Tab still leave the chart, and not into a hidden part of it.
    const chart = chartIn(panel);
    const focusOutsidePlot = () =>
      page.evaluate(() => !!document.activeElement && !document.activeElement.closest("[data-maidr-plot]"));
    await page.keyboard.press("Tab");
    await expect.poll(focusOutsidePlot).toBe(true);
    await chart.focus();
    await expect(chart).toHaveAttribute("role", "application");
    await page.keyboard.press("Shift+Tab");
    await expect.poll(focusOutsidePlot).toBe(true);
  });

  test("students who click rather than Tab see the plain picture: no hint, no outline, no tones", async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto("/coding-studio");
    await setCode(page, "Python", PY_BAR);
    await page.getByRole("button", { name: "Run", exact: true }).click();
    const panel = page.getByRole("tabpanel", { name: "Plots" });
    const chart = chartIn(panel);
    await expect(chart).toBeVisible({ timeout: 180_000 });
    const hint = panel.getByText(/Explore this chart from the keyboard/);
    const hintWidth = async () => (await hint.boundingBox())?.width ?? 0;
    // The hint is for screen readers until the chart has keyboard focus.
    await expect(hint).toBeAttached();
    expect(await hintWidth()).toBeLessThanOrEqual(1);

    // A click focuses nothing in the plot (as clicking a plain image does),
    // so MAIDR stays asleep: arrow keys afterwards announce nothing. Before
    // Nov 17 a click woke it, and the arrows then played tones.
    await panel.locator("[data-maidr-plot] img").click();
    await expect(chart).not.toBeFocused();
    await expect(chart).toHaveAttribute("role", "img");
    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowDown");
    await expect(panel.locator("[data-maidr-plot]")).not.toContainText("Average grade is");
    expect(await hintWidth()).toBeLessThanOrEqual(1);

    // The chart stays a Tab stop: reached from the keyboard, it wakes and the
    // hint appears.
    await chart.focus();
    await page.keyboard.press("Shift+Tab");
    await page.keyboard.press("Tab");
    await expect(chart).toBeFocused();
    await expect(chart).toHaveAttribute("role", "application");
    await expect.poll(hintWidth).toBeGreaterThan(50);
    await page.keyboard.press("ArrowRight");
    await expect(panel.locator('[data-maidr-plot] article[id^="maidr-article-"]')).toContainText(
      "Course is ISA 401, Average grade is 3.2",
    );
  });

  test("an R ggplot in the Coding Studio announces its values", async ({ page }) => {
    // tidyverse loads on first use in webR; give it room.
    test.setTimeout(300_000);
    await page.goto("/coding-studio");
    await setCode(page, "R", R_COL);
    await page.getByRole("button", { name: "Run", exact: true }).click();
    const panel = page.getByRole("tabpanel", { name: "Plots" });
    await expect(chartIn(panel)).toBeVisible({ timeout: 240_000 });
    const article = await firstAnnouncement(panel);
    await expect(article).toContainText("Course is ISA 401, Average grade is 3.2", { timeout: 15_000 });
    // R plots now get figure-based descriptions too (#23).
    await expect(panel.locator("img").first()).toHaveAttribute("alt", /bar chart.*x axis Course/);
  });

  test("the Coding Studio's SQL example (a ggsql dumbbell) is navigable", async ({ page }) => {
    test.setTimeout(180_000);
    await page.goto("/coding-studio");
    await setCode(page, "SQL");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    const panel = page.getByRole("tabpanel", { name: "Plots" });
    const article = await firstAnnouncement(panel);
    await expect(article).toContainText(/Student is \w+/, { timeout: 15_000 });
    await expect(article).toContainText(/Grade is \d+/);
    await expect(article).toContainText(/Course is ISA \d+/);
  });

  test("with a plot on screen, the Coding Studio stays axe-clean and has no stray tab stops", async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto("/coding-studio");
    await setCode(page, "Python", PY_BAR);
    await page.getByRole("button", { name: "Run", exact: true }).click();
    const panel = page.getByRole("tabpanel", { name: "Plots" });
    await expect(chartIn(panel)).toBeVisible({ timeout: 180_000 });
    expect.soft(await axeViolations(page)).toEqual([]);
    const dom = await auditDom(page);
    expect.soft(dom.nonInteractiveTabStops, "non-interactive tab stops").toEqual([]);
    expect.soft(dom.hiddenTabStops, "invisible tab stops").toEqual([]);
    expect.soft(dom.roleNameOnGeneric, "aria-label on a generic element").toEqual([]);
  });

  test("a plot from a runnable block in a Coding Tutor reply is navigable", async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto("/coding-tutor");
    await page.getByLabel("Your message").fill("Please draw a bar chart of the grades");
    await page.getByRole("button", { name: "Send message" }).click();
    const reply = page.getByRole("article", { name: "ChatISA" }).last();
    // Click Run only once the reply has finished streaming: a click while the
    // block is still re-rendering can land on a button that is then replaced
    // (seen under the full parallel suite at 320 px).
    await expect(page.getByText("ChatISA finished responding.")).toBeAttached({ timeout: 60_000 });
    const run = reply.getByRole("button", { name: "Run Python" });
    await run.click({ timeout: 60_000 });
    await expect(reply.getByRole("heading", { name: "Output" })).toBeVisible({ timeout: 150_000 });
    const article = await firstAnnouncement(reply);
    await expect(article).toContainText("Course is ISA 401, Average grade is 3.2", { timeout: 15_000 });
  });

  test("a plot from an Ask Anything tool call is navigable", async ({ page }) => {
    test.setTimeout(240_000);
    await page.goto("/ask-anything");
    await page.getByLabel("Your message").fill("Please plot the grades in python");
    await page.getByRole("button", { name: "Send message" }).click();
    const reply = page.getByRole("article", { name: "ChatISA" }).first();
    const card = reply.locator("details").first();
    await expect(card.locator("summary")).toContainText(/Ran Python/, { timeout: 180_000 });
    await card.locator("summary").click();
    const article = await firstAnnouncement(card);
    await expect(article).toContainText("Course is ISA 401, Average grade is 3.2", { timeout: 15_000 });
  });
});
