import { test, expect } from "@playwright/test";
import {
  HOSTED_CHART_EXAMPLE,
  PYTHON_CHART_EXAMPLE,
  R_CHART_EXAMPLE,
} from "../../lib/ask/chart-examples";

/**
 * The house-style chart exemplars must RUN, not merely look plausible: they are
 * the code the model copies, and they depend on packages we bundle (ggtext and
 * ggrepel in the R mirror, adjustText as a hosted wheel). A typo or a missing
 * package here becomes a broken chart in every student's chat.
 *
 * Opt-in because each test cold-loads a WASM runtime and installs packages.
 *   CHATISA_LIVE_NET=1 npx playwright test tests/e2e/chart-examples.spec.ts
 */
test.describe("chart exemplars render", () => {
  async function runInStudio(
    page: import("@playwright/test").Page,
    language: "R" | "Python",
    code: string,
  ) {
    await page.goto("/coding-studio");
    if (language === "R") await page.getByRole("radio", { name: "R" }).click();
    const editor = page.getByRole("textbox", {
      name: new RegExp(`${language} code`, "i"),
    });
    await expect(page.locator(".cm-content")).toBeVisible();
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    // insertText, not type(): the editor's auto-indent and bracket closing
    // would otherwise mangle multi-line code.
    await page.keyboard.insertText(code);
    await page.getByRole("button", { name: "Run", exact: true }).click();
  }

  test("the R exemplar draws a plot with ggtext and ggrepel (live)", async ({
    page,
  }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs a warm R runtime; opt in with CHATISA_LIVE_NET=1",
    );
    test.setTimeout(420_000);
    await runInStudio(page, "R", R_CHART_EXAMPLE);

    await expect(page.getByRole("img", { name: /^Plot 1 of/ })).toBeVisible({
      timeout: 400_000,
    });
    // A ggplot error surfaces in the console rather than failing the run.
    const output = await page.getByLabel("Console output").innerText();
    expect(output).not.toMatch(/there is no package|could not find function|Error/i);
  });

  test("the Python exemplar draws a plot with adjustText (live)", async ({
    page,
  }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs a warm Python runtime; opt in with CHATISA_LIVE_NET=1",
    );
    test.setTimeout(420_000);
    await runInStudio(page, "Python", PYTHON_CHART_EXAMPLE);

    await expect(page.getByRole("img", { name: /^Plot 1 of/ })).toBeVisible({
      timeout: 400_000,
    });
    const output = await page.getByLabel("Console output").innerText();
    expect(output).not.toMatch(/ModuleNotFoundError|Traceback|NameError/i);
  });

  test("the hosted-sandbox exemplar runs on matplotlib alone (live)", async ({
    page,
  }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs a warm Python runtime; opt in with CHATISA_LIVE_NET=1",
    );
    test.setTimeout(420_000);
    // This is the code that goes INTO a deck, in a container with no network.
    // It must therefore work with nothing but matplotlib. The runtime here is
    // Pyodide, but the constraint being checked is "no extra imports".
    expect(HOSTED_CHART_EXAMPLE).not.toMatch(/adjustText|highlight_text/);
    await runInStudio(
      page,
      "Python",
      `${HOSTED_CHART_EXAMPLE}\nimport os\nprint("PNG_BYTES", os.path.getsize("analytics_stages.png"))\n`,
    );

    const output = page.getByLabel("Console output");
    await expect(output).toContainText("PNG_BYTES", { timeout: 400_000 });
    const text = await output.innerText();
    expect(text).not.toMatch(/ModuleNotFoundError|Traceback/i);
    // A real figure, not an empty file.
    const bytes = Number(/PNG_BYTES (\d+)/.exec(text)?.[1] ?? 0);
    expect(bytes).toBeGreaterThan(10_000);
  });
});
