import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

/**
 * The CodeMirror editor's accessibility (2026-09 audit): gutter and token
 * contrast in both Coding Studio themes (#13), a keyboard-operable scroller
 * (#14), and the caret position status line and shortcut (#21).
 */

/** Replace the script with `text` in one atomic edit (no auto-closed brackets). */
async function replaceScript(page: Page, text: string) {
  const editor = page.locator(".cm-content");
  await editor.click();
  await page.keyboard.press("ControlOrMeta+A");
  await page.keyboard.press("Delete");
  await page.keyboard.insertText(text);
}

async function openStudio(page: Page, language?: "R" | "SQL") {
  await page.goto("/coding-studio");
  if (language) await page.getByRole("radio", { name: language }).click();
  // Wait for CodeMirror itself, not the load-time textarea fallback.
  await expect(page.locator(".cm-content")).toBeVisible();
}

/**
 * The lowest contrast ratio of the gutter's line numbers, the active line's
 * number, and every highlighted token, each against the colour actually
 * painted behind it (translucent layers such as the active-line band are
 * composited over their ancestors).
 */
async function editorContrast(page: Page) {
  return page.evaluate(() => {
    const parse = (c: string) => (c.match(/[\d.]+/g) ?? []).map(Number);
    function lum(rgb: number[]) {
      const c = rgb.map((v) => {
        const x = v / 255;
        return x <= 0.03928 ? x / 12.92 : ((x + 0.055) / 1.055) ** 2.4;
      });
      return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    }
    function background(el: Element) {
      const chain: Element[] = [];
      for (let e: Element | null = el; e; e = e.parentElement) chain.push(e);
      let bg = [255, 255, 255];
      for (const e of chain.reverse()) {
        const p = parse(getComputedStyle(e).backgroundColor);
        if (p.length < 3) continue;
        const a = p[3] ?? 1;
        bg = bg.map((v, i) => p[i] * a + v * (1 - a));
      }
      return bg;
    }
    function ratio(el: Element) {
      const fg = parse(getComputedStyle(el).color).slice(0, 3);
      const [a, b] = [lum(fg), lum(background(el))];
      return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
    }
    const numbers = Array.from(
      document.querySelectorAll(".cm-lineNumbers .cm-gutterElement"),
    ).filter((el) => (el.textContent ?? "").trim() !== "");
    const active = numbers.filter((el) => el.classList.contains("cm-activeLineGutter"));
    const tokens = Array.from(document.querySelectorAll(".cm-line span")).filter(
      (el) =>
        (el.textContent ?? "").trim() !== "" &&
        !el.closest(".cm-ghost-text, .cm-selectionMatch, .cm-searchMatch"),
    );
    const worst = (els: Element[]) =>
      els.length ? Math.min(...els.map(ratio)) : Number.NaN;
    return {
      gutter: worst(numbers.filter((el) => !active.includes(el))),
      activeGutter: worst(active),
      tokens: worst(tokens),
      tokenCount: tokens.length,
    };
  });
}

const SAMPLE = [
  "import re",
  "# a comment",
  "def f(x: int) -> str:",
  '    return re.sub(r"\\d+", "n", str(x))  # done',
  "",
  "print(f(42), True, None)",
].join("\n");

test.describe("Code editor accessibility", () => {
  test("gutter numbers and syntax colours are >=4.5:1 in both themes (#13)", async ({
    page,
  }) => {
    await openStudio(page);
    await replaceScript(page, SAMPLE);
    // Put the caret mid-script so both plain and active-line numbers render.
    await page.keyboard.press("ArrowUp");

    for (const theme of ["light", "dark"] as const) {
      if (theme === "dark") {
        await page.getByRole("button", { name: "Dark theme" }).click();
        await expect(page.getByRole("button", { name: "Light theme" })).toBeVisible();
        await expect(page.locator(".cm-content")).toContainText("print(f(42)");
      }
      // Highlighting is applied after parsing; wait until tokens are coloured.
      await expect
        .poll(async () => (await editorContrast(page)).tokenCount)
        .toBeGreaterThan(5);
      const c = await editorContrast(page);
      expect(c.gutter, `${theme} line numbers`).toBeGreaterThanOrEqual(4.5);
      expect(c.activeGutter, `${theme} active line number`).toBeGreaterThanOrEqual(4.5);
      expect(c.tokens, `${theme} syntax tokens`).toBeGreaterThanOrEqual(4.5);
    }
  });

  test("R keeps keyword, string and comment colours in the light theme", async ({
    page,
  }) => {
    await openStudio(page, "R");
    await replaceScript(page, 'if (TRUE) print("hi") # note');
    // The comment and string are wrapped in highlight spans with a colour
    // other than the body text (they were plain black before #13).
    const coloured = await page.evaluate(() =>
      Array.from(document.querySelectorAll(".cm-line span"))
        .filter((el) => /hi|note/.test(el.textContent ?? ""))
        .map((el) => getComputedStyle(el).color),
    );
    expect(coloured.length).toBeGreaterThanOrEqual(2);
    for (const c of coloured) expect(c).not.toBe("rgb(0, 0, 0)");
  });

  test("the scrolling script region holds focusable content in one tab stop (#14)", async ({
    page,
  }) => {
    await openStudio(page);
    await replaceScript(
      page,
      Array.from({ length: 200 }, (_, i) => `a${i} = ${i}`).join("\n"),
    );
    const scroller = page.locator(".cm-scroller");
    await expect
      .poll(() => scroller.evaluate((el) => el.scrollHeight - el.clientHeight))
      .toBeGreaterThan(100);

    // axe's rule, run on the scroller itself (the module specs exclude it).
    const axe = await new AxeBuilder({ page })
      .include(".cm-editor")
      .withRules(["scrollable-region-focusable"])
      .analyze();
    expect(axe.violations).toEqual([]);

    // The content is the single tab stop; the scroller is not a second one.
    await expect(page.locator(".cm-content")).toHaveAttribute("tabindex", "0");
    await expect(scroller).toHaveAttribute("tabindex", "-1");

    // Keyboard scrolling: moving the caret to the end scrolls the region.
    await page.keyboard.press("ControlOrMeta+Home");
    await expect.poll(() => scroller.evaluate((el) => el.scrollTop)).toBeLessThan(5);
    await page.keyboard.press("ControlOrMeta+End");
    await expect.poll(() => scroller.evaluate((el) => el.scrollTop)).toBeGreaterThan(100);
  });

  test("the status line follows the caret and the shortcut announces it (#21)", async ({
    page,
  }) => {
    await openStudio(page);
    const content = page.locator(".cm-content");
    // Named with the language, and described by the shortcut hint.
    await expect(content).toHaveAttribute("aria-label", /Python code/);
    const hintId = await content.getAttribute("aria-describedby");
    expect(hintId).toBeTruthy();
    await expect(page.locator(`[id="${hintId}"]`)).toContainText(/Shift\+L/);

    await replaceScript(page, "a = 1\nbb = 22\nccc = 333");
    const status = page.getByTestId("editor-caret-status-text");
    await expect(status).toHaveText("Ln 3, Col 10");
    await page.keyboard.press("ArrowUp");
    await expect(status).toHaveText("Ln 2, Col 8");
    await page.keyboard.press("ControlOrMeta+Home");
    await expect(status).toHaveText("Ln 1, Col 1");

    // The status line is not a live region (it would chatter on every key).
    const statusBox = page.getByTestId("editor-caret-status");
    await expect(statusBox).not.toHaveAttribute("aria-live", /.+/);
    await expect(statusBox).not.toHaveAttribute("role", /.+/);

    // The shortcut announces the full position without moving the caret or
    // typing anything.
    await page.keyboard.press("ArrowDown");
    await page.keyboard.press("End");
    await page.keyboard.press("Alt+Shift+L");
    await expect(page.getByTestId("announcer-polite")).toHaveText(
      "Line 2 of 3, column 8.",
    );
    await expect(status).toHaveText("Ln 2, Col 8");
    // Read the text without the AI ghost suggestion, which can appear at a
    // line end after a pause (an aria-hidden widget, not document text).
    const text = await content.evaluate((el) => {
      const copy = el.cloneNode(true) as HTMLElement;
      copy.querySelectorAll(".cm-ghost-text").forEach((g) => g.remove());
      return copy.textContent;
    });
    expect(text).toContain("bb = 22ccc = 333");
  });

  test("the shortcut also reads lint problems on the caret's line (#21)", async ({
    page,
  }) => {
    await openStudio(page, "R");
    await replaceScript(page, "x <- (1 + 2");
    await expect(page.locator(".cm-lintRange-error").first()).toBeVisible({
      timeout: 5000,
    });
    await page.keyboard.press("Alt+Shift+L");
    await expect(page.getByTestId("announcer-polite")).toHaveText(
      "Line 1 of 1, column 12. Error: Unclosed (",
    );
  });
});
