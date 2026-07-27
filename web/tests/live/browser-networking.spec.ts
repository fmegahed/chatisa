import {
  assertLiveServer,
  codeBlocks,
  isCrossOriginIsolated,
  runBlock,
  sendAndSettle,
  test,
  expect,
} from "./support/live";
import type { Page } from "@playwright/test";

/**
 * The regression suite for the bug the professor hit in production on
 * 2026-07-26: rvest::read_html failing with "cannot open the connection" inside
 * a Coding Tutor answer.
 *
 * The cause was a missing pair of response headers. Without cross-origin
 * isolation there is no SharedArrayBuffer, and WebR then falls back to a channel
 * with no networking at all, so every scrape fails with an error that reads like
 * an R problem. lib/run/isolation.ts holds the route list and tests/unit
 * /run-isolation.test.ts guards it statically. What only a live run can prove is
 * that a student pressing "Run R" on a real page actually reaches the internet,
 * so that is what this does: it drives the real button on the real page.
 *
 * The snippets are typed by us, not produced by a model, because the mechanism
 * under test is the runtime's network access, not the model's code. The
 * model-authored variants live in coding-tutor.spec.ts.
 */

/** A tiny, stable page with an h1. Kept boring on purpose. */
const TARGET = "https://example.com";

/**
 * Replaces the first runnable block's code with our own, through "Customize".
 *
 * Every line is column-zero: CodeMirror auto-indents, so a typed block with
 * indentation arrives in the editor with compounding leading whitespace, which
 * is a syntax error in Python and merely ugly in R. Snippets here are written to
 * need no indentation at all rather than fighting the editor.
 */
async function replaceFirstBlock(page: Page, code: string): Promise<void> {
  for (const line of code.split("\n")) {
    expect(
      line,
      "live network snippets must not be indented (CodeMirror auto-indents)",
    ).toBe(line.trimStart());
  }
  await page.getByRole("button", { name: "Customize" }).first().click();
  const editor = page.locator(".cm-content").first();
  await editor.waitFor({ timeout: 120_000 });
  await editor.click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.press("Delete");
  await editor.pressSequentially(code, { delay: 4 });
}

/** Gets any answer with at least one runnable block of the wanted language. */
async function seedRunnableBlock(
  page: Page,
  language: "r" | "python",
): Promise<void> {
  const ask =
    language === "r"
      ? "Show me one line of R that prints the number 1. Just the code, no explanation."
      : "Show me one line of Python that prints the number 1. Just the code, no explanation.";
  await sendAndSettle(page, ask, { timeoutMs: 4 * 60_000 });
  const blocks = await codeBlocks(page);
  expect(
    blocks.some((b) => b.language === language || b.language === null),
    `the model produced no ${language} block to customize: ${JSON.stringify(blocks)}`,
  ).toBe(true);
}

test.describe("browser runtime networking", () => {
  test("the Coding Tutor page is cross-origin isolated", async ({
    page,
    observe,
  }) => {
    await page.goto("/coding-tutor");
    await assertLiveServer(page);
    observe.note("coding-tutor loaded");

    const isolated = await isCrossOriginIsolated(page);
    observe.note(`crossOriginIsolated=${isolated}`);
    // Before the 2026-07-26 fix this was false, and that single boolean is the
    // whole bug: R's networking is absent whenever it is false.
    expect(isolated).toBe(true);
  });

  test("Ask Anything is cross-origin isolated", async ({ page, observe }) => {
    await page.goto("/ask-anything");
    await assertLiveServer(page);
    const isolated = await isCrossOriginIsolated(page);
    observe.note(`crossOriginIsolated=${isolated}`);
    expect(isolated).toBe(true);
  });

  test("the Coding Studio page is still cross-origin isolated", async ({
    page,
    observe,
  }) => {
    // The route that always had the headers. Here as a control: if this ever
    // goes false, the cause is the header plumbing, not the new route list.
    await page.goto("/coding-studio");
    await assertLiveServer(page);
    const isolated = await isCrossOriginIsolated(page);
    observe.note(`crossOriginIsolated=${isolated}`);
    expect(isolated).toBe(true);
  });

  test("AI Comparison is cross-origin isolated too", async ({ page, observe }) => {
    // Covered by the 2026-07-26 follow-up. This test previously asserted the
    // OPPOSITE, recording the gap while the fix was scoped to the Coding Tutor
    // and Ask Anything; the professor then extended it, on the reasoning that a
    // Run button should behave the same wherever it appears.
    await page.goto("/ai-comparison");
    await assertLiveServer(page);
    const isolated = await isCrossOriginIsolated(page);
    observe.note(`crossOriginIsolated=${isolated}`);
    expect(isolated).toBe(true);
  });

  test("R reaches the internet from a Coding Tutor answer", async ({
    page,
    observe,
  }) => {
    await page.goto("/coding-tutor");
    await assertLiveServer(page);
    await seedRunnableBlock(page, "r");
    observe.note("seeded an R block");

    // The exact shape of the professor's failing snippet: rvest over libcurl,
    // which is what needs the SharedArrayBuffer channel.
    await replaceFirstBlock(
      page,
      [
        'if(require(rvest)==FALSE) install.packages("rvest")',
        `doc <- rvest::read_html("${TARGET}")`,
        'cat(rvest::html_text(rvest::html_element(doc, "h1")))',
      ].join("\n"),
    );
    observe.note("typed the rvest snippet");

    const result = await runBlock(page, 0, { timeoutMs: 6 * 60_000 });
    observe.note(`run ok=${result.ok}`);
    await observe.save("r-scrape-output.txt", result.output);

    expect(
      result.output,
      "this is the exact production symptom: R has no network",
    ).not.toContain("cannot open the connection");
    expect(result.ok, `R run failed: ${result.output}`).toBe(true);
    expect(result.output).toContain("Example Domain");
  });

  test("Python reaches the internet from a Coding Tutor answer", async ({
    page,
    observe,
  }) => {
    await page.goto("/coding-tutor");
    await assertLiveServer(page);
    await seedRunnableBlock(page, "python");
    observe.note("seeded a Python block");

    // Python's path is different from R's: requests goes through our own
    // SSRF-guarded /api/py-proxy, a same-origin fetch. It is checked on the same
    // page anyway, because "the Python equivalent path" is what a student
    // actually reaches for, and a same-origin assumption is worth verifying
    // rather than reasoning about.
    await replaceFirstBlock(
      page,
      [
        "import requests",
        `r = requests.get("${TARGET}")`,
        'print(r.status_code, "Example Domain" in r.text)',
      ].join("\n"),
    );
    observe.note("typed the requests snippet");

    const result = await runBlock(page, 0, { timeoutMs: 6 * 60_000 });
    observe.note(`run ok=${result.ok}`);
    await observe.save("python-scrape-output.txt", result.output);

    expect(result.ok, `Python run failed: ${result.output}`).toBe(true);
    expect(result.output).toContain("200 True");
  });
});
