import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

/**
 * Chat flows run against the deterministic mock model (CHATISA_MOCK_LLM=1),
 * so no provider is called and no sponsored budget is spent.
 */
test.describe("Coding Companion chat", () => {
  test("streams a reply, renders markdown, and offers copyable code", async ({
    page,
  }) => {
    await page.goto("/coding-tutor");
    await expect(
      page.getByRole("heading", { level: 1, name: "Coding Tutor" }),
    ).toBeVisible();
    await expect(page.getByText("Start the conversation")).toBeVisible();

    await page.getByLabel("Your message").fill("How do I read a CSV?");
    await page.getByRole("button", { name: "Send message" }).click();

    // The student's message appears immediately.
    await expect(
      page.getByRole("article", { name: "You" }),
    ).toContainText("How do I read a CSV?");

    // The response streams in and renders as markdown, not raw text.
    const reply = page.getByRole("article", { name: "ChatISA" });
    await expect(reply).toContainText("read a CSV in both languages", {
      timeout: 15_000,
    });
    await expect(reply.locator("pre code").first()).toBeVisible();
    await expect(
      reply.getByRole("button", { name: "Copy code" }).first(),
    ).toBeVisible();

    // Status region returns to idle when the stream ends.
    await expect(page.getByText("ChatISA is responding.")).toHaveCount(0, {
      timeout: 15_000,
    });
  });

  test("offers Run on each runnable block, not on prose", async ({ page }) => {
    await page.goto("/coding-tutor");
    await page.getByLabel("Your message").fill("Show me some SQL");
    await page.getByRole("button", { name: "Send message" }).click();

    const reply = page.getByRole("article", { name: "ChatISA" });
    await expect(reply).toContainText("read a CSV in both languages", {
      timeout: 15_000,
    });

    // The reply has an R block, a Python block, and a SQL block, all runnable in
    // the browser. So exactly three Run buttons appear, one per block, and none
    // on the surrounding prose.
    const runButtons = reply.getByRole("button", { name: /^Run / });
    await expect(runButtons).toHaveCount(3, { timeout: 15_000 });
    await expect(reply.getByRole("button", { name: "Run R" })).toBeVisible();
    await expect(
      reply.getByRole("button", { name: "Run Python" }),
    ).toBeVisible();
    await expect(reply.getByRole("button", { name: "Run SQL" })).toBeVisible();

    // Every code block still has its own Copy button; Run is additive.
    const copyButtons = reply.getByRole("button", { name: "Copy code" });
    expect(await copyButtons.count()).toBeGreaterThanOrEqual(3);

    // Each Run button sits inside the figure of the block it runs, not an
    // unrelated one.
    const sqlFigure = reply
      .locator("figure")
      .filter({ has: page.getByRole("button", { name: "Run SQL" }) });
    await expect(sqlFigure.locator("pre")).toContainText("SELECT 1 AS n");
    const pyFigure = reply
      .locator("figure")
      .filter({ has: page.getByRole("button", { name: "Run Python" }) });
    await expect(pyFigure.locator("pre")).toContainText("import pandas");

    // Before running, each runnable block nudges the student that they can
    // bring their own data inline (the sandbox resets each run).
    await expect(sqlFigure).toContainText("CREATE TABLE and INSERT");
    await expect(pyFigure).toContainText("io.StringIO");
  });

  test("renders TeX in replies as equations, leaving code untouched", async ({
    page,
  }) => {
    await page.goto("/coding-tutor");
    await page.getByLabel("Your message").fill("show me math please");
    await page.getByRole("button", { name: "Send message" }).click();

    const reply = page.getByRole("article", { name: "ChatISA" });
    // Wait for the reply to finish streaming before inspecting the markup.
    await expect(reply).toContainText("And code stays code", {
      timeout: 15_000,
    });
    // Inline \( \) and display $$ both render through KaTeX...
    expect(await reply.locator(".katex").count()).toBeGreaterThanOrEqual(3);
    await expect(reply.locator(".katex-display")).toBeVisible();
    // ...while TeX inside inline code is left exactly as written.
    await expect(reply.locator("code", { hasText: "\\(not math\\)" })).toBeVisible();
  });

  test("lets a student customize a runnable block in a code editor", async ({
    page,
  }) => {
    await page.goto("/coding-tutor");
    await page.getByLabel("Your message").fill("Show me some SQL");
    await page.getByRole("button", { name: "Send message" }).click();

    const reply = page.getByRole("article", { name: "ChatISA" });
    // Wait for the reply's final sentence before clicking. While the tail of a
    // reply is still streaming, the markdown re-renders on every chunk and a
    // click between two renders is silently swallowed (verified 2026-07-24:
    // the button never toggles). A student just clicks again; a test must not
    // race it.
    await expect(reply).toContainText("What does your dataset look like?", {
      timeout: 15_000,
    });
    const sqlFigure = reply
      .locator("figure")
      .filter({ has: page.getByRole("button", { name: "Run SQL" }) });
    await expect(
      sqlFigure.getByRole("button", { name: "Customize" }),
    ).toBeVisible({ timeout: 15_000 });

    // Opening the editor lazily loads CodeMirror, seeded with the snippet.
    await sqlFigure.getByRole("button", { name: "Customize" }).click();
    const editor = sqlFigure.locator(".cm-content");
    await expect(editor).toBeVisible({ timeout: 15_000 });
    await expect(editor).toContainText("SELECT 1 AS n");
    // The editor exposes an accessible name to assistive tech.
    await expect(editor).toHaveAttribute("aria-label", /Editable SQL code/i);

    // The lazily loaded editor is still accessible (no WCAG A/AA violations).
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(results.violations).toEqual([]);

    // Editing enables Reset; Reset restores the original snippet.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.type("SELECT 42 AS answer;");
    const reset = sqlFigure.getByRole("button", { name: "Reset" });
    await expect(reset).toBeVisible();
    await expect(editor).toContainText("SELECT 42 AS answer");

    await reset.click();
    await expect(editor).toContainText("SELECT 1 AS n");
    await expect(editor).not.toContainText("42");
    await expect(reset).toHaveCount(0);
  });

  test("can stop a response mid-stream", async ({ page }) => {
    await page.goto("/coding-tutor");
    await page.getByLabel("Your message").fill("Explain joins");
    await page.getByRole("button", { name: "Send message" }).click();

    const stop = page.getByRole("button", { name: "Stop generating" });
    await expect(stop).toBeVisible();

    // Wait for the first tokens so we are genuinely cancelling mid-stream.
    const reply = page.getByRole("article", { name: "ChatISA" });
    await expect(reply).toContainText("Here is how", { timeout: 15_000 });
    await stop.click();

    // Stopping ends the busy state and keeps whatever streamed so far.
    await expect(stop).toHaveCount(0);
    await expect(page.getByText("ChatISA is responding.")).toHaveCount(0);
    await expect(reply).toContainText("Here is how");

    // The student can immediately ask something else.
    await page.getByLabel("Your message").fill("Continue please");
    await expect(
      page.getByRole("button", { name: "Send message" }),
    ).toBeEnabled();
  });

  test("model chooser suggests a default and explains the alternatives", async ({
    page,
  }) => {
    await page.goto("/coding-tutor");

    // A student who recognises no model names must still be able to start, so
    // the suggested model and what it is good for are visible without opening
    // anything.
    await expect(page.getByText("Model", { exact: true })).toBeVisible();
    const chooser = page.getByRole("button", {
      name: "Choose a different model",
    });
    await expect(chooser).toBeVisible();
    await expect(chooser).toHaveAttribute("aria-expanded", "false");

    await chooser.click();
    await expect(chooser).toHaveAttribute("aria-expanded", "true");

    // Grouped, not a flat wall of names.
    await expect(
      page.getByRole("group", { name: /open weight/i }).first(),
    ).toBeVisible();

    const radios = page.getByRole("radio");
    expect(await radios.count()).toBeGreaterThan(5);
    await expect(radios.first()).toBeVisible();

    const panel = await page.locator("body").innerText();
    // Badges are derived from fields now, so the duplicated parenthetical that
    // produced "Gemma 4 31B (open weight, free) (open weight, free tier)"
    // cannot reappear.
    expect(panel).not.toMatch(/\(open weight[^)]*\)\s*\(/);
    expect(panel).not.toContain("free tier)");
    // Every option carries a sentence a student can act on.
    expect(panel).toMatch(/open weight/i);
  });

  test("no WCAG A/AA violations on the chat page, including mid-conversation", async ({
    page,
  }) => {
    await page.goto("/coding-tutor");
    let results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(results.violations).toEqual([]);

    await page.getByLabel("Your message").fill("Show me a table");
    await page.getByRole("button", { name: "Send message" }).click();
    await expect(
      page.getByRole("article", { name: "ChatISA" }),
    ).toContainText("read a CSV", { timeout: 15_000 });

    results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(results.violations).toEqual([]);
  });
});

test.describe("chat API contract", () => {
  const validBody = {
    module: "coding_companion",
    modelId: "claude-sonnet-5",
    messages: [{ role: "user", parts: [{ type: "text", text: "hi" }] }],
  };

  test("rejects a model that is not allowed for the module", async ({
    request,
  }) => {
    const res = await request.post("/api/chat", {
      data: { ...validBody, modelId: "gpt-4o-realtime-preview-2025-06-03" },
    });
    expect(res.status()).toBe(400);
    expect((await res.json()).error).toContain("isn't available");
  });

  test("rejects an unknown module", async ({ request }) => {
    const res = await request.post("/api/chat", {
      data: { ...validBody, module: "totally_made_up" },
    });
    expect(res.status()).toBe(400);
  });

  test("rejects a malformed body without leaking internals", async ({
    request,
  }) => {
    const res = await request.post("/api/chat", { data: { module: 5 } });
    expect(res.status()).toBe(400);
    const body = await res.json();
    expect(body.error).toBe("That request wasn't valid.");
    expect(JSON.stringify(body)).not.toContain("node_modules");
  });

  test("never exposes provider keys in a response", async ({ request }) => {
    const res = await request.post("/api/chat", { data: validBody });
    const text = await res.text();
    expect(text.toLowerCase()).not.toContain("api_key");
    expect(text).not.toMatch(/sk-[A-Za-z0-9]{10,}/);
  });
});

/**
 * The Run button as a promise (2026-07-26, professor's instruction).
 *
 * A snippet needing a package that cannot exist in a WebAssembly runtime can only
 * ever error, so no Run button is offered for it and the reason is stated
 * instead. The professor put it as a trust problem, which is the right frame: a
 * button that always fails teaches students not to believe the app.
 *
 * The mock model returns a pyreadr snippet for any message mentioning it, so
 * this is deterministic and costs nothing. (statsforecast held this role until
 * v6.3.0, when the app started shipping its own wasm build of it.)
 */
test.describe("package availability gates the Run button", () => {
  test("offers no Run button, and says why, for an impossible package", async ({
    page,
  }) => {
    await page.goto("/coding-tutor");
    await page
      .getByLabel("Your message")
      .fill("Show me how to read an RData file with pyreadr");
    await page.getByRole("button", { name: "Send message" }).click();

    const reply = page.getByRole("article", { name: "ChatISA" });
    await expect(reply).toContainText("pyreadr", { timeout: 15_000 });

    const figure = reply.locator("figure").first();
    await expect(figure.locator("pre")).toContainText("import pyreadr");

    // The verdict needs a fetch of the package index, so the button may appear
    // for a moment first. What matters is where it lands.
    await expect(figure.getByRole("button", { name: /^Run / })).toHaveCount(0, {
      timeout: 15_000,
    });
    await expect(figure).toContainText("cannot run here");
    await expect(figure).toContainText("pyreadr");
    // It must say what to do instead, not just refuse.
    await expect(figure).toContainText("on your computer");

    // Copy code stays: the code is fine, it just cannot run in a browser.
    await expect(figure.getByRole("button", { name: "Copy code" })).toBeVisible();
  });

  test("still offers Run for the packages that are available", async ({ page }) => {
    // The guard against over-blocking. The canned answer uses readr (mirrored for
    // R) and pandas (bundled for Python); if either were misclassified, this
    // catches it, and a feature that removes working Run buttons is worse than no
    // feature at all.
    await page.goto("/coding-tutor");
    await page.getByLabel("Your message").fill("Show me some SQL");
    await page.getByRole("button", { name: "Send message" }).click();

    const reply = page.getByRole("article", { name: "ChatISA" });
    await expect(reply).toContainText("read a CSV in both languages", {
      timeout: 15_000,
    });
    await expect(reply.getByRole("button", { name: /^Run / })).toHaveCount(3, {
      timeout: 15_000,
    });
    // And nothing claims a problem.
    await expect(reply).not.toContainText("cannot run here");
  });
});
