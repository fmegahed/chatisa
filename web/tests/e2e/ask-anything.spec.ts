import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

/**
 * Ask Anything shell (slice A): device-stored multi-chat around the shared Chat
 * component, streaming through the mock LLM. Chat content must only ever land
 * in localStorage (ADR-022), which the first test asserts directly.
 */
test.describe("Ask Anything", () => {
  /** On mobile the sidebar sits behind a "Chats" disclosure; open it if so.
   * The shell is client-only (ssr:false), so first wait for it to hydrate:
   * either the desktop nav or the mobile toggle appears. */
  async function openSidebarIfMobile(page: Page) {
    const toggle = page.getByRole("button", { name: /^Chats \(/ });
    const nav = page.getByRole("navigation", { name: "Your chats" });
    await expect(nav.or(toggle).first()).toBeVisible();
    if (await toggle.isVisible()) await toggle.click();
  }

  test("streams a reply and saves the chat on the device", async ({ page }) => {
    await page.goto("/ask-anything");
    await expect(
      page.getByRole("heading", { level: 1, name: "Ask Anything" }),
    ).toBeVisible();

    await page.getByLabel("Your message").fill("Show me some SQL");
    await page.getByRole("button", { name: "Send message" }).click();
    const reply = page.getByRole("article", { name: "ChatISA" });
    await expect(reply).toContainText("SELECT 1 AS n", { timeout: 15_000 });

    // The sidebar lists the chat, titled from the first message.
    await openSidebarIfMobile(page);
    await expect(
      page
        .getByRole("navigation", { name: "Your chats" })
        .getByText("Show me some SQL"),
    ).toBeVisible();

    // Stored on the device only.
    const stored = await page.evaluate(() =>
      window.localStorage.getItem("aa-chats-v1"),
    );
    expect(stored).toContain("Show me some SQL");
  });

  test("multiple chats: new, switch, persist across reload, delete", async ({
    page,
  }) => {
    await page.goto("/ask-anything");
    await page.getByLabel("Your message").fill("First conversation topic");
    await page.getByRole("button", { name: "Send message" }).click();
    await expect(page.getByRole("article", { name: "ChatISA" })).toBeVisible({
      timeout: 15_000,
    });

    await openSidebarIfMobile(page);
    await page.getByRole("button", { name: "New chat" }).click();
    await expect(page.getByRole("article", { name: "ChatISA" })).toHaveCount(0);
    await page.getByLabel("Your message").fill("Second conversation topic");
    await page.getByRole("button", { name: "Send message" }).click();
    await expect(page.getByRole("article", { name: "ChatISA" })).toBeVisible({
      timeout: 15_000,
    });

    // Both listed; switching restores the first transcript.
    const nav = page.getByRole("navigation", { name: "Your chats" });
    await openSidebarIfMobile(page);
    await nav.getByText("First conversation topic").click();
    await expect(
      page
        .getByRole("article", { name: "You" })
        .getByText("First conversation topic"),
    ).toBeVisible();

    // Reload: both chats survive (device storage), and stay openable.
    await page.reload();
    await openSidebarIfMobile(page);
    await expect(nav.getByText("First conversation topic")).toBeVisible();
    await expect(nav.getByText("Second conversation topic")).toBeVisible();

    // Delete the second; it leaves the list.
    await page
      .getByRole("button", { name: "Delete chat: Second conversation topic" })
      .click();
    await expect(nav.getByText("Second conversation topic")).toHaveCount(0);
  });

  test("the agentic loop runs Python in the browser and the model continues", async ({
    page,
  }) => {
    // The mock model scripts the loop (tool call, then an acknowledgement once
    // the result returns), but the execution in between is the real Pyodide
    // worker, cold-loaded from our own mirror. Give it e2e headroom.
    test.setTimeout(180_000);
    await page.goto("/ask-anything");
    await page
      .getByLabel("Your message")
      .fill("Please use python to compute 6 times 7");
    await page.getByRole("button", { name: "Send message" }).click();

    // The tool card appears (running, then done) inside the assistant reply.
    const reply = page.getByRole("article", { name: "ChatISA" }).first();
    const card = reply.locator("details").first();
    await expect(card).toBeVisible({ timeout: 30_000 });
    await expect(card.locator("summary")).toContainText(/Ran Python/, {
      timeout: 150_000,
    });
    // Expanding shows the code the model ran and the real output.
    await card.locator("summary").click();
    await expect(card).toContainText("print(6 * 7)");
    await expect(card).toContainText("42");

    // The loop continued: the model saw the result and acknowledged it.
    await expect(page.getByText(/RESULT_ACK/)).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(/RESULT_ACK/)).toContainText("42");

    // The tool transcript persisted to the device like any other message.
    const stored = await page.evaluate(() =>
      window.localStorage.getItem("aa-chats-v1"),
    );
    expect(stored).toContain("RESULT_ACK");
  });

  // --- Slice C: attachments and research tools -----------------------------

  const TINY_PNG = Buffer.from(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
    "base64",
  );

  // A minimal one-page PDF: enough structure for the client page estimate.
  const TINY_PDF = Buffer.from(
    "%PDF-1.4\n1 0 obj<</Type /Catalog /Pages 2 0 R>>endobj\n2 0 obj<</Type /Pages /Kids [3 0 R] /Count 1>>endobj\n3 0 obj<</Type /Page /Parent 2 0 R /MediaBox [0 0 200 200]>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF",
  );

  async function attach(
    page: Page,
    file: { name: string; mimeType: string; buffer: Buffer },
  ) {
    await page
      .locator("#chat-attach-input")
      .setInputFiles([file]);
  }

  test("attaches a text file: chip, model reads it, survives reload", async ({
    page,
  }) => {
    await page.goto("/ask-anything");
    await expect(page.getByLabel("Your message")).toBeVisible();
    await attach(page, {
      name: "notes.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("The launch code word is PINEAPPLE."),
    });
    // The pending chip reports ready, then the message sends.
    const chips = page.getByRole("list", { name: "Files to send" });
    await expect(chips).toContainText("notes.txt");
    await expect(chips).toContainText("text", { timeout: 15_000 });
    await page.getByLabel("Your message").fill("What does my file say?");
    await page.getByRole("button", { name: "Send message" }).click();

    // The user bubble carries the attachment chip; the mock echoes the block.
    await expect(
      page.getByRole("article", { name: "You" }).getByText("notes.txt"),
    ).toBeVisible();
    const ack = page.getByText(/FILE_ACK/);
    await expect(ack).toBeVisible({ timeout: 15_000 });
    await expect(ack).toContainText("PINEAPPLE");

    // Reload: reopen the chat from the sidebar; the transcript, chip
    // included, came back from device storage.
    await page.reload();
    await openSidebarIfMobile(page);
    await page
      .getByRole("navigation", { name: "Your chats" })
      .getByText("What does my file say?")
      .click();
    await expect(
      page.getByRole("article", { name: "You" }).getByText("notes.txt"),
    ).toBeVisible({ timeout: 15_000 });
  });

  test("attaches code and notebook files: extension beats a blank MIME type", async ({
    page,
  }) => {
    // Windows browsers report no MIME type for .py/.ipynb; the extension
    // decides (v6.1.1). octet-stream here simulates the worst case.
    const notebook = {
      nbformat: 4,
      nbformat_minor: 5,
      metadata: { kernelspec: { language: "python" } },
      cells: [
        {
          cell_type: "code",
          metadata: {},
          execution_count: 1,
          source: "print('KUMQUAT')",
          outputs: [],
        },
      ],
    };
    await page.goto("/ask-anything");
    await expect(page.getByLabel("Your message")).toBeVisible();
    await page.locator("#chat-attach-input").setInputFiles([
      {
        name: "clean.py",
        mimeType: "application/octet-stream",
        buffer: Buffer.from("anchovy_constant = 1\n"),
      },
      {
        name: "analysis.ipynb",
        mimeType: "application/octet-stream",
        buffer: Buffer.from(JSON.stringify(notebook)),
      },
    ]);
    const chips = page.getByRole("list", { name: "Files to send" });
    await expect(chips).toContainText("clean.py");
    await expect(chips).toContainText("analysis.ipynb");
    // The notebook chip reports the extraction, not just "text".
    await expect(chips).toContainText("1 cell, python", { timeout: 15_000 });

    await page.getByLabel("Your message").fill("Review my code files");
    await page.getByRole("button", { name: "Send message" }).click();
    await expect(page.getByText(/FILE_ACK/)).toBeVisible({ timeout: 15_000 });
    // The model saw the .py contents and the notebook's extracted cell. The
    // echo renders as markdown across elements, so scope to the whole reply.
    const reply = page
      .getByRole("article", { name: "ChatISA" })
      .filter({ hasText: "FILE_ACK" });
    await expect(reply).toContainText("anchovy_constant");
    await expect(reply).toContainText("KUMQUAT");
  });

  test("attaches a PDF natively: payload kept out of localStorage, survives reload", async ({
    page,
  }) => {
    await page.goto("/ask-anything");
    await expect(page.getByLabel("Your message")).toBeVisible();
    await attach(page, {
      name: "chapter.pdf",
      mimeType: "application/pdf",
      buffer: TINY_PDF,
    });
    const chips = page.getByRole("list", { name: "Files to send" });
    await expect(chips).toContainText(/PDF/, { timeout: 15_000 });
    await page.getByLabel("Your message").fill("Summarize the attached chapter");
    await page.getByRole("button", { name: "Send message" }).click();

    await expect(page.getByText("PDF_ACK")).toBeVisible({ timeout: 15_000 });

    // ADR-022 economics: the chat record holds an aa-file reference, never the
    // base64 payload (which would blow the localStorage quota).
    const stored = await page.evaluate(() =>
      window.localStorage.getItem("aa-chats-v1"),
    );
    expect(stored).toContain("aa-file:");
    expect(stored).not.toContain("data:application/pdf");

    // Reload: reopen the chat; the chip rehydrates from IndexedDB.
    await page.reload();
    await openSidebarIfMobile(page);
    await page
      .getByRole("navigation", { name: "Your chats" })
      .getByText("Summarize the attached chapter")
      .click();
    await expect(
      page.getByRole("article", { name: "You" }).getByText("chapter.pdf"),
    ).toBeVisible({ timeout: 15_000 });
  });

  test("attaches an image: thumbnail in the bubble, model sees it", async ({
    page,
  }) => {
    await page.goto("/ask-anything");
    await expect(page.getByLabel("Your message")).toBeVisible();
    await attach(page, {
      name: "plot.png",
      mimeType: "image/png",
      buffer: TINY_PNG,
    });
    await expect(
      page.getByRole("list", { name: "Files to send" }),
    ).toContainText("image", { timeout: 15_000 });
    await page.getByLabel("Your message").fill("What is in this picture?");
    await page.getByRole("button", { name: "Send message" }).click();

    await expect(
      page.getByRole("img", { name: "Attached image: plot.png" }),
    ).toBeVisible();
    await expect(page.getByText("IMAGE_ACK")).toBeVisible({ timeout: 15_000 });
  });

  test("attaches a csv into the Python session and the model computes on it", async ({
    page,
  }) => {
    // The dataset import cold-loads the real Pyodide worker at attach time.
    test.setTimeout(180_000);
    await page.goto("/ask-anything");
    await expect(page.getByLabel("Your message")).toBeVisible();
    await attach(page, {
      name: "sales.csv",
      mimeType: "text/csv",
      buffer: Buffer.from(
        "region,month,revenue\nEast,Jan,100\nWest,Jan,90\nEast,Feb,120\n",
      ),
    });
    const chips = page.getByRole("list", { name: "Files to send" });
    await expect(chips).toContainText("loaded as sales", { timeout: 150_000 });
    await page.getByLabel("Your message").fill("Please describe the dataset");
    await page.getByRole("button", { name: "Send message" }).click();

    // The scripted model runs print(sales.shape) on the SAME session the
    // import went into; (3, 3) proves the DataFrame really landed.
    const reply = page.getByRole("article", { name: "ChatISA" }).first();
    const card = reply.locator("details").first();
    await expect(card.locator("summary")).toContainText(/Ran Python/, {
      timeout: 60_000,
    });
    await card.locator("summary").click();
    await expect(card).toContainText("sales.shape");
    await expect(card).toContainText("(3, 3)");
    await expect(page.getByText(/RESULT_ACK/)).toBeVisible({ timeout: 30_000 });
  });

  test("searches the literature and renders linked results", async ({ page }) => {
    await page.goto("/ask-anything");
    await expect(page.getByLabel("Your message")).toBeVisible();
    await page
      .getByLabel("Your message")
      .fill("Please find papers about conformal prediction");
    await page.getByRole("button", { name: "Send message" }).click();

    // search_papers executes on the server (fixtures in mock mode) and the
    // model continues within the same streamed response.
    const reply = page.getByRole("article", { name: "ChatISA" }).first();
    const card = reply.locator("details").first();
    await expect(card.locator("summary")).toContainText(
      /Searched the literature \(2 papers\)/,
      { timeout: 30_000 },
    );
    await card.locator("summary").click();
    await expect(
      card.getByRole("link", { name: /Conformal Prediction Methods/ }),
    ).toBeVisible();
    await expect(page.getByText(/RESULT_ACK/)).toBeVisible({ timeout: 30_000 });
  });

  test("fetches Miami style assets on request", async ({ page }) => {
    await page.goto("/ask-anything");
    await expect(page.getByLabel("Your message")).toBeVisible();
    await page
      .getByLabel("Your message")
      .fill("Draw me a figure in the Miami style");
    await page.getByRole("button", { name: "Send message" }).click();

    const reply = page.getByRole("article", { name: "ChatISA" }).first();
    const card = reply.locator("details").first();
    await expect(card.locator("summary")).toContainText(/Fetched Miami's style/, {
      timeout: 30_000,
    });
    await card.locator("summary").click();
    await expect(card).toContainText("miamired");
    await expect(page.getByText(/RESULT_ACK/)).toBeVisible({ timeout: 30_000 });
  });

  test("scrapes a live no-CORS site through the agentic loop", async ({
    page,
  }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs live network; opt in with CHATISA_LIVE_NET=1",
    );
    // Scripted model turn, REAL everything else: Pyodide, the requests
    // adapter, /api/py-proxy fetching miamioh.edu, bs4 with auto-loaded lxml,
    // and the result flowing back for the model's continuation.
    test.setTimeout(300_000);
    await page.goto("/ask-anything");
    await page
      .getByLabel("Your message")
      .fill("Please scrape the FSB directory for me");
    await page.getByRole("button", { name: "Send message" }).click();

    const reply = page.getByRole("article", { name: "ChatISA" }).first();
    const card = reply.locator("details").first();
    await expect(card.locator("summary")).toContainText(/Ran Python/, {
      timeout: 260_000,
    });
    await card.locator("summary").click();
    await expect(card).toContainText(/FSB_ROWS [1-9]\d*/);
    // The model saw the real row count and acknowledged it.
    const ack = page.getByText(/RESULT_ACK/);
    await expect(ack).toBeVisible({ timeout: 30_000 });
    await expect(ack).toContainText(/FSB_ROWS/);
  });

  test("hosted deck generation: provider card, disclosure, and download", async ({
    page,
  }) => {
    // The professor's real-world failure (2026-07-24): a deck requested as the
    // SECOND message of a chat, with the tool input streamed as a rapid delta
    // burst, blew React's update depth. The mock streams the same burst; this
    // test guards the whole flow plus the absence of console errors.
    const consoleErrors: string[] = [];
    page.on("console", (msg) => {
      if (msg.type() === "error") consoleErrors.push(msg.text());
    });

    await page.goto("/ask-anything");
    await expect(page.getByLabel("Your message")).toBeVisible();
    await page.getByLabel("Your message").fill("Hello there, quick question");
    await page.getByRole("button", { name: "Send message" }).click();
    await expect(page.getByRole("article", { name: "ChatISA" })).toContainText(
      "SELECT 1 AS n",
      { timeout: 15_000 },
    );

    await page
      .getByLabel("Your message")
      .fill("Please make me a PowerPoint deck for my project pitch");
    await page.getByRole("button", { name: "Send message" }).click();

    // The provider-executed run streams call, result, and continuation in one
    // response; the card names whose servers did the work. The deck reply is
    // the SECOND assistant message in this chat.
    const reply = page.getByRole("article", { name: "ChatISA" }).last();
    const card = reply.locator("details").first();
    await expect(card.locator("summary")).toContainText(
      /Ran on Anthropic's servers/,
      { timeout: 30_000 },
    );
    // The deck is the deliverable, so its download sits OUTSIDE the disclosure
    // and is visible before anything is clicked (the 2026-07-25 report: the
    // download was hidden inside "Ran on Anthropic's servers"). The link names
    // the real file, fetched from the route's ?meta=1 view.
    const link = reply.getByRole("link", { name: /Download/ });
    await expect(link).toBeVisible();
    await expect(link).toHaveText(/mock-deck\.pptx/);
    // Still collapsed: the download did not need it opened.
    expect(await card.evaluate((el) => (el as HTMLDetailsElement).open)).toBe(
      false,
    );

    // Opening the card still explains where the run happened.
    await card.locator("summary").click();
    await expect(card).toContainText("Anthropic's hosted sandbox");
    await expect(card).toContainText("miami_template_by_fadel_megahed.pptx");
    await expect(card).toContainText("Saved miami-deck.pptx");

    const href = await link.getAttribute("href");
    expect(href).toContain("/api/ask-anything/files/anthropic/file_mockdeck1");
    const download = await page.request.get(href!);
    expect(download.status()).toBe(200);
    expect(download.headers()["content-disposition"]).toContain("mock-deck.pptx");
    expect((await download.body()).length).toBeGreaterThan(0);

    // The model's continuation disclosed where it ran.
    await expect(page.getByText(/DECK_ACK/)).toBeVisible({ timeout: 15_000 });

    // No update-depth (or any other) console errors under burst streaming.
    expect(
      consoleErrors.filter((e) => e.includes("Maximum update depth")),
    ).toEqual([]);
  });

  test("rejects an unsupported or oversized file with a readable chip", async ({
    page,
  }) => {
    await page.goto("/ask-anything");
    await expect(page.getByLabel("Your message")).toBeVisible();
    await attach(page, {
      name: "song.mp3",
      mimeType: "audio/mpeg",
      buffer: Buffer.from("not really audio"),
    });
    const chips = page.getByRole("list", { name: "Files to send" });
    await expect(chips).toContainText("isn't a supported file type");
    // A failed chip alone must not enable sending.
    await expect(
      page.getByRole("button", { name: "Send message" }),
    ).toBeDisabled();
  });

  test("is axe-clean with attachment chips in the composer", async ({ page }) => {
    await page.goto("/ask-anything");
    await expect(page.getByLabel("Your message")).toBeVisible();
    await attach(page, {
      name: "notes.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("axe pass file"),
    });
    await expect(
      page.getByRole("list", { name: "Files to send" }),
    ).toContainText("notes.txt");
    const axe = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);
  });

  test("is axe-clean, including the sidebar", async ({ page }) => {
    await page.goto("/ask-anything");
    await page.getByLabel("Your message").fill("Accessibility pass");
    await page.getByRole("button", { name: "Send message" }).click();
    await expect(page.getByRole("article", { name: "ChatISA" })).toBeVisible({
      timeout: 15_000,
    });
    await openSidebarIfMobile(page);
    const axe = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);
  });
});
