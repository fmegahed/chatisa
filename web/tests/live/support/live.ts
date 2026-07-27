import { expect, test as base, type Page } from "@playwright/test";
import { Observer } from "./observe";

/**
 * The live-run fixture: an observed page, and a hard guard that the server under
 * test is real.
 */

export const test = base.extend<{ observe: Observer }>({
  // The fixture callback is named `provide`, not Playwright's conventional
  // `use`: the react-hooks lint rule reads a bare `use(...)` call as React's
  // `use` hook and fails the file. The name is arbitrary to Playwright.
  observe: async ({ page }, provide, info) => {
    const observer = new Observer(page, info);
    await provide(observer);
    await observer.writeReport();
  },
});

export { expect };

/**
 * Refuses to continue against a mock server.
 *
 * A live suite pointed at CHATISA_MOCK_LLM=1 passes everything and proves
 * nothing, and it looks exactly like a good run. The app already renders a
 * permanent banner in that state (components/MockModeBanner), so the check is
 * just to read the page rather than trust the operator's shell history.
 */
export async function assertLiveServer(page: Page): Promise<void> {
  const banner = page.getByRole("alert").filter({ hasText: /Test mode/i });
  await expect(
    banner,
    "This server is in CHATISA_MOCK_LLM mode. Live runs against it are meaningless. Restart the server without that flag.",
  ).toHaveCount(0);
}

/**
 * Sends a message in any of the chat-shaped modules and waits for the reply to
 * finish streaming.
 *
 * "Finished" is judged by the Send button becoming available again rather than by
 * a text pattern: a real model's wording is not predictable, and a live run that
 * waits for an expected phrase measures our guess about the model instead of the
 * model. The tool-driven modules can hold the button for minutes, hence the
 * generous default.
 */
export async function sendAndSettle(
  page: Page,
  message: string,
  opts: { timeoutMs?: number } = {},
): Promise<void> {
  const box = page.getByLabel("Your message");
  await box.fill(message);
  // "Send message" in the full-page modules, "Send" in the Coding Studio side
  // chat. Both label the textarea "Your message", so only the button differs, and
  // assuming the longer name made the Studio test hang for its whole 12 minute
  // budget on a control that does not exist there.
  const send = page
    .getByRole("button", { name: "Send message" })
    .or(page.getByRole("button", { name: "Send", exact: true }));
  await send.first().click();
  // The button first goes busy (Stop appears), then comes back. Waiting only for
  // "enabled" can pass instantly on the frame before the request is issued.
  const stop = page.getByRole("button", { name: /^Stop/ });
  await stop.waitFor({ state: "visible", timeout: 60_000 }).catch(() => {
    // Some modules answer fast enough that Stop never paints. Not a failure.
  });
  await expect(stop).toHaveCount(0, { timeout: opts.timeoutMs ?? 8 * 60_000 });

  // A provider failure renders an error panel and no assistant turn. Without
  // this check the next helper reports "no assistant reply on the page", which
  // sent me looking at the test instead of at the server log where the real
  // cause was (2026-07-26: Claude Opus 5 rejecting `temperature`). Surface it
  // here, where the cause is still obvious.
  const failure = page.getByRole("heading", { name: /That response failed/i });
  if (await failure.count()) {
    const panel = page.getByRole("alert").filter({ has: failure });
    const text = (await panel.first().innerText()).replace(/\s+/g, " ").trim();
    throw new Error(
      `the module reported a failed response: ${text}. Check the server log for the provider error.`,
    );
  }
}

/** The text of the most recent assistant turn. */
export async function lastReply(page: Page): Promise<string> {
  const replies = page.getByRole("article", { name: "ChatISA" });
  const count = await replies.count();
  expect(count, "no assistant reply on the page").toBeGreaterThan(0);
  return (await replies.nth(count - 1).innerText()).trim();
}

/** Every fenced code block in the conversation, with its language if labelled. */
export async function codeBlocks(
  page: Page,
): Promise<{ language: string | null; code: string }[]> {
  return page.evaluate(() => {
    const out: { language: string | null; code: string }[] = [];
    for (const pre of Array.from(document.querySelectorAll("pre"))) {
      const code = pre.querySelector("code");
      const text = (code ?? pre).textContent ?? "";
      if (!text.trim()) continue;
      const className = code?.className ?? "";
      const match = /language-([a-z0-9+#]+)/i.exec(className);
      out.push({ language: match ? match[1].toLowerCase() : null, code: text });
    }
    return out;
  });
}

/**
 * Runs a snippet through the module's own "Run" button and returns what the
 * student would see.
 *
 * Uses the real UI rather than calling the run manager directly, because the
 * bug found on 2026-07-26 lived in neither: it was in the page's response
 * headers, and only a real click on a real page can observe it.
 */
export interface RunOutcome {
  ok: boolean;
  output: string;
  hasPlot: boolean;
}

/** Label of the Run button for each runnable language, as the UI writes it. */
const RUN_LABEL = { r: "Run R", python: "Run Python", sql: "Run SQL" } as const;
export type RunLang = keyof typeof RUN_LABEL;

/** Figures carrying a Run button for one language, in document order. */
export function runnableFigures(page: Page, language?: RunLang) {
  const name = language ? RUN_LABEL[language] : /^Run /;
  return page
    .locator("figure")
    .filter({ has: page.getByRole("button", { name }) });
}

/**
 * Runs the LAST block of a language in the conversation.
 *
 * The last one, because these specs are multi-turn: when a run fails the error
 * is pasted back to the tutor, and what matters then is its newest attempt, not
 * the first one that already failed.
 */
export async function runLastBlock(
  page: Page,
  language: RunLang,
  opts: { timeoutMs?: number } = {},
): Promise<RunOutcome> {
  const figures = runnableFigures(page, language);
  const count = await figures.count();
  expect(count, `no runnable ${language} block on the page`).toBeGreaterThan(0);
  return runBlock(page, count - 1, { ...opts, language });
}

export async function runBlock(
  page: Page,
  index: number,
  opts: { timeoutMs?: number; language?: RunLang } = {},
): Promise<RunOutcome> {
  const figure = runnableFigures(page, opts.language).nth(index);
  await figure.scrollIntoViewIfNeeded();
  await figure.getByRole("button", { name: /^Run / }).click();

  const error = figure.getByRole("alert");
  const output = figure.getByText("Output", { exact: true });
  // Whichever lands first. The first run downloads a whole WASM runtime, and
  // the first R run then installs the bundled tidyverse set from our own mirror
  // before any code executes, which has been measured in minutes here.
  await expect(error.or(output).first()).toBeVisible({
    timeout: opts.timeoutMs ?? 5 * 60_000,
  });

  const failed = (await error.count()) > 0;
  // The panel is read by slicing the figure's own text rather than by locating
  // the container element. An earlier version used
  // `figure.locator("div").filter({ has: output })`, which silently matched
  // nothing: a `has:` locator built from `figure` gets re-rooted at each
  // candidate div, so the chain looks for a figure inside a div inside that
  // figure. The run had succeeded; only the reading of it timed out, which is
  // the worst kind of test bug because it reports a healthy feature as broken.
  const whole = (await figure.innerText()).trim();
  const marker = failed ? "Error" : "Output";
  const at = whole.lastIndexOf(marker);
  return {
    ok: !failed,
    output: at === -1 ? whole : whole.slice(at + marker.length).trim(),
    hasPlot: (await figure.locator("img").count()) > 0,
  };
}

/**
 * Switches the module to a named model, by its catalog display name (for
 * example "Claude Opus 5", "GPT-5.6 Sol").
 *
 * The chooser is collapsed by default and its radios are labelled with the
 * display name plus the description, so the label is matched loosely and the
 * selection is verified afterwards. Verifying matters: a silently missed click
 * would run the whole test on the default model while the report claimed
 * otherwise, which is the one failure mode that invalidates a comparison.
 */
export async function chooseModel(page: Page, displayName: string): Promise<void> {
  const open = page.getByRole("button", { name: "Choose a different model" });
  await open.click();
  const radio = page.getByRole("radio", { name: new RegExp(escapeRegExp(displayName)) });
  await radio.first().check();
  await expect(radio.first()).toBeChecked();
  // Collapse again so the composer is not pushed off screen.
  await open.click();
  await expect(
    page.getByText(displayName, { exact: false }).first(),
    `the module does not show ${displayName} as the selected model`,
  ).toBeVisible();
}

function escapeRegExp(text: string): string {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Tool cards in the current reply, as their visible summary lines. */
export async function toolCards(page: Page): Promise<string[]> {
  return page.locator("details > summary").allInnerTexts();
}

/**
 * The panel Ask Anything renders around files a hosted run really created.
 *
 * Scoped to that panel deliberately. A page-wide search for a link named
 * "Download ..." also matches links the MODEL writes in its prose: on
 * 2026-07-26 a run reported one created file when the only match was the
 * model's own markdown link reading "Download the 10-slide PowerPoint", and
 * clicking it produced no download because it was never a file of ours. The
 * question these tests ask is whether the APP produced a file, so the panel is
 * the only honest place to look.
 */
export function createdFilesPanel(page: Page) {
  return page
    .locator("div")
    .filter({ hasText: /^Files this run created/ })
    .last();
}

/** Names of files a hosted run produced and offered for download. */
export async function createdFiles(page: Page): Promise<string[]> {
  const links = createdFilesPanel(page).getByRole("link", { name: /^Download / });
  if ((await links.count()) === 0) return [];
  return (await links.allInnerTexts()).map((t) =>
    t.replace(/^Download\s+/, "").replace(/\(opens in a new tab\)/i, "").trim(),
  );
}

/** True when the page really has a SharedArrayBuffer, which is what R's and
 * Python's networking depend on. */
export async function isCrossOriginIsolated(page: Page): Promise<boolean> {
  return page.evaluate(
    () =>
      globalThis.crossOriginIsolated === true &&
      typeof SharedArrayBuffer !== "undefined",
  );
}
