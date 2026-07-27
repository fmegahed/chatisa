import { closeSync, mkdirSync, openSync, readSync, statSync } from "node:fs";
import { dirname } from "node:path";
import {
  assertLiveServer,
  chooseModel,
  createdFiles,
  createdFilesPanel,
  lastReply,
  sendAndSettle,
  toolCards,
  test,
  expect,
} from "./support/live";
import { inspectPptx, pptxProblems } from "./support/pptx";
import type { Page } from "@playwright/test";
import type { Observer } from "./support/observe";

/**
 * Ask Anything, one real model per task (professor's instruction, 2026-07-26:
 * "use a different model for each, record tool errors, failures, check the
 * output and find issues in it").
 *
 * A different model per test is not decoration. The two providers take different
 * hosted-sandbox paths (OpenAI code_interpreter with a container file, Anthropic
 * code_execution with a container upload), different download routes, and
 * different tool-call streaming shapes. Running all three tasks on one model
 * would exercise one of those and claim to have covered the module.
 *
 * These tests are written to REPORT rather than to police. A model legitimately
 * has several good ways to answer, and the useful output of a live run is the
 * evidence file. What is asserted is only what would be a defect on any path: a
 * tool card that says it failed, a promised file that does not exist or does not
 * open, a page error, or an answer built on data the model could not actually
 * obtain.
 */

/** The one thing no answer may do: claim numbers it never obtained. */
const FABRICATION_HINTS = [
  /illustrative (?:data|numbers|values)/i,
  /synthetic (?:data|dataset)/i,
  /made[- ]up (?:data|numbers)/i,
  /hypothetical (?:data|numbers)/i,
  /simulated data/i,
  /for demonstration purposes/i,
];

/**
 * Downloads every file the reply offers and records its real size and magic
 * bytes.
 *
 * Downloading matters because the failure seen on 2026-07-25 was a deck that
 * appeared as a working link and could not be opened: PowerPoint refused it
 * because each slide declared its layout twice. A test that only checks the link
 * exists would have passed that deck.
 */
async function downloadOffered(
  page: Page,
  observe: Observer,
  prefix: string,
): Promise<{ name: string; bytes: number; kind: string; path: string }[]> {
  // Only the app's own "Files this run created" panel. See createdFilesPanel:
  // a page-wide match also picks up download-shaped links the model writes in
  // prose, which are not files and do not download.
  const links = createdFilesPanel(page).getByRole("link", {
    name: /^Download /,
  });
  const count = await links.count();
  const out: { name: string; bytes: number; kind: string; path: string }[] = [];

  for (let i = 0; i < count; i += 1) {
    const link = links.nth(i);
    const name = (await link.innerText())
      .replace(/^Download\s+/, "")
      .replace(/\(opens in a new tab\)/i, "")
      .trim();
    const waitFor = page.waitForEvent("download", { timeout: 120_000 });
    await link.click();
    let download;
    try {
      download = await waitFor;
    } catch {
      observe.note(`download did not start for ${name}`);
      out.push({ name, bytes: -1, kind: "download never started", path: "" });
      continue;
    }
    const path = `tests/live/.artifacts/files/${prefix}-${name}`;
    mkdirSync(dirname(path), { recursive: true });
    await download.saveAs(path);

    const bytes = statSync(path).size;
    const head = Buffer.alloc(4);
    const fd = openSync(path, "r");
    readSync(fd, head, 0, 4, 0);
    closeSync(fd);
    // A pptx/xlsx/docx is a zip; anything else here is a red flag worth naming.
    const kind =
      head.subarray(0, 2).toString() === "PK"
        ? "zip (ooxml)"
        : `raw ${head.toString("hex")}`;

    observe.note(`downloaded ${name}: ${bytes} bytes, ${kind}`);
    out.push({ name, bytes, kind, path });
  }
  return out;
}

/** Tool cards whose summary says the run failed. */
function failedCards(cards: string[]): string[] {
  return cards.filter((c) => /\bfailed\b|could not be/i.test(c));
}

/**
 * Opens every downloaded deck and asserts it is structurally sound.
 *
 * The expected slide count is checked as a range, not a number: a model asked
 * for ten slides may reasonably deliver ten plus a title, and failing a good
 * deck over an off-by-one would be noise. What is NOT negotiable is the layout
 * wiring, because that is what made every deck unopenable on 2026-07-25.
 */
async function checkDecks(
  observe: Observer,
  files: { name: string; path: string }[],
  expected: { atLeast?: number } = {},
): Promise<void> {
  for (const file of files.filter((f) =>
    f.name.toLowerCase().endsWith(".pptx"),
  )) {
    const report = await inspectPptx(file.path);
    const problems = pptxProblems(report);
    observe.note(
      `${file.name}: ${report.slideCount} slides, ${report.imageCount} images, ` +
        `template theme=${report.usesTemplateTheme}, problems=${problems.length}`,
    );
    await observe.save(
      `pptx-${file.name}.json`,
      JSON.stringify({ report, problems }, null, 2),
    );

    expect(problems, `${file.name}: ${problems.join("; ")}`).toEqual([]);
    if (expected.atLeast !== undefined) {
      expect(
        report.slideCount,
        `${file.name} has ${report.slideCount} slides`,
      ).toBeGreaterThanOrEqual(expected.atLeast);
    }
    if (!report.usesTemplateTheme) {
      // Recorded rather than failed: the prompt requires opening the Miami
      // template, but a deck built from scratch is still a usable deck, and this
      // is the kind of judgement the professor should make from the artifact.
      observe.note(
        `${file.name} does NOT carry the Miami template theme: built from a blank presentation`,
      );
    }
  }
}

async function record(
  page: Page,
  observe: Observer,
  prefix: string,
): Promise<{ reply: string; cards: string[]; files: string[] }> {
  const reply = await lastReply(page);
  const cards = await toolCards(page);
  const files = await createdFiles(page);

  await observe.save(`${prefix}-answer.md`, reply);
  await observe.save(
    `${prefix}-tools.json`,
    JSON.stringify({ cards, failed: failedCards(cards), files }, null, 2),
  );
  observe.note(`tool cards: ${cards.length}, files offered: ${files.length}`);
  if (failedCards(cards).length) {
    observe.note(`FAILED CARDS: ${failedCards(cards).join(" | ")}`);
  }
  return { reply, cards, files };
}

test.describe("Ask Anything, live", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/ask-anything");
    await assertLiveServer(page);
  });

  test("(a) GPT-5.6 Sol: County Business Patterns regression, plot, and a 10-slide deck", async ({
    page,
    observe,
  }) => {
    await chooseModel(page, "GPT-5.6 Sol");
    observe.note("model: GPT-5.6 Sol (OpenAI code_interpreter path)");

    // Known before the run: cbp23st.zip is 11,115,845 bytes, and the browser
    // proxy refuses anything over PROXY_RESPONSE_MAX (4,000,000). So the
    // download CANNOT succeed in the browser runtime, and the hosted sandbox has
    // no network at all. This task is therefore a test of how the app behaves at
    // a hard limit: the acceptable outcomes are that the model finds a smaller
    // official slice, or that it tells the student plainly that the file is too
    // large. The unacceptable outcome is a regression reported on invented
    // numbers, which is what the fabrication check below looks for.
    await sendAndSettle(
      page,
      "Do a regression analysis on a dataset, explain the results, plot it, and make a 10 slide pptx about it. " +
        "This is the data: https://www2.census.gov/programs-surveys/cbp/datasets/2023/cbp23st.zip " +
        // State and industry, not county: cbp23st.zip is the STATE file, and the
        // county records are in cbp23co.zip. The professor's original wording said
        // county, the model caught the mismatch on the 2026-07-26 run, and the
        // professor corrected the example rather than the file.
        "These files contain establishments, employment, first-quarter payroll, and annual payroll by state and industry. " +
        "Select Ohio records using state FIPS code 39, then model annual payroll. " +
        "A possible regression is: Annual Payroll = b0 + b1(Employment) + b2(Establishments) + b3(Industry) + e",
      { timeoutMs: 14 * 60_000 },
    );

    const { reply, cards } = await record(page, observe, "a");
    const files = await downloadOffered(page, observe, "a");
    await observe.save("a-files.json", JSON.stringify(files, null, 2));

    // Did the proxy ceiling actually bite, and did the model handle it?
    const hitCeiling =
      /larger than 4 MB/i.test(reply) ||
      observe.httpErrors.some((e) => e.body.includes("larger than 4 MB")) ||
      /too large|size limit|4 ?MB/i.test(reply);
    observe.note(`hit the 4 MB proxy ceiling: ${hitCeiling}`);

    // Whatever route it took, it must not have quietly invented the data.
    const fabricated = FABRICATION_HINTS.filter((p) => p.test(reply)).map(
      (p) => p.source,
    );
    observe.note(`fabrication language: ${fabricated.join(", ") || "none"}`);
    await observe.save(
      "a-verdict.json",
      JSON.stringify({ hitCeiling, fabricated, cards }, null, 2),
    );

    expect(page.url()).toContain("/ask-anything");
    expect(observe.pageErrors, "the page threw").toEqual([]);
    expect(
      failedCards(cards),
      `tool runs reported failure: ${failedCards(cards).join(" | ")}`,
    ).toEqual([]);

    // If a deck was promised, it must exist and be a real OOXML package.
    const decks = files.filter((f) => f.name.toLowerCase().endsWith(".pptx"));
    if (decks.length === 0) {
      observe.note(
        "NO DECK PRODUCED: the 10-slide pptx was requested and not delivered",
      );
    }
    for (const deck of decks) {
      expect(deck.bytes, `${deck.name} is empty`).toBeGreaterThan(10_000);
      expect(deck.kind, `${deck.name} is not a zip container`).toBe(
        "zip (ooxml)",
      );
    }
    // Ten were asked for. Eight is close enough not to be a defect worth
    // failing; the exact count is in the report either way.
    await checkDecks(observe, decks, { atLeast: 8 });
  });

  test("(b) Claude Sonnet 5: scrape Fantasy Premier League and optimise a lineup", async ({
    page,
    observe,
  }) => {
    await chooseModel(page, "Claude Sonnet 5");
    observe.note("model: Claude Sonnet 5 (Anthropic code_execution path)");

    await sendAndSettle(
      page,
      "Help me scrape data, build an operations research model to optimize my fantasy Premier League lineup. " +
        "Then, create an image of the team with the 11 players in the starting lineup and the backups. " +
        "I want to use this to select my team. Look up the rules so that you can find the constraints, " +
        "define the objective function (maximize points), etc.",
      { timeoutMs: 14 * 60_000 },
    );

    const { reply, cards } = await record(page, observe, "b");
    const files = await downloadOffered(page, observe, "b");
    await observe.save("b-files.json", JSON.stringify(files, null, 2));

    // The three things this task needs, recorded separately so a partial answer
    // is legible rather than just "failed".
    const scraped = cards.some(
      (c) => /^(Python|R)$/.test(c.trim()) || /Read /.test(c),
    );
    const optimised =
      /pulp|scipy|linprog|milp|integer program|knapsack|optimi[sz]/i.test(
        reply,
      );
    const imaged =
      files.some((f) => /\.(png|jpg|jpeg|svg)$/i.test(f.name)) ||
      (await page.locator("img[alt*='Plot produced']").count()) > 0;
    observe.note(`scraped=${scraped} optimised=${optimised} image=${imaged}`);

    // FPL's own endpoint is public JSON and well under the proxy ceiling, so
    // unlike (a) there is no structural reason this cannot work end to end.
    const constraints = {
      budget: /100(\.0)?m|budget|£100/i.test(reply),
      squadSize: /15 players|squad of 15|11 starters|starting (xi|11)/i.test(
        reply,
      ),
      perClub: /three players|3 players (?:from|per)|per club/i.test(reply),
      formation: /formation|1 goalkeeper|at least 3 defenders/i.test(reply),
    };
    observe.note(`constraints stated: ${JSON.stringify(constraints)}`);
    await observe.save(
      "b-verdict.json",
      JSON.stringify(
        { scraped, optimised, imaged, constraints, cards },
        null,
        2,
      ),
    );

    expect(observe.pageErrors, "the page threw").toEqual([]);
    expect(
      failedCards(cards),
      `tool runs reported failure: ${failedCards(cards).join(" | ")}`,
    ).toEqual([]);
    // The optimisation is the substance of the request; an answer without it has
    // not done the task, whatever else it produced.
    expect(optimised, "no optimisation model appears in the answer").toBe(true);
    if (!imaged) {
      observe.note(
        "NO TEAM IMAGE PRODUCED: the request asked for one explicitly",
      );
    }
  });

  /**
   * The deck task, run on BOTH Anthropic models.
   *
   * Parameterised because the single-model version could not answer the question
   * it raised. Opus 5 failed here with `AI_MissingToolResultsError` after its
   * first hosted call, and the obvious next question was whether the hosted deck
   * path is broken generally or only on that model. The Fantasy Premier League
   * test could not settle it: Sonnet 5 answered that one entirely with browser
   * tools and never touched the hosted sandbox.
   *
   * The @ai-sdk/anthropic typings are the reason to suspect the model rather than
   * the path: `codeExecution_20260120` documents "Supported models: Claude Opus
   * 4.6, Sonnet 4.6, Sonnet 4.5, Opus 4.5", and neither of our 5-family ids is on
   * that list.
   */
  for (const model of ["Claude Opus 5", "Claude Sonnet 5"] as const) {
    test(`(c) ${model}: a World Cup economics deck`, async ({
      page,
      observe,
    }) => {
      await chooseModel(page, model);
      observe.note(`model: ${model} (Anthropic code_execution path)`);

      // The simplest of the three on purpose: no dataset, no scraping, no
      // optimisation. If a deck cannot be produced here, the deck path itself is
      // broken rather than the task being hard. It is also the direct regression
      // test for the two fixes made on 2026-07-25: container reuse across steps,
      // and the duplicate slideLayout relationship that made every deck unopenable.
      await sendAndSettle(
        page,
        "make a powerpoint deck on the impacts of the world cup on the U.S. economy",
        { timeoutMs: 14 * 60_000 },
      );

      const prefix = model === "Claude Opus 5" ? "c-opus" : "c-sonnet";
      const { cards } = await record(page, observe, prefix);
      const files = await downloadOffered(page, observe, prefix);
      await observe.save(
        `${prefix}-files.json`,
        JSON.stringify(files, null, 2),
      );

      expect(observe.pageErrors, "the page threw").toEqual([]);
      expect(
        failedCards(cards),
        `tool runs reported failure: ${failedCards(cards).join(" | ")}`,
      ).toEqual([]);

      const decks = files.filter((f) => f.name.toLowerCase().endsWith(".pptx"));
      expect(decks.length, "no .pptx was produced at all").toBeGreaterThan(0);
      for (const deck of decks) {
        expect(deck.bytes, `${deck.name} is empty`).toBeGreaterThan(10_000);
        expect(deck.kind, `${deck.name} is not a zip container`).toBe(
          "zip (ooxml)",
        );
      }
      // The direct regression test for the duplicate-slideLayout bug: every deck
      // this app produced was unopenable until 2026-07-25.
      await checkDecks(observe, decks, { atLeast: 3 });

      // One hosted session per turn is the documented contract. More than a couple
      // of hosted cards means the model is iterating the sandbox, which is what
      // ran for 15 minutes before the container-reuse fix.
      const hostedCards = cards.filter((c) => /servers/i.test(c));
      observe.note(`hosted sandbox cards: ${hostedCards.length}`);
      await observe.save(
        `${prefix}-verdict.json`,
        JSON.stringify({ model, hostedCards, decks }, null, 2),
      );
    });
  }
});
