import {
  assertLiveServer,
  codeBlocks,
  lastReply,
  runLastBlock,
  runnableFigures,
  sendAndSettle,
  test,
  expect,
  type RunLang,
  type RunOutcome,
} from "./support/live";
import { chartStyleProblems, inspectChartCode } from "./support/chart-check";
import type { Page } from "@playwright/test";
import type { Observer } from "./support/observe";

/**
 * Coding Tutor, driven by a real model on the three tasks the professor
 * specified (2026-07-26).
 *
 * The point is not that the tutor replies. It is that the code it hands a
 * student actually RUNS, in this app's own runtimes, and follows the house chart
 * style that was added to this module's prompt the same day. So every block the
 * model emits is executed through the real "Run" button, and a failure is pasted
 * back to the tutor the way a student would, up to a bounded number of attempts.
 *
 * Assertions are on artifacts, never on wording: whether a plot appeared,
 * whether the palette is ours, whether the maths rendered, whether the scrape
 * produced the four columns asked for. A live spec that asserted phrasing would
 * be measuring our guess about the model.
 */

/** Attempts allowed per task. Enough to recover from a genuine mistake, few
 * enough that a model looping on one bad idea ends the test instead of the
 * budget. */
const MAX_ATTEMPTS = 4;

/**
 * Opens a task with the professor's wording, then gets past the scoping
 * question.
 *
 * The Coding Tutor's first reply is ALWAYS an introduction plus one question:
 * its prompt is the legacy Socratic text, "Only ask one question at a time. Ask
 * them about the subject title and topic they want to learn about. Wait for
 * their response." So no first answer to this module ever contains code or
 * maths, and a spec that asserts on it is asserting on the wrong turn. (My first
 * version of the matrix test did exactly that and reported "no LaTeX rendered"
 * against a reply that was only a greeting.)
 *
 * Answering it is what a student does, so the helper answers it, once, with the
 * level context the tutor asked for.
 */
async function askTutor(
  page: Page,
  observe: Observer,
  prompt: string,
): Promise<void> {
  await sendAndSettle(page, prompt, { timeoutMs: 6 * 60_000 });
  observe.note("first reply received");

  const reply = await lastReply(page);
  const hasCode = (await runnableFigures(page).count()) > 0;
  const asksSomething = /\?\s*$/.test(reply.trim()) || /\?/.test(reply.slice(-400));
  if (hasCode || !asksSomething) return;

  observe.note("the tutor asked a scoping question first; answering it");
  await sendAndSettle(
    page,
    "I am a senior business analytics student. I am comfortable with the basics, " +
      "so please go ahead with the full explanation and give me complete, runnable code.",
    { timeoutMs: 6 * 60_000 },
  );
  observe.note("substantive reply received");
}

interface Attempt {
  attempt: number;
  ok: boolean;
  output: string;
  hasPlot: boolean;
  code: string;
}

/**
 * Runs the newest block of a language, and on failure hands the error back to
 * the tutor and tries again. Returns every attempt, so the report shows what it
 * took rather than only the final state.
 */
async function runUntilItWorks(
  page: Page,
  observe: Observer,
  language: RunLang,
  opts: {
    needPlot?: boolean;
    nudge?: string;
    /**
     * An extra goal beyond "it ran". When it returns a reason, the run counts as
     * unsatisfied and the reason plus the actual output is sent back to the
     * tutor. This is what makes the loop pursue the TASK rather than merely a
     * clean exit code: a scrape that runs and prints package-loading noise has
     * not produced a CSV.
     */
    goal?: (result: RunOutcome) => string | null;
  } = {},
): Promise<Attempt[]> {
  const attempts: Attempt[] = [];

  for (let n = 1; n <= MAX_ATTEMPTS; n += 1) {
    const before = await runnableFigures(page, language).count();
    if (before === 0) {
      observe.note(`attempt ${n}: no ${language} block was offered`);
      await sendAndSettle(
        page,
        `Please give me the complete ${language === "r" ? "R" : "Python"} code in a single code block so I can run it.`,
      );
      continue;
    }

    const result: RunOutcome = await runLastBlock(page, language, {
      timeoutMs: 6 * 60_000,
    });
    const blocks = await codeBlocks(page);
    const code =
      blocks.filter((b) => b.language === language).at(-1)?.code ??
      blocks.at(-1)?.code ??
      "";

    attempts.push({ attempt: n, ...result, code });
    observe.note(
      `attempt ${n}: ok=${result.ok} plot=${result.hasPlot} ${result.output.slice(0, 120).replace(/\s+/g, " ")}`,
    );

    const goalMiss = result.ok ? (opts.goal?.(result) ?? null) : null;
    const satisfied =
      result.ok && (!opts.needPlot || result.hasPlot) && goalMiss === null;
    if (satisfied) return attempts;
    if (n === MAX_ATTEMPTS) return attempts;

    // Exactly what a student would send back: the error, or the output plus
    // what is still missing.
    let complaint: string;
    if (!result.ok) {
      complaint = `That code failed when I ran it. Here is the error:\n\n${result.output.slice(0, 1_500)}\n\nPlease fix it and give me the full corrected code.`;
    } else if (opts.needPlot && !result.hasPlot) {
      complaint = `It ran but produced no plot. ${opts.nudge ?? "Please give me code that actually draws the figure."}`;
    } else {
      complaint = `I ran it. ${goalMiss}\n\nHere is exactly what it printed:\n\n${result.output.slice(0, 2_000)}\n\nPlease use this to finish the job and give me the complete corrected code in one block.`;
    }
    await sendAndSettle(page, complaint, { timeoutMs: 6 * 60_000 });
  }

  return attempts;
}

test.describe("Coding Tutor, live", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/coding-tutor");
    await assertLiveServer(page);
  });

  test("(a) plots the AirPassengers trend in the house style", async ({
    page,
    observe,
  }) => {
    // The professor's wording, verbatim. Note what it asks for: months OR
    // seasons. Months is twelve categories, and the house style says nine or
    // more must NOT be coloured (group, facet, or rank instead), while seasons
    // is four and gets the Miami palette plus a secondary encoding. So this
    // prompt is also a test of whether the model applies the escalation rule
    // rather than reaching for a twelve-colour ramp.
    await askTutor(
      page,
      observe,
      "Plot the trend and color code the months/seasons in the airpassengers dataset.",
    );

    const attempts = await runUntilItWorks(page, observe, "r", {
      needPlot: true,
      nudge:
        "Please give me R code that draws the plot, and make sure the last expression prints the plot object.",
    });

    const final = attempts.at(-1);
    expect(final, "the tutor never produced a runnable R block").toBeDefined();

    await observe.save("a-airpassengers.R", final?.code ?? "");
    await observe.save("a-transcript.md", await lastReply(page));
    await observe.save(
      "a-attempts.json",
      JSON.stringify(
        attempts.map((a) => ({ ...a, code: a.code.slice(0, 4_000) })),
        null,
        2,
      ),
    );

    const findings = inspectChartCode(final?.code ?? "");
    const problems = chartStyleProblems(findings);
    observe.note(`title: ${findings.title ?? "(none)"}`);
    observe.note(`palette: ${findings.miamiHexes.join(", ") || "(none)"}`);
    observe.note(`style problems: ${problems.join("; ") || "none"}`);
    await observe.save(
      "a-chart-findings.json",
      JSON.stringify({ findings, problems }, null, 2),
    );

    // The app's part of the contract: the code runs and a plot comes back.
    expect(final?.ok, `R failed: ${final?.output}`).toBe(true);
    expect(final?.hasPlot, "no plot image was produced").toBe(true);

    // The prompt's part: the chart is in the house style. This is the check that
    // the chart rules added to this module's prompt on 2026-07-26 actually bite.
    expect(problems, `house style problems: ${problems.join("; ")}`).toEqual([]);

    // A title that merely restates the variables is the failure the style rules
    // call out by name, so the title is recorded above for a human and only its
    // presence and length are asserted here.
    expect((findings.title ?? "").length).toBeGreaterThan(15);
  });

  test("(b) explains matrix multiplication with rendered maths and runnable code", async ({
    page,
    observe,
  }) => {
    await askTutor(
      page,
      observe,
      "Explain to me the basics of matrix multiplication, its role in ML/DL, and how we can perform dot products in both R and Python with a simple example dataset.",
    );

    const reply = await lastReply(page);
    await observe.save("b-transcript.md", reply);

    // KaTeX must have rendered. The two failures worth catching are opposite:
    // no maths at all, and maths that leaked as raw source next to the prose.
    const katex = await page.locator(".katex").count();
    observe.note(`katex nodes: ${katex}`);
    expect(katex, "no LaTeX was rendered anywhere in the answer").toBeGreaterThan(0);

    // Raw delimiters surviving into the visible text means the delimiter
    // normalisation (lib/chat/math.ts) missed this model's dialect.
    const rawMath = /\\\[|\\\(|\$\$/.test(reply);
    observe.note(`raw LaTeX delimiters visible: ${rawMath}`);
    await observe.save(
      "b-math.json",
      JSON.stringify({ katexNodes: katex, rawDelimitersVisible: rawMath }, null, 2),
    );
    expect(rawMath, "LaTeX delimiters leaked into the rendered text").toBe(false);

    // Both languages must be offered, and both must run. The module's whole
    // premise is R and Python side by side.
    for (const language of ["r", "python"] as RunLang[]) {
      const attempts = await runUntilItWorks(page, observe, language);
      const final = attempts.at(-1);
      await observe.save(
        `b-${language}${language === "r" ? ".R" : ".py"}`,
        final?.code ?? "",
      );
      expect(
        final?.ok,
        `${language} failed after ${attempts.length} attempts: ${final?.output}`,
      ).toBe(true);
      // A dot product that printed nothing has not shown the student anything.
      expect(final?.output.length, `${language} produced no output`).toBeGreaterThan(0);
    }
  });

  test("(c) teaches CSS selectors and scrapes the ISA faculty table to CSV", async ({
    page,
    observe,
  }) => {
    const url =
      "https://miamioh.edu/fsb/directory/?up=/query/all/all/Information_Systems_and_Analytics/all";

    await askTutor(
      page,
      observe,
      `Help me understand how to select the correct CSS selectors and scrape the information within the ISA faculty list table in ${url}. I want to have a CSV with Department, Faculty Name, Faculty Position, Faculty Webpage.`,
    );
    await observe.save("c-transcript.md", await lastReply(page));

    // R first: rvest is the module's house tool for scraping, and this is the
    // exact path that was broken until the 2026-07-26 isolation fix.
    //
    // The goal is the CSV, not a clean exit. The first live run of this task
    // "succeeded" while printing nothing but package-loading noise, because the
    // model had guessed selectors that matched no elements. So the goal check
    // looks for the four requested columns AND a real profile URL, and anything
    // short of that goes back to the tutor with the actual output attached, the
    // way a student would reply.
    // Note on what can and cannot be asserted from the printed output: R prints
    // a tibble truncated to the console width, so a URL column shows as
    // "http://www.fsb.m…" and the full address never appears. An earlier version
    // of this check demanded the whole domain and so failed a scrape that was
    // completely correct, which is a test bug of the worst kind: it reports
    // working software as broken. What is checked instead is that a URL-shaped
    // value reached the column at all, with the href extraction confirmed
    // separately against the code.
    const wantsColumns = (result: RunOutcome): string | null => {
      const out = result.output;
      const missing: string[] = [];
      if (!/department/i.test(out)) missing.push("Department");
      if (!/name/i.test(out)) missing.push("Faculty Name");
      if (!/position|title|rank/i.test(out)) missing.push("Faculty Position");
      if (!/https?:\/\//i.test(out)) missing.push("Faculty Webpage (a real link)");
      // A table with a header and no rows still matches every pattern above, so
      // the row count is what proves data was actually extracted.
      if (!/\b(?:[1-9]|[1-9]\d)\s+Information Systems/i.test(out)) {
        missing.push("any data rows (the selectors may match nothing on this page)");
      }
      return missing.length
        ? `It printed no ${missing.join(", ")}.`
        : null;
    };

    const attempts = await runUntilItWorks(page, observe, "r", {
      goal: wantsColumns,
    });
    const final = attempts.at(-1);
    await observe.save("c-scrape.R", final?.code ?? "");
    await observe.save("c-output.txt", final?.output ?? "");
    await observe.save(
      "c-attempts.json",
      JSON.stringify(
        attempts.map((a) => ({ ...a, code: a.code.slice(0, 4_000) })),
        null,
        2,
      ),
    );

    expect(
      final?.output,
      "R has no network access: the cross-origin isolation headers are missing again",
    ).not.toContain("cannot open the connection");
    expect(
      final?.ok,
      `the scrape failed after ${attempts.length} attempts: ${final?.output}`,
    ).toBe(true);

    // The four columns the professor asked for. Matched loosely on purpose: the
    // model may name them Name or Faculty Name, Webpage or URL. What must be
    // there is the DATA, so a real faculty member and a real profile URL are
    // required too, which no amount of plausible column naming can fake.
    const output = final?.output ?? "";
    const columnSignals = [
      /department/i,
      /name/i,
      /position|title|rank/i,
      /webpage|url|link|profile|href/i,
    ];
    const missing = columnSignals.filter((p) => !p.test(output)).map((p) => p.source);
    observe.note(`missing column signals: ${missing.join(", ") || "none"}`);
    expect(missing, `output lacked columns: ${missing.join(", ")}`).toEqual([]);

    // Real scraped content, not a fabricated example. "Information Systems" is
    // the Department value on that page, and the row count proves rows were
    // found rather than an empty table printed.
    expect(output).toMatch(/Information Systems/i);
    expect(
      output,
      "the table printed no data rows, so the selectors matched nothing",
    ).toMatch(/\b(?:[1-9]|[1-9]\d)\s+Information Systems/i);

    // The webpage column is the one that cannot come from html_table(): it needs
    // the href off the anchor, which is the actual lesson the professor asked
    // for. Asserted against the CODE, because the printed tibble truncates a URL
    // column to the console width and never shows a full address.
    const code = final?.code ?? "";
    expect(
      code,
      "the code never reads an href, so the webpage column cannot be real",
    ).toMatch(/html_attr\s*\([^)]*href/);
    expect(output, "no URL reached the output at all").toMatch(/https?:\/\//i);
  });
});
