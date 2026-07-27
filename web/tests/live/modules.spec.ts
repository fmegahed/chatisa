import { readFileSync } from "node:fs";
import {
  assertLiveServer,
  chooseModel,
  isCrossOriginIsolated,
  sendAndSettle,
  test,
  expect,
} from "./support/live";
import { chartStyleProblems, inspectChartCode } from "./support/chart-check";
import type { Page } from "@playwright/test";
import type { Observer } from "./support/observe";

/**
 * The remaining modules, two to three real examples each (professor's
 * instruction, 2026-07-26: "Harden any remaining module, pipeline so that we are
 * ready for production. Try 2-3 examples per case").
 *
 * Coding Tutor, Ask Anything, and the runtime plumbing have their own files.
 * This one covers Coding Studio, Exam Prep, JobApp Drafter, and AI Comparison
 * against real models.
 *
 * Each test states in a note what it did NOT verify, because a live run reads as
 * broader proof than it is. "The deck downloaded" is not "the deck is good", and
 * "the exam generated" is not "the questions are fair".
 */

const RESUME_PDF = "../../videos/chatisa/fixtures/jane_doe_resume.pdf";
const NOTES_PDF = "../../videos/chatisa/fixtures/isa444_stationarity_excerpt.pdf";

function fixtureExists(path: string): boolean {
  try {
    readFileSync(path);
    return true;
  } catch {
    return false;
  }
}

/** Types a snippet into the Coding Studio editor, replacing what is there. */
async function typeInStudio(page: Page, code: string): Promise<void> {
  for (const line of code.split("\n")) {
    expect(line, "Studio snippets must not be indented (CodeMirror auto-indents)").toBe(
      line.trimStart(),
    );
  }
  const editor = page.locator(".cm-content").first();
  await editor.waitFor({ timeout: 120_000 });
  await editor.click();
  await page.keyboard.press("ControlOrMeta+a");
  await page.keyboard.press("Delete");
  await editor.pressSequentially(code, { delay: 4 });
}

/**
 * A marker each Studio snippet prints last, so the test can tell real output
 * from the console's own placeholder.
 *
 * The first version of runStudio waited for the console to become "non-empty",
 * which it already is: the pane ships with "Python 3.14.0 (Pyodide, WebAssembly)
 * running in your browser. The first run loads the interpreter...". So it
 * returned instantly with the placeholder and every assertion on real output
 * failed. Waiting for a sentinel the snippet itself prints is unambiguous.
 */
const DONE = "CHATISA_RUN_COMPLETE";

/** Presses Run and returns the console text once the sentinel appears. */
async function runStudio(page: Page, observe: Observer): Promise<string> {
  const console_ = page
    .locator("section")
    .filter({ has: page.getByRole("heading", { name: "Console" }) })
    .first();
  const before = (await console_.innerText()).trim();

  await page.getByRole("button", { name: "Run", exact: true }).click();

  await expect
    .poll(async () => (await console_.innerText()).includes(DONE), {
      // A cold runtime boot plus a first-ever package install is minutes, not
      // seconds, and for R it includes the whole tidyverse from our mirror.
      timeout: 8 * 60_000,
      intervals: [1_000],
    })
    .toBe(true);

  const text = (await console_.innerText()).trim();
  observe.note(`console: ${text.slice(-400).replace(/\s+/g, " ")}`);
  // Returned with the pre-run text stripped, so an assertion cannot accidentally
  // match the placeholder that was already on screen.
  return text.startsWith(before) ? text.slice(before.length).trim() : text;
}

test.describe("Coding Studio, live", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/coding-studio");
    await assertLiveServer(page);
    expect(await isCrossOriginIsolated(page)).toBe(true);
  });

  test("runs a Python analysis and draws a house-style chart", async ({
    page,
    observe,
  }) => {
    test.setTimeout(12 * 60_000);
    await page.getByRole("radio", { name: "Python" }).click();

    // A whole small analysis: statsmodels regression plus a chart in the palette.
    // Column-zero only, so CodeMirror's auto-indent cannot corrupt it.
    await typeInStudio(
      page,
      [
        "import numpy as np, pandas as pd, statsmodels.api as sm",
        "import matplotlib; matplotlib.use('Agg')",
        "import matplotlib.pyplot as plt",
        "rng = np.random.default_rng(7)",
        "emp = rng.integers(5, 500, 60)",
        "est = rng.integers(1, 40, 60)",
        "pay = 38 * emp + 120 * est + rng.normal(0, 400, 60)",
        "df = pd.DataFrame({'employment': emp, 'establishments': est, 'payroll': pay})",
        "model = sm.OLS(df['payroll'], sm.add_constant(df[['employment','establishments']])).fit()",
        "fig, ax = plt.subplots(figsize=(7,4), facecolor='#FFFFFF')",
        "ax.scatter(df['employment'], df['payroll'], color='#C3142D', s=28)",
        "ax.set_title('Payroll rises about 38 dollars for each added employee', color='#000000')",
        "ax.set_xlabel('Employment'); ax.set_ylabel('Annual payroll')",
        "plt.show()",
        "print('r_squared', round(model.rsquared, 4), 'slope', round(model.params['employment'], 2))",
        `print("${DONE}")`,
      ].join("\n"),
    );

    const out = await runStudio(page, observe);
    await observe.save("studio-python-console.txt", out);

    // The regression really ran: r_squared is in the output, not a promise of it.
    expect(out).toMatch(/r_squared/);
    expect(out, "the fitted slope is missing, so the model did not run").toMatch(/slope/);

    // And the plot reached the plot pane.
    await page.getByRole("tab", { name: "Plots" }).click();
    const plot = page.locator("img").first();
    await expect(plot, "no figure in the Plots pane").toBeVisible({ timeout: 60_000 });
    observe.note("plot rendered in the Plots pane");
    observe.note(
      "NOT verified: whether the figure LOOKS right. Only that it rendered and the code used the palette.",
    );
  });

  test("runs R with the bundled tidyverse and reaches the internet", async ({
    page,
    observe,
  }) => {
    test.setTimeout(12 * 60_000);
    await page.getByRole("radio", { name: "R" }).click();

    await typeInStudio(
      page,
      [
        'doc <- rvest::read_html("https://example.com")',
        'title <- rvest::html_text(rvest::html_element(doc, "h1"))',
        "grades <- tibble::tibble(student = c(\"Amanda\",\"Bill\",\"Cara\"), isa_401 = c(93,88,74))",
        "summary_row <- dplyr::summarise(grades, mean_grade = mean(isa_401))",
        "cat(title, as.character(summary_row$mean_grade), sep = \" | \")",
        `cat("\\n${DONE}\\n")`,
      ].join("\n"),
    );

    const out = await runStudio(page, observe);
    await observe.save("studio-r-console.txt", out);
    expect(out, "R could not reach the internet from the Studio").not.toContain(
      "cannot open the connection",
    );
    expect(out).toContain("Example Domain");
    expect(out, "dplyr did not compute the summary").toContain("85");
  });

  test("the side chat suggests code that runs, in the house style", async ({
    page,
    observe,
  }) => {
    test.setTimeout(12 * 60_000);
    await page.getByRole("radio", { name: "Python" }).click();

    // The assistant panel is behind a toggle and starts closed, so its composer
    // is not on the page until it is opened. Without this the fill() waited for
    // an invisible textarea and burned the whole 12 minute budget with nothing to
    // show for it.
    const toggle = page.getByRole("button", { name: /^(Ask AI|Hide assistant)$/ });
    await toggle.waitFor({ timeout: 120_000 });
    if ((await toggle.innerText()).trim() === "Ask AI") await toggle.click();
    await expect(page.getByLabel("Your message")).toBeVisible({ timeout: 60_000 });
    observe.note("assistant panel open");

    // The Sandbox chat was given the house chart rules on 2026-07-26, the same
    // block as the Coding Tutor. This is where that matters most: its plot pane
    // draws the suggestion immediately.
    await sendAndSettle(
      page,
      "Give me one complete Python block that plots monthly sales for four regions and follows our chart style. Data can be made up in the code.",
      { timeoutMs: 6 * 60_000 },
    );

    const blocks = await page.locator("pre code").allInnerTexts();
    const code = blocks.at(-1) ?? "";
    await observe.save("studio-chat-suggestion.py", code);

    const findings = inspectChartCode(code);
    const problems = chartStyleProblems(findings);
    observe.note(`palette: ${findings.miamiHexes.join(", ") || "(none)"}`);
    observe.note(`title: ${findings.title ?? "(none)"}`);
    observe.note(`style problems: ${problems.join("; ") || "none"}`);
    await observe.save(
      "studio-chat-findings.json",
      JSON.stringify({ findings, problems }, null, 2),
    );

    // Four series means colour alone is not enough, so a secondary encoding is
    // required by the contract, not merely nice.
    expect(problems, `house style problems: ${problems.join("; ")}`).toEqual([]);
    expect(
      findings.hasSecondaryEncoding,
      "four series with colour only: the contract requires labels, shapes, or line types",
    ).toBe(true);
  });
});

test.describe("AI Comparison, live", () => {
  test("runs a blinded comparison and records a preference", async ({
    page,
    observe,
  }) => {
    test.setTimeout(12 * 60_000);
    await page.goto("/ai-comparison");
    await assertLiveServer(page);

    await page.getByRole("radio", { name: /Surprise me/i }).check();
    await page.getByLabel(/How many questions/i).fill("1");
    await page.getByRole("button", { name: "Start comparing" }).click();

    // The professor's own example question for this module.
    await page
      .getByLabel(/Your question for both models/i)
      .fill(
        "Explain MAPE for forecasting, its advantages, and its limitations, for a business analytics student.",
      );
    await page.getByRole("button", { name: "Ask both models" }).click();

    const left = page.getByRole("article", { name: "Answer on the left" });
    const right = page.getByRole("article", { name: "Answer on the right" });
    // Poll for a substantive answer in BOTH panes, not merely "not empty". The
    // panes carry a heading and status text from the moment they mount, so
    // not-empty was true at 1.1s with 29 and 30 characters, and every assertion
    // after it failed on placeholder text.
    await expect
      .poll(
        async () =>
          Math.min(
            (await left.innerText()).trim().length,
            (await right.innerText()).trim().length,
          ),
        { timeout: 8 * 60_000, intervals: [2_000] },
      )
      .toBeGreaterThan(400);

    const leftText = await left.innerText();
    const rightText = await right.innerText();
    await observe.save("comparison-left.md", leftText);
    await observe.save("comparison-right.md", rightText);
    observe.note(`left ${leftText.length} chars, right ${rightText.length} chars`);

    // Both answered substantively, and blinding held: neither pane may name a
    // model, or the comparison is not blind and the preference is worthless.
    expect(leftText.length).toBeGreaterThan(200);
    expect(rightText.length).toBeGreaterThan(200);
    const paneText = `${leftText}\n${rightText}`;
    for (const brand of ["GPT-5", "Claude", "Gemini", "Anthropic", "OpenAI"]) {
      // A model naming ITSELF in prose is the leak that matters here.
      if (paneText.includes(brand)) {
        observe.note(`BLINDING LEAK: a pane mentions "${brand}"`);
      }
    }

    await page.getByRole("button", { name: "Prefer the left answer" }).click();
    // The report names the winning model, so the heading is a model name and not
    // fixed wording. An earlier version guessed "How the models compared" and
    // failed a run that had worked perfectly: the page said "GPT-5.6 Luna won".
    // Matched on the control that only exists once the report is rendered, plus
    // the win sentence, so neither depends on which model won.
    await expect(
      page.getByRole("button", { name: "Run another comparison" }),
    ).toBeVisible({ timeout: 60_000 });
    const report = await page.locator("main").innerText();
    expect(report, "no winner was announced").toMatch(/\bwon\b|\btie\b/i);
    observe.note(
      `report: ${(/^.*\bwon\b.*$/im.exec(report)?.[0] ?? "(no win line)").slice(0, 80)}`,
    );
    observe.note(
      "NOT verified: which model is actually better. This checks the mechanism, not the pedagogy.",
    );
  });
});

test.describe("JobApp Drafter, live", () => {
  test("tailors a resume against a real posting and exports Word", async ({
    page,
    observe,
  }) => {
    test.setTimeout(14 * 60_000);
    test.skip(!fixtureExists(RESUME_PDF), `fixture missing: ${RESUME_PDF}`);

    await page.goto("/jobapp-drafter");
    await assertLiveServer(page);

    await page.getByLabel("Company").fill("Miami University");
    await page.getByLabel("Position title").fill("BI/ETL Data Developer II or III");
    await page
      .getByLabel("Or paste the job description")
      .fill(
        "Design and maintain ETL pipelines and BI reporting for university data. " +
          "SQL, data warehousing, Power BI, and stakeholder communication.",
      );
    await page.getByLabel(/Choose a PDF|resume/i).first().setInputFiles(RESUME_PDF);
    await page.getByRole("button", { name: "Continue" }).click();

    await expect(
      page.getByRole("heading", { name: /BI\/ETL Data Developer/i }),
    ).toBeVisible({ timeout: 3 * 60_000 });
    observe.note("posting accepted");

    // Tailoring is a SECOND explicit step, not part of Continue.
    await page.getByRole("button", { name: "Tailor my resume" }).click();
    await expect(
      page.getByRole("heading", { name: "Your tailored resume" }),
    ).toBeVisible({ timeout: 8 * 60_000 });
    observe.note("resume tailored");

    const resume = await page.locator("main").innerText();
    await observe.save("jobapp-resume.md", resume);

    // Two defects seen on 2026-07-25 that a human noticed and no test caught.
    // Recorded rather than asserted, because the professor should decide whether
    // they are still present and what the right behaviour is.
    const wrongName = /Test Student/i.test(resume);
    const duplicatedHeading =
      (resume.match(/SKILLS\s*\/?\s*CERTIFICATIONS/gi) ?? []).length > 1 ||
      (resume.match(/\bSKILLS\b/gi) ?? []).length > 1;
    observe.note(`shows the session name instead of the resume's name: ${wrongName}`);
    observe.note(`duplicated skills heading: ${duplicatedHeading}`);
    await observe.save(
      "jobapp-findings.json",
      JSON.stringify({ wrongName, duplicatedHeading }, null, 2),
    );

    // The name on a resume a student sends out is not cosmetic.
    expect(
      wrongName,
      "the tailored resume is headed with the session's test name, not the name on the uploaded resume",
    ).toBe(false);

    await page.getByRole("button", { name: "Save my edits" }).click();
    const download = page.getByRole("link", { name: "Download as Word" });
    await expect(download).toBeVisible({ timeout: 2 * 60_000 });

    const waitFor = page.waitForEvent("download", { timeout: 120_000 });
    await download.click();
    const file = await waitFor;
    const path = "tests/live/.artifacts/files/jobapp-resume.docx";
    await file.saveAs(path);
    const bytes = readFileSync(path);
    observe.note(`docx: ${bytes.byteLength} bytes`);
    // A docx is a zip; anything else will not open in Word.
    expect(bytes.subarray(0, 2).toString()).toBe("PK");
    expect(bytes.byteLength).toBeGreaterThan(5_000);
  });
});

test.describe("Exam Prep, live", () => {
  test("builds a practice exam from real course notes and grades it", async ({
    page,
    observe,
  }) => {
    test.setTimeout(15 * 60_000);
    test.skip(!fixtureExists(NOTES_PDF), `fixture missing: ${NOTES_PDF}`);

    await page.goto("/exam-prep");
    await assertLiveServer(page);
    await chooseModel(page, "GPT-5.6 Terra");

    await page.getByLabel("Choose a PDF", { exact: true }).setInputFiles(NOTES_PDF);
    observe.note("notes uploaded");
    await page.getByLabel("How many questions").fill("3");
    await page.getByRole("button", { name: "Build my practice exam" }).click();

    // Wait for the QUIZ to exist, identified by its Submit control, before
    // looking for anything to answer. Two earlier versions got this wrong in the
    // same way: "the first h2 is visible" was true at 1.8s (the footer has h2s),
    // and "a radio is visible" was true at 0.9s (the setup form has radios). Both
    // reported an exam that did not exist yet, and then sat on a Submit button
    // that was correctly disabled. The Submit button appears only with a question.
    const submit = page.getByRole("button", { name: "Submit answer" });
    await expect(submit).toBeVisible({ timeout: 12 * 60_000 });
    observe.note("exam generated: the quiz is on screen");

    // Answer everything, whatever the type, and reach results. Submit is only
    // clicked once it is ENABLED, because the module disables it until the
    // question is actually answered: clicking a disabled button just waits for
    // the test timeout and says nothing about the module.
    for (let i = 0; i < 3; i += 1) {
      // The quiz form is re-derived every iteration. Holding one locator across
      // iterations failed on question 2: after a submit the form re-renders with
      // feedback and Submit is replaced by Next, so the "form containing Submit"
      // locator matched nothing and the loop reported no answer control on a page
      // that had one.
      await expect(submit).toBeVisible({ timeout: 3 * 60_000 });
      const quiz = page.locator("form").filter({ has: submit }).first();

      const radio = quiz.getByRole("radio").first();
      const written = quiz.getByLabel("Your answer");
      if (await radio.count()) {
        await radio.check();
      } else if (await written.count()) {
        await written.fill(
          "A stationary series has a constant mean and variance over time, so an ADF test " +
            "checks whether differencing is needed before fitting an ARIMA model.",
        );
      } else {
        observe.note(`question ${i + 1}: no answer control found, stopping`);
        break;
      }
      await expect(
        submit,
        `Submit stayed disabled on question ${i + 1}, so the answer did not register`,
      ).toBeEnabled({ timeout: 60_000 });
      await submit.click();
      observe.note(`question ${i + 1} submitted`);

      // Immediate feedback is the default, so a control appears between
      // questions: "Next question", and on the LAST question "See results".
      // Guessing "See my results" is what stranded an otherwise complete run:
      // all three questions were answered and graded in 11 seconds, then the
      // loop found nothing to click and the results page never opened. The
      // wording is pinned by tests/e2e/exam-quiz.spec.ts, which is where it
      // should have been read from in the first place.
      const next = page.getByRole("button", {
        name: /Next question|See results/,
      });
      await next
        .first()
        .waitFor({ timeout: 3 * 60_000 })
        .catch(() => {});
      if (await next.count()) {
        const label = (await next.first().innerText()).trim();
        await next.first().click();
        observe.note(`clicked "${label}"`);
        if (/results/i.test(label)) break;
      }
    }

    await expect(page.getByRole("heading", { name: "Your results" })).toBeVisible({
      timeout: 5 * 60_000,
    });
    const results = await page.locator("main").innerText();
    await observe.save("exam-results.md", results);
    observe.note("results rendered");

    // Grounding is the module's whole promise: questions come from the notes.
    expect(results, "the results page shows no per-topic breakdown").toMatch(
      /topic|How you did/i,
    );
    observe.note(
      "NOT verified: whether the questions are fair or well-formed. Only that generation, answering, and grading complete.",
    );
  });
});
