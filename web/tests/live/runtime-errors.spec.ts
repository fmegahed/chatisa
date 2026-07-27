import { assertLiveServer, test, expect } from "./support/live";
import type { Page } from "@playwright/test";

/**
 * Does a failing run actually REPORT as a failure?
 *
 * Found while driving the Coding Tutor on 2026-07-26: a scrape whose pipeline
 * errored came back as a successful run whose output text happened to contain
 * "Error: In argument: ...". Three consequences, in increasing order of harm:
 *
 *   1. The student sees the neutral "Output" panel, not the red "Error" panel,
 *      so success and failure look the same.
 *   2. The message is announced through aria-live="polite" instead of
 *      role="alert", so a screen reader user is not told something went wrong.
 *   3. Nothing downstream can tell either. Our own multi-turn harness never
 *      pasted the error back to the tutor, so the model was never given the
 *      chance to fix its code.
 *
 * The professor's original production screenshot shows the same shape: "Output"
 * followed by "Error: cannot open the connection". So the reporting defect was
 * sitting in plain sight underneath the networking bug.
 *
 * These tests talk to the language workers directly, using the same message
 * protocol lib/run/manager.ts uses. That isolates the question (does the worker
 * classify this run as failed?) from the UI, and it is the layer a fix belongs
 * in.
 */

const WS_PROXY = "socks5h://test:yolo@ws.r-universe.dev:443";

interface WorkerReply {
  ok?: boolean;
  error?: string;
  result?: { text?: string; imageDataUrl?: string };
}

/**
 * Runs snippets through one worker instance, in order, reusing it so the R
 * package install is paid once.
 */
async function runInWorker(
  page: Page,
  workerUrl: string,
  snippets: string[],
  extra: Record<string, unknown> = {},
): Promise<WorkerReply[]> {
  return page.evaluate(
    async ({ workerUrl, snippets, extra }) => {
      const worker = new Worker(workerUrl, { type: "module" });
      const replies: WorkerReply[] = [];
      try {
        for (let i = 0; i < snippets.length; i += 1) {
          const id = i + 1;
          const reply = await new Promise<WorkerReply>((resolve) => {
            const onMessage = (event: MessageEvent) => {
              const data = event.data as { id?: number };
              if (data?.id !== id) return;
              worker.removeEventListener("message", onMessage);
              resolve(event.data as WorkerReply);
            };
            worker.addEventListener("message", onMessage);
            worker.postMessage({
              id,
              code: snippets[i],
              keepState: false,
              withVariables: false,
              ...extra,
            });
          });
          replies.push(reply);
        }
      } finally {
        worker.terminate();
      }
      return replies;
    },
    { workerUrl, snippets, extra },
  );
}

test.describe("runtime error reporting", () => {
  test.beforeEach(async ({ page }) => {
    // Any isolated page: the workers need the SharedArrayBuffer either way.
    await page.goto("/coding-studio");
    await assertLiveServer(page);
  });

  test("R reports a failing run as failed", async ({ page, observe }) => {
    test.setTimeout(10 * 60_000);

    const cases = [
      // A plain top-level stop(): the simplest possible R error.
      'stop("boom")',
      // A missing symbol, which is what a typo produces.
      "this_function_does_not_exist(1)",
      // The shape that surfaced the bug: an error raised inside a dplyr verb,
      // which arrives as an rlang condition rather than a base R one.
      'dplyr::if_else(character(0), "a", "b", missing = stop("inner"))',
      // Controls. These must stay successes AND keep their text, because the fix
      // works by turning condition capture ON, which takes message() and
      // warning() off the stderr stream and hands them over as R objects. Get
      // the rendering wrong and every "Loading required package: ..." line,
      // every dplyr masking note, and every warning disappears from the panel:
      // a quieter regression than the one being fixed, and easier to miss.
      'message("just a message"); cat("fine\\n")',
      'warning("careful"); cat("still ran\\n")',
      // Autoprint, which is how `1:10` shows its value at all.
      "1:3",
    ];

    const replies = await runInWorker(page, "/workers/webr-worker.mjs", cases, {
      wsProxy: WS_PROXY,
    });

    const summary = replies.map((r, i) => ({
      code: cases[i],
      ok: r.ok,
      error: r.error ?? null,
      text: r.result?.text ?? null,
    }));
    await observe.save("r-error-cases.json", JSON.stringify(summary, null, 2));
    for (const row of summary) {
      observe.note(
        `R ok=${row.ok} ${row.code.slice(0, 40)} -> ${(row.error ?? row.text ?? "").slice(0, 80).replace(/\s+/g, " ")}`,
      );
    }

    // The three failures must be reported as failures.
    for (const i of [0, 1, 2]) {
      expect(
        replies[i].ok,
        `R run "${cases[i]}" was reported as a SUCCESS. Its message: ${
          replies[i].result?.text ?? replies[i].error
        }`,
      ).toBe(false);
      expect(replies[i].error, `no error text for "${cases[i]}"`).toBeTruthy();
    }
    // And the controls must still be successes, with their text intact.
    expect(replies[3].ok, "a plain message() was misreported as a failure").toBe(true);
    expect(
      replies[3].result?.text ?? "",
      "message() text was dropped from the output",
    ).toContain("just a message");
    expect(replies[3].result?.text ?? "").toContain("fine");

    expect(replies[4].ok, "a warning() was misreported as a failure").toBe(true);
    expect(
      replies[4].result?.text ?? "",
      "warning() text was dropped from the output",
    ).toContain("careful");
    expect(replies[4].result?.text ?? "").toContain("still ran");

    expect(replies[5].ok).toBe(true);
    expect(
      replies[5].result?.text ?? "",
      "autoprint stopped showing a value",
    ).toContain("[1] 1 2 3");

    // The cleaned-up error text: no webR type tag, no internal eval() call.
    const firstError = replies[0].error ?? "";
    expect(firstError, "webR's one-letter type tag leaked to the student").not.toMatch(
      /^[A-Z]:\s/,
    );
    expect(firstError, "webR's internal call leaked to the student").not.toContain(
      "eval(ei, envir)",
    );
    expect(firstError).toContain("boom");
  });

  test("Python reports a failing run as failed", async ({ page, observe }) => {
    test.setTimeout(10 * 60_000);

    const cases = [
      'raise ValueError("boom")',
      "this_name_does_not_exist(1)",
      "print('fine')",
    ];
    const replies = await runInWorker(page, "/workers/pyodide-worker.mjs", cases);

    const summary = replies.map((r, i) => ({
      code: cases[i],
      ok: r.ok,
      error: r.error ?? null,
      text: r.result?.text ?? null,
    }));
    await observe.save("python-error-cases.json", JSON.stringify(summary, null, 2));
    for (const row of summary) {
      observe.note(`Python ok=${row.ok} ${row.code.slice(0, 40)}`);
    }

    // Python is the comparison case: if it already classifies correctly, the
    // defect is specific to the R worker rather than to the protocol.
    for (const i of [0, 1]) {
      expect(
        replies[i].ok,
        `Python run "${cases[i]}" was reported as a SUCCESS`,
      ).toBe(false);
    }
    expect(replies[2].ok).toBe(true);
  });
});
