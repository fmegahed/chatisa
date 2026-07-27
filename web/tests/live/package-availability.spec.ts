import { assertLiveServer, test, expect } from "./support/live";
import type { Page } from "@playwright/test";

/**
 * Evidence gathering, before building anything: what ACTUALLY happens today when
 * a runnable block needs a package that is not bundled?
 *
 * Three cases matter and they need different handling:
 *   - bundled: works immediately.
 *   - installable but not bundled: must be fetched on first use. Both workers
 *     claim to do this (Pyodide via loadPackagesFromImports plus our hosted
 *     wheels, R via webr::shim_install), so the question is whether it works and
 *     how long it takes.
 *   - impossible: needs a native toolchain. Nothing can make this run, so
 *     offering a Run button is a promise the app cannot keep.
 *
 * Run against the workers directly, same protocol as lib/run/manager.ts, so the
 * answer is about the runtime rather than the UI.
 */

const WS_PROXY = "socks5h://test:yolo@ws.r-universe.dev:443";

interface Reply {
  ok?: boolean;
  error?: string;
  result?: { text?: string };
}

async function runOne(
  page: Page,
  workerUrl: string,
  code: string,
  extra: Record<string, unknown> = {},
): Promise<Reply & { seconds: number }> {
  return page.evaluate(
    async ({ workerUrl, code, extra }) => {
      const worker = new Worker(workerUrl, { type: "module" });
      const started = performance.now();
      try {
        const reply = await new Promise<Reply>((resolve) => {
          worker.addEventListener("message", (event: MessageEvent) => {
            if ((event.data as { id?: number })?.id === 1) resolve(event.data as Reply);
          });
          worker.postMessage({
            id: 1,
            code,
            keepState: false,
            withVariables: false,
            ...extra,
          });
        });
        return {
          ...reply,
          seconds: Math.round((performance.now() - started) / 100) / 10,
        };
      } finally {
        worker.terminate();
      }
    },
    { workerUrl, code, extra },
  );
}

test.describe("package availability today", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/coding-studio");
    await assertLiveServer(page);
  });

  test("Python: bundled, installable, and impossible", async ({ page, observe }) => {
    test.setTimeout(12 * 60_000);

    const cases: { label: string; code: string }[] = [
      // Bundled: in BUNDLED_PYTHON, loaded from our origin.
      { label: "bundled (pandas)", code: "import pandas as pd\nprint(pd.__version__)" },
      // Installable: in Pyodide's lock but not preloaded.
      { label: "installable (requests)", code: "import requests\nprint(requests.__name__)" },
      // A hosted wheel of ours, not in Pyodide's lock at all.
      { label: "hosted wheel (openpyxl)", code: "import openpyxl\nprint(openpyxl.__name__)" },
      // Impossible: needs compiling. KNOWN_UNAVAILABLE_PYTHON names it.
      { label: "impossible (statsforecast)", code: "import statsforecast" },
    ];

    const results = [];
    for (const c of cases) {
      const reply = await runOne(page, "/workers/pyodide-worker.mjs", c.code);
      const row = {
        label: c.label,
        ok: reply.ok,
        seconds: reply.seconds,
        text: (reply.result?.text ?? reply.error ?? "").slice(0, 300),
      };
      results.push(row);
      observe.note(`${c.label}: ok=${row.ok} in ${row.seconds}s :: ${row.text.slice(0, 120).replace(/\s+/g, " ")}`);
    }
    await observe.save("python-availability.json", JSON.stringify(results, null, 2));

    // The first three must work, or "install in the background" is already broken.
    expect(results[0].ok, "pandas is bundled and must run").toBe(true);
    expect(results[1].ok, "requests is installable and must be fetched on use").toBe(true);
    expect(results[2].ok, "openpyxl is a hosted wheel and must install").toBe(true);
    // The fourth cannot work. What matters is that the failure is legible, since
    // that is the text a Run-button gate would be replacing.
    expect(results[3].ok).toBe(false);
    observe.note(`impossible-package error text: ${results[3].text}`);
  });

  test("R: bundled, installable from webR's repo, and impossible", async ({
    page,
    observe,
  }) => {
    test.setTimeout(14 * 60_000);

    const cases: { label: string; code: string }[] = [
      // In our mirror, installed at session start.
      { label: "bundled (dplyr)", code: 'cat(class(dplyr::tibble(x=1))[1])' },
      // NOT in our mirror (checked: no zoo_*.tgz). Must come from webR's repo
      // over the public internet, which is the case worth measuring.
      { label: "not mirrored (zoo)", code: 'if(require(zoo)==FALSE) install.packages("zoo")\ncat(class(zoo::zoo(1:3)))' },
      // Not built for WebAssembly at all.
      { label: "impossible (rJava)", code: 'if(require(rJava)==FALSE) install.packages("rJava")\ncat("loaded")' },
    ];

    const results = [];
    for (const c of cases) {
      const reply = await runOne(page, "/workers/webr-worker.mjs", c.code, {
        wsProxy: WS_PROXY,
      });
      const row = {
        label: c.label,
        ok: reply.ok,
        seconds: reply.seconds,
        text: (reply.result?.text ?? reply.error ?? "").slice(0, 400),
      };
      results.push(row);
      observe.note(`${c.label}: ok=${row.ok} in ${row.seconds}s :: ${row.text.slice(0, 140).replace(/\s+/g, " ")}`);
    }
    await observe.save("r-availability.json", JSON.stringify(results, null, 2));

    expect(results[0].ok, "dplyr is bundled and must run").toBe(true);
    // The other two are RECORDED, not asserted: this test exists to find out
    // what happens, and both answers are informative. If zoo installs, the
    // background-install story already works for R; if it does not, that is the
    // gap to close.
    observe.note(`not-mirrored package installed: ${results[1].ok}`);
    observe.note(`impossible package failed as expected: ${results[2].ok === false}`);
  });
});
