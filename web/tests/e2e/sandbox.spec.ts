import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { readFileSync } from "node:fs";

/**
 * The Sandbox UI, without running code (executing a runtime loads WASM, which
 * is heavy and flaky in CI; real execution is verified manually). This covers
 * the four-pane shell, the language switch, and the theme toggle, with axe.
 */
test.describe("AI Sandbox", () => {
  /** The exact textContent of the nth rendered editor line (0-based), spaces kept. */
  async function editorLineText(
    page: import("@playwright/test").Page,
    n: number,
  ) {
    return page.locator(".cm-line").nth(n).evaluate((el) => el.textContent ?? "");
  }

  test("keeps each language's script across a reload, on the device (localStorage)", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    await expect(page.locator(".cm-content")).toBeVisible();

    // Python is the default language: replace its script with a marker.
    await page.locator(".cm-content").click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("my_python_marker = 42");

    // Switch to R and give it a different marker; each language has its own script.
    await page.getByRole("radio", { name: "R" }).click();
    await expect(page.locator(".cm-content")).toBeVisible();
    await page.locator(".cm-content").click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("my_r_marker <- 7");

    // Nothing was sent to the server: the scripts live only in localStorage.
    const stored = await page.evaluate(() =>
      window.localStorage.getItem("sb-drafts-v1"),
    );
    expect(stored).toContain("my_python_marker");
    expect(stored).toContain("my_r_marker");

    // A full reload (as if the tab was closed and reopened) restores the work.
    await page.reload();
    await expect(page.locator(".cm-content")).toBeVisible();
    // The last language (R) is remembered, and its script survived.
    await expect(page.getByRole("radio", { name: "R" })).toHaveAttribute(
      "aria-checked",
      "true",
    );
    await expect(page.locator(".cm-content")).toContainText("my_r_marker <- 7");
    // Python still has its own, separate script.
    await page.getByRole("radio", { name: "Python" }).click();
    await expect(page.locator(".cm-content")).toContainText("my_python_marker = 42");
  });

  test("shows a four-pane workspace and switches language and theme", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    await expect(
      page.getByRole("heading", { level: 1, name: "Coding Studio" }),
    ).toBeVisible();

    // Four quadrants: editor, console, variables, plots.
    await expect(page.getByText("Python script")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Console" })).toBeVisible();
    await expect(
      page.getByRole("heading", { name: "Environment" }),
    ).toBeVisible();
    // The bottom-right pane is now a two-tab Plots | Help panel, so Plots is a
    // tab rather than a pane heading.
    await expect(page.getByRole("tab", { name: "Plots" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Help" })).toBeVisible();
    await expect(
      page.getByRole("textbox", { name: /Python code/i }),
    ).toBeVisible();

    // No WCAG A/AA violations in the workspace shell.
    let axe = await new AxeBuilder({ page })
      .exclude(".cm-scroller") // CodeMirror scroller; its contenteditable is keyboard-operable
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);

    // Switching to SQL relabels the editor and turns the variables pane into a
    // Tables pane.
    await page.getByRole("radio", { name: "SQL" }).click();
    await expect(page.getByText("SQL script")).toBeVisible();
    await expect(page.getByRole("heading", { name: "Tables" })).toBeVisible();
    await expect(
      page.getByRole("textbox", { name: /SQL code/i }),
    ).toBeVisible();

    // The dark theme toggles and stays accessible.
    await page.getByRole("button", { name: "Dark theme" }).click();
    await expect(
      page.getByRole("button", { name: "Light theme" }),
    ).toBeVisible();
    axe = await new AxeBuilder({ page })
      .exclude(".cm-scroller") // CodeMirror scroller; its contenteditable is keyboard-operable
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);
  });

  test("the package checker says whether a Python package can be used", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "Python" }).click();
    await page.getByRole("button", { name: "What can I install?" }).click();
    const box = page.getByRole("textbox", { name: /Check a package/i });
    await expect(box).toBeVisible();

    // A preloaded package is ready now.
    await box.fill("seaborn");
    await expect(page.getByText(/seaborn is ready to use/i)).toBeVisible();
    // An import alias resolves to the canonical package (sklearn -> scikit-learn).
    await box.fill("sklearn");
    await expect(page.getByText(/scikit-learn is ready to use/i)).toBeVisible();
    // A built-but-not-preloaded package is available on import.
    await box.fill("requests");
    await expect(page.getByText(/requests is available/i)).toBeVisible();
    // A package that needs compiling cannot be installed.
    await box.fill("statsforecast");
    await expect(page.getByText(/statsforecast cannot be installed/i)).toBeVisible();
  });

  test("the R mirror lists the bundled packages", async ({ request }) => {
    // Cheap integrity check on the mirror's own index: if a package is listed
    // here, webR installs it from our origin. The roots are tidyverse, readxl,
    // janitor, httr2 (tidymodels/fpp2/fpp3 were bundled and reverted on
    // 2026-07-24; they install on demand from the public webR repo instead).
    const res = await request.get(
      "/runtimes/webr-packages/bin/emscripten/contrib/4.6/PACKAGES",
    );
    expect(res.status()).toBe(200);
    const index = await res.text();
    for (const name of ["dplyr", "ggplot2", "rvest", "readxl", "janitor", "httr2"]) {
      expect(index, name).toContain(`Package: ${name}`);
    }
    expect(index).not.toContain("Package: tidymodels");
  });

  test("console shows a version banner and the runtimes explainer opens", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    const console_ = page.getByLabel("Console output");
    // The studio remembers the last language on this device, so pick each
    // explicitly; every banner names its bundled version.
    await page.getByRole("radio", { name: "SQL" }).click();
    await expect(console_).toContainText("SQLite 3.53.0 (WebAssembly)");
    await page.getByRole("radio", { name: "Python" }).click();
    await expect(console_).toContainText("Python 3.14.0 (Pyodide, WebAssembly)");
    await expect(console_).toContainText("first run loads the interpreter");
    await page.getByRole("radio", { name: "R", exact: true }).click();
    await expect(console_).toContainText("R version 4.6.0 (WebR 0.6.0");

    // The runtimes explainer sits next to the package help toggle.
    await page.getByRole("button", { name: "About these runtimes" }).click();
    await expect(page.getByText(/compiled to/)).toBeVisible();
    await expect(
      page.getByRole("link", { name: "webR 0.6.0" }),
    ).toHaveAttribute("href", /docs\.r-wasm\.org/);
    await expect(
      page.getByRole("link", { name: "Pyodide" }),
    ).toHaveAttribute("href", /pyodide\.org/);
  });

  test("Python requests works in the worker (urllib3 emscripten backend)", async ({
    page,
  }) => {
    // Verified live 2026-07-24: requests rides urllib3's native Pyodide
    // support, no shim. Cross-origin obeys per-site CORS (raw.githubusercontent
    // worked, arXiv's export API refused), so the deterministic regression
    // check here is same-origin: it exercises the whole HTTP stack without
    // leaving the test server. If a Pyodide or urllib3 upgrade breaks the
    // emscripten backend, this is the test that says so.
    test.setTimeout(180_000);
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "Python" }).click();
    const editor = page.locator(".cm-content").first();
    await expect(editor).toBeVisible({ timeout: 30_000 });
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      [
        "import requests",
        "from js import location",
        "r = requests.get(str(location.origin) + '/api/health')",
        "print('HTTP_OK', r.status_code, 'json' in r.headers.get('content-type', ''))",
        // 127.0.0.1 is a DIFFERENT origin than localhost, so this request
        // exercises the whole cross-origin path: the worker's adapter rewrite,
        // /api/py-proxy (test env allows local targets), and the relay back.
        // r.url restored to what the student asked for proves the rewrite is
        // invisible.
        "p = requests.get('http://127.0.0.1:3100/api/health')",
        "print('PROXY_OK', p.status_code, p.url == 'http://127.0.0.1:3100/api/health', 'json' in p.headers.get('content-type', ''))",
      ].join("\n"),
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    const consoleOut = page.getByLabel("Console output");
    await expect(consoleOut).toContainText("HTTP_OK 200 True", {
      timeout: 150_000,
    });
    await expect(consoleOut).toContainText("PROXY_OK 200 True True", {
      timeout: 30_000,
    });
  });

  test("console shows a Clear button, distinct from Restart, on every language", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    await expect(
      page.getByRole("heading", { name: "Console" }),
    ).toBeVisible();

    // The visible Clear affordance (the priority of requirement A).
    await expect(
      page.getByRole("button", { name: "Clear console" }),
    ).toBeVisible();
    // It is not the full-session reset; both exist and are separate. The
    // full reset now lives in the Environment/Tables pane header.
    await expect(
      page.getByRole("button", { name: "Restart" }),
    ).toBeVisible();

    // Present for SQL too (its console holds query results/messages/errors).
    await page.getByRole("radio", { name: "SQL" }).click();
    await expect(
      page.getByRole("button", { name: "Clear console" }),
    ).toBeVisible();

    // The console UI stays WCAG AA clean.
    const axe = await new AxeBuilder({ page })
      .exclude(".cm-scroller")
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);
  });

  test("Clear empties SQL console output but keeps tables and the db session", async ({
    page,
  }) => {
    // Real sqlite-wasm execution; give it generous headroom for first compile.
    test.setTimeout(120_000);

    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();

    // Create a table and query it.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "CREATE TABLE t(n INTEGER);\n" +
        "INSERT INTO t VALUES (1),(2),(3);\n" +
        "SELECT COUNT(*) AS c FROM t;",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();

    const output = page.getByLabel("Console output");
    await expect(output).toContainText("CREATE TABLE t", { timeout: 60_000 });
    // The table now exists, so the Tables pane is no longer empty.
    await expect(
      page.getByText("Tables you create appear here."),
    ).toHaveCount(0);

    // Clear the console.
    await page.getByRole("button", { name: "Clear console" }).click();

    // Console output is gone and the empty-state placeholder is back.
    await expect(output).not.toContainText("CREATE TABLE t");
    await expect(output).toContainText("Output appears here");
    // The table survived (variables/tables were not reset).
    await expect(
      page.getByText("Tables you create appear here."),
    ).toHaveCount(0);

    // The db session is still connected: a follow-up query still sees the table.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("SELECT SUM(n) AS s FROM t;");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("6", { timeout: 60_000 });
  });

  test("exports a SQL table to a real CSV download without resetting the session", async ({
    page,
  }) => {
    // Real sqlite-wasm execution; generous headroom for the first compile.
    test.setTimeout(120_000);

    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    // Build a table with values that exercise CSV quoting (a comma in a field).
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "CREATE TABLE grades(student TEXT, course TEXT, grade REAL);\n" +
        "INSERT INTO grades VALUES ('Amanda','ISA 401',91),('Bill, Jr','ISA 444',79);\n" +
        "SELECT * FROM grades;",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();

    const output = page.getByLabel("Console output");
    await expect(output).toContainText("Amanda", { timeout: 60_000 });

    // The table shows in the Tables pane; open its Export menu and choose CSV.
    await page.getByRole("button", { name: "Export grades" }).click();
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("menuitem", { name: "Export as CSV" }).click(),
    ]);

    // Named after the student, the object, and a timestamp.
    expect(download.suggestedFilename()).toMatch(
      /^student-grades-\d{8}-\d{4}\.csv$/,
    );

    // The downloaded bytes are a correct CSV: header, both rows, and the field
    // with a comma is quoted.
    const path = await download.path();
    const text = readFileSync(path, "utf8");
    expect(text).toContain("student,course,grade");
    expect(text).toContain("Amanda,ISA 401,91");
    expect(text).toContain('"Bill, Jr",ISA 444,79');

    // Cheap second format from the same warm session: TSV, tab-separated.
    await page.getByRole("button", { name: "Export grades" }).click();
    const [tsvDownload] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("menuitem", { name: "Export as TSV" }).click(),
    ]);
    expect(tsvDownload.suggestedFilename()).toMatch(
      /^student-grades-\d{8}-\d{4}\.tsv$/,
    );
    const tsvText = readFileSync(await tsvDownload.path(), "utf8");
    expect(tsvText).toContain("student\tcourse\tgrade");
    expect(tsvText).toContain("Amanda\tISA 401\t91");

    // The export did not reset the session: the table and db are still there.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("SELECT COUNT(*) AS c FROM grades;");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("2", { timeout: 60_000 });
  });

  test("exports the whole SQL database as a real .sqlite file without resetting the session", async ({
    page,
  }) => {
    // Real sqlite-wasm execution; generous headroom for the first compile.
    test.setTimeout(120_000);

    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    // Two tables, so "whole database" means more than one object.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "CREATE TABLE grades(student TEXT, grade REAL);\n" +
        "INSERT INTO grades VALUES ('Amanda',91),('Bill',79);\n" +
        "CREATE TABLE courses(code TEXT);\n" +
        "INSERT INTO courses VALUES ('ISA 401'),('ISA 444');\n" +
        "SELECT * FROM grades;",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();

    const output = page.getByLabel("Console output");
    await expect(output).toContainText("Amanda", { timeout: 60_000 });

    // The pane-header workspace button downloads the whole database.
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Export database" }).click(),
    ]);

    // Named chatisa-workspace-sql-<stamp>.sqlite.
    expect(download.suggestedFilename()).toMatch(
      /^chatisa-workspace-sql-\d{8}-\d{4}\.sqlite$/,
    );

    // The downloaded bytes are a genuine SQLite database: the file header is the
    // ASCII string "SQLite format 3" followed by a NUL.
    const bytes = readFileSync(await download.path());
    expect(bytes.subarray(0, 16).toString("latin1")).toBe("SQLite format 3\0");
    expect(bytes.length).toBeGreaterThan(512); // a real DB, not an empty stub

    // The export did not reset the session: both tables are still there.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "SELECT (SELECT COUNT(*) FROM grades) + (SELECT COUNT(*) FROM courses) AS total;",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("4", { timeout: 60_000 });
  });

  test("workspace export shows a friendly note for an empty database", async ({
    page,
  }) => {
    test.setTimeout(120_000);

    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "SQL" }).click();
    await expect(page.locator(".cm-content")).toBeVisible();

    // No tables created yet. The button is present and, when clicked, explains
    // rather than downloading a confusing near-empty file.
    await page.getByRole("button", { name: "Export database" }).click();
    await expect(page.getByLabel("Console output")).toContainText(
      "no tables yet",
      { timeout: 60_000 },
    );
  });

  test("restores an uploaded .sqlite workspace, merging tables per the conflict choice", async ({
    page,
  }) => {
    test.setTimeout(120_000);
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(page.locator(".cm-content")).toBeVisible();
    const output = page.getByLabel("Console output");

    // Build two tables and export the whole database.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "CREATE TABLE grades(student TEXT, grade REAL);\n" +
        "INSERT INTO grades VALUES ('Amanda',91);\n" +
        "CREATE TABLE courses(code TEXT);\n" +
        "INSERT INTO courses VALUES ('ISA 401');\n" +
        "SELECT 'built' AS s;",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("built", { timeout: 60_000 });

    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Export database" }).click(),
    ]);
    const buffer = readFileSync(await download.path());

    // Change the session: drop courses and edit grades, so the restore has both a
    // clash (grades) and a clean add (courses) to handle.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "DROP TABLE courses;\nUPDATE grades SET student='EDITED';\nSELECT student FROM grades;",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("EDITED", { timeout: 60_000 });

    // Re-upload the exported database; Upload Dataset opens in restore mode.
    await page
      .locator('input[type="file"]')
      .setInputFiles({ name: "backup.sqlite", mimeType: "application/vnd.sqlite3", buffer });
    const dialog = page.getByRole("dialog");
    await expect(
      dialog.getByRole("heading", { level: 2 }),
    ).toContainText("Restore backup.sqlite");
    // grades exists (collides); courses does not.
    await expect(dialog.getByText("grades", { exact: true })).toBeVisible();
    await expect(dialog.getByText("already exists", { exact: true })).toBeVisible();
    // The restore dialog (member list, conflict radios) is WCAG AA clean.
    const axe = await new AxeBuilder({ page })
      .include('[role="dialog"]')
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);
    // Replace the existing grades.
    await dialog.getByRole("radio", { name: /Replace the existing/ }).check();
    await dialog.getByRole("button", { name: "Restore" }).click();
    await expect(dialog).toBeHidden();

    // grades is back to Amanda (overwritten) and courses is restored.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "SELECT student FROM grades;\nSELECT code FROM courses;",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("Amanda", { timeout: 60_000 });
    await expect(output).toContainText("ISA 401");
  });

  test("exports only the selected tables (export selected)", async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(page.locator(".cm-content")).toBeVisible();
    const output = page.getByLabel("Console output");

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "CREATE TABLE grades(student TEXT);\n" +
        "INSERT INTO grades VALUES ('Amanda');\n" +
        "CREATE TABLE courses(code TEXT);\n" +
        "INSERT INTO courses VALUES ('ISA 401');\n" +
        "SELECT 'built' AS s;",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("built", { timeout: 60_000 });

    // Select only grades, then Export selected (not the whole database).
    await page.getByRole("checkbox", { name: "Select grades to export" }).check();
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: /Export selected \(1\)/ }).click(),
    ]);
    const buffer = readFileSync(await download.path());

    // Start fresh, then restore the selected-only file: courses must not appear.
    await page.getByRole("button", { name: "Restart" }).click();
    await page
      .locator('input[type="file"]')
      .setInputFiles({ name: "picked.sqlite", mimeType: "application/vnd.sqlite3", buffer });
    const dialog = page.getByRole("dialog");
    await expect(dialog.getByText("grades", { exact: true })).toBeVisible();
    await expect(dialog.getByText("courses", { exact: true })).toHaveCount(0);
    await dialog.getByRole("button", { name: "Restore" }).click();
    await expect(dialog).toBeHidden();

    await page.getByRole("button", { name: "Clear console" }).click();
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "SELECT group_concat(name) AS tables FROM sqlite_schema WHERE type='table' AND name NOT LIKE 'sqlite_%';",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("grades", { timeout: 60_000 });
    await expect(output).not.toContainText("courses");
  });

  test("R restores an uploaded .RData, renaming a clash (live)", async ({ page }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs a warm R runtime; opt in with CHATISA_LIVE_NET=1",
    );
    test.setTimeout(300_000);
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "R" }).click();
    const editor = page.getByRole("textbox", { name: /R code/i });
    await expect(page.locator(".cm-content")).toBeVisible();
    const output = page.getByLabel("Console output");

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("grades <- data.frame(x = c(1, 2))");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("data.frame", { timeout: 260_000 });

    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Export workspace" }).click(),
    ]);
    const buffer = readFileSync(await download.path());

    // Edit grades so restore has a clash to resolve.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("grades <- data.frame(x = 99)");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await page.waitForTimeout(500);

    await page
      .locator('input[type="file"]')
      .setInputFiles({ name: "backup.RData", mimeType: "application/octet-stream", buffer });
    const dialog = page.getByRole("dialog");
    await expect(dialog.getByRole("heading", { level: 2 })).toContainText("Restore");
    // Default rule is rename (keep both).
    await dialog.getByRole("button", { name: "Restore" }).click();
    await expect(dialog).toBeHidden();
    await expect(output).toContainText("grades_2", { timeout: 60_000 });
  });

  test("Python restores an uploaded .pkl, gated on trust (live)", async ({ page }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs a warm Python runtime; opt in with CHATISA_LIVE_NET=1",
    );
    test.setTimeout(300_000);
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "Python" }).click();
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(page.locator(".cm-content")).toBeVisible();
    const output = page.getByLabel("Console output");

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      'import pandas as pd\ngrades = pd.DataFrame({"x": [1, 2]})\nprint("R1")',
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("R1", { timeout: 260_000 });

    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Export workspace" }).click(),
    ]);
    const buffer = readFileSync(await download.path());

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText('grades = "edited"\nprint("R2")');
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(output).toContainText("R2", { timeout: 60_000 });

    await page
      .locator('input[type="file"]')
      .setInputFiles({ name: "backup.pkl", mimeType: "application/octet-stream", buffer });
    const dialog = page.getByRole("dialog");
    await expect(dialog.getByRole("heading", { level: 2 })).toContainText("Restore");
    // A pickle can run code, so Restore is disabled until the file is trusted.
    const restoreBtn = dialog.getByRole("button", { name: "Restore" });
    await expect(restoreBtn).toBeDisabled();
    await dialog.getByRole("checkbox", { name: /I trust this file/ }).check();
    await expect(restoreBtn).toBeEnabled();
    await restoreBtn.click();
    await expect(dialog).toBeHidden();
    await expect(output).toContainText("grades_2", { timeout: 60_000 });
  });

  test("R exports the whole environment as a real .RData file (live)", async ({
    page,
  }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs a warm R runtime; opt in with CHATISA_LIVE_NET=1",
    );
    test.setTimeout(300_000);
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "R" }).click();
    const editor = page.getByRole("textbox", { name: /R code/i });
    await expect(page.locator(".cm-content")).toBeVisible();
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "grades <- data.frame(student = c('Amanda','Bill'), grade = c(91, 79))",
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    // The run echoes the code once it finishes (assignment prints nothing itself).
    await expect(page.getByLabel("Console output")).toContainText("data.frame", {
      timeout: 260_000,
    });
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Export workspace" }).click(),
    ]);
    expect(download.suggestedFilename()).toMatch(
      /^chatisa-workspace-r-\d{8}-\d{4}\.RData$/,
    );
    const bytes = readFileSync(await download.path());
    expect(bytes.length).toBeGreaterThan(50); // a real RData image, not an empty stub
  });

  test("Python exports the whole environment as a real pickle and reports unpicklable objects (live)", async ({
    page,
  }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs a warm Python runtime; opt in with CHATISA_LIVE_NET=1",
    );
    test.setTimeout(300_000);
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "Python" }).click();
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(page.locator(".cm-content")).toBeVisible();
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    // grades (a DataFrame) and notes (a str) pickle fine; a generator does not, so
    // it must be reported rather than silently dropped.
    await page.keyboard.insertText(
      'import pandas as pd\ngrades = pd.DataFrame({"g": [1, 2]})\nnotes = "hello"\ngen = (i for i in range(3))\nprint("READY")',
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(page.getByLabel("Console output")).toContainText("READY", {
      timeout: 260_000,
    });
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Export workspace" }).click(),
    ]);
    expect(download.suggestedFilename()).toMatch(
      /^chatisa-workspace-python-\d{8}-\d{4}\.pkl$/,
    );
    // A protocol-4 pickle begins with the PROTO opcode 0x80 0x04.
    const bytes = readFileSync(await download.path());
    expect(bytes[0]).toBe(0x80);
    expect(bytes[1]).toBe(0x04);
    // The unpicklable global is reported, and the trust warning is shown.
    const output = page.getByLabel("Console output");
    await expect(output).toContainText("gen", { timeout: 60_000 });
    await expect(output).toContainText(/trust/i);
  });

  test("Ctrl+Enter runs the whole multi-line SQL statement from a middle line", async ({
    page,
  }) => {
    // Real sqlite-wasm execution; generous headroom for the first compile.
    test.setTimeout(120_000);

    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();
    // Wait for CodeMirror itself, not the load-time textarea fallback that shares
    // the same accessible name. The Mod-Enter keymap only lives in CodeMirror, and
    // typing into the fallback would swap out mid-interaction under load.
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    // A CTE that only yields a value if the WHOLE statement runs. Running any
    // single physical line here is a syntax error, so a passing result proves
    // the complete logical statement executed.
    await page.keyboard.insertText(
      "WITH nums(n) AS (\n" +
        "  VALUES (1), (2), (3)\n" +
        ")\n" +
        "SELECT SUM(n) AS total FROM nums;",
    );

    // Put the cursor on a middle line (the VALUES row), not the start.
    await page.getByText("VALUES (1), (2), (3)").click();
    await page.keyboard.press("ControlOrMeta+Enter");

    const output = page.getByLabel("Console output");
    await expect(output).toContainText("total", { timeout: 60_000 });
    await expect(output).toContainText("6");
  });

  test("Ctrl+Enter runs only the statement at the cursor, not the next one", async ({
    page,
  }) => {
    test.setTimeout(120_000);

    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();
    // Wait for CodeMirror itself, not the load-time textarea fallback that shares
    // the same accessible name. The Mod-Enter keymap only lives in CodeMirror, and
    // typing into the fallback would swap out mid-interaction under load.
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      "SELECT 111 AS first_only;\nSELECT 222 AS second_only;",
    );

    // Cursor on the first statement.
    await page.getByText("SELECT 111 AS first_only;").click();
    await page.keyboard.press("ControlOrMeta+Enter");

    const output = page.getByLabel("Console output");
    await expect(output).toContainText("111", { timeout: 60_000 });
    // The second statement did not run.
    await expect(output).not.toContainText("222");
  });

  test("side chat opens, sends code context, and shows a reply", async ({
    page,
  }) => {
    let sentBody: { module?: string; context?: string } | null = null;
    await page.route("**/api/chat", async (route) => {
      try {
        sentBody = route.request().postDataJSON();
      } catch {
        sentBody = null;
      }
      await route.continue();
    });

    await page.goto("/coding-studio");
    await expect(
      page.getByRole("heading", { level: 1, name: "Coding Studio" }),
    ).toBeVisible();

    await page.getByRole("button", { name: "Ask AI" }).click();
    const chat = page.getByRole("complementary", { name: "Sandbox assistant" });
    await expect(chat).toBeVisible();
    await expect(
      chat.getByRole("button", { name: "Choose a different model" }),
    ).toBeVisible();

    // No WCAG A/AA violations with the assistant open.
    const axe = await new AxeBuilder({ page })
      .exclude(".cm-scroller") // CodeMirror scroller; its contenteditable is keyboard-operable
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);

    await chat.getByRole("textbox", { name: "Your message" }).fill("Explain df");
    await chat.getByRole("button", { name: "Send" }).click();

    // The deterministic mock model streams a reply into the assistant panel.
    await expect(chat.getByText(/read a CSV in both languages/i)).toBeVisible({
      timeout: 15_000,
    });

    // The request carried the sandbox module and the code context (script and
    // language), never persisted but sent per message.
    const body = sentBody as { module?: string; context?: string } | null;
    expect(body?.module).toBe("sandbox_chat");
    expect(body?.context).toContain("Language: Python");
    // The starter is the guidance comment; the runnable sample is added on
    // demand with "Insert Coding Example". Either way the script is in context.
    expect(body?.context).toContain("Python runs in your browser");

    await chat.getByRole("button", { name: "Close" }).click();
    await expect(chat).toHaveCount(0);
  });

  test("downloads the current script with the language's extension", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    await page.locator(".cm-content").waitFor();
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.getByRole("button", { name: "Download Script" }).click(),
    ]);
    // Named after the student (email local part) and a timestamp, e.g.
    // student-20260727-1430.py.
    expect(download.suggestedFilename()).toMatch(
      /^student-\d{8}-\d{4}\.py$/,
    );
  });

  test("inline completions show ghost text and Tab accepts it", async ({
    page,
  }) => {
    // Observe the completion request without intercepting it (interception
    // perturbs the fetch timing the ghost-text relies on).
    let requestLanguage: string | undefined;
    page.on("request", (r) => {
      if (r.url().includes("/api/complete")) {
        try {
          requestLanguage = (r.postDataJSON() as { language?: string })?.language;
        } catch {
          // ignore
        }
      }
    });

    await page.goto("/coding-studio");
    // Wait for CodeMirror itself (not the textarea fallback), since the
    // completion extension only lives in the CodeMirror editor.
    const editor = page.locator(".cm-content");
    await expect(editor).toBeVisible();

    // Type at the end of the script; a completion request follows the pause.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+End");
    const completed = page.waitForResponse(
      (r) => r.url().includes("/api/complete") && r.status() === 200,
      // Generous: the route compiles lazily on its first hit in dev.
      { timeout: 30_000 },
    );
    await page.keyboard.type("\nx = ");
    await completed;

    // The reply shows as dimmed ghost text after the cursor.
    const ghost = page.locator(".cm-ghost-text");
    await expect(ghost).toHaveText("# ai suggestion", { timeout: 5_000 });

    // The request named the current language.
    expect(requestLanguage).toBe("python");

    // Tab accepts the suggestion into the document.
    await page.keyboard.press("Tab");
    await expect(editor).toContainText("# ai suggestion");
  });

  test("Coding Studio is cross-origin isolated (enables R networking)", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    await expect(
      page.getByRole("heading", { level: 1, name: "Coding Studio" }),
    ).toBeVisible();
    expect(
      await page.evaluate(
        () =>
          (globalThis as { crossOriginIsolated?: boolean })
            .crossOriginIsolated === true,
      ),
    ).toBe(true);

    // Entering from another page via the nav must be a full load, so isolation
    // still applies (an SPA navigation would leave it un-isolated).
    await page.goto("/");
    // The nav tab's accessible name is exactly "Coding Studio" (the home-page
    // module card also links there, with a longer name); target the nav tab.
    await page
      .getByRole("link", { name: "Coding Studio", exact: true })
      .click();
    await expect(
      page.getByRole("heading", { level: 1, name: "Coding Studio" }),
    ).toBeVisible();
    expect(
      await page.evaluate(
        () =>
          (globalThis as { crossOriginIsolated?: boolean })
            .crossOriginIsolated === true,
      ),
    ).toBe(true);
  });

  for (const lang of [
    { radio: "Python", name: /Python code/i },
    { radio: "R", name: /R code/i },
    { radio: "SQL", name: /SQL code/i },
  ]) {
    test(`${lang.radio} editor scrolls a long script within its panel`, async ({
      page,
    }) => {
      await page.goto("/coding-studio");
      if (lang.radio !== "Python") {
        await page.getByRole("radio", { name: lang.radio }).click();
      }
      const editor = page.getByRole("textbox", { name: lang.name });
      await expect(editor).toBeVisible();
      // Wait for CodeMirror itself, not the load-time textarea fallback that
      // shares the same accessible name. If we typed into the fallback it would
      // swap out mid-interaction under load and the keystrokes would be lost.
      await expect(page.locator(".cm-content")).toBeVisible();

      // Replace the starter with a script far taller than the panel.
      await editor.click();
      await page.keyboard.press("ControlOrMeta+A");
      await page.keyboard.press("Delete");
      await page.keyboard.insertText(
        Array.from({ length: 200 }, (_, i) => `a${i} <- ${i}`).join("\n"),
      );
      // Confirm the replacement landed before measuring layout. CodeMirror
      // virtualizes rows, so after the insert the cursor sits at the end and the
      // last line (not the first) is the one rendered in the DOM.
      await expect(editor).toContainText("a199 <- 199");

      // The content overflows: the scroller is taller inside than its box.
      // Inserting 200 lines and laying them out can lag a frame under parallel
      // load, so poll the measurement instead of reading a single mid-render one.
      const scroller = page.locator(".cm-scroller");
      await expect
        .poll(() =>
          scroller.evaluate((el) => el.scrollHeight - el.clientHeight),
        )
        .toBeGreaterThan(100);

      // The editor does not expand past the visible layout.
      const box = await page.locator(".cm-editor").boundingBox();
      const vp = page.viewportSize()!;
      expect(box!.y + box!.height).toBeLessThanOrEqual(vp.height + 1);

      // Moving to the end keeps the cursor within the visible viewport.
      await page.keyboard.press("ControlOrMeta+End");
      const cursorVisible = await page.evaluate(() => {
        const cur = document.querySelector(".cm-cursor-primary");
        const sc = document.querySelector(".cm-scroller");
        if (!cur || !sc) return false;
        const c = cur.getBoundingClientRect();
        const s = sc.getBoundingClientRect();
        return c.top >= s.top - 1 && c.bottom <= s.bottom + 1;
      });
      expect(cursorVisible).toBe(true);
    });
  }

  test("editor scrolls horizontally for a very long line (no wrapping)", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(editor).toBeVisible();
    // Wait for CodeMirror itself, not the load-time textarea fallback.
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("x = " + "1234567890".repeat(80));
    await expect(editor).toContainText("1234567890");

    const scroller = page.locator(".cm-scroller");
    await expect
      .poll(() => scroller.evaluate((el) => el.scrollWidth - el.clientWidth))
      .toBeGreaterThan(50);
  });

  test("R scrapes a no-CORS site via the ws-proxy", async ({ page }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs live network; opt in with CHATISA_LIVE_NET=1",
    );
    test.setTimeout(300_000);

    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "R" }).click();
    const editor = page.locator(".cm-content");
    await expect(editor).toBeVisible();
    await editor.click();
    await page.keyboard.press("Control+A");
    await page.keyboard.press("Delete");
    // No Sys.setenv: ALL_PROXY is auto-wired by the WebR worker. That the row
    // count appears proves the auto-wiring and isolation work end to end.
    await page.keyboard.insertText(
      'cat("ROWS=", nrow(rvest::read_html(' +
        '"https://miamioh.edu/fsb/directory/?up=/query/all/all/Information_Systems_and_Analytics/all"' +
        ') |> rvest::html_element("table") |> rvest::html_table()), "\\n")',
    );
    await page.getByRole("button", { name: "Run" }).click();
    await expect(page.locator('[aria-label="Console output"]')).toContainText(
      /ROWS=\s*\d+/,
      { timeout: 260_000 },
    );
  });

  test("Python scrapes a no-CORS site via the py-proxy", async ({ page }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs live network; opt in with CHATISA_LIVE_NET=1",
    );
    test.setTimeout(300_000);

    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "Python" }).click();
    const editor = page.locator(".cm-content").first();
    await expect(editor).toBeVisible({ timeout: 30_000 });
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    // The twin of the R live test above, same no-CORS target: plain requests
    // plus BeautifulSoup, with the worker's adapter routing the fetch through
    // /api/py-proxy. A row count proves the entire scrape flow.
    await page.keyboard.insertText(
      [
        "import requests",
        "from bs4 import BeautifulSoup",
        "r = requests.get('https://miamioh.edu/fsb/directory/?up=/query/all/all/Information_Systems_and_Analytics/all')",
        // The 'lxml' STRING is deliberate: the worker sees the word and loads
        // the lxml package before running (2026-07-24), because bs4 asks for
        // the parser by name and Pyodide's load-on-import cannot see strings.
        "soup = BeautifulSoup(r.text, 'lxml')",
        "table = soup.find('table')",
        "rows = table.find_all('tr') if table else []",
        "print('PY_STATUS=', r.status_code)",
        "print('PY_ROWS=', len(rows))",
      ].join("\n"),
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    const output = page.getByLabel("Console output");
    await expect(output).toContainText(/PY_ROWS=\s*[1-9]\d*/, {
      timeout: 260_000,
    });
    const text = await output.innerText();
    console.log(
      "live-scrape: " +
        text
          .split("\n")
          .filter((l) => l.startsWith("PY_"))
          .join(" "),
    );
  });

  test("Enter after an R pipe indents the continuation line", async ({ page }) => {
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "R" }).click();
    const editor = page.getByRole("textbox", { name: /R code/i });
    await expect(editor).toBeVisible();
    // Wait for CodeMirror itself: the indentService only lives in the editor, not
    // the load-time textarea fallback that shares the accessible name.
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("df |>");
    await page.keyboard.press("Enter");
    await page.keyboard.type("filter(x)");

    // The second line was auto-indented two spaces before "filter(x)".
    expect(await editorLineText(page, 1)).toBe("  filter(x)");
  });

  test("Enter after a Python colon header indents four spaces", async ({ page }) => {
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "Python" }).click();
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("def f(x):");
    await page.keyboard.press("Enter");
    await page.keyboard.type("return x");

    expect(await editorLineText(page, 1)).toBe("    return x");
  });

  test("Enter after SQL SELECT indents the column list", async ({ page }) => {
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("SELECT");
    await page.keyboard.press("Enter");
    await page.keyboard.type("a,");

    expect(await editorLineText(page, 1)).toBe("  a,");
  });

  test("an unclosed R bracket draws a lint underline", async ({ page }) => {
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "R" }).click();
    const editor = page.getByRole("textbox", { name: /R code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    // insertText makes one atomic edit, so basicSetup's closeBrackets does not
    // auto-insert a matching ")" (which would balance the statement and defeat the
    // test). This leaves a genuinely unclosed "(" for the linter to flag.
    await page.keyboard.insertText("x <- (1 + 2");

    // The linter runs on a 400ms debounce and underlines the unclosed "(".
    await expect(page.locator(".cm-lintRange-error").first()).toBeVisible({
      timeout: 5000,
    });
  });

  // The clicked token often has no dedicated span in the CodeMirror light theme
  // (only keywords and strings are wrapped), so getByText resolves to the
  // full-width line div and a center click misses the word. These helpers click
  // the exact glyph coordinates of a token via a DOM Range, which is what a user
  // does when they Ctrl/Cmd+click a specific identifier.
  async function tokenCoords(
    page: import("@playwright/test").Page,
    token: string,
  ) {
    const box = await page.evaluate((tok) => {
      const content = document.querySelector(".cm-content");
      if (!content) return null;
      const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT);
      let node: Node | null;
      while ((node = walker.nextNode())) {
        const idx = node.textContent?.indexOf(tok) ?? -1;
        if (idx >= 0) {
          const range = document.createRange();
          range.setStart(node, idx);
          range.setEnd(node, idx + tok.length);
          const r = range.getBoundingClientRect();
          return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
        }
      }
      return null;
    }, token);
    if (!box) throw new Error(`token not found: ${token}`);
    return box;
  }
  // Ctrl on Windows/Linux, Cmd on macOS, matching the requirement.
  const HELP_MOD = process.platform === "darwin" ? "Meta" : "Control";
  async function modClickToken(
    page: import("@playwright/test").Page,
    token: string,
  ) {
    const { x, y } = await tokenCoords(page, token);
    await page.keyboard.down(HELP_MOD);
    await page.mouse.click(x, y);
    await page.keyboard.up(HELP_MOD);
  }

  test("Ctrl/Cmd+Click opens a HELP tab beside PLOTS with the symbol and a new-tab link", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(editor).toBeVisible();
    // Wait for CodeMirror itself, not the load-time textarea fallback: the
    // mousedown/F1 handlers only live in the CodeMirror editor.
    await expect(page.locator(".cm-content")).toBeVisible();

    // The Help tab sits next to Plots from the start (one instance, reused).
    await expect(page.getByRole("tab", { name: "Plots" })).toBeVisible();
    await expect(page.getByRole("tab", { name: "Help" })).toHaveCount(1);

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      'import pandas as pd\ndf = pd.DataFrame()\nout = df.groupby("k")\nn = len(out)\n',
    );
    await expect(editor).toContainText("groupby");

    // Record the caret box so we can prove a modified click does not move it.
    const caretBefore = await page.evaluate(() => {
      const c = document.querySelector(".cm-cursor-primary");
      const r = c?.getBoundingClientRect();
      return r ? { x: Math.round(r.left), y: Math.round(r.top) } : null;
    });

    // Modified click on the method name.
    await modClickToken(page, "groupby");

    // The Help tab is now selected and shows the symbol, the source, and the link.
    await expect(page.getByRole("tab", { name: "Help" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    const help = page.getByRole("tabpanel", { name: "Help" });
    await expect(help.getByText("df.groupby", { exact: true })).toBeVisible();
    await expect(help.getByText("pandas", { exact: true })).toBeVisible();
    const link = help.getByRole("link", { name: "Open full documentation" });
    await expect(link).toHaveAttribute("target", "_blank");
    await expect(link).toHaveAttribute(
      "href",
      "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html",
    );

    // The caret did not move (script position and cursor preserved).
    const caretAfter = await page.evaluate(() => {
      const c = document.querySelector(".cm-cursor-primary");
      const r = c?.getBoundingClientRect();
      return r ? { x: Math.round(r.left), y: Math.round(r.top) } : null;
    });
    expect(caretAfter).toEqual(caretBefore);

    // No WCAG A/AA violations with the Help tab open.
    const axe = await new AxeBuilder({ page })
      .exclude(".cm-scroller")
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);
  });

  test("a second Ctrl/Cmd+Click reuses the one HELP tab", async ({ page }) => {
    await page.goto("/coding-studio");
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      'import pandas as pd\ndf = pd.DataFrame()\nout = df.groupby("k")\nn = len(out)\n',
    );

    await modClickToken(page, "groupby");
    const help = page.getByRole("tabpanel", { name: "Help" });
    await expect(help.getByText("df.groupby", { exact: true })).toBeVisible();

    // Click a different symbol: still one Help tab, contents replaced.
    await modClickToken(page, "len");
    await expect(page.getByRole("tab", { name: "Help" })).toHaveCount(1);
    await expect(help.getByText("len", { exact: true })).toBeVisible();
    await expect(
      help.getByRole("link", { name: "Open full documentation" }),
    ).toHaveAttribute(
      "href",
      "https://docs.python.org/3/library/functions.html#len",
    );
    // The previous symbol is gone (the tab was reused, not duplicated).
    await expect(help.getByText("groupby")).toHaveCount(0);
  });

  test("F1 opens docs for the symbol at the cursor (keyboard path)", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("n = len(items)\n");

    // Put the cursor inside "len" (click its exact glyph), then press F1.
    const { x, y } = await tokenCoords(page, "len");
    await page.mouse.click(x, y);
    await page.keyboard.press("F1");

    const help = page.getByRole("tabpanel", { name: "Help" });
    await expect(page.getByRole("tab", { name: "Help" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    await expect(help.getByText("len", { exact: true })).toBeVisible();
    await expect(help.getByText("Python", { exact: true })).toBeVisible();
  });

  test("the HELP pane shows a loading state then a doc region or fallback (SQL)", async ({
    page,
  }) => {
    test.setTimeout(120_000);
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "SQL" }).click();
    const editor = page.getByRole("textbox", { name: /SQL code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("SELECT COUNT(*) FROM t;");

    // Ctrl/Cmd+Click COUNT to open its docs in the HELP tab.
    const count = page.locator(".cm-content").getByText("COUNT", { exact: false }).first();
    await count.click({ modifiers: ["ControlOrMeta"] });

    // The HELP tab is selected and shows the symbol and source.
    await expect(page.getByRole("tab", { name: "Help" })).toHaveAttribute("aria-selected", "true");
    const helpPanel = page.getByRole("tabpanel", { name: "Help" });
    await expect(helpPanel.getByText("COUNT", { exact: true })).toBeVisible();

    // SQLite has no runtime help, so after the brief loading state the pane falls back
    // to the blurb + link (the deterministic no-local-docs path). Assert the stable
    // end state: the "Open full documentation" link and the honest no-doc line.
    await expect(helpPanel.getByRole("link", { name: "Open full documentation" })).toBeVisible();
    await expect(helpPanel.getByText(/No documentation text is available/i)).toBeVisible();
  });

  test("R HELP resolves a bundled, non-attached function (try.all.packages)", async ({
    page,
  }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs a warm R runtime; opt in with CHATISA_LIVE_NET=1",
    );
    test.setTimeout(300_000);
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "R" }).click();
    const editor = page.getByRole("textbox", { name: /R code/i });
    await expect(page.locator(".cm-content")).toBeVisible();
    // Run once so the bundled package library is installed (tidyr among them).
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("1 + 1");
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(page.locator('[aria-label="Console output"]')).toContainText(
      /\b2\b/,
      { timeout: 260_000 },
    );
    // Bare and unqualified, and tidyr is installed but NOT attached: this resolves
    // only because the worker searches the whole installed library
    // (try.all.packages = TRUE), not just attached packages. Not curated either.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("pivot_longer(x)\n");
    // Place the cursor on the token and press F1 (deterministic; no click-coordinate
    // or autocomplete-popup interference).
    {
      const { x, y } = await tokenCoords(page, "pivot_longer");
      await page.mouse.click(x, y);
    }
    await page.keyboard.press("F1");
    await expect(page.getByRole("tab", { name: "Help" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    const doc = page
      .getByRole("tabpanel", { name: "Help" })
      .getByRole("region", { name: /Documentation for/i });
    await expect(doc).toBeVisible({ timeout: 60_000 });
    await expect(doc).toContainText(/pivot/i);
  });

  test("Python HELP resolves a non-curated method via live introspection (df.nlargest)", async ({
    page,
  }) => {
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "needs a warm Python runtime; opt in with CHATISA_LIVE_NET=1",
    );
    test.setTimeout(300_000);
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "Python" }).click();
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(page.locator(".cm-content")).toBeVisible();
    // Build a live DataFrame, then click a method that is NOT in the curated map
    // (nlargest). The worker introspects the live object for its docstring, so any
    // real method resolves, not just the handful the curated list knows.
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText(
      'import pandas as pd\ndf = pd.DataFrame({"a": [3, 1, 2]})\ndf.shape',
    );
    await page.getByRole("button", { name: "Run", exact: true }).click();
    await expect(page.locator('[aria-label="Console output"]')).toContainText(
      /\(3, 1\)/,
      { timeout: 260_000 },
    );
    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText('df.nlargest(2, "a")\n');
    // Place the cursor on the token and press F1 (deterministic; no click-coordinate
    // or autocomplete-popup interference).
    {
      const { x, y } = await tokenCoords(page, "nlargest");
      await page.mouse.click(x, y);
    }
    await page.keyboard.press("F1");
    await expect(page.getByRole("tab", { name: "Help" })).toHaveAttribute(
      "aria-selected",
      "true",
    );
    const doc = page
      .getByRole("tabpanel", { name: "Help" })
      .getByRole("region", { name: /Documentation for/i });
    await expect(doc).toBeVisible({ timeout: 60_000 });
    await expect(doc).toContainText(/largest|rows/i);
  });

  test("the Shortcuts dialog lists editor shortcuts, closes on Esc, and is axe-clean", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    await expect(
      page.getByRole("heading", { level: 1, name: "Coding Studio" }),
    ).toBeVisible();

    await page.getByRole("button", { name: "Shortcuts" }).click();
    const dialog = page.getByRole("dialog", { name: "Keyboard shortcuts" });
    await expect(dialog).toBeVisible();

    // The real, verified bindings are listed (keys rendered for the test platform:
    // Ctrl on Linux/Windows CI, Cmd on macOS). Assert the actions, which are
    // platform-independent, and that the pipe is marked R only.
    await expect(dialog.getByText("Run statement or selection")).toBeVisible();
    await expect(dialog.getByText("Run whole script")).toBeVisible();
    await expect(dialog.getByText("Source silently")).toBeVisible();
    await expect(dialog.getByText("Insert pipe")).toBeVisible();
    await expect(dialog.getByText("R only")).toBeVisible();
    await expect(dialog.getByText("Toggle comment")).toBeVisible();
    await expect(dialog.getByText("Documentation for symbol")).toBeVisible();
    // exact: the footnote paragraph also mentions "Autocomplete" and "Ctrl+Space",
    // so target the definition and key rows precisely (strict-mode safe).
    await expect(dialog.getByText("Autocomplete", { exact: true })).toBeVisible();
    // Autocomplete is Ctrl on every platform.
    await expect(dialog.getByText("Ctrl+Space", { exact: true })).toBeVisible();

    // No WCAG A/AA violations with the dialog open.
    const axe = await new AxeBuilder({ page })
      .exclude(".cm-scroller")
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);

    // Esc closes it and returns focus to the trigger.
    await page.keyboard.press("Escape");
    await expect(dialog).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Shortcuts" })).toBeFocused();
  });

  test("Ctrl/Cmd+Shift+M inserts the native pipe in R", async ({ page }) => {
    await page.goto("/coding-studio");
    await page.getByRole("radio", { name: "R" }).click();
    const editor = page.getByRole("textbox", { name: /R code/i });
    await expect(editor).toBeVisible();
    // Wait for CodeMirror itself; the keymap only lives in the editor, not the
    // load-time textarea fallback that shares the accessible name.
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("df");
    await page.keyboard.press("ControlOrMeta+Shift+M");

    // The native pipe with surrounding spaces was inserted after "df".
    await expect(editor).toContainText("df |>");
    // And it is the native pipe, not magrittr.
    await expect(editor).not.toContainText("%>%");
  });

  test("Ctrl/Cmd+Shift+M does not insert a pipe in Python or SQL", async ({
    page,
  }) => {
    await page.goto("/coding-studio");
    // Python
    const py = page.getByRole("textbox", { name: /Python code/i });
    await expect(py).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();
    await py.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("x = 1");
    await page.keyboard.press("ControlOrMeta+Shift+M");
    await expect(py).not.toContainText("|>");

    // SQL
    await page.getByRole("radio", { name: "SQL" }).click();
    const sql = page.getByRole("textbox", { name: /SQL code/i });
    await expect(sql).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();
    await sql.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.type("SELECT 1");
    await page.keyboard.press("ControlOrMeta+Shift+M");
    await expect(sql).not.toContainText("|>");
  });

  test("Ctrl/Cmd+/ toggles a line comment in R, Python, and SQL", async ({
    page,
  }) => {
    await page.goto("/coding-studio");

    // R and Python use '#', SQL uses '--'.
    const cases = [
      { radio: "R", name: /R code/i, line: "x <- 1", token: "# " },
      { radio: "Python", name: /Python code/i, line: "x = 1", token: "# " },
      { radio: "SQL", name: /SQL code/i, line: "SELECT 1", token: "-- " },
    ];

    for (const c of cases) {
      await page.getByRole("radio", { name: c.radio }).click();
      const editor = page.getByRole("textbox", { name: c.name });
      await expect(editor).toBeVisible();
      await expect(page.locator(".cm-content")).toBeVisible();

      await editor.click();
      await page.keyboard.press("ControlOrMeta+A");
      await page.keyboard.press("Delete");
      await page.keyboard.type(c.line);
      // Comment on.
      await page.keyboard.press("ControlOrMeta+/");
      expect(await editorLineText(page, 0)).toContain(`${c.token}${c.line}`);
      // Comment off (toggles back).
      await page.keyboard.press("ControlOrMeta+/");
      expect(await editorLineText(page, 0)).toBe(c.line);
    }
  });

  test("Ctrl/Cmd+/ comments a multi-line selection", async ({ page }) => {
    await page.goto("/coding-studio");
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(editor).toBeVisible();
    await expect(page.locator(".cm-content")).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("a = 1\nb = 2");
    // Select everything, then toggle.
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("ControlOrMeta+/");

    expect(await editorLineText(page, 0)).toContain("# a = 1");
    expect(await editorLineText(page, 1)).toContain("# b = 2");
  });
});
