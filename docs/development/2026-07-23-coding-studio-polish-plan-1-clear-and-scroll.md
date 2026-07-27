# Slice 1 — Clear Console + Scrollable Editor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a visible Clear button to the Coding Studio console that removes only
the console/query output, and make the editor scroll normally when a script is
taller or wider than its panel, consistently for R, Python, and SQL.

**Architecture:** Both changes are local and low-risk. Clear is a pure React state
reset (`setEntries([])`) wired from `Workspace` into `ConsolePane`; it never touches
the session worker, the database, variables, or plots. Scrolling is a CodeMirror
theme fix: give the `.cm-editor` a real `height: 100%` in fill-height mode so its
default-scrolling `.cm-scroller` is bounded by the panel instead of overflowing a
clipped wrapper.

**Tech Stack:** Next.js (this repo's vendored build — read `node_modules/next/dist/docs/`
before touching any Next API; this slice does not), React 19, CodeMirror 6
(`@codemirror/view`, `@codemirror/state`, `@codemirror/language`), Playwright +
`@axe-core/playwright` for the sandbox e2e.

## Global Constraints

- No git commits. The working tree stays uncommitted; each task ends by running
  verification commands instead of committing.
- WCAG 2.1 AA, keyboard and mouse. Run axe on the sandbox after UI changes.
- Miami brand tokens only for colors: use the existing `var(--sb-...)` CSS variables
  (`--sb-border`, `--sb-accent`, `--sb-muted`, `--sb-header`, `--sb-panel`,
  `--sb-text`). Do not introduce raw hex in the console UI.
- No em dashes in any user-facing copy (button labels, titles, tooltips).
- Keyboard shortcuts, where added, support Ctrl (Windows/Linux) and Cmd (macOS). The
  visible Clear button is the priority; a shortcut is optional and out of scope here.
- Consistent behavior across R, Python, and SQL.
- All commands run from `web/` (the Next app root). The Playwright config starts its
  own dev server on port 3100, so no separate server needs starting.

## File Structure

- `web/components/sandbox/Sandbox.tsx` — Modify. Add `clearConsole` in `Workspace`,
  pass it to `ConsolePane`, and add the Clear button to the `ConsolePane` header.
- `web/components/run/CodeEditor.tsx` — Modify. In `themeExtensions`, give
  `.cm-editor` a real height in fill-height mode so the editor scrolls within its
  panel.
- `web/tests/e2e/sandbox.spec.ts` — Modify. Add the Clear tests and the scroll tests
  to the existing `AI Sandbox` describe block, following the existing patterns.

---

## Task 1: Clear Console button

Removes all console/query output (echoed code, results, tables, messages, errors)
without modifying the script or resetting variables, plots, the database, or the
session. For SQL this clears displayed results/messages/errors without disconnecting
the active database.

**Files:**
- Modify: `web/components/sandbox/Sandbox.tsx` (add `clearConsole` near line 467;
  wire `onClear` into `ConsolePane` at line 676; add the button in `ConsolePane`,
  header at lines 982-1003, props at lines 954-966)
- Test: `web/tests/e2e/sandbox.spec.ts`

**Interfaces:**
- Consumes: existing `entries` state and its `setEntries` setter in `Workspace`
  (Sandbox.tsx:300); the `ConsolePane` component (Sandbox.tsx:954).
- Produces: `ConsolePane` gains an `onClear: () => void` prop; a button with
  accessible name `Clear console` in the console header.

- [ ] **Step 1: Write the failing presence test**

Add to the `test.describe("AI Sandbox", ...)` block in
`web/tests/e2e/sandbox.spec.ts`:

```typescript
  test("console shows a Clear button, distinct from Restart, on every language", async ({
    page,
  }) => {
    await page.goto("/ai-sandbox");
    await expect(
      page.getByRole("heading", { name: "Console" }),
    ).toBeVisible();

    // The visible Clear affordance (the priority of requirement A).
    await expect(
      page.getByRole("button", { name: "Clear console" }),
    ).toBeVisible();
    // It is not the full-session reset; both exist and are separate.
    await expect(
      page.getByRole("button", { name: "Restart session" }),
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
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `npm run test:e2e -- sandbox.spec.ts -g "Clear button"`
Expected: FAIL. The `getByRole("button", { name: "Clear console" })` assertion times
out because no such button exists yet.

- [ ] **Step 3: Add the `clearConsole` callback in `Workspace`**

In `web/components/sandbox/Sandbox.tsx`, next to `clearPlots` (around line 467), add:

```typescript
  // Clears only the console output. Variables, plots, the database, and the
  // session are untouched: this sets React state, never the worker. For SQL it
  // removes displayed results, messages, and errors without disconnecting.
  const clearConsole = useCallback(() => setEntries([]), []);
```

- [ ] **Step 4: Pass `onClear` into `ConsolePane`**

In the same file, in the `<ConsolePane ... />` usage (around line 676), add the prop:

```tsx
                <ConsolePane
                  entries={entries}
                  running={running}
                  preparing={preparing}
                  label={language.label}
                  onRun={(code) => void execute(code)}
                  onClear={clearConsole}
                />
```

- [ ] **Step 5: Accept `onClear` in `ConsolePane` and render the button**

Update the `ConsolePane` prop list (around line 954) to include `onClear`:

```tsx
function ConsolePane({
  entries,
  running,
  preparing,
  label,
  onRun,
  onClear,
}: {
  entries: ConsoleEntry[];
  running: boolean;
  preparing: boolean;
  label: string;
  onRun: (code: string) => void;
  onClear: () => void;
}) {
```

Replace the `ConsolePane` header (the `<header>...</header>` at lines 982-1003) with a
version that keeps the status indicator and adds the Clear button beside it:

```tsx
      <header className="flex items-center justify-between border-b border-[var(--sb-border)] bg-[var(--sb-header)] px-3 py-1.5">
        <h2 className="text-xs font-bold uppercase tracking-wide text-[var(--sb-muted)]">
          Console
        </h2>
        <div className="flex items-center gap-2">
          {running ? (
            <span
              role="status"
              className="flex items-center gap-1.5 text-xs font-bold text-[var(--sb-accent)]"
            >
              <Spinner />
              Running
            </span>
          ) : preparing ? (
            <span
              role="status"
              className="flex items-center gap-1.5 text-xs font-bold text-[var(--sb-muted)]"
            >
              <Spinner />
              Preparing {label}
            </span>
          ) : null}
          <button
            type="button"
            onClick={onClear}
            aria-label="Clear console"
            title="Clear the console output. Your variables, plots, database, and session stay as they are."
            className="rounded border border-[var(--sb-border)] px-1.5 py-0.5 text-xs font-bold text-[var(--sb-muted)] hover:border-[var(--sb-accent)] hover:text-[var(--sb-accent)]"
          >
            Clear
          </button>
        </div>
      </header>
```

Note: the accessible name is `Clear console` (from `aria-label`), while the visible
label is the shorter `Clear`. The title copy has no em dashes.

- [ ] **Step 6: Run the presence test and confirm it passes**

Run: `npm run test:e2e -- sandbox.spec.ts -g "Clear button"`
Expected: PASS. Button visible on Python and SQL; Restart still separate; axe clean.

- [ ] **Step 7: Write the failing behavioral test (SQL, real execution)**

SQL is the right language to prove the "does not reset variables/db/session"
requirement, and it is lightweight: `@sqlite.org/sqlite-wasm`, no prewarm, no network,
deterministic. Add to the same describe block:

```typescript
  test("Clear empties SQL console output but keeps tables and the db session", async ({
    page,
  }) => {
    // Real sqlite-wasm execution; give it generous headroom for first compile.
    test.setTimeout(120_000);

    await page.goto("/ai-sandbox");
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
```

- [ ] **Step 8: Run the behavioral test and confirm it passes**

Run: `npm run test:e2e -- sandbox.spec.ts -g "keeps tables"`
Expected: PASS. Output clears, the Tables pane stays populated, and `SELECT SUM(n)`
returns 6 after Clear, proving the session and database survived.

Fallback if this proves flaky in CI: the surrounding suite deliberately avoids WASM
execution. If sqlite-wasm flakes, gate this one test behind an opt-in env flag like
the existing live-network test does, at the top of the test body:

```typescript
    test.skip(
      process.env.CHATISA_LIVE_NET !== "1",
      "runs real sqlite-wasm; opt in with CHATISA_LIVE_NET=1",
    );
```

- [ ] **Step 9: Verify the task (no commit)**

This project does not commit. Verify instead:

Run: `npm run typecheck`
Expected: no errors.

Run: `npm run lint`
Expected: no errors for the changed files.

Run: `npm run test:e2e -- sandbox.spec.ts`
Expected: the full sandbox suite passes, including the two new Clear tests. Leave the
working tree uncommitted.

---

## Task 2: Scrollable editor for long and wide scripts

Give the editor normal vertical scrolling when a script is taller than its panel, and
horizontal scrolling when a line is wider than the editor (line wrapping is not
enabled), consistently for R, Python, and SQL. The editor must not grow past its
panel, and the cursor must stay visible while navigating.

**Files:**
- Modify: `web/components/run/CodeEditor.tsx` (`themeExtensions`, lines 340-423)
- Test: `web/tests/e2e/sandbox.spec.ts`

**Interfaces:**
- Consumes: `themeExtensions(view, lang, tags, dark, fillHeight, languageId)` and the
  `fillHeight` flag the Sandbox passes (`CodeEditor ... fillHeight`,
  Sandbox.tsx:657). The host wrapper is already `h-full overflow-hidden`
  (CodeEditor.tsx:164).
- Produces: no signature change. `.cm-editor` gets `height: 100%` and `.cm-scroller`
  an explicit `overflow: auto` in fill-height mode, so the panel bounds the editor and
  the scroller scrolls.

**Why this is the fix (root cause).** In fill-height mode the code sets
`.cm-scroller { maxHeight: "none" }` but never gives `.cm-editor` (the `"&"` selector)
a height. CodeMirror's editor defaults to auto height, so it grows with the content;
the wrapper is `overflow-hidden`, so a long script is clipped rather than scrolled.
Setting `.cm-editor` to `height: 100%` bounds it to the panel, and its `.cm-scroller`
(which already defaults to `overflow: auto`) then scrolls. This is CodeMirror's
documented fixed-height pattern.

- [ ] **Step 1: Write the failing vertical-scroll test**

Add to the `test.describe("AI Sandbox", ...)` block:

```typescript
  for (const lang of [
    { radio: "Python", name: /Python code/i },
    { radio: "R", name: /R code/i },
    { radio: "SQL", name: /SQL code/i },
  ]) {
    test(`${lang.radio} editor scrolls a long script within its panel`, async ({
      page,
    }) => {
      await page.goto("/ai-sandbox");
      if (lang.radio !== "Python") {
        await page.getByRole("radio", { name: lang.radio }).click();
      }
      const editor = page.getByRole("textbox", { name: lang.name });
      await expect(editor).toBeVisible();

      // Replace the starter with a script far taller than the panel.
      await editor.click();
      await page.keyboard.press("ControlOrMeta+A");
      await page.keyboard.press("Delete");
      await page.keyboard.insertText(
        Array.from({ length: 200 }, (_, i) => `a${i} <- ${i}`).join("\n"),
      );

      // The content overflows: the scroller is taller inside than its box.
      const scroller = page.locator(".cm-scroller");
      const overflow = await scroller.evaluate(
        (el) => el.scrollHeight - el.clientHeight,
      );
      expect(overflow).toBeGreaterThan(100);

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
```

- [ ] **Step 2: Run the vertical-scroll test and confirm it fails**

Run: `npm run test:e2e -- sandbox.spec.ts -g "scrolls a long script"`
Expected: FAIL. Before the fix, `.cm-editor` grows with the 200-line content, so
`box.y + box.height` exceeds the viewport height (the editor expands past its panel)
and/or the scroller does not overflow. At least one assertion fails for each language.

- [ ] **Step 3: Apply the height fix in `themeExtensions`**

In `web/components/run/CodeEditor.tsx`, inside `themeExtensions`, just below the
existing `const maxHeight = fillHeight ? "none" : "30rem";` (line 348), add:

```typescript
  // In fill-height mode the editor must match its panel so the scroller (which
  // defaults to overflow:auto) scrolls a long or wide script instead of growing
  // past the overflow-hidden wrapper. Inline (capped) mode keeps auto height.
  const height = fillHeight ? "100%" : "auto";
```

In the light-theme branch (the `if (!dark)` return, lines 368-380), update the `"&"`
rule and the `.cm-scroller` rule:

```typescript
      view.EditorView.theme({
        "&": { backgroundColor: "transparent", fontSize: "0.875rem", height },
        "&.cm-focused": { outline: "none" },
        ".cm-content": { fontFamily: MONO },
        ".cm-gutters": { backgroundColor: "transparent", border: "none" },
        ".cm-scroller": { maxHeight, overflow: "auto" },
        // Ghost (AI suggestion) text: clearly visible on the light background.
        ".cm-ghost-text": { color: "#6f685c" },
      }),
```

In the dark-theme branch (lines 401-419), update the same two rules:

```typescript
      {
        "&": { backgroundColor: "transparent", color: "#eaeaea", fontSize: "0.875rem", height },
        "&.cm-focused": { outline: "none" },
        ".cm-content": { fontFamily: MONO, caretColor: "#eaeaea" },
        ".cm-cursor, .cm-dropCursor": { borderLeftColor: "#eaeaea" },
        "&.cm-focused .cm-selectionBackground, .cm-selectionBackground, .cm-content ::selection":
          { backgroundColor: "#3a3a3a" },
        ".cm-gutters": {
          backgroundColor: "transparent",
          color: "#5a5a5a",
          border: "none",
        },
        ".cm-activeLine": { backgroundColor: "rgba(255,255,255,0.04)" },
        ".cm-activeLineGutter": { backgroundColor: "rgba(255,255,255,0.05)" },
        ".cm-scroller": { maxHeight, overflow: "auto" },
        ".cm-ghost-text": { color: "#8a8a8a" },
      },
```

Leave inline (non-fill) mode unchanged: `height` is `"auto"` and `maxHeight` is
`"30rem"` there, preserving the capped, page-scrolling behavior used by the inline
Customize editor.

- [ ] **Step 4: Run the vertical-scroll test and confirm it passes**

Run: `npm run test:e2e -- sandbox.spec.ts -g "scrolls a long script"`
Expected: PASS for Python, R, and SQL. The editor stays within the viewport, the
scroller overflows (is scrollable), and the cursor is visible after ControlOrMeta+End.

- [ ] **Step 5: Write and run the horizontal-scroll test**

Horizontal scrolling must work when a line is wider than the editor and wrapping is
off (no `lineWrapping` extension is configured, so lines do not wrap). Add:

```typescript
  test("editor scrolls horizontally for a very long line (no wrapping)", async ({
    page,
  }) => {
    await page.goto("/ai-sandbox");
    const editor = page.getByRole("textbox", { name: /Python code/i });
    await expect(editor).toBeVisible();

    await editor.click();
    await page.keyboard.press("ControlOrMeta+A");
    await page.keyboard.press("Delete");
    await page.keyboard.insertText("x = " + "1234567890".repeat(80));

    const scroller = page.locator(".cm-scroller");
    const overflowX = await scroller.evaluate(
      (el) => el.scrollWidth - el.clientWidth,
    );
    expect(overflowX).toBeGreaterThan(50);
  });
```

Run: `npm run test:e2e -- sandbox.spec.ts -g "scrolls horizontally"`
Expected: PASS. The long line makes the scroller wider inside than its box, so it
scrolls horizontally.

- [ ] **Step 6: Verify the task (no commit)**

Run: `npm run typecheck`
Expected: no errors.

Run: `npm run lint`
Expected: no errors for the changed files.

Run: `npm run test:e2e -- sandbox.spec.ts`
Expected: the full sandbox suite passes, including the new scroll tests and the
existing shell/axe tests (confirming the height change did not regress the layout or
accessibility). Leave the working tree uncommitted.

---

## Self-Review

**Spec coverage (requirement A and C):**

- A: visible Clear button in the console panel — Task 1, Steps 5-6. Removes all
  console/query output — Task 1 Step 3 (`setEntries([])`). Does not modify the script,
  variables, libraries, db connections, plots, or session — Task 1 Steps 3, 7-8
  (`clearConsole` touches only `entries`; the SQL behavioral test proves tables and
  the db session survive). SQL clears results/messages/errors without disconnecting —
  Task 1 Steps 7-8. Optional shortcut — intentionally out of scope; noted in Global
  Constraints, the visible button is the priority.
- C: normal vertical scrolling when a script exceeds the visible area — Task 2 Steps
  1-4. Cursor stays visible — Task 2 Step 1 (cursor-visible assertion). Editor does not
  expand beyond its panel — Task 2 Step 1 (viewport-bound assertion). Horizontal
  scrolling when a line is wider and wrapping is off — Task 2 Step 5. Consistent for R,
  Python, SQL — Task 2 Step 1 loops all three languages. Mouse-wheel / trackbar /
  Page Up-Down all use the same bounded `.cm-scroller`, so they follow from the same
  fix; keyboard navigation visibility is directly asserted.

**Placeholder scan:** No TBDs. Every code step shows the final code. Commands are exact
and adapted to the no-commit rule (verify steps replace commit steps).

**Type consistency:** `clearConsole: () => void` in `Workspace` is passed as
`onClear`; `ConsolePane`'s prop is `onClear: () => void`. Accessible name is
`Clear console` (aria-label) in both the implementation and every test. `themeExtensions`
keeps its existing signature; only rule values change.

## Handoff note

Per the task that produced this plan, this is a planning document only; do not build
from it in this session. When execution is scheduled, use
superpowers:subagent-driven-development (fresh subagent per task, review between) or
superpowers:executing-plans (batched with checkpoints), and remember: no git commits,
verify with the commands above instead.
