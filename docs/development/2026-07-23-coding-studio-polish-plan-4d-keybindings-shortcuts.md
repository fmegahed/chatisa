# Slice 4 — D — Editor keybindings (native pipe, comment toggle) and a Shortcuts helper — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is a planning document only; do not build from it in the session that produced it.

**Goal.** Add three editor improvements plus a discoverability helper to the Coding
Studio: (1) the native R pipe ` |> ` on Ctrl/Cmd+Shift+M in R mode only; (2) a
toggle-comment on Ctrl/Cmd+/ for R, Python and SQL; and (3) a Shortcuts button in the
toolbar that opens a modal listing every editor shortcut so students can discover them.

**Architecture.** All three keys live in the CodeMirror editor. The pipe and the comment
toggle are added as one new, small `Prec.high` keymap (`editorKeymap`) in
`components/run/CodeEditor.tsx`, alongside the existing `runKeymap` and the separately
built `helpKeymap`. This keymap needs no React callbacks and no new component props: the
comment toggle is a self-contained CodeMirror command and the pipe is a self-contained
document edit. The pure, DOM-free pieces (the pipe-insertion transaction spec, the
platform modifier label, and the shortcut list) move to two small `lib/sandbox` modules
so Vitest can exercise them. The Shortcuts helper is a new client modal component
(`ShortcutsDialog.tsx`) opened by a new toolbar button in `Sandbox.tsx`; it is additive
and does not touch the concurrently built HELP-pane work.

**Tech Stack:** Next.js (this repo's vendored build; read `node_modules/next/dist/docs/`
before touching any Next API, though this slice touches none), React 19, CodeMirror 6
(`codemirror` basicSetup, `@codemirror/view`, `@codemirror/state`, `@codemirror/commands`,
`@codemirror/language` `StreamLanguage`, `@codemirror/legacy-modes/mode/r`,
`@codemirror/lang-python`, `@codemirror/lang-sql`), Vitest for the pure helpers,
Playwright + `@axe-core/playwright` for e2e (the `tests/e2e/sandbox.spec.ts` pattern).

## Global Constraints

- **No em dashes in any user-facing copy** (button labels, tooltips, dialog text, aria
  labels). Use commas, parentheses, or separate sentences. Code and this plan may use them.
- **No new git commits.** The repo's rule is that commits are made only when the user asks.
  Every task below ends with a **Verify** step (run the tests / typecheck) in place of a commit.
- **WCAG 2.1 AA.** New UI must pass `@axe-core/playwright` with tags
  `wcag2a, wcag2aa, wcag21a, wcag21aa`, matching the existing sandbox e2e assertions.
- **SSR-safe platform detection.** The app is Next App Router; the dialog is a client
  component. Any read of `navigator` must be guarded (`typeof navigator !== "undefined"`)
  and must default to Ctrl on the server / first render.
- **Additive only.** Do not modify `runKeymap`, `helpKeymap`, `helpMouse`, the `onHelp`
  prop, or the `helpTarget` state machine. Append a new keymap, a new toolbar button, a new
  dialog component, and two new pure lib modules.
- **Native pipe, not magrittr.** The inserted token is exactly ` |> ` (space, pipe, greater
  than, space), never `%>%`.

---

## Verified findings (read before implementing)

These were confirmed against the installed `node_modules`, not assumed. They shape the plan.

1. **Comment toggle already works in all three languages today.** `basicSetup` (from the
   `codemirror` package, `dist/index.js:67-75`) bundles `defaultKeymap`, and `defaultKeymap`
   binds `Mod-/` to `toggleComment` (`@codemirror/commands/dist/index.js:1797`).
   `toggleComment` reads the line-comment token from the language via
   `state.languageDataAt("commentTokens", pos)` (`@codemirror/commands/dist/index.js:61`).
   - **Python** (`@codemirror/lang-python/dist/index.js:293`) provides `commentTokens: {line: "#"}`.
   - **SQL** (`@codemirror/lang-sql/dist/index.js:679`) provides `commentTokens: {line: "--", block: {...}}`.
   - **R** — contrary to the original assumption, the legacy stream mode
     **does** provide `commentTokens: {line: "#"}` (`@codemirror/legacy-modes/mode/r.js:167-171`),
     and `StreamLanguage.define(spec)` propagates `spec.languageData` into the language-data
     facet (`@codemirror/language/dist/index.js:2230-2249`, `defineLanguageFacet(parser.languageData)`).
     So `Mod-/` already toggles `#` comments in R with `@codemirror/legacy-modes@6.5.3`.

   **Decision: still add an explicit `Mod-/ -> toggleComment` binding (in our own
   `Prec.high` keymap) AND defensively attach R's `commentTokens`.** Justification:
   - *Robustness.* The R comment token is incidental to the legacy grammar. The original
     spec author believed R lacked it, which shows how fragile a cross-version assumption is;
     a future `@codemirror/legacy-modes` bump could drop or change it. Attaching
     `commentTokens: {line: "#"}` to the R language ourselves (`rLang.data.of(...)`) removes
     that dependency. `toggleComment` reads `languageDataAt`, which returns the nearest match,
     and both sources agree on `{line: "#"}`, so there is no conflict.
   - *Ownership and honesty.* The Shortcuts dialog will claim "Toggle comment (Ctrl/Cmd+/)".
     Binding it in our code at `Prec.high` makes that claim true-by-construction and immune to
     any future extension that might also want `Mod-/`.
   - *No double-toggle.* Keymaps run in precedence order and stop at the first command that
     returns true (`@codemirror/view/dist/index.js:9161-9184`, `runHandlers`). Our `Prec.high`
     binding runs before `basicSetup`'s `defaultKeymap`, returns true, and `defaultKeymap`'s
     `Mod-/` never fires. One toggle, once.

2. **`Mod-Shift-M` is NOT free.** `basicSetup` includes `lintKeymap`
   (`codemirror/dist/index.js:74`), and `lintKeymap` binds `Mod-Shift-m` to `openLintPanel`
   (`@codemirror/lint/dist/index.js:294`). `defaultKeymap` binds only `Ctrl-m` / mac
   `Shift-Alt-m` to `toggleTabFocusMode` (a different chord), and `runKeymap` does not use it.
   The collision is with the lint panel.

   **Design that resolves it cleanly (first-match-wins):** register the pipe in our
   `Prec.high` keymap, gated on `languageId === "r"`.
   - In **R**, the handler inserts ` |> ` and returns `true`; because our keymap has higher
     precedence, `openLintPanel` never runs. R students get the pipe.
   - In **Python/SQL**, the handler returns `false`. Per `runHandlers` semantics, a `false`
     return with **no** binding-level `preventDefault` lets the event fall through to the
     lower-precedence `lintKeymap`, so `Mod-Shift-M` still opens the lint panel exactly as it
     does today. This is why the pipe binding must **not** set `preventDefault: true`: with
     `preventDefault: true` and a `false` return, `runHandlers` sets `prevented = true` which
     forces `handled = true` (`@codemirror/view/dist/index.js:9172-9207`), swallowing the key
     and breaking the fall-through. Returning `true` in the R branch is what stops the event;
     CodeMirror prevents the browser default for a handled key on its own.

   **Caveat (note in the dialog testing section, not the UI):** in Firefox, `Ctrl+Shift+M`
   is a browser-level shortcut (Responsive Design Mode) that fires in the browser chrome and
   may not be preventable from the page. Chrome and Safari do not claim it. This is inherent
   to running an RStudio-style chord in a browser tab; document it as a known limitation.

3. **Autocomplete is `Ctrl-Space` on every platform.** `basicSetup` includes
   `completionKeymap` (via `autocompletion()`), which binds `Ctrl-Space` to `startCompletion`
   (`@codemirror/autocomplete/dist/index.js:2063-2064`), with mac-only alternates
   `Alt-\`` / `Alt-i`. So the dialog lists autocomplete as **Ctrl+Space** (Ctrl, not Cmd) on
   all platforms.

4. **`@codemirror/commands` is installed (6.10.4) but is not a direct dependency** in
   `package.json` (it arrives transitively through `codemirror`). Task 3 adds it explicitly so
   the direct `import("@codemirror/commands")` for `toggleComment` is declared, not implicit.

### Final verified shortcut list (only real bindings)

| Action | Keys | Source | Scope |
| --- | --- | --- | --- |
| Run statement or selection | Ctrl/Cmd+Enter | `runKeymap` `Mod-Enter` (CodeEditor.tsx:325) | all |
| Run whole script | Ctrl/Cmd+Shift+Enter | `runKeymap` `Mod-Shift-Enter` (CodeEditor.tsx:354) | all |
| Source silently | Ctrl/Cmd+Shift+S | `runKeymap` `Mod-Shift-s` (CodeEditor.tsx:362) | all |
| Insert pipe | Ctrl/Cmd+Shift+M | **new** `editorKeymap` `Mod-Shift-m` (this slice) | **R only** |
| Toggle comment | Ctrl/Cmd+/ | **new** `editorKeymap` `Mod-/` + `basicSetup` default | all |
| Documentation for symbol | Ctrl/Cmd+Click or F1 | `helpMouse` + `helpKeymap` (built in the concurrent HELP slice) | all |
| Autocomplete | Ctrl+Space | `completionKeymap` (basicSetup) | all |

The "Documentation for symbol" row is listed even though its feature ships in a separate,
concurrent slice, per the requirement that the discoverability helper cover it. If, at
implementation time, `onHelp`/`helpKeymap` are not yet merged into `CodeEditor.tsx`, keep the
row (the shortcut is part of the product) but the e2e assertion for it may be skipped until
that slice lands.

---

## File structure

- **Create** `lib/sandbox/editor-keys.ts` — pure helpers for the editor keymap: the
  pipe-insertion transaction spec builder. No DOM, no React, no CodeMirror runtime imports
  (plain data in, plain data out). Vitest-tested.
- **Create** `lib/sandbox/shortcuts.ts` — the platform modifier label and the grouped
  shortcut list given an `isMac` flag. Pure. Vitest-tested.
- **Create** `tests/unit/sandbox-editor-keys.test.ts` — Vitest for the pipe builder.
- **Create** `tests/unit/sandbox-shortcuts.test.ts` — Vitest for the label and list.
- **Create** `components/sandbox/ShortcutsDialog.tsx` — the modal, mirroring
  `components/sandbox/ImportDialog.tsx` conventions (`role="dialog"`, `aria-modal`, Esc and
  backdrop close, `sb-root` themed panel) plus a focus trap and focus-return.
- **Modify** `components/run/CodeEditor.tsx` — (a) return R `commentTokens` from
  `loadLanguageMode`; (b) import `@codemirror/commands` in `loadCodeMirror`; (c) add the
  `editorKeymap(view, cmState, commands, languageId)` function and wire it into the
  `EditorView` extensions array.
- **Modify** `components/sandbox/Sandbox.tsx` — add a "Shortcuts" toolbar button in
  `Toolbar` and the open/close state plus the `<ShortcutsDialog>` render in `Workspace`.
- **Modify** `package.json` — add `@codemirror/commands` to `dependencies`.
- **Modify** `tests/e2e/sandbox.spec.ts` — add the dialog and keybinding e2e tests.

---

## Task 1: Pure pipe-insertion transaction builder

**Files:**
- Create: `lib/sandbox/editor-keys.ts`
- Test: `tests/unit/sandbox-editor-keys.test.ts`

**Interfaces:**
- Produces: `PIPE_TOKEN: " |> "` and
  `buildPipeInsertion(sel: { from: number; to: number }): { from: number; to: number; insert: string; anchor: number }`.
  Task 3 consumes both to build a CodeMirror transaction: it replaces `[from, to)` with
  `insert` and sets the caret to `anchor`. `anchor` is `from + PIPE_TOKEN.length`, so the caret
  lands immediately after the inserted ` |> ` whether the caret was empty or a range was selected.

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/sandbox-editor-keys.test.ts
import { describe, expect, it } from "vitest";
import { PIPE_TOKEN, buildPipeInsertion } from "@/lib/sandbox/editor-keys";

describe("native pipe insertion", () => {
  it("uses the native pipe with surrounding spaces, not magrittr", () => {
    expect(PIPE_TOKEN).toBe(" |> ");
    expect(PIPE_TOKEN).not.toContain("%>%");
  });

  it("inserts at an empty caret and places the caret after the pipe", () => {
    // caret at offset 5, nothing selected
    expect(buildPipeInsertion({ from: 5, to: 5 })).toEqual({
      from: 5,
      to: 5,
      insert: " |> ",
      anchor: 9, // 5 + 4
    });
  });

  it("replaces a selection and places the caret after the pipe", () => {
    // "df|filter" style: a 5-char selection [2,7) is replaced by the pipe
    expect(buildPipeInsertion({ from: 2, to: 7 })).toEqual({
      from: 2,
      to: 7,
      insert: " |> ",
      anchor: 6, // 2 + 4
    });
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run tests/unit/sandbox-editor-keys.test.ts`
Expected: FAIL, cannot resolve `@/lib/sandbox/editor-keys`.

- [ ] **Step 3: Write the minimal implementation**

```ts
// lib/sandbox/editor-keys.ts

/** The native R pipe, inserted with surrounding spaces (RStudio's insert-pipe
 *  behaviour). Deliberately the native `|>`, never the magrittr `%>%`. */
export const PIPE_TOKEN = " |> ";

/**
 * Pure description of the edit that inserts the native pipe. Given the current
 * main selection range, it replaces that range (empty or not) with ` |> ` and
 * reports where the caret should land: immediately after the inserted token.
 * DOM-free so it can be unit tested; Task 3 turns this into a CodeMirror
 * transaction via `view.dispatch`.
 */
export function buildPipeInsertion(sel: { from: number; to: number }): {
  from: number;
  to: number;
  insert: string;
  anchor: number;
} {
  return {
    from: sel.from,
    to: sel.to,
    insert: PIPE_TOKEN,
    anchor: sel.from + PIPE_TOKEN.length,
  };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run tests/unit/sandbox-editor-keys.test.ts`
Expected: PASS (3 tests).

- [ ] **Step 5: Verify (in place of a commit)**

Run: `npx vitest run tests/unit/sandbox-editor-keys.test.ts && npx tsc --noEmit`
Expected: tests PASS, no type errors introduced by the new file.

---

## Task 2: Pure platform label and shortcut list

**Files:**
- Create: `lib/sandbox/shortcuts.ts`
- Test: `tests/unit/sandbox-shortcuts.test.ts`

**Interfaces:**
- Produces:
  - `modLabel(isMac: boolean): "Cmd" | "Ctrl"` — the platform modifier glyph text.
  - `type Shortcut = { action: string; keys: string; scope?: "R only" }`.
  - `type ShortcutGroup = { title: string; items: Shortcut[] }`.
  - `shortcutGroups(isMac: boolean): ShortcutGroup[]` — the full grouped list, with keys
    already rendered for the platform (Cmd vs Ctrl). Autocomplete is always "Ctrl+Space".
  - `detectIsMac(): boolean` — SSR-safe navigator probe, default `false` (Ctrl).
- Consumed by: Task 4 (`ShortcutsDialog`).

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/sandbox-shortcuts.test.ts
import { describe, expect, it } from "vitest";
import { modLabel, shortcutGroups } from "@/lib/sandbox/shortcuts";

describe("platform modifier label", () => {
  it("is Cmd on macOS and Ctrl elsewhere", () => {
    expect(modLabel(true)).toBe("Cmd");
    expect(modLabel(false)).toBe("Ctrl");
  });
});

describe("shortcut list", () => {
  it("renders the modifier per platform", () => {
    const mac = shortcutGroups(true).flatMap((g) => g.items);
    const win = shortcutGroups(false).flatMap((g) => g.items);
    expect(mac.find((s) => s.action.startsWith("Run statement"))?.keys).toBe(
      "Cmd+Enter",
    );
    expect(win.find((s) => s.action.startsWith("Run statement"))?.keys).toBe(
      "Ctrl+Enter",
    );
  });

  it("lists exactly the real bindings, and marks the pipe R only", () => {
    const items = shortcutGroups(false).flatMap((g) => g.items);
    const byAction = Object.fromEntries(items.map((s) => [s.action, s]));
    expect(byAction["Run statement or selection"].keys).toBe("Ctrl+Enter");
    expect(byAction["Run whole script"].keys).toBe("Ctrl+Shift+Enter");
    expect(byAction["Source silently"].keys).toBe("Ctrl+Shift+S");
    expect(byAction["Insert pipe"].keys).toBe("Ctrl+Shift+M");
    expect(byAction["Insert pipe"].scope).toBe("R only");
    expect(byAction["Toggle comment"].keys).toBe("Ctrl+/");
    expect(byAction["Documentation for symbol"].keys).toBe(
      "Ctrl+Click or F1",
    );
    // Autocomplete is Ctrl on every platform, never Cmd.
    expect(byAction["Autocomplete"].keys).toBe("Ctrl+Space");
    expect(shortcutGroups(true).flatMap((g) => g.items).find(
      (s) => s.action === "Autocomplete",
    )?.keys).toBe("Ctrl+Space");
  });

  it("contains no em dashes in any copy", () => {
    const text = shortcutGroups(true)
      .flatMap((g) => [g.title, ...g.items.map((s) => `${s.action} ${s.keys} ${s.scope ?? ""}`)])
      .join(" ");
    expect(text).not.toContain("—");
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `npx vitest run tests/unit/sandbox-shortcuts.test.ts`
Expected: FAIL, cannot resolve `@/lib/sandbox/shortcuts`.

- [ ] **Step 3: Write the minimal implementation**

```ts
// lib/sandbox/shortcuts.ts

export type Shortcut = {
  /** What the shortcut does, in plain words (no em dashes). */
  action: string;
  /** The rendered keys for the current platform, e.g. "Ctrl+Enter". */
  keys: string;
  /** Present only when the shortcut is language-scoped. */
  scope?: "R only";
};

export type ShortcutGroup = { title: string; items: Shortcut[] };

/** The platform modifier glyph text: Cmd on macOS, Ctrl elsewhere. */
export function modLabel(isMac: boolean): "Cmd" | "Ctrl" {
  return isMac ? "Cmd" : "Ctrl";
}

/**
 * SSR-safe platform probe. Returns false (Ctrl) on the server and on any
 * environment without a navigator, so the first render is deterministic; the
 * dialog refines it in an effect after mount.
 */
export function detectIsMac(): boolean {
  if (typeof navigator === "undefined") return false;
  const s = `${navigator.platform ?? ""} ${navigator.userAgent ?? ""}`;
  return /Mac|iPhone|iPad|iPod/i.test(s);
}

/**
 * The full grouped shortcut list, keys already rendered for the platform. Only
 * bindings that actually exist in the editor are listed (see the plan's verified
 * table). Autocomplete is Ctrl+Space on every platform, so it is not templated.
 */
export function shortcutGroups(isMac: boolean): ShortcutGroup[] {
  const mod = modLabel(isMac);
  return [
    {
      title: "Run code",
      items: [
        { action: "Run statement or selection", keys: `${mod}+Enter` },
        { action: "Run whole script", keys: `${mod}+Shift+Enter` },
        { action: "Source silently", keys: `${mod}+Shift+S` },
      ],
    },
    {
      title: "Edit",
      items: [
        { action: "Insert pipe", keys: `${mod}+Shift+M`, scope: "R only" },
        { action: "Toggle comment", keys: `${mod}+/` },
      ],
    },
    {
      title: "Assist",
      items: [
        { action: "Documentation for symbol", keys: `${mod}+Click or F1` },
        { action: "Autocomplete", keys: "Ctrl+Space" },
      ],
    },
  ];
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run tests/unit/sandbox-shortcuts.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 5: Verify (in place of a commit)**

Run: `npx vitest run tests/unit/sandbox-shortcuts.test.ts && npx tsc --noEmit`
Expected: tests PASS, no type errors.

---

## Task 3: Wire the pipe and comment keymap into CodeEditor

**Files:**
- Modify: `package.json` (add `@codemirror/commands`)
- Modify: `components/run/CodeEditor.tsx`
  - `loadLanguageMode` (currently `CodeEditor.tsx:283-300`) — attach R `commentTokens`.
  - `LoadedEditor` interface + `loadCodeMirror` (`CodeEditor.tsx:203-246`) — import `@codemirror/commands`.
  - The `EditorView` extensions array (`CodeEditor.tsx:106-140`) — add `editorKeymap(...)`.
  - Add the `editorKeymap` function next to `runKeymap`.

**Interfaces:**
- Consumes: `PIPE_TOKEN`, `buildPipeInsertion` from `@/lib/sandbox/editor-keys` (Task 1).
- Produces: an always-on `editorKeymap(view, cmState, commands, languageId): Extension`
  registered at `Prec.high` in the editor. No new component props.

- [ ] **Step 1: Add `@codemirror/commands` to dependencies**

In `package.json`, under `"dependencies"`, add the line (keep alphabetical order, it sits
just before `@codemirror/lang-python`). Use the already-installed version:

```json
    "@codemirror/commands": "^6.10.4",
```

- [ ] **Step 2: Attach R `commentTokens` in `loadLanguageMode`**

Rationale in the verified-findings section: this removes R's dependency on the legacy
grammar incidentally shipping `commentTokens`. Replace the R branch of `loadLanguageMode`
(`CodeEditor.tsx:290-298`):

```ts
  if (languageId === "r") {
    // R has no first-party CodeMirror 6 grammar; the legacy stream mode gives
    // comment/keyword/string/function highlighting, which is what students expect.
    const [{ StreamLanguage }, { r }] = await Promise.all([
      import("@codemirror/language"),
      import("@codemirror/legacy-modes/mode/r"),
    ]);
    const rLang = StreamLanguage.define(r);
    // Own the comment token rather than relying on the legacy grammar to keep
    // providing it, so Mod-/ toggles `#` comments in R regardless of the mode's
    // version. `toggleComment` reads this via state.languageDataAt.
    return [rLang, rLang.data.of({ commentTokens: { line: "#" } })];
  }
```

(`loadLanguageMode`'s return type is `Extension`, and a `Extension[]` is a valid `Extension`,
so returning the array needs no signature change. It is consumed as `langExt` at
`CodeEditor.tsx:108`.)

- [ ] **Step 3: Import `@codemirror/commands` in `loadCodeMirror`**

Add `commands` to the `LoadedEditor` interface (`CodeEditor.tsx:203-215`):

```ts
  commands: typeof import("@codemirror/commands");
```

Add the import to the `Promise.all` in `loadCodeMirror` (`CodeEditor.tsx:219-232`). Add
`import("@codemirror/commands")` to the array and destructure it, then include it in the
returned object:

```ts
async function loadCodeMirror(languageId: string): Promise<LoadedEditor> {
  const [cm, view, lang, state, autocomplete, highlight, commands, inline, langStructure, helpDocs, lint, langExt] =
    await Promise.all([
      import("codemirror"),
      import("@codemirror/view"),
      import("@codemirror/language"),
      import("@codemirror/state"),
      import("@codemirror/autocomplete"),
      import("@lezer/highlight"),
      import("@codemirror/commands"),
      import("@/lib/sandbox/inline-completion"),
      import("@/lib/sandbox/lang-structure"),
      import("@/lib/sandbox/help-docs"),
      import("@codemirror/lint"),
      loadLanguageMode(languageId),
    ]);
  return {
    view,
    cm,
    lang,
    state,
    autocomplete,
    tags: highlight.tags,
    commands,
    inline,
    langStructure,
    helpDocs,
    lint,
    langExt,
  };
}
```

- [ ] **Step 4: Add the `editorKeymap` function**

Add this next to `runKeymap` (for example after it, near `CodeEditor.tsx:371`). It imports
the pure pipe builder at the top of the file:

```ts
import { PIPE_TOKEN, buildPipeInsertion } from "@/lib/sandbox/editor-keys";
```

```ts
/**
 * Editor edit-keys, added at Prec.high so they win over basicSetup:
 *  - Mod-/ toggles a line comment in every language. R, Python and SQL all carry
 *    a commentTokens line token, so `toggleComment` knows the prefix. Ours runs
 *    first and returns true, so basicSetup's default Mod-/ never double-toggles.
 *  - Mod-Shift-m inserts the native R pipe ` |> ` in R only. On Python and SQL it
 *    returns false (no preventDefault) so the key falls through to basicSetup's
 *    lint-panel binding exactly as before. It is R-gated the same way runKeymap
 *    derives its language.
 */
function editorKeymap(
  view: LoadedEditor["view"],
  cmState: LoadedEditor["state"],
  commands: LoadedEditor["commands"],
  languageId: string,
): Extension {
  const isR = languageId === "r";
  return cmState.Prec.high(
    view.keymap.of([
      {
        key: "Mod-/",
        preventDefault: true,
        run: commands.toggleComment,
      },
      {
        key: "Mod-Shift-m",
        // No preventDefault: on non-R we must return false AND let the event fall
        // through to basicSetup's lint-panel binding, which requires that this
        // binding not mark the event prevented.
        run: (editor) => {
          if (!isR) return false;
          const sel = editor.state.selection.main;
          const edit = buildPipeInsertion({ from: sel.from, to: sel.to });
          editor.dispatch(
            editor.state.update({
              changes: { from: edit.from, to: edit.to, insert: edit.insert },
              selection: { anchor: edit.anchor },
              scrollIntoView: true,
              userEvent: "input.pipe",
            }),
          );
          return true;
        },
      },
    ]),
  );
}
```

(`PIPE_TOKEN` is imported for parity/reference; the length lives inside `buildPipeInsertion`,
so it is fine if `editorKeymap` uses only `buildPipeInsertion`. If your linter flags an unused
import, import only `buildPipeInsertion`.)

- [ ] **Step 5: Register `editorKeymap` in the extensions array**

In the `EditorView` construction (`CodeEditor.tsx:106-140`), add one line to the `extensions`
array. It is unconditional (the comment toggle applies to all languages; the pipe self-gates
to R), so place it right after the `themeExtensions(...)` spread and before the `hasRun` block:

```ts
            ...themeExtensions(view, lang, tags, dark, fillHeight, props.languageId),
            editorKeymap(view, state, commands, props.languageId),
            ...(hasRun
              ? [runKeymap(view, state, runRef, langStructure.statementRangeAt, props.languageId)]
              : []),
```

The effect that builds the editor already re-runs on `props.languageId` change
(`CodeEditor.tsx:158` dependency array), so the R-gating and R `commentTokens` refresh on a
language switch with no new dependency.

- [ ] **Step 6: Verify (in place of a commit)**

Run: `npm run lint && npx tsc --noEmit && npx vitest run tests/unit/sandbox-editor-keys.test.ts`
Expected: lint clean, no type errors, Task 1 tests still PASS.
Manual smoke (optional here, formalized in Task 6): in R, `Ctrl+Shift+M` inserts ` |> `; in
Python/SQL it does nothing to the text; `Ctrl+/` toggles `#` (R/Python) or `--` (SQL).

---

## Task 4: The Shortcuts dialog component

**Files:**
- Create: `components/sandbox/ShortcutsDialog.tsx`

**Interfaces:**
- Consumes: `detectIsMac`, `shortcutGroups`, `type ShortcutGroup` from `@/lib/sandbox/shortcuts` (Task 2).
- Produces: `ShortcutsDialog(props: { dark: boolean; onClose: () => void }): JSX.Element`.
  A modal that Task 5 renders when its open state is true and hides otherwise. It is
  self-contained: `role="dialog"`, `aria-modal="true"`, labelled by its heading, Esc and
  backdrop close, focus moved into the panel on mount and trapped, focus returned to the
  trigger on close.

- [ ] **Step 1: Write the component**

Mirror `ImportDialog.tsx` for the themed shell (`sb-root`, `data-sb-theme`, the fixed
backdrop, Esc/backdrop close), and add a focus trap plus focus-return. Platform detection is
an effect (default Ctrl for SSR / first paint, refined to Cmd on macOS after mount).

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { detectIsMac, shortcutGroups } from "@/lib/sandbox/shortcuts";

/**
 * A modal listing every Coding Studio editor shortcut, so students can discover
 * them. Platform-aware (Cmd on macOS, Ctrl elsewhere), SSR-safe (defaults to
 * Ctrl until an effect confirms the platform), and accessible: labelled dialog,
 * focus trapped inside, Esc and backdrop close, focus returned to the trigger.
 */
export function ShortcutsDialog(props: { dark: boolean; onClose: () => void }) {
  const panelRef = useRef<HTMLDivElement | null>(null);
  // The element focused before the dialog opened, restored on close.
  const returnFocusRef = useRef<HTMLElement | null>(null);
  // Default Ctrl (SSR / first render is deterministic); refine after mount.
  const [isMac, setIsMac] = useState(false);

  useEffect(() => {
    setIsMac(detectIsMac());
  }, []);

  useEffect(() => {
    returnFocusRef.current = document.activeElement as HTMLElement | null;
    panelRef.current?.focus();
    return () => {
      returnFocusRef.current?.focus?.();
    };
  }, []);

  // Trap Tab within the panel and close on Esc.
  function onKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      e.stopPropagation();
      props.onClose();
      return;
    }
    if (e.key !== "Tab") return;
    const panel = panelRef.current;
    if (!panel) return;
    const focusable = panel.querySelectorAll<HTMLElement>(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
    );
    if (focusable.length === 0) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && active === first) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && active === last) {
      e.preventDefault();
      first.focus();
    }
  }

  const groups = shortcutGroups(isMac);

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="sb-shortcuts-title"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      onKeyDown={onKeyDown}
      onClick={(e) => e.target === e.currentTarget && props.onClose()}
    >
      <div
        ref={panelRef}
        tabIndex={-1}
        data-sb-theme={props.dark ? "dark" : "light"}
        className="sb-root flex max-h-[85vh] w-full max-w-md flex-col rounded-card border border-[var(--sb-border)] bg-[var(--sb-bg)] text-[var(--sb-text)] shadow-xl outline-none"
      >
        <div className="flex items-center justify-between border-b border-[var(--sb-border)] px-4 py-3">
          <h2 id="sb-shortcuts-title" className="text-lg font-bold">
            Keyboard shortcuts
          </h2>
          <button
            type="button"
            onClick={props.onClose}
            className="rounded-card border border-[var(--sb-border)] px-2 py-1 text-sm font-bold hover:border-[var(--sb-accent)]"
          >
            Close
          </button>
        </div>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
          {groups.map((group) => (
            <section key={group.title}>
              <h3 className="mb-1 text-xs font-bold uppercase tracking-wide text-[var(--sb-muted)]">
                {group.title}
              </h3>
              <dl className="divide-y divide-[var(--sb-border)]">
                {group.items.map((s) => (
                  <div
                    key={s.action}
                    className="flex items-center justify-between gap-4 py-1.5"
                  >
                    <dt className="text-sm">
                      {s.action}
                      {s.scope ? (
                        <span className="ml-2 rounded border border-[var(--sb-border)] px-1.5 py-0.5 text-xs font-bold uppercase tracking-wide text-[var(--sb-muted)]">
                          {s.scope}
                        </span>
                      ) : null}
                    </dt>
                    <dd>
                      <kbd className="rounded border border-[var(--sb-border)] bg-[var(--sb-panel)] px-2 py-0.5 font-mono text-xs">
                        {s.keys}
                      </kbd>
                    </dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
          <p className="text-xs text-[var(--sb-muted)]">
            Shortcuts work while the editor has focus. On macOS, Cmd replaces Ctrl,
            except Autocomplete which stays Ctrl+Space.
          </p>
        </div>
      </div>
    </div>
  );
}
```

Copy check: no em dashes appear in any string above (the list copy comes from Task 2, also
em-dash-free; verified by that task's test).

- [ ] **Step 2: Verify (in place of a commit)**

Run: `npm run lint && npx tsc --noEmit`
Expected: lint clean, no type errors. (Behaviour is exercised end to end in Task 6.)

---

## Task 5: Add the Shortcuts toolbar button

**Files:**
- Modify: `components/sandbox/Sandbox.tsx`
  - Import `ShortcutsDialog`.
  - `Workspace` (`Sandbox.tsx:296`) — add `shortcutsOpen` state and render the dialog.
  - `Toolbar` (`Sandbox.tsx:812`) — add an `onShortcuts` prop and a "Shortcuts" button.

**Interfaces:**
- Consumes: `ShortcutsDialog` (Task 4). `Toolbar` gains one prop `onShortcuts: () => void`.

- [ ] **Step 1: Import the dialog**

Add near the other `components/sandbox` imports (for example after the `ImportDialog` import
at `Sandbox.tsx:52`):

```ts
import { ShortcutsDialog } from "@/components/sandbox/ShortcutsDialog";
```

- [ ] **Step 2: Add open/close state in `Workspace`**

Add alongside the other `useState` hooks in `Workspace` (for example near `Sandbox.tsx:325`):

```ts
  const [shortcutsOpen, setShortcutsOpen] = useState(false);
```

- [ ] **Step 3: Pass the opener to the Toolbar**

In the `<Toolbar ... />` render (`Sandbox.tsx:637-653`), add the prop:

```tsx
      <Toolbar
        languageId={props.languageId}
        onLanguage={props.onLanguage}
        onRun={() => void runAll()}
        onSource={() => void source()}
        onInsertExample={insertExample}
        onUpload={() => fileInputRef.current?.click()}
        onDownload={download}
        onRestart={restart}
        onShortcuts={() => setShortcutsOpen(true)}
        running={running}
        theme={theme}
        onToggleTheme={props.onToggleTheme}
        chatOpen={props.chatOpen}
        onToggleChat={props.onToggleChat}
        completionsOn={props.completionsOn}
        onToggleCompletions={props.onToggleCompletions}
      />
```

- [ ] **Step 4: Render the dialog**

Add right after the `{uploadFile ? (<ImportDialog .../>) : null}` block (`Sandbox.tsx:691`):

```tsx
      {shortcutsOpen ? (
        <ShortcutsDialog
          dark={theme === "dark"}
          onClose={() => setShortcutsOpen(false)}
        />
      ) : null}
```

- [ ] **Step 5: Add the button and the prop to `Toolbar`**

In the `Toolbar` prop type (`Sandbox.tsx:812-828`), add:

```ts
  onShortcuts: () => void;
```

Add the button in the right-hand button cluster (for example just before the "Ask AI"
button at `Sandbox.tsx:918`). Keep the copy em-dash-free:

```tsx
        <button
          type="button"
          onClick={props.onShortcuts}
          title="See every keyboard shortcut for the editor"
          className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-panel)] px-3 py-1 text-sm font-bold hover:border-[var(--sb-accent)]"
        >
          Shortcuts
        </button>
```

- [ ] **Step 6: Verify (in place of a commit)**

Run: `npm run lint && npx tsc --noEmit`
Expected: lint clean, no type errors.

---

## Task 6: End-to-end tests

**Files:**
- Modify: `tests/e2e/sandbox.spec.ts`

**Interfaces:**
- Consumes the running app at `/ai-sandbox` and the `editorLineText` helper already defined
  in the spec (`sandbox.spec.ts:12`).

Add the tests below inside the existing `test.describe("AI Sandbox", ...)` block so they can
reuse `editorLineText`.

- [ ] **Step 1: Shortcuts dialog opens, lists shortcuts, is Esc-dismissible, and is axe-clean**

```ts
  test("the Shortcuts dialog lists editor shortcuts, closes on Esc, and is axe-clean", async ({
    page,
  }) => {
    await page.goto("/ai-sandbox");
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
    await expect(dialog.getByText("Autocomplete")).toBeVisible();
    // Autocomplete is Ctrl on every platform.
    await expect(dialog.getByText("Ctrl+Space")).toBeVisible();

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
```

- [ ] **Step 2: Native pipe on Ctrl+Shift+M in R**

```ts
  test("Ctrl/Cmd+Shift+M inserts the native pipe in R", async ({ page }) => {
    await page.goto("/ai-sandbox");
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
```

- [ ] **Step 3: Ctrl+Shift+M is a no-op for the text in Python and SQL**

```ts
  test("Ctrl/Cmd+Shift+M does not insert a pipe in Python or SQL", async ({
    page,
  }) => {
    await page.goto("/ai-sandbox");
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
```

- [ ] **Step 4: Toggle comment on Ctrl+/ in each language**

```ts
  test("Ctrl/Cmd+/ toggles a line comment in R, Python, and SQL", async ({
    page,
  }) => {
    await page.goto("/ai-sandbox");

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
```

- [ ] **Step 5: Multi-line comment toggle across a selection (Python)**

```ts
  test("Ctrl/Cmd+/ comments a multi-line selection", async ({ page }) => {
    await page.goto("/ai-sandbox");
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
```

- [ ] **Step 6: Run the e2e suite**

Run: `npx playwright test tests/e2e/sandbox.spec.ts`
Expected: the new tests PASS along with the existing ones.

**Harness caveats to record when running:**
- The comment and pipe keybinding tests type into the real CodeMirror on the page, exactly
  like the existing `Ctrl+Enter` and indentation tests in this spec, so they are practical in
  the harness.
- **Firefox only:** `Ctrl+Shift+M` is the browser's Responsive Design Mode shortcut and may
  be intercepted before the page sees it. If the suite runs a Firefox project, mark Step 2's
  pipe test `test.skip` for Firefox (or scope it to Chromium/WebKit) and verify the R pipe
  manually in Firefox. Chromium and WebKit are unaffected.
- The "Documentation for symbol" row is asserted only as **listed** in the dialog (Step 1),
  not exercised as a keybinding here; the Ctrl/Cmd+Click and F1 behaviours are covered by the
  concurrent HELP slice's own tests (`sandbox.spec.ts` help tests). If that slice has not
  landed when this runs, keep the dialog row assertion (the shortcut is part of the product).

---

## Self-review

**Spec coverage.**
- Native pipe on Ctrl/Cmd+Shift+M, R only, ` |> ` with caret after, selection replaced,
  no-op elsewhere: Task 1 (pure builder) + Task 3 (R-gated keymap) + Task 6 Steps 2, 3.
- Toggle comment on Ctrl/Cmd+/ for R/Python/SQL, `#` and `--`, line and multi-line: Task 3
  (explicit binding + R commentTokens) + Task 6 Steps 4, 5.
- Shortcuts helper button + dialog listing every shortcut: Task 2 (list) + Task 4 (dialog) +
  Task 5 (button) + Task 6 Step 1.
- Platform awareness, SSR-safe: Task 2 `detectIsMac` (guarded, default Ctrl) + Task 4 effect.
- Accessibility (role, aria-modal, labelled, focus trap, Esc, focus return, axe): Task 4 +
  Task 6 Step 1.
- No em dashes: Global Constraints + Task 2's em-dash test + copy checks in Tasks 4, 5.
- No git commits: every task ends in a Verify step.

**Placeholder scan.** Every code step contains complete code; every run step has an exact
command and expected result. No TODO/TBD.

**Type consistency.** `buildPipeInsertion` returns `{ from, to, insert, anchor }` in Task 1
and is consumed with those exact fields in Task 3. `shortcutGroups`/`modLabel`/`detectIsMac`
signatures in Task 2 match their use in Task 4. `ShortcutsDialog(props: { dark, onClose })` in
Task 4 matches the render in Task 5. `Toolbar`'s new `onShortcuts: () => void` prop is added
to both the type and the call site in Task 5.

**One deviation from the original spec, called out deliberately.** The spec assumed R lacks
`commentTokens` and that Mod-/ does nothing in R. Verified against `node_modules`, the
installed `@codemirror/legacy-modes@6.5.3` R mode **does** provide `commentTokens: {line: "#"}`,
and `StreamLanguage.define` propagates it, so Mod-/ already comments R today. The plan still
attaches R's `commentTokens` itself and adds an explicit `Mod-/` binding, for robustness and
ownership (justified in the verified-findings section), rather than relying on that incidental
behaviour.
