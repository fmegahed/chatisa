# Slice 4 — B (revision) — Render runtime documentation inside the HELP pane — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. This is a planning document only; do not build from it in the session that produced it.

**Goal.** Professor feedback on the just-shipped Slice 4 (B): the HELP tab should
*render* documentation in the pane, not only offer a click-out link. Revise the HELP
pane so that, on Ctrl/Cmd+Click or F1, it shows the symbol, its source, **the actual
documentation text pulled from the running language runtime with no network**, and
still keeps the "Open full documentation" link as a secondary affordance. For Python
this is the docstring (Pyodide `inspect.getdoc`); for R this is the rendered help page
(WebR `tools::Rd2txt`); for SQL there is no runtime help, so SQL keeps the curated
one-line blurb plus the link, honestly labelled. Unknown symbols and any runtime that
returns no local docs fall back to the blurb plus link with a friendly "No
documentation found for X" message.

**What already shipped (Slice 4 B), and stays.** Symbol resolution is unchanged:
`lib/sandbox/help-docs/symbol-at.ts` (the Lezer/word-scan token finder) and
`lib/sandbox/help-docs/resolve.ts` (the curated symbol-to-URL-and-blurb map) keep
working exactly as they do now. The editor plumbing (`onHelp`, the Ctrl/Cmd+Click
`domEventHandlers` extension, the F1 keybinding in `components/run/CodeEditor.tsx`) is
untouched. The two-tab Plots | Help panel (`PlotsHelpPane`) and the `helpTarget` state
in `components/sandbox/Sandbox.tsx` stay. This revision **adds** a runtime doc-text
fetch and rewrites only `HelpBody` and the small piece of `Workspace` that drives it.

**Architecture.** The documentation text lives inside each language Web Worker's
runtime, so it round-trips through the worker exactly as the just-built `exportRequest`
and the older `dataRequest` / `completeAt` do. A new read-only worker op,
`docRequest: { name, qualifier?, source? }`, asks the runtime to produce the doc text
for one symbol and replies `{ doc: { found, text?, signature?, truncated? } }`:

- **Python (Pyodide):** resolve the symbol to a live object and read its docstring
  with `inspect.getdoc`. This is the same introspection technique the shipped
  `completeAt` autocomplete already uses (it calls `inspect.getdoc` on members); the
  doc fetch reuses that precedent. It runs only introspection, never the student's
  code.
- **R (WebR):** render the Rd help page for the topic to plain text with
  `tools::Rd2txt`, resolving the topic through `utils::help` and
  `utils:::.getHelpFile`.
- **SQL (SQLite):** SQLite has no runtime help database, so the SQLite worker answers
  `{ found: false }` immediately. The pane falls back to the curated blurb plus the
  link. This asymmetry is real and is stated in the UI.

The main thread already resolved a `DocEntry` (with `source`) in `onHelp`; the revision
passes `entry.source` to the worker as a hint so R can pick the right package
(`summarise` -> `dplyr`) and Python can fall back to the module method
(`pandas.DataFrame.groupby`) when the live object is not defined yet. The fetch is
async and non-blocking: `onHelp` only sets React state and kicks off the fetch, never
focuses or scrolls the editor. A request token discards stale replies when a newer
click supersedes an in-flight one. While the fetch is out, the pane shows a small
"Loading documentation" status (`aria-live="polite"`, so it does not steal focus). On a
symbol with no local doc, a timeout, or an error, the pane shows the friendly fallback,
never an error dialog.

**Why runtime introspection and not the network (unchanged COEP constraint).**
`/ai-sandbox` is cross-origin isolated (`next.config.ts` sets COOP `same-origin` + COEP
`require-corp` to give WebR its SharedArrayBuffer channel). Under COEP `require-corp`,
an `<iframe>` to a cross-origin docs site is blocked and a cross-origin `fetch()` of a
doc page is CORS-blocked, so we still cannot embed or pull the external HTML. The
runtimes, however, already carry their documentation locally in the browser tab: the
Python docstrings ship inside the loaded modules, and R's help pages ship (when built
in) inside the installed packages. Reading them is a same-tab introspection call with
no network at all, so COEP does not apply. The "Open full documentation" link stays as
the path to the full, formatted, canonical page (it opens a new top-level browsing
context, which is not governed by our embedder policy).

**Tech Stack:** Next.js (this repo's vendored build; read `node_modules/next/dist/docs/`
before touching any Next API, though this slice touches none), React 19, Web Workers +
Pyodide (Python docstrings via `inspect`/`pydoc`), WebR (R help via
`utils::help` + `utils:::.getHelpFile` + `tools::Rd2txt`), `@sqlite.org/sqlite-wasm`
(SQL, no runtime help), Vitest for the pure helpers and the manager protocol
(FakeWorker), Playwright + `@axe-core/playwright` for the sandbox e2e.

## Global Constraints

- **No git commits.** The working tree stays uncommitted; each task ends by running
  verification commands instead of committing.
- **Read-only introspection only.** `docRequest` runs `keepState: true` so it can see
  the session's live objects (for example `df` in `df.groupby`), but it must never run
  the student's code, never mutate state, and never reset the session. It runs only
  the small introspection snippet, in the same spirit as the existing `completeAt`
  autocomplete op.
- **COEP require-corp is in force on `/ai-sandbox`.** No `<iframe>` to and no `fetch()`
  of a cross-origin docs site. Documentation text comes from the local runtime; the
  external doc page is reached only through the existing `<a target="_blank"
  rel="noopener noreferrer">` link.
- **Symbol resolution is frozen.** Do not change `symbol-at.ts` or `resolve.ts`
  behavior; this slice consumes them. The curated blurb and URL remain the fallback.
- **Non-blocking, no focus theft.** `onHelp` sets state and starts the fetch; it never
  calls `view.focus()` or `scrollIntoView`, and the caret and editor scroll are
  preserved (as in Slice 4 B). The loading indicator is `role="status"`
  `aria-live="polite"`, which announces without moving focus.
- **Graceful, never an error dialog.** A symbol that does not exist, a runtime with no
  local doc, a timeout, or an introspection error all resolve to a friendly "No
  documentation found for X" plus the link, not a thrown error or alert.
- **Truncate sensibly.** Docstrings (pandas methods) and R help pages can be long. Cap
  the rendered text (hard cap in the worker to bound the postMessage; a line/char cap
  on the main thread with a visible "Showing the first part; the full documentation is
  one click away." note). Never silently drop text without saying so.
- **WCAG 2.1 AA.** The doc-text region is labelled (`role="region"`,
  `aria-label="Documentation for {symbol}"`) and keyboard reachable/scrollable
  (`tabIndex={0}`, matching the existing `Pane` pattern). The loading status is a live
  region. The link stays a real anchor.
- **Miami brand tokens.** Reuse the `--sb-*` CSS variables already used by the panes;
  no new colors, no raw hex.
- **No em dashes in any user-facing copy** (loading text, region labels, fallback
  messages, truncation note, the link text, tooltips). Use commas, colons, or sentence
  breaks.
- All commands run from `web/`. Unit tests: `npm run test`. The Playwright config
  starts its own dev server on port 3100, so `npm run test:e2e` needs no separate
  server. Heavy-runtime (Pyodide/WebR) checks follow the project's existing pattern:
  gated behind an opt-in flag or verified manually; SQLite is the cheap, deterministic
  language for the e2e.

## File Structure

- `web/lib/sandbox/help-docs/doc-text.ts` — **Create.** Pure, runtime-free helpers and
  types: `DocRequest`, `DocText`, `DOC_MAX_CHARS`, `DOC_MAX_LINES`, `buildDocRequest`,
  `truncateDocText`. Unit-tested in Vitest's `node` environment.
- `web/lib/sandbox/help-docs/index.ts` — **Modify.** Re-export the new `doc-text`
  types and helpers alongside the existing exports.
- `web/lib/run/manager.ts` — **Modify.** Add `doc?` to `WorkerReply` and to the
  `onmessage` destructure and `resolve`; add `DOC_TIMEOUT_MS`; add
  `LanguageRunner.fetchDoc`; add `fetchDoc` to the `RunSession` interface and
  `createSession`.
- `web/public/workers/pyodide-worker.mjs` — **Modify.** Handle `docRequest`: resolve
  the symbol to a live/module object and return `inspect.getdoc` text.
- `web/public/workers/webr-worker.mjs` — **Modify.** Handle `docRequest`: render the Rd
  help page to plain text via `tools::Rd2txt`, or `{ found: false }`.
- `web/public/workers/sqlite-worker.mjs` — **Modify.** Handle `docRequest`: reply
  `{ found: false }` (SQLite has no runtime help), so the pane falls back deterministically.
- `web/components/sandbox/Sandbox.tsx` — **Modify.** `helpTarget` carries the raw
  `HelpRequest`; add `helpDoc` state and a token-guarded fetch in `onHelp`; rewrite
  `HelpBody` to show the doc-text region, the loading status, the fallback, and the link.
- `web/tests/unit/help-docs-doc-text.test.ts` — **Create.** Vitest for `buildDocRequest`
  and `truncateDocText`.
- `web/tests/unit/run-harness.test.ts` — **Modify.** Add a FakeWorker test for the
  `fetchDoc` protocol (mirrors the existing `getData` / `exportObject` tests).
- `web/tests/e2e/sandbox.spec.ts` — **Modify.** Add a Playwright test that the HELP pane
  shows a loading state then a doc region, proven deterministically with SQL's
  no-local-docs fallback, and that axe stays clean.

---

## Task 1: Pure doc-text helpers (request builder and truncation)

The main-thread pieces are pure: building the `docRequest` payload from a resolved
`HelpRequest` + `DocEntry`, and capping long text with an honest "truncated" flag. The
runtime introspection itself lives in the workers and is proven end to end (SQL) or by
gated/manual runtime checks (Python/R), not here.

**Files:**
- Create: `web/lib/sandbox/help-docs/doc-text.ts`
- Test: `web/tests/unit/help-docs-doc-text.test.ts`

**Interfaces:**
- Produces:
  - `interface DocRequest { name: string; qualifier?: string; source?: string }`
  - `interface DocText { found: boolean; text?: string; signature?: string; truncated?: boolean }`
  - `const DOC_MAX_CHARS = 8000`, `const DOC_MAX_LINES = 160`
  - `buildDocRequest(req: HelpRequest, entry: DocEntry): DocRequest`
  - `truncateDocText(raw: string, opts?: { maxChars?: number; maxLines?: number }): { text: string; truncated: boolean }`

- [ ] **Step 1: Write the failing helper tests**

Create `web/tests/unit/help-docs-doc-text.test.ts`:

```typescript
import { describe, expect, it } from "vitest";
import {
  buildDocRequest,
  truncateDocText,
  DOC_MAX_LINES,
} from "@/lib/sandbox/help-docs/doc-text";
import type { DocEntry, HelpRequest } from "@/lib/sandbox/help-docs/types";

function req(p: Partial<HelpRequest> & { name: string }): HelpRequest {
  return { kind: "function", language: p.language ?? "python", qualifier: p.qualifier, name: p.name };
}
function entry(p: Partial<DocEntry> & { source: string }): DocEntry {
  return { symbol: p.symbol ?? "x", source: p.source, url: p.url ?? "https://example.test", blurb: p.blurb };
}

describe("buildDocRequest", () => {
  it("carries name, qualifier, and the resolved source as a hint", () => {
    const r = buildDocRequest(
      req({ name: "groupby", qualifier: "df", language: "python" }),
      entry({ source: "pandas" }),
    );
    expect(r).toEqual({ name: "groupby", qualifier: "df", source: "pandas" });
  });

  it("passes the source so R can pick a package for a bare name", () => {
    const r = buildDocRequest(
      req({ name: "summarise", language: "r" }),
      entry({ source: "dplyr" }),
    );
    expect(r.name).toBe("summarise");
    expect(r.qualifier).toBeUndefined();
    expect(r.source).toBe("dplyr");
  });
});

describe("truncateDocText", () => {
  it("leaves short text unchanged and not truncated", () => {
    const { text, truncated } = truncateDocText("one\ntwo\nthree");
    expect(text).toBe("one\ntwo\nthree");
    expect(truncated).toBe(false);
  });

  it("caps by line count and flags truncation", () => {
    const raw = Array.from({ length: DOC_MAX_LINES + 40 }, (_, i) => `line ${i}`).join("\n");
    const { text, truncated } = truncateDocText(raw);
    expect(truncated).toBe(true);
    expect(text.split("\n").length).toBeLessThanOrEqual(DOC_MAX_LINES);
  });

  it("caps by character count and flags truncation", () => {
    const raw = "x".repeat(20000);
    const { text, truncated } = truncateDocText(raw, { maxChars: 100 });
    expect(truncated).toBe(true);
    expect(text.length).toBeLessThanOrEqual(100);
  });
});
```

- [ ] **Step 2: Run the tests and confirm they fail**

Run: `npm run test -- help-docs-doc-text`
Expected: FAIL. The module `@/lib/sandbox/help-docs/doc-text` does not exist yet.

- [ ] **Step 3: Implement the helpers**

Create `web/lib/sandbox/help-docs/doc-text.ts`:

```typescript
import type { DocEntry, HelpRequest } from "./types";

/**
 * The read-only documentation request sent to a language worker. `name` is the
 * clicked symbol, `qualifier` the receiver it hangs off when known (Python
 * `df.groupby` -> `df`; R `dplyr::summarise` -> `dplyr`), and `source` is the
 * resolver's label (pandas, NumPy, Python, dplyr, ggplot2, base R, SQLite), used
 * as a hint so the worker can find the topic even when no live object exists yet.
 */
export interface DocRequest {
  name: string;
  qualifier?: string;
  source?: string;
}

/** The runtime documentation for one symbol, as returned by the worker. */
export interface DocText {
  /** True when the runtime produced help/docstring text for the symbol. */
  found: boolean;
  /** The plain-text documentation (docstring or rendered help page). */
  text?: string;
  /** The call signature, when the runtime can produce one (Python). */
  signature?: string;
  /** True when the text was capped; the pane says the full docs are one click away. */
  truncated?: boolean;
}

/** Main-thread caps. The worker also hard-caps to bound the postMessage size; these
 *  are the display caps and the source of the "truncated" note in the pane. */
export const DOC_MAX_CHARS = 8000;
export const DOC_MAX_LINES = 160;

/** Builds the worker doc request from a resolved click. The resolver already ran on
 *  the main thread, so its `source` rides along as a topic hint. */
export function buildDocRequest(req: HelpRequest, entry: DocEntry): DocRequest {
  return {
    name: req.name,
    qualifier: req.qualifier,
    source: entry.source,
  };
}

/**
 * Caps long documentation text by line and character count, reporting whether it
 * was shortened so the pane can note that the full documentation is one click away.
 * Pure; the actual doc text is produced in the worker runtime.
 */
export function truncateDocText(
  raw: string,
  opts: { maxChars?: number; maxLines?: number } = {},
): { text: string; truncated: boolean } {
  const maxChars = opts.maxChars ?? DOC_MAX_CHARS;
  const maxLines = opts.maxLines ?? DOC_MAX_LINES;
  let truncated = false;
  let text = raw;

  const lines = text.split("\n");
  if (lines.length > maxLines) {
    text = lines.slice(0, maxLines).join("\n");
    truncated = true;
  }
  if (text.length > maxChars) {
    text = text.slice(0, maxChars);
    truncated = true;
  }
  return { text, truncated };
}
```

- [ ] **Step 4: Re-export from the barrel**

In `web/lib/sandbox/help-docs/index.ts`, add:

```typescript
export type { DocRequest, DocText } from "./doc-text";
export { buildDocRequest, truncateDocText, DOC_MAX_CHARS, DOC_MAX_LINES } from "./doc-text";
```

- [ ] **Step 5: Run the tests to green and verify (no commit)**

Run: `npm run test -- help-docs-doc-text` — PASS.
Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors for the new file.

---

## Task 2: The `fetchDoc` protocol in the manager

Add the `fetchDoc` round-trip to `LanguageRunner` and `RunSession`, mirroring
`exportObject` and `getData`. Prove it with a FakeWorker, exactly as those do. The
request is read-only (`keepState: true`) and carries a dedicated, generous-but-bounded
timeout so a slow first lookup falls back gracefully rather than hanging the pane.

**Files:**
- Modify: `web/lib/run/manager.ts`
- Test: `web/tests/unit/run-harness.test.ts`

**Interfaces:**
- Consumes: `LanguageRunner.dispatch`; the `WorkerReply` shape and the `onmessage`
  destructure; `DocRequest` / `DocText` from `@/lib/sandbox/help-docs/doc-text`.
- Produces: `WorkerReply.doc?: DocText`; `LanguageRunner.fetchDoc(req: DocRequest): Promise<DocText>`;
  `RunSession.fetchDoc(req: DocRequest): Promise<DocText>`.

- [ ] **Step 1: Write the failing protocol tests**

In `web/tests/unit/run-harness.test.ts`, first widen the `FakeWorker.posted` element
type and the `postMessage` parameter type to include
`docRequest?: { name: string; qualifier?: string; source?: string }`. Then add, inside
the `run manager timeout and worker reuse` describe block:

```typescript
  it("fetches runtime documentation through the session worker", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);

    const pending = session.fetchDoc({ name: "groupby", qualifier: "df", source: "pandas" });
    const worker = FakeWorker.instances[0];
    // The request carries the symbol, receiver, and source hint, and keeps state
    // (read-only introspection: it must see the session's live objects).
    expect(worker.posted[0].docRequest).toEqual({
      name: "groupby",
      qualifier: "df",
      source: "pandas",
    });
    expect(worker.posted[0].keepState).toBe(true);

    worker.reply(worker.posted[0].id, {
      ok: true,
      doc: { found: true, text: "Group DataFrame using a mapper.", signature: "(by=None)" },
    });
    const res = await pending;
    expect(res.found).toBe(true);
    expect(res.text).toMatch(/Group DataFrame/);
    expect(res.signature).toBe("(by=None)");
  });

  it("reports no-local-docs as a clean not-found, never an error", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);

    const pending = session.fetchDoc({ name: "COUNT", source: "SQLite" });
    const worker = FakeWorker.instances[0];
    // SQLite has no runtime help; the worker answers found:false.
    worker.reply(worker.posted[0].id, { ok: true, doc: { found: false } });
    const res = await pending;
    expect(res.found).toBe(false);
    expect(res.text).toBeUndefined();
  });

  it("treats a worker failure as not-found so the pane falls back", async () => {
    const { createSession } = await load();
    const session = createSession(SQL);

    const pending = session.fetchDoc({ name: "nope" });
    const worker = FakeWorker.instances[0];
    worker.reply(worker.posted[0].id, { ok: false, error: "boom" });
    const res = await pending;
    // No throw, no error surfaced: the pane shows the blurb + link fallback.
    expect(res.found).toBe(false);
  });
```

- [ ] **Step 2: Run the test and confirm it fails**

Run: `npm run test -- run-harness`
Expected: FAIL. `session.fetchDoc` does not exist.

- [ ] **Step 3: Add `doc` to the reply shape and wiring**

In `web/lib/run/manager.ts`, import the types near the top:

```typescript
import type { DocRequest, DocText } from "@/lib/sandbox/help-docs/doc-text";
```

Add to `WorkerReply` (around line 118):

```typescript
  doc?: DocText;
```

In `LanguageRunner.ensureWorker`, add `doc` to the destructure and to the resolve
(around lines 165 and 171):

```typescript
      const { id, ok, result, error, data, completions, tables, preview, exported, doc } =
        event.data ?? {};
      // ...
      entry.resolve({ ok, result, error, data, completions, tables, preview, exported, doc });
```

- [ ] **Step 4: Add the `DOC_TIMEOUT_MS` and `fetchDoc` to `LanguageRunner`**

Near `RUN_TIMEOUT_MS` / `PREWARM_TIMEOUT_MS`, add:

```typescript
/** A documentation lookup is read-only introspection, but a first-ever one may wait
 * on the runtime (or pandas) finishing loading. Give it generous headroom, but bound
 * it so a slow or missing doc falls back to the blurb + link rather than hanging the
 * pane. The fetch is off the editor's critical path, so this never blocks typing. */
const DOC_TIMEOUT_MS = 20_000;
```

Next to `exportObject` (manager.ts:275), add:

```typescript
  /** Fetches the runtime documentation text for one symbol. Read-only: keepState is
   * on so the runtime can introspect the session's live objects, and it runs only the
   * introspection snippet, never the student's code and never a state change. Any
   * failure or missing doc resolves to { found: false } so the pane can fall back to
   * the curated blurb and the open-in-new-tab link. */
  fetchDoc(req: DocRequest): Promise<DocText> {
    return this.dispatch({ docRequest: req, keepState: true }, DOC_TIMEOUT_MS).then(
      (reply) =>
        reply.ok && reply.doc ? reply.doc : { found: false },
    );
  }
```

- [ ] **Step 5: Expose it on `RunSession`**

Add to the `RunSession` interface (near `exportObject`, manager.ts:431):

```typescript
  /** Fetch the runtime documentation text for one symbol (docstring for Python, help
   * page for R; SQL has none). Read-only; never runs code or resets the session. */
  fetchDoc(req: DocRequest): Promise<DocText>;
```

Wire it in `createSession` (near the `exportObject` line):

```typescript
    fetchDoc: (req) => runner.fetchDoc(req),
```

- [ ] **Step 6: Run the protocol tests to green and verify (no commit)**

Run: `npm run test -- run-harness` — PASS (the three new tests plus the existing suite).
Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors for the changed file.

---

## Task 3: Python worker `docRequest` (Pyodide docstring)

The Pyodide worker resolves the clicked symbol to a live object and returns its
docstring via `inspect.getdoc`. Resolution tries, in order: the live receiver
(`eval(qualifier)` then `getattr(name)`, so `df.groupby` reads the actual bound method
when `df` exists), the bare name in the session globals/builtins (so `len` reads the
builtin docstring), and finally the curated module fallback keyed by the `source` hint
(so `df.groupby` still resolves to `pandas.DataFrame.groupby` before any `df` is
defined). This is the same introspection surface the shipped `completeAt` autocomplete
uses; it runs entirely inside the student's own Pyodide sandbox and only ever evaluates
a dotted identifier path, never a call.

**Verification of feasibility (design note).** In Pyodide, `inspect.getdoc(len)`
returns the builtin docstring ("Return the number of items in a container."), and once
pandas is loaded (the prewarm loads it), `inspect.getdoc(pandas.DataFrame.groupby)`
returns the rich method docstring. `inspect.signature(...)` yields a signature for most
callables and raises `ValueError`/`TypeError` for some builtins, which is caught. These
are confirmed to work headlessly in the worker; no network is involved because the
docstrings ship inside the already-loaded modules.

**Files:**
- Modify: `web/public/workers/pyodide-worker.mjs`

**Interfaces:**
- Consumes: the `{ docRequest: { name, qualifier?, source? }, keepState }` message.
- Produces: `{ id, ok: true, doc: { found, text?, signature? } }`.

- [ ] **Step 1: Add the introspection code builder**

In `web/public/workers/pyodide-worker.mjs`, next to `exportFrameCode` (around line 202),
add a doc-resolver builder. It hard-caps the text so the postMessage stays small:

```javascript
/**
 * Resolves a clicked symbol to a live object and returns its docstring as JSON:
 * {found, text?, signature?}. Tries the live receiver (df.groupby), then the bare
 * name (len), then a curated module fallback from the source hint (pandas/numpy), so
 * a pandas method resolves to pandas.DataFrame.<name> even before any df is defined.
 * Runs only introspection (the same technique as the autocomplete op); never the
 * student's code. `__doc_req` (name, qualifier, source) is set from JS.
 */
const DOC_CODE = `
def __chatisa_doc(req):
    import json, inspect, importlib
    name = req.get("name") or ""
    qualifier = req.get("qualifier")
    source = (req.get("source") or "").lower()
    obj = None
    # 1) Live receiver: df.groupby reads the actual bound method when df exists.
    if qualifier:
        try:
            base = eval(qualifier, globals())
            obj = getattr(base, name, None)
        except Exception:
            obj = None
    # 2) Bare name in the session globals or builtins: len, print, a user function.
    if obj is None:
        try:
            obj = eval(name, globals())
        except Exception:
            obj = None
    # 3) Curated module fallback from the source hint, no live object needed.
    if obj is None:
        mod_name = {"pandas": "pandas", "numpy": "numpy"}.get(source)
        if mod_name:
            try:
                mod = importlib.import_module(mod_name)
                # A DataFrame method (groupby) lives on the class; a top-level
                # function (read_csv, array) lives on the module.
                obj = getattr(getattr(mod, "DataFrame", object), name, None) or getattr(mod, name, None)
            except Exception:
                obj = None
    if obj is None:
        return json.dumps({"found": False})
    doc = inspect.getdoc(obj) or ""
    if not doc:
        return json.dumps({"found": False})
    sig = ""
    try:
        sig = str(inspect.signature(obj))
    except (ValueError, TypeError):
        sig = ""
    # Hard cap so the message stays small; the pane also caps and notes truncation.
    if len(doc) > 12000:
        doc = doc[:12000]
    return json.dumps({"found": True, "text": doc, "signature": sig})
__chatisa_doc(__doc_req)
`;
```

- [ ] **Step 2: Handle `docRequest` in `onmessage`**

Add `docRequest` to the `onmessage` destructure (around line 319):

```javascript
  const { id, code, keepState, withVariables, dataRequest, completeAt, prewarm, exportRequest, docRequest, fileOp } =
    event.data ?? {};
```

Add a branch next to the `completeAt` branch (around line 434). It loads the runtime
(and, when the source is pandas/numpy and the module is not present, that one package)
so a static fallback can resolve, then returns the docstring:

```javascript
  // Documentation request: return the docstring for one symbol. Read-only
  // introspection (same surface as autocomplete); never runs the student's code.
  if (docRequest) {
    try {
      const pyodide = await getPyodide();
      // If we will need a module fallback (no live object) make sure it is present.
      const src = (docRequest.source || "").toLowerCase();
      if (src === "pandas") await pyodide.loadPackage(["pandas"], QUIET);
      if (src === "numpy") await pyodide.loadPackage(["numpy"], QUIET);
      pyodide.globals.set(
        "__doc_req",
        pyodide.toPy({
          name: docRequest.name || "",
          qualifier: docRequest.qualifier ?? null,
          source: docRequest.source ?? null,
        }),
      );
      const parsed = JSON.parse(pyodide.runPython(DOC_CODE));
      self.postMessage({ id, ok: true, doc: parsed });
    } catch {
      // Never surface an error for a doc lookup: the pane falls back to the blurb.
      self.postMessage({ id, ok: true, doc: { found: false } });
    }
    return;
  }
```

- [ ] **Step 3: Verify the worker still loads (no commit)**

Workers are static `.mjs` loaded by URL, not type-checked by `tsc`. Sanity-check by
opening the Python tab in dev and confirming no worker console error. The behavioral
proof for Python docstrings is the gated/manual runtime check in Task 6.

Run: `npm run typecheck` — no errors (the manager change still checks; the worker is untyped).
Run: `npm run lint` — no errors for changed files.

---

## Task 4: R worker `docRequest` (WebR rendered help)

The WebR worker renders the Rd help page for the topic to plain text with
`tools::Rd2txt`. It resolves the topic through `utils::help`, picking the package from
the qualifier (`dplyr::summarise`) or, for a bare name, from the `source` hint
(`summarise` -> package `dplyr`; `mean` -> base R, no package needed), then reads the
parsed Rd with `utils:::.getHelpFile` and renders it. Everything is local: the Rd help
ships inside the installed package; no network.

**Honesty caveat (must be verified during implementation).** Base R help (mean, sum,
paste, c, seq) is reliably present in WebR, so those render. Package help for dplyr and
ggplot2 depends on whether the WebR binary build of the package includes its help
database (`help/` index and `Rd.rds`); some WebR package binaries strip it to save
space. dplyr is bundled here (installed from our mirror in `installBundledPackages`),
but if its build lacks the Rd DB, `.getHelpFile` fails and this handler returns
`{ found: false }`, so the pane falls back to the curated blurb plus the link. This is
the honest behavior; do not fake help text. During implementation, run the manual check
in Task 6 for `mean` (expected to render) and `dplyr::summarise` (renders if the build
carries help; otherwise falls back), and record the outcome in the interaction log.

**Files:**
- Modify: `web/public/workers/webr-worker.mjs`

**Interfaces:**
- Consumes: the `{ docRequest: { name, qualifier?, source? }, keepState }` message.
- Produces: `{ id, ok: true, doc: { found, text? } }`.

- [ ] **Step 1: Add the help-render code builder**

In `web/public/workers/webr-worker.mjs`, next to `buildRExportCode` (around line 311),
add. It maps the source hint to a package, resolves the topic with `do.call` (so a
character topic passes cleanly through `help`'s non-standard evaluation), renders with
`tools::Rd2txt`, and hard-caps the text:

```javascript
/** Maps the resolver's source label to an R package name for help lookup, or "" when
 * none is needed (base R). */
function sourceToPackage(source) {
  const s = String(source || "").toLowerCase();
  if (s === "dplyr") return "dplyr";
  if (s === "ggplot2") return "ggplot2";
  return ""; // base R and unknown: let help() search without a package
}

/** R that renders the Rd help page for one topic to plain text, or signals no help
 * with {found:false}. Uses do.call so a character topic passes through help()'s NSE,
 * .getHelpFile to read the parsed Rd, and Rd2txt to render. Returns a JSON string. */
function buildRDocCode(name, qualifier, source) {
  const pkg = qualifier ? String(qualifier) : sourceToPackage(source);
  const pkgArg = pkg ? `, package = ${rStr(pkg)}` : "";
  return `
local({
  topic <- ${rStr(name)}
  paths <- tryCatch(
    as.character(do.call(utils::help, list(topic${pkgArg}))),
    error = function(e) character(0)
  )
  if (length(paths) == 0) return(jsonlite::toJSON(list(found = FALSE), auto_unbox = TRUE))
  rd <- tryCatch(utils:::.getHelpFile(paths[[1]]), error = function(e) NULL)
  if (is.null(rd)) return(jsonlite::toJSON(list(found = FALSE), auto_unbox = TRUE))
  tmp <- tempfile()
  ok <- tryCatch({ tools::Rd2txt(rd, out = tmp, package = ${pkg ? rStr(pkg) : "\"\""}); TRUE },
                 error = function(e) FALSE)
  if (!ok) return(jsonlite::toJSON(list(found = FALSE), auto_unbox = TRUE))
  txt <- paste(readLines(tmp, warn = FALSE), collapse = "\\n")
  if (nchar(txt) > 12000) txt <- substr(txt, 1, 12000)
  jsonlite::toJSON(list(found = TRUE, text = txt), auto_unbox = TRUE)
})
`;
}
```

- [ ] **Step 2: Handle `docRequest` in `onmessage`**

Add `docRequest` to the `onmessage` destructure (around line 323):

```javascript
  const { id, code, keepState, withVariables, dataRequest, completeAt, prewarm, exportRequest, docRequest, fileOp, wsProxy } =
    event.data ?? {};
```

Add a branch next to the `exportRequest` branch (around line 449). For a package topic
(dplyr, ggplot2) it makes sure the bundled packages are installed so their help exists;
for base R it does not need them:

```javascript
  // Documentation request: render the Rd help page for one topic to plain text.
  // Read-only; never runs the student's code. Falls back to {found:false} on any
  // failure (including a WebR build that ships without the package help database).
  if (docRequest) {
    const shelter = await new webR.Shelter();
    try {
      const src = String(docRequest.source || "").toLowerCase();
      if (src === "dplyr" || src === "ggplot2" || docRequest.qualifier) {
        await ensurePackages(webR); // dplyr/ggplot2 (via tidyverse) carry their help
      }
      const obj = await shelter.captureR(
        buildRDocCode(docRequest.name, docRequest.qualifier, docRequest.source),
        { withAutoprint: false, captureStreams: false, captureConditions: false, captureGraphics: false },
      );
      const parsed = JSON.parse((await obj.result.toArray())[0]);
      self.postMessage({ id, ok: true, doc: parsed });
    } catch {
      self.postMessage({ id, ok: true, doc: { found: false } });
    } finally {
      await shelter.purge();
    }
    return;
  }
```

- [ ] **Step 3: Verify the worker still loads (no commit)**

Open the R tab in dev; confirm no worker console error on tab open. Behavioral proof is
the manual check in Task 6.

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors for changed files.

---

## Task 5: SQL worker `docRequest` (honest no-local-docs)

SQLite has no runtime help database, so the SQLite worker answers every `docRequest`
with `{ found: false }`. This is deterministic and cheap, which makes SQL the language
that proves the pane's loading -> fallback flow in the e2e (Task 7). The pane then shows
the curated blurb plus the link, exactly as it does today, with an honest note that
SQLite has no built-in help text.

**Files:**
- Modify: `web/public/workers/sqlite-worker.mjs`

**Interfaces:**
- Consumes: the `{ docRequest, keepState }` message.
- Produces: `{ id, ok: true, doc: { found: false } }`.

- [ ] **Step 1: Handle `docRequest` in `onmessage`**

Add `docRequest` to the `onmessage` destructure (sqlite-worker.mjs:348):

```javascript
  const { id, code, keepState, withVariables, dataRequest, completeAt, dumpTablesRequest, exportRequest, docRequest, fileOp } =
    event.data ?? {};
```

Add a branch alongside `exportRequest` / `dumpTablesRequest` (around line 353). It does
not even need the database; SQLite simply has no help to render:

```javascript
  // SQLite has no runtime help database, so there is no local doc text to render.
  // Answer found:false so the pane falls back to the curated blurb and the link.
  if (docRequest) {
    self.postMessage({ id, ok: true, doc: { found: false } });
    return;
  }
```

- [ ] **Step 2: Verify (no commit)**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors for changed files.

---

## Task 6: The HELP pane renders the doc text, with loading and fallback

Rewrite `HelpBody` and the small piece of `Workspace` that drives it. `helpTarget`
carries the raw `HelpRequest` so the pane can fetch. A `helpDoc` state holds the fetch
status. `onHelp` resolves the entry (as today), selects the Help tab, sets the loading
state, and fires the fetch, guarded by a request token so a newer click wins. The pane
shows: the symbol and source (as today), then the loading status or the rendered doc
region or the fallback blurb, and always the "Open full documentation" link.

**Files:**
- Modify: `web/components/sandbox/Sandbox.tsx`

**Interfaces:**
- Consumes: `session.fetchDoc`; `buildDocRequest`, `truncateDocText`, and the
  `DocText` type from `@/lib/sandbox/help-docs`.
- Produces: `helpTarget` shape gains `req: HelpRequest`; new `helpDoc` state; a
  token-guarded fetch in `onHelp`; a rewritten `HelpBody`.

- [ ] **Step 1: Import the new helpers**

Extend the existing `@/lib/sandbox/help-docs` import (Sandbox.tsx:30):

```typescript
import {
  resolveDoc,
  referenceHome,
  buildDocRequest,
  truncateDocText,
  type DocEntry,
  type DocText,
  type HelpRequest,
} from "@/lib/sandbox/help-docs";
```

- [ ] **Step 2: Carry the request and add the fetch state**

Change the `helpTarget` state (Sandbox.tsx:319) to carry the raw request, and add a
`helpDoc` status slot below it:

```typescript
  const [helpTarget, setHelpTarget] = useState<{
    symbol: string;
    entry: DocEntry;
    req: HelpRequest;
  } | null>(null);
  const [rightLowerTab, setRightLowerTab] = useState<"plots" | "help">("plots");
  // The runtime doc-text fetch for the current HELP target. `status` drives the pane:
  // "loading" shows a spinner note, "loaded" shows the doc region, "none" falls back
  // to the blurb + link. A monotonically increasing token discards stale replies when
  // a newer click supersedes an in-flight fetch.
  const [helpDoc, setHelpDoc] = useState<{
    status: "idle" | "loading" | "loaded" | "none";
    text?: string;
    signature?: string;
    truncated?: boolean;
  }>({ status: "idle" });
  const helpTokenRef = useRef(0);
```

- [ ] **Step 3: Rewrite `onHelp` to resolve, select, and fetch**

Replace the current `onHelp` (Sandbox.tsx:509). It stays synchronous for the state
updates (so the tab switches and the caret is untouched) and starts the fetch without
awaiting in a way that blocks:

```typescript
  // Ctrl/Cmd+Click or F1 in the editor lands here. Resolve the clicked symbol to a
  // doc entry (or the reference home if not curated), show it in the one HELP tab, and
  // select that tab. Then fetch the runtime documentation text (docstring for Python,
  // help page for R; SQL has none) without blocking the editor or stealing focus. A
  // request token guards against a stale reply when a newer click arrives first.
  const onHelp = useCallback((req: HelpRequest) => {
    const entry = resolveDoc(req) ?? referenceHome(req.language);
    const symbol = req.qualifier ? `${req.qualifier}.${req.name}` : req.name;
    setHelpTarget({ symbol, entry, req });
    setRightLowerTab("help");
    setHelpDoc({ status: "loading" });

    const token = ++helpTokenRef.current;
    const session = sessionRef.current;
    if (!session) {
      setHelpDoc({ status: "none" });
      return;
    }
    void session
      .fetchDoc(buildDocRequest(req, entry))
      .then((doc: DocText) => {
        if (helpTokenRef.current !== token) return; // superseded by a newer click
        if (doc.found && doc.text) {
          const { text, truncated } = truncateDocText(doc.text);
          setHelpDoc({ status: "loaded", text, signature: doc.signature, truncated });
        } else {
          setHelpDoc({ status: "none" });
        }
      })
      .catch(() => {
        if (helpTokenRef.current === token) setHelpDoc({ status: "none" });
      });
  }, []);
```

- [ ] **Step 4: Pass the doc state into the pane**

Where `PlotsHelpPane` is rendered (Sandbox.tsx:781), pass `helpDoc` through:

```tsx
                <PlotsHelpPane
                  tab={rightLowerTab}
                  onTab={setRightLowerTab}
                  help={helpTarget}
                  helpDoc={helpDoc}
                  plots={plots}
                  index={plotIndex}
                  onIndex={setPlotIndex}
                  onDelete={deletePlot}
                  onClear={clearPlots}
                />
```

Add `helpDoc` to the `PlotsHelpPane` prop type and thread it to `HelpBody`:

```tsx
  help,
  helpDoc,
  // ...
}: {
  // ...
  help: { symbol: string; entry: DocEntry; req: HelpRequest } | null;
  helpDoc: { status: "idle" | "loading" | "loaded" | "none"; text?: string; signature?: string; truncated?: boolean };
  // ...
}) {
  // ...
          <HelpBody help={help} helpDoc={helpDoc} />
```

- [ ] **Step 5: Rewrite `HelpBody` to render doc text, loading, and fallback**

Replace the current `HelpBody` (Sandbox.tsx:1465). It keeps the symbol + source header
and the link, and inserts the runtime doc region, the loading status, or the fallback
blurb between them:

```tsx
/**
 * The HELP tab body. It renders the documentation text the language runtime carries
 * locally (a Python docstring, an R help page), fetched with no network. While the
 * fetch is out it shows a small loading status; when the runtime has no local doc for
 * the symbol (SQLite always, or an unknown name) it falls back to the curated blurb.
 * The "Open full documentation" link is always present as the path to the full,
 * formatted, canonical page, which cannot be embedded here because /ai-sandbox is
 * cross-origin isolated (COEP require-corp blocks an iframe and a cross-origin fetch).
 */
function HelpBody({
  help,
  helpDoc,
}: {
  help: { symbol: string; entry: DocEntry; req: HelpRequest } | null;
  helpDoc: { status: "idle" | "loading" | "loaded" | "none"; text?: string; signature?: string; truncated?: boolean };
}) {
  if (!help) {
    return (
      <div className="p-3 text-sm text-[var(--sb-muted)]">
        <p>
          Ctrl or Cmd click a function or keyword in your script, or put the cursor on
          it and press F1, to see its documentation here.
        </p>
      </div>
    );
  }
  const { symbol, entry } = help;
  const loading = helpDoc.status === "loading";
  const loaded = helpDoc.status === "loaded" && !!helpDoc.text;
  return (
    <div className="flex flex-col gap-2 p-3 text-sm">
      <div className="flex flex-wrap items-baseline gap-2">
        <span className="font-mono text-base font-bold text-[var(--sb-text)]">
          {symbol}
        </span>
        <span className="rounded border border-[var(--sb-border)] px-1.5 py-0.5 text-xs font-bold uppercase tracking-wide text-[var(--sb-muted)]">
          {entry.source}
        </span>
      </div>

      {helpDoc.signature ? (
        <p className="font-mono text-xs text-[var(--sb-muted)]">
          {symbol}
          {helpDoc.signature}
        </p>
      ) : null}

      {loading ? (
        <p
          role="status"
          aria-live="polite"
          className="flex items-center gap-1.5 text-xs font-bold text-[var(--sb-muted)]"
        >
          Loading documentation
        </p>
      ) : null}

      {loaded ? (
        <>
          {/* Labelled and keyboard-scrollable, so a keyboard user can read a long
              doc (WCAG 2.1.1); the runtime produced this text with no network. */}
          <div
            role="region"
            aria-label={`Documentation for ${symbol}`}
            tabIndex={0}
            className="max-h-[22rem] overflow-auto rounded-card border border-[var(--sb-border)] bg-[var(--sb-header)] p-2"
          >
            <pre className="whitespace-pre-wrap font-mono text-xs text-[var(--sb-text)]">
              {helpDoc.text}
            </pre>
          </div>
          {helpDoc.truncated ? (
            <p className="text-xs text-[var(--sb-muted)]">
              Showing the first part of the documentation. The full page is one click
              away below.
            </p>
          ) : null}
        </>
      ) : null}

      {helpDoc.status === "none" ? (
        <>
          {entry.blurb ? (
            <p className="text-[var(--sb-text)]">{entry.blurb}</p>
          ) : null}
          <p className="text-xs text-[var(--sb-muted)]">
            No documentation text is available for {symbol} in this runtime. Open the
            full documentation below.
          </p>
        </>
      ) : null}

      {entry.note ? (
        <p className="rounded-card border border-[var(--sb-border)] bg-[var(--sb-header)] px-2 py-1.5 text-xs text-[var(--sb-muted)]">
          {entry.note}
        </p>
      ) : null}

      <a
        href={entry.url}
        target="_blank"
        rel="noopener noreferrer"
        className="inline-flex w-fit items-center gap-1 rounded-card bg-[var(--sb-accent)] px-3 py-1 text-sm font-bold text-white hover:opacity-90"
      >
        Open full documentation
      </a>
      <p className="text-xs text-[var(--sb-muted)]">
        Opens the official {entry.source} page in a new tab.
      </p>
    </div>
  );
}
```

- [ ] **Step 6: Verify types, lint, and manual runtime checks (no commit)**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors for `Sandbox.tsx`.

Manual runtime check (follow the project's heavy-runtime pattern; record results in
`docs/development/interaction-log.md`):

- Python: open the Python tab, Ctrl/Cmd+Click `len` in `n = len(items)`. Expect the
  builtin docstring in the doc region. Run the coding example, then Ctrl/Cmd+Click
  `groupby` in a `df.groupby(...)` line: expect the pandas method docstring (rich, so
  the truncation note appears). Ctrl/Cmd+Click a name with no docstring: expect the
  blurb fallback.
- R: open the R tab, Ctrl/Cmd+Click `mean`: expect the base R help page rendered as
  text (reliable). Ctrl/Cmd+Click `summarise`: expect the dplyr help page if the WebR
  dplyr build carries its help database; otherwise the blurb fallback. Record which
  behavior actually occurred (the honesty caveat in Task 4).
- SQL: Ctrl/Cmd+Click `COUNT`: expect a brief loading state then the blurb plus the
  "No documentation text is available ... in this runtime" line and the link.

---

## Task 7: End-to-end (SQL loading then fallback) and accessibility

SQL is deterministic (found:false immediately) and cheap (no prewarm, no network), so
it proves the pane's loading -> doc-region-or-fallback flow and that axe stays clean
with the new region. Heavy-runtime doc rendering (Python/R) is covered by the gated or
manual checks in Task 6, per the project's pattern of not running Pyodide/WebR in the
default e2e suite.

**Files:**
- Modify: `web/tests/e2e/sandbox.spec.ts`

- [ ] **Step 1: Write the e2e test**

Add to the `test.describe("AI Sandbox", ...)` block:

```typescript
  test("the HELP pane shows a loading state then a doc region or fallback (SQL)", async ({
    page,
  }) => {
    test.setTimeout(120_000);
    await page.goto("/ai-sandbox");
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
    await expect(helpPanel.getByText("COUNT")).toBeVisible();

    // SQLite has no runtime help, so after the brief loading state the pane falls back
    // to the blurb + link (the deterministic no-local-docs path). Assert the stable
    // end state: the "Open full documentation" link and the honest no-doc line.
    await expect(helpPanel.getByRole("link", { name: "Open full documentation" })).toBeVisible();
    await expect(helpPanel.getByText(/No documentation text is available/i)).toBeVisible();
  });
```

Notes: the loading state is very brief for SQL (found:false returns immediately), so the
assertion targets the deterministic end state rather than racing the spinner. If a test
needs to observe the loading text reliably, mock `session.fetchDoc` to delay, or assert
the loading path in a unit test of `onHelp`'s state machine instead. For a Python/R
render assertion, gate a separate test behind the existing opt-in flag:

```typescript
    test.skip(process.env.CHATISA_LIVE_NET !== "1", "runs a real heavy runtime; opt in with CHATISA_LIVE_NET=1");
```

- [ ] **Step 2: Confirm accessibility stays clean**

The existing sandbox axe test covers the four-pane shell. Confirm it still passes with
the HELP doc region present:

Run: `npm run test:e2e -- sandbox.spec.ts -g "four-pane"`
Expected: PASS. The doc region is a labelled `role="region"` with a heading-labelled
tabpanel around it, so axe stays clean.

- [ ] **Step 3: Run it and confirm it passes**

Run: `npm run test:e2e -- sandbox.spec.ts -g "HELP pane shows a loading state"`
Expected: PASS. If sqlite-wasm flakes in CI, gate this one test behind the same opt-in
flag the live-network test uses.

- [ ] **Step 4: Full verification (no commit)**

Run: `npm run typecheck` — no errors.
Run: `npm run lint` — no errors.
Run: `npm run test -- help-docs-doc-text run-harness help-docs` — the unit suites pass.
Run: `npm run test:e2e -- sandbox.spec.ts` — the sandbox suite passes, including the
existing HELP-tab tests from Slice 4 B (the symbol, source, and link still render), the
new loading/fallback test, and the shell/axe test. Leave the working tree uncommitted.

---

## Self-Review

**Feedback coverage (the professor's ask).**

- Documentation renders in the HELP pane, not only a click-out link. Task 3 (Python
  docstring), Task 4 (R help page), Task 6 (`HelpBody` renders the doc region). The
  link stays as a secondary affordance in every state (Task 6 Step 5).
- No-network local docs. Python docstrings and R help ship inside the loaded
  runtime/packages; the fetch is same-tab introspection, so COEP does not apply
  (Architecture, Tasks 3 and 4). No iframe, no cross-origin fetch is added.
- SQL honesty. SQLite has no runtime help, so it answers found:false and the pane keeps
  the blurb + link with an honest line (Task 5, Task 6 Step 5). The asymmetry is stated,
  not hidden.
- Async, non-blocking, no focus theft. The fetch runs off the click handler; `onHelp`
  only sets state and starts the fetch; the loading status is a polite live region; the
  editor caret and scroll are untouched (Task 6 Step 3, Global Constraints).
- Read-only, no session reset. `docRequest` uses `keepState: true` and runs only
  introspection (the same surface as the shipped `completeAt`), never the student's code
  and never a state change (Task 2, Tasks 3 to 5, Global Constraints).
- Graceful guards. A missing symbol, a runtime with no local doc, a timeout, or an
  introspection error all resolve to `{ found: false }` -> the friendly "No
  documentation text is available for X" plus the link, never an error dialog (Task 2
  Step 1 tests, Tasks 3 to 5 catch branches, Task 6 Step 5).
- WCAG AA. The doc region is labelled and keyboard-scrollable; the loading status is a
  live region; the link is a real anchor (Task 6 Step 5, Task 7 Step 2).
- Truncation. Worker hard-cap (12000 chars) bounds the message; `truncateDocText`
  applies the display cap and drives the visible "the full page is one click away" note
  (Task 1, Task 3/4 caps, Task 6 Step 5).
- No em dashes in any new copy (loading text, region label, fallback line, truncation
  note, link text). Verified against the strings in Task 6 Step 5.

**Honesty caveats (called out, not hidden).**

- WebR help availability: base R help renders reliably; dplyr/ggplot2 help renders only
  if the WebR package binary carries its help database, and falls back to the blurb + link
  otherwise. Must be verified manually during Task 4/Task 6 and recorded. The code never
  fabricates help text.
- Docstring/help size: pandas method docstrings and R help pages are long; they are
  capped in the worker and on the main thread, with a visible note and the full page one
  click away.
- Pyodide static resolution: `df.groupby` before any `df` exists resolves through the
  `pandas.DataFrame.<name>` fallback using the source hint; if neither the live object
  nor the module resolves (or the docstring is empty), it is found:false -> fallback.
- SQL: no runtime help at all, by design; blurb + link only.

**Placeholder scan.** No TBDs. Every step shows final code. The two verification-only
points (WebR dplyr help availability; the brief SQL loading state in the e2e) are called
out explicitly with the chosen handling (fallback; assert the stable end state). Commands
are exact and adapted to the no-commit rule (verify steps replace commit steps).

**Type consistency.** `DocRequest` and `DocText` flow from `doc-text.ts` through
`buildDocRequest`, `LanguageRunner.fetchDoc`, `RunSession.fetchDoc`, `WorkerReply.doc`,
`onHelp`, and `HelpBody` unchanged. The workers post exactly `{ doc: { found, text?,
signature? } }`. `helpTarget` gains `req: HelpRequest`; `helpDoc.status` is the single
"idle" | "loading" | "loaded" | "none" state machine that drives the pane.

## Handoff note

Per the task that produced this plan, this is a planning document only; do not build
from it in this session. When execution is scheduled, use
superpowers:subagent-driven-development (a fresh subagent per task, review between) or
superpowers:executing-plans (batched with checkpoints), and remember: no git commits,
verify with the commands above instead. This revision keeps Slice 4 B's frozen symbol
resolution (`symbol-at.ts`, `resolve.ts`) and editor plumbing; it adds the read-only
`docRequest` worker op and rewrites only `HelpBody` and the `Workspace` glue. Keep the
`docRequest` / `doc` message shape and the `doc-text.ts` helpers stable, and record the
WebR package-help availability outcome in the interaction log so the honest fallback is
documented, not a surprise.
