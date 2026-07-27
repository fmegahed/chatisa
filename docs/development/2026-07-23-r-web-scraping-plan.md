# R Web Scraping in Coding Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let R's `rvest` / `httr2` / `curl` reach real websites and APIs from Coding Studio (including sites with no CORS headers), by making the Coding Studio page cross-origin isolated and routing R's libcurl through a ws-proxy.

**Architecture:** Serve `/ai-sandbox` with COOP/COEP so WebR gets the SharedArrayBuffer channel (synchronous networking); make the Coding Studio nav entry a full page load so those headers actually apply; set `ALL_PROXY` automatically in the WebR worker from a configurable proxy value (default: the public `ws.r-universe.dev`); bundle `httr2`; and correct the Limitations notice. Proven end to end by a spike on 2026-07-23 (`rvest::read_html` scraped the FSB directory, 30 rows).

**Tech Stack:** Next.js 16, WebR 0.6.0, TypeScript, Vitest, Playwright.

## Global Constraints

- **No git commits, no deploys, no production access.** Working tree stays uncommitted; each task ends by running its gate. (Git repo at `webapp/`; `web/` and `docs/` untracked; never run git write commands.)
- **No secrets in the client;** the proxy string is public (public proxy, public creds), not a secret.
- **No em dashes in any user-facing text.**
- **This is a customized Next.js;** follow existing patterns.
- **Scope: R only.** Python scraping and the self-hosted proxy are deferred (roadmap).
- **Cross-origin isolation is scoped to the Coding Studio route** (`/ai-sandbox`), not app-wide (the coding GUI).
- **Header: `require-corp`** (universal browser support). The app has no cross-origin subresources (fully self-hosted, header shows the email as text, brand logo is same-origin), so nothing needs a CORP fallback. Confirmed by grep and by the spike running `require-corp` app-wide with the suite green.
- **Reference:** design spec `2026-07-23-r-web-scraping-design.md`; deferred items in `roadmap.md`.

---

## File Structure

**Created:**
- `lib/sandbox/network.ts` — the ws-proxy value (config, public default).
- `tests/unit/sandbox-network.test.ts`.

**Modified:**
- `next.config.ts` — COOP/COEP headers on `/ai-sandbox`.
- `components/ModuleNav.tsx` — full page load for the Coding Studio tab.
- `lib/run/manager.ts` — thread the proxy value to the WebR worker.
- `public/workers/webr-worker.mjs` — set `ALL_PROXY` in R once.
- `scripts/setup-runtimes.mjs` — add `httr2` to the bundled R mirror.
- `components/sandbox/LimitationsNotice.tsx` — accurate networking copy.
- `tests/e2e/sandbox.spec.ts` — isolation assertion (+ an opt-in live-scrape test).

## Interfaces produced

```ts
// lib/sandbox/network.ts
export const WS_PROXY: string; // "socks5h://user:pass@host:443"
```

---

### Task 1: Cross-origin isolation on the Coding Studio route

**Files:**
- Modify: `next.config.ts`
- Modify: `components/ModuleNav.tsx`

- [ ] **Step 1: Add the headers (scoped to `/ai-sandbox`)**

```ts
// next.config.ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async headers() {
    return [
      {
        // Cross-origin isolation gives WebR the SharedArrayBuffer channel, which
        // is what enables R's synchronous networking (rvest/httr2/curl through
        // the ws-proxy). Scoped to Coding Studio: no other route needs it, and
        // this keeps the isolation blast radius to one page. require-corp is
        // safe here because the app is fully self-hosted (no cross-origin
        // subresources).
        // NOTE (corrected during execution): the page alone is not enough.
        // WebR spawns a nested worker and we spawn our own; each worker script
        // needs COEP too, or it is not isolated and networking fails. So apply
        // the same headers to "/ai-sandbox", "/workers/:path*", and
        // "/runtimes/:path*" (see the final next.config.ts and the design spec).
        source: "/ai-sandbox",
        headers: [
          { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
          { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
        ],
      },
    ];
  },
};

export default nextConfig;
```

- [ ] **Step 2: Make the Coding Studio tab a full page load**

Cross-origin isolation only applies to a document actually loaded with the headers. A client-side (SPA) navigation into `/ai-sandbox` keeps the previous, non-isolated document, so the page would not be isolated and R networking would silently not work. Render the Coding Studio nav item as a plain anchor (full navigation) instead of `next/link`.

In `components/ModuleNav.tsx`, where each item renders `<Link href={href}>`, special-case the Coding Studio slug:

```tsx
            <li key={href}>
              {slug === "ai-sandbox" ? (
                // Full page load so the cross-origin isolation headers apply
                // (SPA navigation would leave the page un-isolated and disable
                // R networking).
                <a
                  href={href}
                  aria-current={isCurrent ? "page" : undefined}
                  className={/* same className as the Link below */ linkClass}
                >
                  {label}
                </a>
              ) : (
                <Link
                  href={href}
                  aria-current={isCurrent ? "page" : undefined}
                  className={linkClass}
                >
                  {label}
                </Link>
              )}
            </li>
```

(Match the existing className/props on the current `<Link>`; the only change is `<a>` for the `ai-sandbox` slug. If the class is inline rather than a `linkClass` variable, repeat it verbatim on both branches.)

- [ ] **Step 3: Verify**

Run: `npm run typecheck && npm run lint`
Expected: clean. Then confirm isolation manually or via the e2e in Task 5: a hard load of `/ai-sandbox` has `crossOriginIsolated === true`.

- [ ] **Step 4: Checkpoint (no commit).** Working tree stays uncommitted.

---

### Task 2: Bundle httr2 into the R mirror

**Files:**
- Modify: `scripts/setup-runtimes.mjs`

- [ ] **Step 1: Add httr2 to the seed list**

`rvest`, `curl`, and `xml2` already arrive via the tidyverse closure; `httr2` is separate. In `scripts/setup-runtimes.mjs`:

```js
const WEBR_PACKAGES = ["tidyverse", "readxl", "janitor", "httr2"];
```

- [ ] **Step 2: Rebuild the mirror**

Run: `npm run setup:runtimes webr-packages`
Expected: the script resolves the closure (now including `httr2` and its dependencies) and writes them plus a filtered `PACKAGES`/`PACKAGES.gz` under `public/runtimes/webr-packages/...contrib/4.6/`. It is idempotent, so it only adds the new packages. Note the added package count and size in the run output.

- [ ] **Step 3: Confirm httr2 is in the mirror**

Run: `ls public/runtimes/webr-packages/bin/emscripten/contrib/4.6/ | grep -E "^httr2_|^curl_|^rvest_|^xml2_"`
Expected: `httr2_*.tgz` present (and curl/rvest/xml2, already there).

- [ ] **Step 4: Checkpoint.** `npm run typecheck` clean (script is not typechecked, but ensure nothing else broke). Working tree uncommitted.

---

### Task 3: Configure and wire ALL_PROXY

**Files:**
- Create: `lib/sandbox/network.ts`
- Test: `tests/unit/sandbox-network.test.ts`
- Modify: `lib/run/manager.ts`
- Modify: `public/workers/webr-worker.mjs`

- [ ] **Step 1: Write the failing test**

```ts
// tests/unit/sandbox-network.test.ts
import { afterEach, describe, expect, it, vi } from "vitest";

const PUBLIC_DEFAULT = "socks5h://test:yolo@ws.r-universe.dev:443";

afterEach(() => {
  vi.resetModules();
  delete process.env.NEXT_PUBLIC_WS_PROXY;
});

describe("ws-proxy config", () => {
  it("defaults to the public ws-proxy", async () => {
    const { WS_PROXY } = await import("@/lib/sandbox/network");
    expect(WS_PROXY).toBe(PUBLIC_DEFAULT);
  });

  it("honors NEXT_PUBLIC_WS_PROXY when set", async () => {
    process.env.NEXT_PUBLIC_WS_PROXY = "socks5h://u:p@wsproxy.example:443";
    vi.resetModules();
    const { WS_PROXY } = await import("@/lib/sandbox/network");
    expect(WS_PROXY).toBe("socks5h://u:p@wsproxy.example:443");
  });
});
```

- [ ] **Step 2: Run it (fails)** — `npx vitest run tests/unit/sandbox-network.test.ts` (module missing).

- [ ] **Step 3: Write the config module**

```ts
// lib/sandbox/network.ts
/**
 * The ws-proxy that WebR routes R's libcurl through, so rvest/httr2/curl can
 * reach the internet (including sites without CORS headers). The value is
 * public, not a secret. Defaults to the public rOpenSci proxy; override at build
 * time with NEXT_PUBLIC_WS_PROXY. A runtime-swappable value ships with the
 * self-hosted proxy (see roadmap.md).
 */
export const WS_PROXY =
  process.env.NEXT_PUBLIC_WS_PROXY ?? "socks5h://test:yolo@ws.r-universe.dev:443";
```

- [ ] **Step 4: Run it (passes)** — `npx vitest run tests/unit/sandbox-network.test.ts`.

- [ ] **Step 5: Thread the value to the WebR worker (manager)**

In `lib/run/manager.ts`, import the value and attach it to messages sent to the R worker only (other workers ignore unknown fields, but scoping keeps it clean). Add the import near the top:

```ts
import { WS_PROXY } from "@/lib/sandbox/network";
```

In the `dispatch` method, include `wsProxy` for the R language when posting:

```ts
      this.pending.set(id, { resolve, timer });
      worker.postMessage({
        id,
        ...(this.language.id === "r" ? { wsProxy: WS_PROXY } : {}),
        ...payload,
      });
```

(Every R run and the prewarm now carry the proxy; the worker applies it once.)

- [ ] **Step 6: Set ALL_PROXY in the worker**

In `public/workers/webr-worker.mjs`:

- Add a memoized setter near `ensurePackages`:

```js
// Point R's libcurl at the ws-proxy exactly once, so rvest/httr2/curl reach the
// internet. Harmless when the page is not cross-origin isolated: the setenv
// succeeds, but R's synchronous networking still cannot run, and the request
// fails with libcurl's own error rather than silently.
let networkingProxy = null;
async function ensureNetworking(webR, wsProxy) {
  if (!wsProxy || networkingProxy === wsProxy) return;
  networkingProxy = wsProxy;
  await webR.evalRVoid(`Sys.setenv(ALL_PROXY=${rStr(wsProxy)})`);
}
```

- Destructure `wsProxy` from the incoming message (add it to the existing destructure at the top of `self.onmessage`):

```js
  const { id, code, keepState, withVariables, dataRequest, completeAt, prewarm, fileOp, wsProxy } =
    event.data;
```

- Call it right after `ensurePackages(webR)` in both the **run** branch and the **prewarm** branch:

```js
      await ensurePackages(webR);
      await ensureNetworking(webR, wsProxy);
```

- [ ] **Step 7: Checkpoint** — `npm run typecheck && npm run lint`. Working tree uncommitted.

---

### Task 4: Correct the Limitations notice

**Files:**
- Modify: `components/sandbox/LimitationsNotice.tsx`

- [ ] **Step 1: Rewrite the networking copy**

Update the notice so the "cannot reach the internet" message is gone and the caveat is Python-only. Two concrete edits:

1. The top-of-file banner/intro that currently says code cannot reach the internet due to CORS: rewrite to state that **R can reach the internet** (rvest/httr2/curl are tunneled through a built-in proxy, so even sites without CORS headers work), and that **Python** web requests are still limited by CORS. No em dashes.

2. In the per-language help:
   - R "Already installed" list: add `httr2` (now `tidyverse, readxl, janitor and httr2`), and add a line that R can fetch web pages and APIs (for example `rvest::read_html(url)`), routed through the proxy.
   - Python help: add that `requests` and `beautifulsoup4` are available, but Python web requests go through the browser and are limited by CORS, so `requests.get()` works for CORS-enabled APIs and fails on sites that block cross-origin requests; `beautifulsoup4` parses fine once you have the HTML. Include an accurate link:

```tsx
        <a
          href="https://pyodide.org/en/stable/usage/faq.html#why-can-t-i-load-files-from-the-local-file-system"
          className="underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          why browser networking is limited
        </a>
```

(If that exact FAQ anchor has moved, link the current Pyodide networking/FAQ page; the point is an accurate networking/CORS reference, not the packages list. Verify the link resolves.)

- [ ] **Step 2: Checkpoint** — `npm run typecheck && npm run lint`. Confirm the component still has no em dashes and the axe-relevant structure (labels, toggle) is unchanged.

---

### Task 5: Tests, gate, and log

**Files:**
- Modify: `tests/e2e/sandbox.spec.ts`

- [ ] **Step 1: Isolation assertion (default suite)**

Add a test that a hard load of Coding Studio is cross-origin isolated, and that entering via the nav tab (full load) is isolated too:

```ts
test("Coding Studio is cross-origin isolated (enables R networking)", async ({ page }) => {
  await page.goto("/ai-sandbox");
  await expect(
    page.getByRole("heading", { level: 1, name: "Coding Studio" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true,
    ),
  ).toBe(true);

  // Entering from another page via the nav must be a full load, so isolation
  // still applies (an SPA navigation would leave it un-isolated).
  await page.goto("/");
  await page.getByRole("link", { name: /Coding Studio/ }).click();
  await expect(
    page.getByRole("heading", { level: 1, name: "Coding Studio" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true,
    ),
  ).toBe(true);
});
```

- [ ] **Step 2: Opt-in live-scrape test (not in the default gate)**

Add a test that reproduces the spike, skipped unless `CHATISA_LIVE_NET=1`, because it needs live network to the proxy and the target:

```ts
test("R scrapes a no-CORS site via the ws-proxy", async ({ page }) => {
  test.skip(process.env.CHATISA_LIVE_NET !== "1", "needs live network; opt in with CHATISA_LIVE_NET=1");
  test.setTimeout(300_000);

  await page.goto("/ai-sandbox");
  await page.getByRole("radio", { name: "R" }).click();
  const editor = page.locator(".cm-content");
  await expect(editor).toBeVisible();
  await editor.click();
  await page.keyboard.press("Control+A");
  await page.keyboard.press("Delete");
  await page.keyboard.insertText(
    'cat("ROWS=", nrow(rvest::read_html(' +
      '"https://miamioh.edu/fsb/directory/?up=/query/all/all/Information_Systems_and_Analytics/all"' +
      ') |> rvest::html_element("table") |> rvest::html_table()), "\\n")',
  );
  await page.getByRole("button", { name: "Run" }).click();
  await expect(page.locator('[aria-label="Console output"]')).toContainText(/ROWS=\s*\d+/, {
    timeout: 260_000,
  });
});
```

(The proxy is now set automatically by the worker, so the test does not call `Sys.setenv`. That the row count appears proves the auto-wiring and isolation work end to end.)

- [ ] **Step 3: Full gate**

Run: `npm run typecheck && npm run lint && npm test && npm run test:e2e`
Expected: green, real counts quoted. The default `test:e2e` now runs with `/ai-sandbox` isolated (the feature is on), so confirm the existing sandbox specs still pass under isolation. The live-scrape test is skipped by default; run it deliberately once with `CHATISA_LIVE_NET=1 npm run test:e2e -- sandbox` to confirm the real scrape, and report that result separately.

- [ ] **Step 4: Migration log**

Append a dated entry: cross-origin isolation on the Coding Studio route (require-corp, full-load nav entry), the auto-wired `ALL_PROXY` to the public ws-proxy (config via `NEXT_PUBLIC_WS_PROXY`), `httr2` bundling, and the corrected Limitations notice. Note the self-hosted proxy and Python scraping remain deferred (roadmap).

---

## Self-Review

**1. Spec coverage (`2026-07-23-r-web-scraping-design.md`):**
- Cross-origin isolation, require-corp, scoped to Coding Studio (section 4.1): Task 1. The full-load-entry gotcha is handled (Task 1 Step 2). Covered.
- Proxy as config, public default (4.2): `lib/sandbox/network.ts` (Task 3). Covered.
- Automatic ALL_PROXY, no student `Sys.setenv` (4.3): worker sets it from the threaded value (Task 3). Covered.
- Bundle httr2 (4.4): Task 2. rvest/curl/xml2 already present. Covered.
- Limitations notice, Python-only caveat, accurate link (4.5): Task 4. Covered.
- Error handling: libcurl's own error surfaces; `ensureNetworking` is harmless when not isolated (Task 3 comment). Covered.
- Testing (section 6): isolation assertion, opt-in live scrape, regression under isolation (Task 5). Covered.
- Deferred correctly: self-hosted proxy, Python (roadmap), not in this plan.

**2. Placeholder scan:** none. The two "match the existing className / verify the link" notes are explicit, with the concrete action stated.

**3. Type/name consistency:** `WS_PROXY` is defined once and imported by the manager. `wsProxy` is the message field name in both the manager's `postMessage` and the worker's destructure and `ensureNetworking`. The header route `/ai-sandbox` matches the nav slug `ai-sandbox` and the page path. `httr2` is added in one place (the seed list) and surfaced in the notice.

---

## Execution Handoff

**Plan saved to `webapp/docs/development/2026-07-23-r-web-scraping-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks.

**2. Inline Execution** — execute here with checkpoints.

Which approach would you like? Note that the meaningful confirmation is the opt-in live-scrape (`CHATISA_LIVE_NET=1`), which reproduces the already-passing spike, plus a manual check in a real browser once it is wired.
