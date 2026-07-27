# R web scraping in Coding Studio — design spec

Date: 2026-07-23. Status: approved in brainstorming (mechanism proven by spike),
pending user review of this document before an implementation plan.

Standing constraints apply: no git commits, no production access, no secrets in
the client, WCAG 2.1 AA, Miami brand tokens, and no em dashes in user-facing text.

## 1. What it is

Make R networking work in Coding Studio so `rvest`, `httr2`, and `curl` can reach
real websites and APIs from the student's browser, including sites that send no
CORS headers (like the FSB directory). Today R runs on WebR's PostMessage channel
with no proxy, so all R networking fails. This closes that gap for R.

Python (`requests` / `beautifulsoup4`) is explicitly out of scope here and
deferred (section 8), because Pyodide's networking is a different, CORS-bound
mechanism that does not benefit from this work.

## 2. The mechanism (confirmed by spike, 2026-07-23)

Two independent things are required, and both were verified end to end in our own
WebR 0.6.0 (`rvest::read_html` scraped the FSB directory, 30 rows, which sends no
CORS headers):

1. **Cross-origin isolation.** Serving the page with
   `Cross-Origin-Opener-Policy: same-origin` and
   `Cross-Origin-Embedder-Policy: require-corp` makes the page cross-origin
   isolated, which gives WebR the SharedArrayBuffer channel, which is what allows
   R to make a synchronous network request at all. Without these headers WebR
   falls back to the PostMessage channel and R networking cannot happen.
2. **A ws-proxy.** R's libcurl is pointed at a SOCKS5-over-WebSocket proxy via
   `ALL_PROXY=socks5h://<user>:<pass>@<host>:443`. The proxy makes the real
   TCP/TLS connection server-side and tunnels the bytes back, so CORS never
   applies. This is a socket tunnel, not an HTTP/CORS proxy.

## 3. Goals and non-goals

Goals:
- R `rvest` / `httr2` / `curl` reach arbitrary sites from Coding Studio.
- Student code just works: `rvest::read_html("https://...")` with no manual proxy
  setup.
- Proxy target is configuration, so we can move from the public proxy to a
  self-hosted one later without code changes.

Non-goals (deferred, tracked in `roadmap.md`):
- **Self-hosted ws-proxy.** Ship on the public `ws.r-universe.dev`. The
  self-hosted version is a native Node WebSocket-to-SOCKS5 relay designed with
  the Next.js production deployment (Windows VM, no Docker/reverse proxy).
- **Python web scraping.** Separate mechanism, separate slice.
- **Arbitrary sockets** beyond what R's libcurl uses over the tunnel.

## 4. Design

### 4.1 Cross-origin isolation (the main real work)

Enable COOP + COEP so Coding Studio is cross-origin isolated. This is the piece
that needs care, because under COEP `require-corp` any cross-origin subresource
that does not send `Cross-Origin-Resource-Policy` (or CORS) is blocked. Two
decisions:

- **Scope (decided): tie the headers to the coding GUI's execution assets, not
  app-wide.** This means the Coding Studio page AND the worker/runtime scripts it
  loads: `/ai-sandbox`, `/workers/:path*`, and `/runtimes/:path*`. The
  page-response-only version (first attempt) is NOT enough: WebR spawns a nested
  worker (`/runtimes/webr/webr-worker.js`) and we spawn our own
  (`/workers/webr-worker.mjs`), and each worker script needs its own COEP header
  or that worker is not cross-origin isolated and WebR falls back to the
  PostMessage channel, so synchronous networking fails even though the page
  reports `crossOriginIsolated`. Confirmed during implementation: page-only
  isolation passed a page-level check but broke the live scrape; covering the
  worker/runtime paths fixed it. This still keeps the blast radius to the coding
  GUI, not the whole app. Note: the same runtimes power the "Run" buttons in the
  Coding Tutor chat (`/coding-companion`); isolating that document too (the
  workers are already covered) would let scraping work there as well. This slice:
  Coding Studio only; extend to the Tutor later if wanted.
- **Header choice / browser support.** `require-corp` works in all modern
  browsers (including Safari) but requires fixing cross-origin subresources.
  `credentialless` avoids the subresource problem (cross-origin resources load
  without credentials) but is not supported in Safari, where the page would then
  not be isolated and R networking would be off. Recommendation: `require-corp`
  for universal support, and handle the known cross-origin subresource.

- **The known subresource to handle: the Google profile avatar.** The app shell
  shows the signed-in user's Google avatar (`googleusercontent.com`), which sends
  no CORP and would be blocked on an isolated page. Options: render initials or a
  same-origin icon on the isolated page, or proxy the avatar through our origin.
  Recommendation: on the Coding Studio page, fall back to initials rather than the
  remote avatar, which is the smallest change. **An audit task** confirms this is
  the only cross-origin subresource on that route (fonts, images, scripts, and
  the runtime assets are all same-origin already, which is why the spike's
  `require-corp` run still loaded WebR fine).

Interaction with the future **Slice 10 CSP**: COOP/COEP are set alongside the CSP
headers; both are recorded together so production hardening sets them coherently.

### 4.2 Proxy target as configuration

A single server-read setting holds the proxy, for example an env var
`CHATISA_WS_PROXY` defaulting to `socks5h://test:yolo@ws.r-universe.dev:443`. The
value is passed to the WebR worker at startup. Swapping to a self-hosted relay
later is a config change, no code change. The value is not a secret (it is a
public proxy with public credentials), but it is read server-side and handed to
the client, never hard-coded in multiple places.

### 4.3 Automatic ALL_PROXY wiring

The WebR worker sets `ALL_PROXY` in the R session during initialization (the same
place it installs bundled packages), so student code needs no `Sys.setenv`. If
the page is not cross-origin isolated (for example an unsupported browser),
setting it is harmless: R networking simply fails as it does today, and the error
message explains that networking is unavailable in this browser.

### 4.4 Package bundling

`rvest`, `curl`, and `xml2` are already in our bundled tidyverse dependency
closure. Add **`httr2`** (and its dependency closure) to the WebR package mirror
built by `scripts/setup-runtimes.mjs`, so `httr2` is available offline like the
rest. Confirm the added size and that the closure resolves (same idempotent
mirror build already used for tidyverse/readxl/janitor).

### 4.5 Fix the Limitations notice

`components/sandbox/LimitationsNotice.tsx` currently tells students the opposite
of the truth (that code cannot reach the internet due to CORS). Rewrite it to be
accurate, and make the "scraping is limited" caveat **Python-only**, since R now
works:

- **R:** can reach the internet. `rvest` / `httr2` / `curl` requests are tunneled
  through a built-in proxy, so even sites without CORS headers work.
- **Python:** web requests are limited by CORS, unlike R. `requests` and
  `beautifulsoup4` are both available (Pyodide 0.25+ gives `requests` a fetch
  based adapter), but because Python's HTTP goes through the browser's fetch,
  `requests.get()` works only for sites that send CORS headers and fails on sites
  that block cross origin requests (like the FSB directory); `beautifulsoup4`
  parses fine once you have the HTML. A socket tunnel for Python is a future item.
  Link to Pyodide's networking/CORS documentation (accurate), not the packages
  list (the packages are supported; CORS is the limit).

No em dashes.

## 5. Error handling and accessibility

- If the proxy is unreachable or a request fails, R surfaces libcurl's own error;
  the console shows it verbatim (the error is the teaching signal), consistent
  with how run output already works.
- If the browser is not cross-origin isolated (unsupported browser), the notice
  and, on a failed network call, the console explain that in-browser networking is
  unavailable there, rather than failing silently.
- The Limitations notice stays keyboard-accessible and labelled; the e2e axe scans
  already cover that component.

## 6. Testing

- **Unit:** the proxy-config helper (default value, env override, the exact
  `ALL_PROXY` string it produces).
- **Isolation header:** an e2e assertion that the Coding Studio route responds
  with the COOP/COEP headers and that `crossOriginIsolated` is true on that page.
- **Live scrape (opt-in, not in the default suite):** a spike-style test, gated
  behind an env flag, that enables isolation and runs `rvest::read_html` against a
  known page through the public proxy, asserting a real result. It stays out of
  the default gate because it needs live network to the proxy and target; it is
  the reproduction of the spike that already passed.
- **Regression:** the normal suite runs with isolation off (default), so existing
  behavior is unchanged; and a check that the avatar fallback renders on the
  isolated Coding Studio page.

## 7. Security and privacy

- The public proxy carries **public-website scraping**, not sensitive student
  data, so the data-sensitivity concern is low. The real exposures are the public
  proxy's reliability and terms under class load, which is why self-hosting is on
  the roadmap.
- No secret values are added; the proxy string is public. Cross-origin isolation
  is a hardening posture (it also enables high-resolution timers, noted for the
  CSP slice).

## 8. Deferred, in `roadmap.md`

- **Self-hosted ws-proxy:** a native Node WebSocket-to-SOCKS5 relay co-located
  with the app, egress-filtered (block RFC1918 / loopback / link-local / cloud
  metadata) and gated to authenticated sessions, designed with the Next.js
  production deployment.
- **Python web scraping:** `requests` is fetch-based and CORS-bound in Pyodide;
  needs its own HTTP proxy or socket shim; `beautifulsoup4` parses fine once the
  HTML is in hand.

## 9. Decisions (settled 2026-07-23)

1. **Header: `require-corp`** (universal browser support, including Safari;
   requires the Google-avatar fallback to initials on the isolated page).
2. **Proxy: ship on the public `ws.r-universe.dev`**, self-hosting deferred to the
   production deployment.
3. **Scope: the Coding Studio route only** (tie isolation to the coding GUI), with
   the Coding Tutor as an easy later extension.
4. **Python notice wording corrected:** `requests`/`beautifulsoup4` are supported;
   the limit is CORS on Python's fetch, not package availability. R is enabled.
