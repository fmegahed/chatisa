/**
 * Which routes are served cross-origin isolated, and why.
 *
 * Cross-origin isolation (COOP same-origin + COEP require-corp) is what gives
 * the page a SharedArrayBuffer. WebR needs one for its synchronous channel, and
 * that channel is what makes R's networking work at all: rvest, httr2 and curl
 * all go through libcurl, which WebR services over the ws-proxy with a blocking
 * call. Without a SharedArrayBuffer, WebR silently falls back to the PostMessage
 * channel, R's networking is simply absent, and every scrape fails with
 * "cannot open the connection". Python's urllib3 emscripten transport uses the
 * same SharedArrayBuffer for its streaming path, so both languages depend on it.
 *
 * The bug this constant exists to prevent (reported by the professor from
 * production on 2026-07-26, rvest::read_html failing on /coding-tutor): the
 * headers were attached to /coding-studio only, while FOUR pages render runnable
 * code blocks. A student pressing "Run R" on a Coding Tutor answer got a runtime
 * with no network, and nothing in the UI explained why. The page list therefore
 * lives here, next to a test that walks the components which mount
 * RunnableCode, rather than being retyped in next.config.ts.
 *
 * The asset routes matter as much as the pages: our worker script AND WebR's own
 * nested worker must each be served with COEP, or they are not isolated even
 * when their parent page is.
 */

/** Page routes that must be cross-origin isolated. Next.js `source` patterns. */
export const ISOLATED_PAGE_SOURCES = [
  // The Coding Studio: the plot pane, the runtimes, the whole workbench.
  "/coding-studio",
  // The Coding Tutor: answers carry Run R / Run Python blocks (2026-07-26).
  "/coding-tutor",
  // Ask Anything: run_r, run_python and run_sql execute in the student's tab.
  "/ask-anything",
  // AI Comparison and the Project Assistant coach sessions (2026-07-26, second
  // pass). Both render assistant Markdown, so both carry the same Run buttons as
  // the Coding Tutor and had the same broken networking. There is no reason for
  // the guarantee to differ by module: a student pressing "Run R" should get a
  // runtime that works wherever the button appears.
  "/ai-comparison",
  "/project-assistant/:projectId/coach/:coachType",
] as const;

/** Worker and runtime asset routes that must carry COEP. */
export const ISOLATED_ASSET_SOURCES = [
  "/workers/:path*",
  "/runtimes/:path*",
] as const;

/**
 * Pages that render runnable code blocks but are deliberately NOT isolated, so R
 * and Python networking is unavailable in them.
 *
 * EMPTY, and that is the intended state: every page that can execute code is
 * isolated. The list stays because the guard test needs somewhere to record a
 * deliberate exception, and because an empty list is a stronger statement than no
 * list at all. Adding a route here is a decision that needs a reason written next
 * to it.
 *
 * It briefly held /ai-comparison and the Project Assistant coach route, between
 * the first 2026-07-26 fix (scoped by the professor to the Coding Tutor and Ask
 * Anything) and the same day's follow-up that covered them too.
 */
export const KNOWN_UNISOLATED_RUN_PAGES: readonly string[] = [];

export const ISOLATION_HEADERS = [
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  { key: "Cross-Origin-Embedder-Policy", value: "require-corp" },
] as const;
