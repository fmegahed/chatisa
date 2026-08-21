import type { NextConfig } from "next";
import {
  ISOLATED_ASSET_SOURCES,
  ISOLATED_PAGE_SOURCES,
  ISOLATION_HEADERS,
} from "./lib/run/isolation";

const nextConfig: NextConfig = {
  // Production ships as a self-contained folder (server.js + only the
  // node_modules it needs, native better-sqlite3 binary included), assembled
  // by scripts/make-deploy-bundle.mjs. The server machine needs Node only:
  // no npm install, no build step there. Dev and e2e (which run `next dev`)
  // are unaffected.
  output: "standalone",
  async headers() {
    // Which routes are isolated, and the full reasoning, live in
    // lib/run/isolation.ts beside the test that checks no page rendering
    // runnable code is left out. require-corp is safe here because the app is
    // fully self-hosted: every image, font and script on these pages is
    // same-origin, so there is no cross-origin subresource to be blocked.
    // COOP is ignored on the worker/runtime (non-document) responses, which is
    // harmless.
    const isolation = [...ISOLATION_HEADERS];
    return [...ISOLATED_PAGE_SOURCES, ...ISOLATED_ASSET_SOURCES].map(
      (source) => ({ source, headers: isolation }),
    );
  },
  async redirects() {
    // Route slugs were renamed 2026-07-24 to match the modules' display names
    // (a bookmark or shared link should land where it used to). Temporary (307)
    // rather than permanent, so browsers do not cache the mapping forever if a
    // name changes again.
    const renames: [string, string][] = [
      ["/coding-companion", "/coding-tutor"],
      ["/ai-sandbox", "/coding-studio"],
      ["/exam-ally", "/exam-prep"],
      ["/project-coach", "/project-assistant"],
      ["/jobapp-assistant", "/jobapp-drafter"],
      ["/general-chat", "/ask-anything"],
      ["/ai-comparisons", "/ai-comparison"],
      ["/job-scout/github-connected", "/portfolio/github-connected"],
    ];
    return [
      ...renames.flatMap(([from, to]) => [
        { source: from, destination: to, permanent: false },
        {
          source: `${from}/:path*`,
          destination: `${to}/:path*`,
          permanent: false,
        },
      ]),
      // The Portfolio Builder took over Job Scout's portfolio tab (2026-08-20),
      // so a saved deep link to that tab lands on the new module.
      {
        source: "/job-scout",
        has: [{ type: "query", key: "tab", value: "portfolio" }],
        destination: "/portfolio",
        permanent: false,
      },
    ];
  },
};

export default nextConfig;
