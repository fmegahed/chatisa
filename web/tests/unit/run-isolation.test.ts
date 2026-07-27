import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import {
  ISOLATED_ASSET_SOURCES,
  ISOLATED_PAGE_SOURCES,
  ISOLATION_HEADERS,
  KNOWN_UNISOLATED_RUN_PAGES,
} from "@/lib/run/isolation";
import nextConfig from "@/next.config";

/**
 * The bug these tests exist for, reported from production on 2026-07-26:
 * rvest::read_html failed with "cannot open the connection" on /coding-tutor.
 * The cause was not in the R code, the ws-proxy, or the worker. It was one
 * missing line of configuration: the cross-origin isolation headers were
 * attached to /coding-studio only, and without a SharedArrayBuffer WebR quietly
 * drops to a channel that has no networking at all.
 *
 * That failure mode is invisible to every other kind of test. The page renders,
 * the worker boots, code runs, and only the network calls fail, with an error
 * that points at R rather than at a header. So the check has to be static: find
 * every page that can execute code, and prove it is either isolated or listed
 * as a known gap.
 */

const ROOT = resolve(__dirname, "..", "..");
const SKIP = new Set(["node_modules", ".next", "test-results", "playwright-report"]);

function sourceFiles(dir: string, found: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    if (SKIP.has(entry)) continue;
    const path = join(dir, entry);
    if (statSync(path).isDirectory()) sourceFiles(path, found);
    else if (/\.tsx?$/.test(entry)) found.push(path);
  }
  return found;
}

/**
 * Import specifiers a file pulls in from our own alias or a relative path.
 *
 * Both static `from "x"` and dynamic `import("x")` forms are matched. The
 * dynamic form is not an edge case here: the two most important pages, Coding
 * Studio and Ask Anything, reach their runtimes ONLY through next/dynamic
 * (`dynamic(() => import("./Sandbox"))`), because those components must not
 * server-render. A scanner that reads static imports alone reports both pages as
 * unable to run code, which is the opposite of the truth.
 */
function localImports(file: string): string[] {
  const text = readFileSync(file, "utf8");
  const out: string[] = [];
  const patterns = [
    /from\s+"((?:@\/|\.\/|\.\.\/)[^"]+)"/g,
    /\bimport\s*\(\s*"((?:@\/|\.\/|\.\.\/)[^"]+)"\s*\)/g,
  ];
  for (const pattern of patterns) {
    for (let m = pattern.exec(text); m; m = pattern.exec(text)) out.push(m[1]);
  }
  return out;
}

/** Resolves an import specifier to a source path on disk, if it is one. */
function resolveTsx(fromFile: string, spec: string): string | null {
  const base = spec.startsWith("@/")
    ? join(ROOT, spec.slice(2))
    : resolve(fromFile, "..", spec);
  for (const candidate of [
    `${base}.tsx`,
    `${base}.ts`,
    join(base, "index.tsx"),
    join(base, "index.ts"),
  ]) {
    try {
      if (statSync(candidate).isFile()) return candidate;
    } catch {
      // Not one of ours (a package, or a type-only path). Nothing to follow.
    }
  }
  return null;
}

/**
 * Every page.tsx that can reach the run manager, directly or through any number
 * of imports. The manager is the target rather than the RunnableCode component
 * because the app has THREE ways to reach a runtime, and only one of them is
 * that component:
 *
 *   - Coding Tutor, AI Comparison, Project Assistant: assistant Markdown wraps
 *     code fences in RunnableCode ("Run R" buttons).
 *   - Coding Studio: the Sandbox workbench calls the manager itself.
 *   - Ask Anything: the run_python / run_r / run_sql tools call the manager.
 *
 * Targeting the component found only the first group, which is how a scan can
 * look thorough and still miss the module the professor cares about most. The
 * manager is the one place all three converge, so it is the honest target.
 */
function pagesThatCanRunCode(): string[] {
  const RUNNER = resolve(ROOT, "lib", "run", "manager.ts");

  // Reverse reachability by fixpoint rather than a depth-first walk. A DFS that
  // shares one "visited" set across sibling branches reports false for any file
  // whose only path to the target runs through a subtree an earlier sibling
  // already entered, and component graphs have plenty of shared nodes. Iterating
  // to a fixpoint has no such ordering dependence and terminates on cycles.
  const files = [
    ...sourceFiles(join(ROOT, "app")),
    ...sourceFiles(join(ROOT, "components")),
    ...sourceFiles(join(ROOT, "lib")),
  ];
  const edges = new Map<string, string[]>();
  for (const file of files) {
    edges.set(
      file,
      localImports(file)
        .map((spec) => resolveTsx(file, spec))
        .filter((f): f is string => f !== null),
    );
  }

  const reaches = new Set<string>([RUNNER]);
  for (let changed = true; changed; ) {
    changed = false;
    for (const [file, imports] of edges) {
      if (reaches.has(file)) continue;
      if (imports.some((i) => reaches.has(i))) {
        reaches.add(file);
        changed = true;
      }
    }
  }

  return files
    .filter((f) => /[\\/]page\.tsx$/.test(f))
    .filter((f) => reaches.has(f))
    .map((f) => relative(ROOT, f).replace(/\\/g, "/"))
    .sort();
}

/** "app/(app)/coding-tutor/page.tsx" -> "/coding-tutor". */
function routeOf(pageFile: string): string {
  const segments = pageFile
    .replace(/^app\//, "")
    .replace(/\/page\.tsx$/, "")
    .split("/")
    .filter((s) => !/^\(.*\)$/.test(s)) // route groups are not in the URL
    // A dynamic segment becomes "*", NOT dropped. Dropping it turned
    // app/(app)/project-assistant/[projectId]/coach/[coachType]/page.tsx into
    // "/project-assistant/coach", which matches no Next.js source pattern, so a
    // route that HAD been isolated still reported as unaccounted for.
    .map((s) => (/^\[.*\]$/.test(s) ? "*" : s));
  return `/${segments.join("/")}`;
}

/** A Next.js header `source` in the same canonical form as routeOf's output, so
 * "/project-assistant/:projectId/coach/:coachType" and the page path above
 * compare equal. */
function canonicalSource(source: string): string {
  return source
    .split("/")
    .map((s) => (s.startsWith(":") ? "*" : s))
    .join("/");
}

describe("cross-origin isolation coverage", () => {
  it("finds the pages that can execute code", () => {
    // A canary on the scanner itself: if this ever comes back empty (an import
    // style it cannot parse, a moved file), every other assertion here would
    // pass vacuously and the guard would be silently dead.
    const pages = pagesThatCanRunCode();
    expect(pages.length).toBeGreaterThanOrEqual(4);
    expect(pages).toContain("app/(app)/coding-studio/page.tsx");
    expect(pages).toContain("app/(app)/coding-tutor/page.tsx");
    expect(pages).toContain("app/(app)/ask-anything/page.tsx");
  });

  it("isolates every code-running page, or records it as a known gap", () => {
    const isolated = new Set<string>(
      ISOLATED_PAGE_SOURCES.map(canonicalSource),
    );
    const known = new Set<string>(KNOWN_UNISOLATED_RUN_PAGES.map(canonicalSource));
    const unaccounted = pagesThatCanRunCode()
      .map(routeOf)
      // A dynamic route collapses to its static prefix, e.g.
      // /project-assistant/[projectId]/coach/[coachType] -> /project-assistant.
      .map((route) => {
        const match = [...isolated, ...known].find(
          (src) => route === src || route.startsWith(`${src}/`),
        );
        return match ?? route;
      })
      .filter((route) => !isolated.has(route) && !known.has(route));

    // Anything here can run R or Python with no network and no explanation to
    // the student. Either add the route to ISOLATED_PAGE_SOURCES or, if that is
    // deliberate, to KNOWN_UNISOLATED_RUN_PAGES with the reason.
    expect(unaccounted).toEqual([]);
  });

  it("keeps the isolated pages and the known gaps disjoint", () => {
    const overlap = ISOLATED_PAGE_SOURCES.filter((s) =>
      (KNOWN_UNISOLATED_RUN_PAGES as readonly string[]).includes(s),
    );
    expect(overlap).toEqual([]);
  });
});

describe("next.config headers", () => {
  async function headerRules() {
    const rules = await nextConfig.headers?.();
    return rules ?? [];
  }

  it("serves both isolation headers on every isolated source", async () => {
    const rules = await headerRules();
    for (const source of [...ISOLATED_PAGE_SOURCES, ...ISOLATED_ASSET_SOURCES]) {
      const rule = rules.find((r) => r.source === source);
      expect(rule, `no header rule for ${source}`).toBeDefined();
      for (const { key, value } of ISOLATION_HEADERS) {
        expect(rule?.headers).toEqual(
          expect.arrayContaining([expect.objectContaining({ key, value })]),
        );
      }
    }
  });

  it("covers the worker and runtime assets, not only the pages", () => {
    // Our worker script and WebR's nested worker each need COEP of their own:
    // an isolated page that loads a non-isolated worker is not isolated inside
    // that worker, which is exactly where R runs.
    expect(ISOLATED_ASSET_SOURCES).toContain("/workers/:path*");
    expect(ISOLATED_ASSET_SOURCES).toContain("/runtimes/:path*");
  });

  it("requires corp rather than credentialless", async () => {
    // credentialless would let cross-origin subresources through without CORP,
    // but strips credentials from them. The app has no cross-origin
    // subresources, so require-corp is both safe and the stricter choice.
    const rules = await headerRules();
    const coep = rules
      .flatMap((r) => r.headers)
      .filter((h) => h.key === "Cross-Origin-Embedder-Policy");
    expect(coep.length).toBeGreaterThan(0);
    for (const header of coep) expect(header.value).toBe("require-corp");
  });
});
