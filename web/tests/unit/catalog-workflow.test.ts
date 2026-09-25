import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
// js-yaml arrives with the lint toolchain and ships no types; v4's load()
// is the safe schema (no code-running tags).
// @ts-expect-error no declaration file
import yaml from "js-yaml";
import pkg from "../../package.json";

/**
 * The catalog refresh Action (v6.7.0). No actionlint on this machine, so the
 * contract is pinned here: when it runs, what it may touch, that only the
 * mapping step sees the API keys, and that every npm script it calls exists.
 */
type Step = { name?: string; run?: string; env?: Record<string, string>; uses?: string };
const file = path.join(__dirname, "../../../.github/workflows/catalog-refresh.yml");
const wf = yaml.load(readFileSync(file, "utf8")) as {
  on: { schedule: { cron: string }[]; workflow_dispatch: unknown };
  permissions: Record<string, string>;
  jobs: { refresh: { steps: Step[]; defaults: { run: { "working-directory": string } } } };
};
const steps = wf.jobs.refresh.steps;

describe("catalog refresh workflow", () => {
  it("runs one week into each term (professor, 2026-09-24), and on demand", () => {
    // Classes start on a Monday between the 22nd and 28th (Fall Aug 24 2026
    // and Aug 23 2027, Spring Jan 25 2027 and Jan 24 2028, Summer's main
    // sessions May 24 2027), so a daily window from the 29th to the 4th plus
    // a Monday guard lands exactly one week in, once per term. One run per
    // window matters: a second run before the first PR is merged would map
    // the same courses again and open a duplicate PR (review fix). Winter
    // term (from Jan 2) runs on Jan 9.
    expect(wf.on.schedule).toEqual([
      { cron: "0 12 29-31 1,5,8 *" },
      { cron: "0 12 1-4 2,6,9 *" },
      { cron: "0 12 9 1 *" },
    ]);
    expect(wf.on).toHaveProperty("workflow_dispatch");
  });

  it("guards scheduled runs to the Monday one week in (or Jan 9), never manual ones", () => {
    const when = steps.find((s) => s.name === "Decide whether this run is due");
    expect(when?.run).toContain("workflow_dispatch");
    expect(when?.run).toContain("date -u +%u");
    // It runs before checkout, when the web/ default directory does not exist.
    expect((when as { "working-directory"?: string })["working-directory"]).toBe(".");
    const guarded = steps.filter((s) => s.run && s !== when);
    for (const s of guarded) expect((s as { if?: string }).if, s.name).toBe("steps.when.outputs.due == 'true'");
  });

  it("uses current action versions (Node 20 actions are retired)", () => {
    const uses = steps.map((s) => s.uses).filter(Boolean);
    expect(uses).toEqual(["actions/checkout@v7", "actions/setup-node@v7"]);
  });

  it("may write contents and pull requests, nothing else", () => {
    expect(wf.permissions).toEqual({ contents: "write", "pull-requests": "write" });
  });

  it("gives the API keys to the mapping step only", () => {
    const withSecrets = steps.filter((s) => JSON.stringify(s.env ?? {}).includes("secrets."));
    expect(withSecrets.map((s) => s.name)).toEqual(["Map changed courses to skills"]);
  });

  it("opens a PR only for a real change: new files count, the timestamped manifest alone does not (review fix)", () => {
    const pr = steps.find((s) => s.name === "Open a pull request")?.run ?? "";
    expect(pr).toContain("git status --porcelain -- catalog ':!catalog/manifest.json'");
    expect(pr).not.toContain("git diff --quiet");
  });

  it("calls only npm scripts that exist, from the web directory", () => {
    expect(wf.jobs.refresh.defaults.run["working-directory"]).toBe("web");
    const called = steps.flatMap((s) => [...(s.run ?? "").matchAll(/npm run ([\w:-]+)/g)].map((m) => m[1]));
    expect(called).toEqual(["catalog:collect", "catalog:map", "catalog:report"]);
    for (const script of called) expect(pkg.scripts).toHaveProperty(script);
  });
});
