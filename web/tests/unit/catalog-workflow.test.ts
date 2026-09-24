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
  it("runs Feb 1, Jun 1 and Sep 1, and on demand", () => {
    expect(wf.on.schedule).toEqual([{ cron: "0 12 1 2,6,9 *" }]);
    expect(wf.on).toHaveProperty("workflow_dispatch");
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
