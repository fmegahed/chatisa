import { describe, expect, it } from "vitest";
import {
  SUMMARY_LIMITS, authorshipFrom, clipSummary, isSubstantial, pickCodePaths, pickDependencyPaths, stripNotebook,
  type RepoSummary,
} from "@/lib/scout/github-summary";

const tree = [
  { path: "README.md", size: 900 },
  { path: "requirements.txt", size: 120 },
  { path: "src/model.py", size: 9_000 },
  { path: "src/utils.py", size: 2_000 },
  { path: "analysis.ipynb", size: 400_000 },
  { path: "huge.ipynb", size: 14_000_000 },
  { path: "mid.ipynb", size: 5_000_000 },
  { path: "enormous.py", size: 150_000_000 },
  { path: "node_modules/x/index.js", size: 50_000 },
  { path: ".venv/lib/site.py", size: 50_000 },
  { path: "renv/activate.R", size: 40_000 },
  { path: "queries/load.sql", size: 3_000 },
  { path: "data/raw.csv", size: 80_000 },
];

const base: RepoSummary = {
  fullName: "ada/churn", description: "", topics: [], fork: false, archived: false,
  languages: { Python: 10_000 }, authorship: { studentCommits: 12, totalCommits: 15 },
  tree, treeTruncated: false, readme: "# Churn", dependencyFiles: [], codeFiles: [], skippedLarge: [],
};

describe("github summary", () => {
  it("picks dependency files by name, anywhere in the tree", () => {
    expect(pickDependencyPaths(tree)).toEqual(["requirements.txt"]);
  });

  it("picks code files largest first up to 6 MB, skipping vendored paths, and lists what it skipped for size", () => {
    expect(pickCodePaths(tree)).toEqual({
      paths: ["mid.ipynb", "analysis.ipynb", "src/model.py", "queries/load.sql", "src/utils.py"],
      skippedLarge: [{ path: "huge.ipynb", size: 14_000_000 }],
    });
  });

  it("reads a large file the student chose, up to GitHub's 100 MB ceiling, and never beyond it", () => {
    const out = pickCodePaths(tree, ["huge.ipynb", "enormous.py"]);
    expect(out.paths[0]).toBe("huge.ipynb");
    expect(out.paths).not.toContain("enormous.py");
    expect(out.skippedLarge).toEqual([]);
  });

  it("reduces a notebook to its code and markdown, without outputs", () => {
    const nb = JSON.stringify({ cells: [
      { cell_type: "markdown", source: ["# Churn model"] },
      { cell_type: "code", source: ["import pandas as pd\n", "df = pd.read_csv('x')"], outputs: [{ text: "SECRET OUTPUT" }] },
    ] });
    const out = stripNotebook(nb);
    expect(out).toContain("# Churn model");
    expect(out).toContain("import pandas as pd");
    expect(out).not.toContain("SECRET OUTPUT");
    expect(stripNotebook("not json")).toBe("not json");
  });

  it("counts the student's commits against human commits, bots and anonymous authors in the total", () => {
    expect(authorshipFrom([
      { login: "ada", type: "User", contributions: 30 },
      { login: "dependabot[bot]", type: "Bot", contributions: 40 },
      { type: "Anonymous", contributions: 10 },
    ], "Ada")).toEqual({ studentCommits: 30, totalCommits: 40 });
    expect(authorshipFrom([], "ada")).toEqual({ studentCommits: 0, totalCommits: 0 });
  });

  it("calls a repository substantial only when it is the student's own work", () => {
    expect(isSubstantial(base)).toBe(true);
    expect(isSubstantial({ ...base, fork: true })).toBe(false);
    expect(isSubstantial({ ...base, authorship: { studentCommits: 9, totalCommits: 9 } })).toBe(false);
    expect(isSubstantial({ ...base, authorship: { studentCommits: 12, totalCommits: 21 } })).toBe(false);
  });

  it("clips every field and the total", () => {
    const big = "x".repeat(60_000);
    const s = clipSummary({
      ...base,
      readme: big,
      tree: Array.from({ length: 900 }, (_, i) => ({ path: `f${i}.py`, size: 1 })),
      dependencyFiles: [{ path: "requirements.txt", text: big }],
      codeFiles: Array.from({ length: 9 }, (_, i) => ({ path: `c${i}.py`, text: big })),
    });
    expect(s.readme.length).toBe(SUMMARY_LIMITS.readmeChars);
    expect(s.tree.length).toBe(SUMMARY_LIMITS.treeEntries);
    expect(s.treeTruncated).toBe(true);
    expect(s.codeFiles.length).toBeLessThanOrEqual(SUMMARY_LIMITS.codeFiles);
    const total = s.readme.length + [...s.dependencyFiles, ...s.codeFiles].reduce((n, f) => n + f.text.length, 0);
    expect(total).toBeLessThanOrEqual(SUMMARY_LIMITS.totalChars);
  });

  it("clips and coerces every field a browser could send (review fix)", () => {
    const s = clipSummary({
      ...base,
      tree: [{ path: "p".repeat(5_000), size: "12" as unknown as number }, null as unknown as { path: string; size: number }],
      languages: Object.fromEntries(Array.from({ length: 80 }, (_, i) => [`L${i}`.repeat(50), i])),
      authorship: { studentCommits: -5 as number, totalCommits: "9" as unknown as number },
      codeFiles: [null as unknown as { path: string; text: string }, { path: "a.py", text: "x" }],
    });
    expect(s.tree).toEqual([{ path: "p".repeat(300), size: 12 }]);
    expect(Object.keys(s.languages).length).toBeLessThanOrEqual(30);
    for (const k of Object.keys(s.languages)) expect(k.length).toBeLessThanOrEqual(40);
    expect(s.authorship).toEqual({ studentCommits: 0, totalCommits: 9 });
    expect(s.codeFiles).toEqual([{ path: "a.py", text: "x" }]);
  });

  it("reads common code types beyond data science: React, Java, Go, C++, SAS (review fix)", () => {
    const t = ["App.jsx", "Main.java", "main.go", "sim.cpp", "model.sas", "index.html"].map((path) => ({ path, size: 100 }));
    expect(pickCodePaths(t).paths.sort()).toEqual(["App.jsx", "Main.java", "main.go", "model.sas", "sim.cpp"]);
  });
});
