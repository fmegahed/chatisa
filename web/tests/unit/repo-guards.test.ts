import { describe, expect, it } from "vitest";
import { guardRepoSkills } from "@/lib/scout/repo-guards";
import type { RepoSummary } from "@/lib/scout/github-summary";

const summary: RepoSummary = {
  fullName: "ada/churn", description: "", topics: [], fork: false, archived: false,
  languages: { Python: 10_000 }, authorship: { studentCommits: 20, totalCommits: 25 },
  tree: [{ path: "src/model.py", size: 10 }, { path: "requirements.txt", size: 10 }], treeTruncated: false,
  readme: "Uses Tableau dashboards and scikit-learn.",
  dependencyFiles: [{ path: "requirements.txt", text: "pandas\nscikit-learn" }],
  codeFiles: [{ path: "src/model.py", text: "import pandas as pd\nfrom sklearn.ensemble import GradientBoostingClassifier" }],
  skippedLarge: [],
};
const p = (skillId: string, level: "anchor" | "applied" | "exposure", evidence: string) => ({ skillId, level, evidence });

describe("guardRepoSkills", () => {
  it("keeps an anchor from a substantial repository whose evidence names a code file", () => {
    const out = guardRepoSkills(summary, [p("machine_learning", "anchor", "trained a churn classifier in src/model.py")]);
    expect(out.suggestions).toEqual([{ skillId: "machine_learning", suggested: "anchor", evidence: "trained a churn classifier in src/model.py" }]);
    expect(out.substantial).toBe(true);
  });

  it("lowers an anchor whose evidence is only the README, or cites a file that was not read", () => {
    const out = guardRepoSkills(summary, [
      p("machine_learning", "anchor", "described the model in README.md"),
      p("data_wrangling", "anchor", "cleaned data in src/clean.py"),
    ]);
    expect(out.suggestions.map((s) => s.suggested)).toEqual(["applied", "applied"]);
  });

  it("lowers every anchor when the repository is not substantially the student's", () => {
    const out = guardRepoSkills({ ...summary, authorship: { studentCommits: 5, totalCommits: 30 } }, [p("machine_learning", "anchor", "trained in src/model.py")]);
    expect(out.suggestions[0].suggested).toBe("applied");
    expect(out.substantial).toBe(false);
  });

  it("drops unknown skills and tools with no proof beyond the README", () => {
    const out = guardRepoSkills(summary, [p("made_up", "applied", "x"), p("tableau", "applied", "dashboards"), p("scikit_learn", "applied", "src/model.py")]);
    expect(out.suggestions.map((s) => s.skillId)).toEqual(["scikit_learn"]);
  });

  it("proves version control by the student's own commits", () => {
    expect(guardRepoSkills(summary, [p("version_control", "applied", "git")]).suggestions).toHaveLength(1);
    expect(guardRepoSkills({ ...summary, authorship: { studentCommits: 3, totalCommits: 3 } }, [p("version_control", "applied", "git")]).suggestions).toHaveLength(0);
  });

  it("keeps at most 3 anchors, in the model's order, and at most 8 skills", () => {
    const ids = ["machine_learning", "classification", "predictive_modeling", "data_wrangling", "regression", "python", "pandas", "data_analysis", "statistical_analysis"];
    const out = guardRepoSkills(summary, ids.map((id) => p(id, "anchor", "in src/model.py")));
    expect(out.suggestions).toHaveLength(8);
    expect(out.suggestions.filter((s) => s.suggested === "anchor").map((s) => s.skillId)).toEqual(["machine_learning", "classification", "predictive_modeling"]);
  });

  it("gives exposure only when no code or data file was read", () => {
    const out = guardRepoSkills({ ...summary, codeFiles: [], dependencyFiles: [], languages: {} }, [p("machine_learning", "anchor", "README")]);
    expect(out.suggestions).toEqual([{ skillId: "machine_learning", suggested: "exposure", evidence: "README" }]);
    expect(out.codeRead).toBe(false);
  });

  it("keeps one suggestion per skill", () => {
    const out = guardRepoSkills(summary, [p("python", "applied", "src/model.py"), p("Python", "exposure", "y")]);
    expect(out.suggestions).toHaveLength(1);
  });

  it("does not let a dependency file alone support an anchor or count as code (review fix)", () => {
    const depsOnly = { ...summary, codeFiles: [], languages: {} };
    const out = guardRepoSkills(depsOnly, [p("machine_learning", "anchor", "listed in requirements.txt")]);
    expect(out.codeRead).toBe(false);
    expect(out.suggestions[0].suggested).toBe("exposure");
    const withCode = guardRepoSkills(summary, [p("machine_learning", "anchor", "declared in requirements.txt")]);
    expect(withCode.suggestions[0].suggested).toBe("applied");
  });
});
