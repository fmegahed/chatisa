import { describe, expect, it } from "vitest";
import { extraSourceLabel, mergeRepoExtras, type ProfileExtra } from "@/lib/scout/profile-store";

const resume: ProfileExtra = { skillId: "sql", level: "applied", source: "resume", evidence: "queried" };

describe("GitHub extras", () => {
  it("stores confirmed skills with the repository and marks levels the student raised", () => {
    const out = mergeRepoExtras([resume], "ada/churn", [
      { skillId: "machine_learning", level: "anchor", suggested: "anchor", evidence: "in src/model.py" },
      { skillId: "python", level: "anchor", suggested: "applied", evidence: "in src/model.py" },
      { skillId: "pandas", level: "exposure", suggested: "applied", evidence: "in src/model.py" },
    ]);
    expect(out).toEqual([
      resume,
      { skillId: "machine_learning", level: "anchor", source: "github", repo: "ada/churn", evidence: "in src/model.py" },
      { skillId: "python", level: "anchor", source: "github", repo: "ada/churn", evidence: "in src/model.py", setByStudent: true },
      { skillId: "pandas", level: "exposure", source: "github", repo: "ada/churn", evidence: "in src/model.py" },
    ]);
  });

  it("replaces a repository's earlier skills when it is analyzed again, leaving other repositories alone", () => {
    const first = mergeRepoExtras([], "ada/churn", [{ skillId: "python", level: "applied", suggested: "applied", evidence: "a" }]);
    const other = mergeRepoExtras(first, "ada/web", [{ skillId: "javascript", level: "applied", suggested: "applied", evidence: "b" }]);
    const again = mergeRepoExtras(other, "ada/churn", [{ skillId: "sql", level: "applied", suggested: "applied", evidence: "c" }]);
    expect(again.map((e) => `${e.repo}:${e.skillId}`)).toEqual(["ada/web:javascript", "ada/churn:sql"]);
  });

  it("names the source for the skills panel", () => {
    expect(extraSourceLabel(resume)).toBe("your resume");
    expect(extraSourceLabel({ skillId: "x", level: "applied", source: "github", repo: "ada/churn" })).toBe("from ada/churn");
    expect(extraSourceLabel({ skillId: "x", level: "anchor", source: "github", repo: "ada/churn", setByStudent: true })).toBe("from ada/churn, set by you");
    expect(extraSourceLabel({ skillId: "x", level: "applied", source: "manual" })).toBe("added by you");
  });
});
