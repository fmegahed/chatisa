import { describe, expect, it } from "vitest";
import { canPick, MAX_REPOS, ruleText } from "@/components/scout/GithubSkills";

/** The five-repository cap (v6.8.0): unticking is always allowed. */
describe("canPick", () => {
  it("allows up to five and always allows unticking", () => {
    expect(MAX_REPOS).toBe(5);
    expect(canPick(["a", "b", "c", "d", "e"], "f")).toBe(false);
    expect(canPick(["a"], "b")).toBe(true);
    expect(canPick(["a", "b", "c", "d", "e"], "a")).toBe(true);
  });
});

describe("ruleText", () => {
  const base = { codeRead: true, substantial: false };
  it("names too few commits as the reason, not the email, for a small repository (review fix)", () => {
    expect(ruleText({ ...base, authorship: { studentCommits: 8, totalCommits: 8 } })).toBe(
      "You wrote all 8 commits here. A repository needs at least 10 commits of yours to support an anchor, so its skills are suggested as applied.",
    );
  });
  it("names a low share, and the email caveat, when others wrote most of it", () => {
    expect(ruleText({ ...base, authorship: { studentCommits: 12, totalCommits: 40 } })).toMatch(/^You wrote 30% of the commits here, so its skills are suggested as applied\. Commits made under another email/);
  });
  it("says when a repository can support an anchor, and when no code was read", () => {
    expect(ruleText({ codeRead: true, substantial: true, authorship: { studentCommits: 20, totalCommits: 20 } })).toBe("You wrote 100% of 20 commits here, so this repository can support an anchor.");
    expect(ruleText({ codeRead: false, substantial: false, authorship: { studentCommits: 0, totalCommits: 0 } })).toMatch(/No code or data files were found here/);
  });
});
