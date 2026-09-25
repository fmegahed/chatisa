import { describe, expect, it } from "vitest";
import { GITHUB_CONTENTS_MAX_BYTES, githubRepoUrl, importCandidates, importedName } from "@/lib/portfolio/github-import";

const tree = [
  { path: "README.md", size: 900 },
  { path: "src/model.py", size: 9_000 },
  { path: "src/utils.py", size: 2_000 },
  { path: "tests/utils.py", size: 1_000 },
  { path: "notebooks/eda.ipynb", size: 400_000 },
  { path: "figures/roc.png", size: 50_000 },
  { path: "report/final.pdf", size: 2_000_000 },
  { path: "data/raw.csv", size: 30 * 1024 * 1024 },
  { path: "data/huge.parquet", size: 150_000_000 },
  { path: "node_modules/x/index.js", size: 10 },
  { path: ".git/config", size: 10 },
  { path: ".gitignore", size: 10 },
];

describe("importCandidates", () => {
  it("hides vendored and hidden paths, and pre-ticks the README and code within the room", () => {
    const c = importCandidates(tree, 4);
    expect(c.map((x) => x.path)).not.toContain("node_modules/x/index.js");
    expect(c.map((x) => x.path)).not.toContain(".git/config");
    expect(c.filter((x) => x.preselected).map((x) => x.path)).toEqual(["README.md", "src/model.py", "src/utils.py", "notebooks/eda.ipynb"]);
  });

  it("never pre-ticks data, and never more than the room left", () => {
    const c = importCandidates(tree, 2);
    expect(c.filter((x) => x.preselected)).toHaveLength(2);
    expect(c.find((x) => x.path === "data/raw.csv")?.preselected).toBe(false);
    expect(importCandidates(tree, 0).some((x) => x.preselected)).toBe(false);
  });

  it("lists files GitHub cannot serve as not selectable, with a reason", () => {
    const huge = importCandidates(tree, 10).find((x) => x.path === "data/huge.parquet");
    expect(huge).toMatchObject({ selectable: false, preselected: false });
    expect(huge?.note).toMatch(/over 100 MB/);
    expect(GITHUB_CONTENTS_MAX_BYTES).toBe(100_000_000);
  });

  it("refuses files over 25 MB, the per-file publishing limit, and says why (professor, 2026-09-24)", () => {
    const raw = importCandidates(tree, 10).find((x) => x.path === "data/raw.csv");
    expect(raw).toMatchObject({ selectable: false, preselected: false });
    expect(raw?.note).toBe("30.0 MB: over the 25 MB limit for one file on a published page, so it cannot be imported");
  });

  it("guesses roles from names like an upload does", () => {
    const c = importCandidates(tree, 10);
    expect(c.find((x) => x.path === "figures/roc.png")?.role).toBe("figure");
    expect(c.find((x) => x.path === "notebooks/eda.ipynb")?.role).toBe("notebook");
  });
});

describe("importedName", () => {
  it("uses the file name, and the folder too when two chosen files share a name", () => {
    const chosen = ["src/utils.py", "tests/utils.py", "src/model.py"];
    expect(importedName("src/model.py", chosen)).toBe("model.py");
    expect(importedName("src/utils.py", chosen)).toBe("src_utils.py");
    expect(importedName("tests/utils.py", chosen)).toBe("tests_utils.py");
  });

  it("never takes a name a file already in the project has (review fix)", () => {
    expect(importedName("src/model.py", ["src/model.py"], ["model.py"])).toBe("src_model.py");
    expect(importedName("src/model.py", ["src/model.py"], ["model.py", "src_model.py"])).toBe("src_model (2).py");
    expect(importedName("README.md", ["README.md"], ["README.md"])).toBe("README (2).md");
  });
});

describe("githubRepoUrl", () => {
  it("accepts a GitHub repository address and normalizes it", () => {
    expect(githubRepoUrl("https://github.com/ada/churn-model")).toBe("https://github.com/ada/churn-model");
    expect(githubRepoUrl("https://github.com/ada/churn.model/")).toBe("https://github.com/ada/churn.model");
  });
  it("rejects anything else", () => {
    for (const bad of ["http://github.com/ada/x", "https://evil.com/ada/x", "https://github.com/ada", "javascript:alert(1)", "https://github.com/ada/x\"><script>", "", null, undefined]) {
      expect(githubRepoUrl(bad)).toBeNull();
    }
  });
});
