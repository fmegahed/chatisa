import { describe, expect, it } from "vitest";
import {
  careerFileSet, dedupePaths, guessRole, measure, rolePath, safeFileName, showcaseFileSet,
  showcaseRepoName, slugify,
} from "@/lib/portfolio/files";

describe("roles and names", () => {
  it("guesses roles from extensions", () => {
    expect(guessRole("train.csv")).toBe("data");
    expect(guessRole("model.R")).toBe("code");
    expect(guessRole("Final Project.ipynb")).toBe("notebook");
    expect(guessRole("report.docx")).toBe("report");
    expect(guessRole("deck.pptx")).toBe("slides");
    expect(guessRole("roc.png")).toBe("figure");
    expect(guessRole("notes")).toBe("other");
  });
  it("slugifies and names safely", () => {
    expect(slugify("Churn Model: Final!")).toBe("churn-model-final");
    expect(slugify("!!")).toBe("project");
    expect(safeFileName("../Final Project.ipynb")).toBe("Final-Project.ipynb");
    expect(rolePath("notebook", "Final Project.ipynb")).toBe("code/Final-Project.ipynb");
    expect(rolePath("figure", "roc.png")).toBe("figures/roc.png");
    expect(showcaseRepoName("ISA 401", "Churn Model")).toBe("isa-401-churn-model");
  });
});

describe("measure", () => {
  it("flags files over the per-file cap and totals", () => {
    const big = "x".repeat(400_001);
    const m = measure([
      { path: "a.txt", contents: "abc" },
      { path: "b.txt", contents: big },
    ]);
    expect(m.ok).toBe(false);
    expect(m.over).toEqual([{ path: "b.txt", bytes: 400_001 }]);
    expect(m.totalBytes).toBe(400_004);
    expect(m.count).toBe(2);
  });
});

describe("file sets", () => {
  it("career: fixed paths, opt-in resume, project folders, unpublished files skipped", () => {
    const files = careerFileSet({
      html: "<p/>",
      photoBase64: "cGhvdG8=",
      resumeBase64: null,
      projects: [
        {
          slug: "churn",
          files: [
            { name: "model.R", role: "code", publish: true, bytes: 3, text: "x<-1", base64: null },
            { name: "secret.csv", role: "data", publish: false, bytes: 3, text: "a,b", base64: null },
          ],
        },
      ],
    });
    const paths = files.map((f) => f.path);
    expect(paths).toEqual(["index.html", ".nojekyll", "README.md", "assets/photo.jpg", "projects/churn/model.R"]);
    expect(files.find((f) => f.path === "assets/photo.jpg")?.encoding).toBe("base64");
  });
  it("suffixes repo paths that would collide, keeping every ticked file", () => {
    // Two different uploads clean to the same repo path; a repeated path in
    // one tree silently drops a file, so the second gets a suffix.
    const showcase = showcaseFileSet({
      html: "<p/>", readme: "# r", gitignore: "",
      files: [
        { name: "Final Report.pdf", role: "report", publish: true, bytes: 2, text: null, base64: "YQ==" },
        { name: "Final-Report.pdf", role: "report", publish: true, bytes: 2, text: null, base64: "Yg==" },
        { name: "Final Report.pdf", role: "report", publish: true, bytes: 2, text: null, base64: "Yw==" },
        { name: "notes", role: "other", publish: true, bytes: 1, text: "x", base64: null },
        { name: "notes", role: "other", publish: true, bytes: 1, text: "y", base64: null },
      ],
    });
    expect(showcase.map((f) => f.path)).toEqual([
      "index.html", ".nojekyll", "README.md", ".gitignore",
      "report/Final-Report.pdf", "report/Final-Report-2.pdf", "report/Final-Report-3.pdf",
      "other/notes", "other/notes-2",
    ]);
    const career = careerFileSet({
      html: "<p/>", photoBase64: null, resumeBase64: null,
      projects: [{
        slug: "churn",
        files: [
          { name: "model.R", role: "code", publish: true, bytes: 1, text: "a", base64: null },
          { name: "model.R", role: "code", publish: true, bytes: 1, text: "b", base64: null },
        ],
      }],
    });
    expect(career.map((f) => f.path)).toEqual([
      "index.html", ".nojekyll", "README.md", "projects/churn/model.R", "projects/churn/model-2.R",
    ]);
  });
  it("dedupePaths reports the paths a push will really write", () => {
    expect(dedupePaths(["code/a.R", "code/a.R", "code/a.R"])).toEqual([
      "code/a.R", "code/a-2.R", "code/a-3.R",
    ]);
  });
  it("showcase: role folders, README, gitignore, figures kept as base64", () => {
    const files = showcaseFileSet({
      html: "<p/>",
      readme: "# Churn",
      gitignore: ".Rproj.user/\n",
      files: [
        { name: "roc.png", role: "figure", publish: true, bytes: 4, text: null, base64: "aW1n" },
        { name: "train.csv", role: "data", publish: false, bytes: 3, text: "a,b", base64: null },
        { name: "model.R", role: "code", publish: true, bytes: 4, text: "x<-1", base64: null },
      ],
    });
    expect(files.map((f) => f.path)).toEqual([
      "index.html", ".nojekyll", "README.md", ".gitignore", "figures/roc.png", "code/model.R",
    ]);
  });
});
