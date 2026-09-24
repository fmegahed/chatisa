import { describe, expect, it } from "vitest";
import { buildPublishPlan } from "@/lib/portfolio/publish-plan";
import { emptyCareer } from "@/lib/portfolio/content";

const baseDraft = {
  siteId: "s", mode: "career" as const, step: "review" as const,
  resume: null, resumeLink: true, courses: ["ISA 401"],
  projects: [{ slug: "churn", title: "Churn", externalUrl: "", files: [{ name: "m.R", role: "code" as const, publish: true, bytes: 3, text: "x<-1", base64: null }] }],
  photo: { base64: "cGhvdG8=", bytes: 5 }, name: "Ada", links: [{ label: "GitHub", url: "https://github.com/ada" }],
  course: "", semester: "", team: [], files: [], prompts: { problem: "", hardest: "", next: "" },
  content: { kind: "career" as const, content: { ...emptyCareer(), siteTitle: "Ada", headline: "h", about: "a", projects: [{ slug: "churn", title: "Churn", blurb: "b", skills: [], externalUrl: null }] } },
  readme: null, skillIds: [], html: "",
};

describe("buildPublishPlan", () => {
  it("career: portfolio repo, login-based project links, resume only when opted in", () => {
    const plan = buildPublishPlan(baseDraft, "ada", { resumeBase64: "cmVzdW1l", existingRepoName: null });
    expect(plan.repoName).toBe("portfolio");
    expect(plan.html).toContain("https://github.com/ada/portfolio/tree/main/projects/churn");
    expect(plan.files.map((f) => f.path)).toEqual(["index.html", ".nojekyll", "README.md", "assets/photo.jpg", "resume.pdf", "projects/churn/m.R"]);
    const without = buildPublishPlan({ ...baseDraft, resumeLink: false }, "ada", { resumeBase64: "cmVzdW1l", existingRepoName: null });
    expect(without.files.some((f) => f.path === "resume.pdf")).toBe(false);
  });
  it("career: honours the repository name the student had to pick, links included", () => {
    // "portfolio" was taken on the account, so the publish goes to the name
    // the student chose and the Files links must point at it.
    const plan = buildPublishPlan(baseDraft, "ada", { resumeBase64: null, existingRepoName: "portfolio-2" });
    expect(plan.repoName).toBe("portfolio-2");
    expect(plan.html).toContain("https://github.com/ada/portfolio-2/tree/main/projects/churn");
    expect(plan.html).not.toContain("/ada/portfolio/tree");
  });
  it("showcase: course-title repo name, kept on republish, README from the draft", () => {
    const draft = {
      ...baseDraft, mode: "showcase" as const, course: "ISA 401", team: ["Ada"],
      files: [{ name: "roc.png", role: "figure" as const, publish: true, bytes: 4, text: null, base64: "aW1n" }],
      readme: "# Churn",
      content: { kind: "showcase" as const, content: { v: 1 as const, title: "Churn Model", tagline: "t", problem: "p", data: "d", approach: "a", findings: [{ heading: "x", body: "y", figure: "figures/roc.png" }], deliverables: [], skills: [], nextSteps: "" } },
    };
    const plan = buildPublishPlan(draft, "ada", { resumeBase64: null, existingRepoName: null });
    expect(plan.repoName).toBe("isa-401-churn-model");
    expect(plan.html).toContain('src="figures/roc.png"');
    expect(plan.html).toContain("https://github.com/ada/isa-401-churn-model");
    expect(plan.files.find((f) => f.path === "README.md")?.contents).toBe("# Churn");
    expect(buildPublishPlan(draft, "ada", { resumeBase64: null, existingRepoName: "old-name" }).repoName).toBe("old-name");
  });

  describe("showcase origin (v6.6.0)", () => {
    const showcase = {
      ...baseDraft, mode: "showcase" as const, course: "ISA 401", readme: null,
      content: { kind: "showcase" as const, content: { v: 1 as const, title: "Churn Model", tagline: "Who leaves, and why", problem: "p", data: "d", approach: "a", findings: [], deliverables: [], skills: [], nextSteps: "" } },
    };
    const readme = (plan: { files: { path: string; contents?: string }[] }) =>
      plan.files.find((f) => f.path === "README.md")?.contents ?? "";

    it("a draft saved before origins existed publishes exactly as a Miami course did", () => {
      const plan = buildPublishPlan(showcase, "ada", { resumeBase64: null, existingRepoName: null });
      expect(plan.repoName).toBe("isa-401-churn-model");
      expect(readme(plan)).toContain("Built for ISA 401.");
      expect(plan.html).toContain("ISA 401 - ");
    });

    it("a course at another school: title-only repo, typed course in header and README", () => {
      const plan = buildPublishPlan({ ...showcase, origin: "other", course: "STAT 4520, Ohio State" }, "ada", { resumeBase64: null, existingRepoName: null });
      expect(plan.repoName).toBe("churn-model");
      expect(plan.html).toContain("STAT 4520, Ohio State");
      expect(readme(plan)).toContain("Built for STAT 4520, Ohio State.");
    });

    it("self-study and personal projects ignore course text left from another origin", () => {
      const self = buildPublishPlan({ ...showcase, origin: "self" }, "ada", { resumeBase64: null, existingRepoName: null });
      expect(self.repoName).toBe("churn-model");
      expect(self.html).toContain("Self-study project");
      expect(self.html).not.toContain("ISA 401");
      expect(readme(self)).toContain("A self-study project.");
      expect(readme(self)).not.toContain("ISA 401");
      const personal = buildPublishPlan({ ...showcase, origin: "personal" }, "ada", { resumeBase64: null, existingRepoName: null });
      expect(personal.html).toContain("Personal project");
      expect(readme(personal)).toContain("A personal project.");
    });

    it("a republish keeps the stored repository name whatever the origin", () => {
      expect(buildPublishPlan({ ...showcase, origin: "self" }, "ada", { resumeBase64: null, existingRepoName: "isa-401-churn-model" }).repoName)
        .toBe("isa-401-churn-model");
    });
  });
});
