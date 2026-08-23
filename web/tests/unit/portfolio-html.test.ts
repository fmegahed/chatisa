import { describe, expect, it } from "vitest";
import { renderCareer, renderShowcase } from "@/lib/portfolio/html";
import { emptyCareer } from "@/lib/portfolio/content";

const career = {
  ...emptyCareer(),
  siteTitle: "Ada <script>alert(1)</script>",
  headline: "Analytics",
  about: "Hello",
  projects: [
    { slug: "churn", title: "Churn", blurb: "b", skills: ["R"], externalUrl: "javascript:alert(1)" },
  ],
  experience: [{ org: "Acme", role: "Intern", dates: "2025", bullets: ["Did x"] }],
  education: [{ school: "Miami", degree: "BS", dates: "2027" }],
};

describe("renderCareer", () => {
  it("escapes text and drops unsafe hrefs", () => {
    const html = renderCareer(career, { name: "Ada", links: [], hasPhoto: false, resumeLink: false, login: "ada", folders: ["churn"], repoName: "portfolio" });
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
    expect(html).not.toContain("javascript:");
    expect(html).toContain("https://github.com/ada/portfolio/tree/main/projects/churn");
  });
  it("links Files only for projects that actually have a pushed folder", () => {
    // The student added this one in the editor, so nothing was ever pushed
    // under projects/typed-in/ and a Files link would be a 404.
    const typed = {
      ...career,
      projects: [
        ...career.projects,
        { slug: "typed-in", title: "Typed in", blurb: "b", skills: [], externalUrl: null },
      ],
    };
    const html = renderCareer(typed, { name: "Ada", links: [], hasPhoto: false, resumeLink: false, login: "ada", folders: ["churn"], repoName: "portfolio" });
    expect(html).toContain("https://github.com/ada/portfolio/tree/main/projects/churn");
    expect(html).not.toContain("projects/typed-in");
  });
  it("adds the photo and resume link only when asked", () => {
    const without = renderCareer(career, { name: "Ada", links: [], hasPhoto: false, resumeLink: false, login: null, folders: [], repoName: "portfolio" });
    expect(without).not.toContain("assets/photo.jpg");
    expect(without).not.toContain("resume.pdf");
    const withBoth = renderCareer(career, { name: "Ada", links: [], hasPhoto: true, resumeLink: true, login: null, folders: [], repoName: "portfolio" });
    expect(withBoth).toContain('src="assets/photo.jpg"');
    expect(withBoth).toContain('href="resume.pdf"');
  });
  it("renders experience and education sections", () => {
    const html = renderCareer(career, { name: "Ada", links: [], hasPhoto: false, resumeLink: false, login: null, folders: [], repoName: "portfolio" });
    expect(html).toContain("Experience");
    expect(html).toContain("Acme");
    expect(html).toContain("Education");
  });

  it("renames a repository the student had to pick, in the Files links", () => {
    // The account already owned "portfolio", so the publish went elsewhere
    // and every Files link has to follow it.
    const html = renderCareer(career, { name: "Ada", links: [], hasPhoto: false, resumeLink: false, login: "ada", folders: ["churn"], repoName: "portfolio-2" });
    expect(html).toContain("https://github.com/ada/portfolio-2/tree/main/projects/churn");
    expect(html).not.toContain("/ada/portfolio/tree");
  });

  it("drops a section whose only paragraph is empty", () => {
    const blank = { ...career, about: "   " };
    const html = renderCareer(blank, { name: "Ada", links: [], hasPhoto: false, resumeLink: false, login: null, folders: [], repoName: "portfolio" });
    expect(html).not.toContain("About");
    expect(html).not.toContain("<p></p>");
  });
  it("has no script tags or external requests", () => {
    const html = renderCareer(career, { name: "Ada", links: [], hasPhoto: true, resumeLink: false, login: null, folders: [], repoName: "portfolio" });
    expect(html).not.toMatch(/<script/i);
    expect(html).not.toMatch(/https?:\/\/[^"']*\.(css|js|woff)/i);
  });
});

describe("renderShowcase", () => {
  const content = {
    v: 1 as const, title: "Churn", tagline: "t", problem: "p", data: "d", approach: "a",
    findings: [
      { heading: "Lift", body: "b", figure: "figures/roc.png" },
      { heading: "Fake", body: "b", figure: "figures/../../etc" },
    ],
    deliverables: [{ label: "Report", path: "report/final.pdf" }],
    skills: ["R"], nextSteps: "n",
  };
  it("only renders figures from the allow-list and links deliverables relatively", () => {
    const html = renderShowcase(content, { course: "ISA 401", semester: "Spring 2026", team: ["Ada", "Grace"], repoUrl: null, figures: ["figures/roc.png"], deliverablePaths: ["report/final.pdf", "figures/roc.png"] });
    expect(html).toContain('src="figures/roc.png"');
    expect(html).not.toContain("etc");
    expect(html).toContain('href="report/final.pdf"');
    expect(html).toContain("Grace");
  });


  it("links only deliverables whose path will actually be published", () => {
    // The student unticked the report after generating, so the model's
    // deliverable would have been a 404 on the published page.
    const html = renderShowcase(content, { course: "ISA 401", semester: "", team: [], repoUrl: null, figures: [], deliverablePaths: ["code/model.R"] });
    expect(html).not.toContain('href="report/final.pdf"');
    expect(html).not.toContain("Deliverables");
  });
  it("rejects url schemes and traversal in deliverable paths, keeping only safe relative ones", () => {
    const unsafeDeliverables = {
      ...content,
      deliverables: [
        { label: "Scheme", path: "javascript:alert(1)" },
        { label: "Absolute", path: "/etc/passwd" },
        { label: "Traversal", path: "../../etc/passwd" },
        { label: "Embedded traversal", path: "report/../x.pdf" },
        { label: "Report", path: "report/final.pdf" },
      ],
    };
    const html = renderShowcase(unsafeDeliverables, { course: "ISA 401", semester: "Spring 2026", team: [], repoUrl: null, figures: [], deliverablePaths: ["report/final.pdf"] });
    expect(html).toContain('href="report/final.pdf"');
    expect(html).not.toContain("javascript:");
    expect(html).not.toContain("etc");
    const hrefCount = (html.match(/<a href="/g) ?? []).length;
    expect(hrefCount).toBe(1);
  });

  it("rejects a traversal figure path even when it is in the allow-list", () => {
    const unsafeFigure = {
      ...content,
      findings: [{ heading: "Bad figure", body: "b", figure: "figures/../x.png" }],
    };
    const html = renderShowcase(unsafeFigure, { course: "ISA 401", semester: "Spring 2026", team: [], repoUrl: null, figures: ["figures/../x.png"], deliverablePaths: [] });
    expect(html).not.toContain("<img");
  });

  it("prints the catalog title next to a course code", () => {
    const html = renderCareer(
      { ...career, courses: [{ code: "ISA 444", why: "Forecasting." }, { code: "XYZ 100", why: "Unknown." }] },
      { name: "Ada", links: [], hasPhoto: false, resumeLink: false, login: "ada", folders: [], repoName: "portfolio" },
    );
    expect(html).toContain("<strong>ISA 444 - Business Forecasting</strong>: Forecasting.");
    expect(html).toContain("<strong>XYZ 100</strong>: Unknown.");
  });
});
