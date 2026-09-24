import { describe, expect, it } from "vitest";
import {
  careerContentSchema,
  careerGenerationSchema,
  migrateCareerV1,
  showcaseContentSchema,
} from "@/lib/portfolio/content";

describe("career content schema", () => {
  it("accepts a full v2 document", () => {
    const ok = careerContentSchema.safeParse({
      v: 2,
      siteTitle: "Ada Lovelace",
      headline: "Analytics student",
      about: "I like data.",
      skillGroups: [{ title: "Tools", skills: ["R", "SQL"] }],
      projects: [
        { slug: "churn-model", title: "Churn", blurb: "Built a model.", skills: ["R"], externalUrl: null },
      ],
      courses: [{ code: "ISA 401", why: "Machine learning." }],
      experience: [{ org: "Acme", role: "Intern", dates: "2025", bullets: ["Did things."] }],
      education: [{ school: "Miami University", degree: "BS Business Analytics", dates: "2027" }],
    });
    expect(ok.success).toBe(true);
  });

  it("rejects a slug with a path separator", () => {
    const bad = careerContentSchema.safeParse({
      v: 2, siteTitle: "x", headline: "x", about: "x", skillGroups: [],
      projects: [{ slug: "../etc", title: "x", blurb: "x", skills: [], externalUrl: null }],
      courses: [], experience: [], education: [],
    });
    expect(bad.success).toBe(false);
  });
});

describe("migrateCareerV1", () => {
  it("lifts the v6.3.0 shape into v2 with empty new sections", () => {
    const v1 = {
      siteTitle: "Ada", headline: "h", about: "a",
      skillGroups: [{ title: "T", skills: ["R"] }],
      projectCards: [
        { repoName: "retail-demand", title: "Retail", blurb: "b", skillLabels: ["R"], repoUrl: "https://github.com/a/retail-demand" },
      ],
      courseHighlights: [{ course: "ISA 401", why: "w" }],
    };
    const out = migrateCareerV1(v1);
    expect(out?.v).toBe(2);
    expect(out?.projects[0]).toEqual({
      slug: "retail-demand", title: "Retail", blurb: "b", skills: ["R"],
      externalUrl: "https://github.com/a/retail-demand",
    });
    expect(out?.experience).toEqual([]);
    expect(out?.courses[0].code).toBe("ISA 401");
  });

  it("returns null for garbage", () => {
    expect(migrateCareerV1({ nope: true })).toBeNull();
    expect(migrateCareerV1(null)).toBeNull();
  });
});

describe("showcase content schema", () => {
  it("accepts findings with a null figure", () => {
    const ok = showcaseContentSchema.safeParse({
      v: 1, title: "Churn", tagline: "t", problem: "p", data: "d", approach: "a",
      findings: [{ heading: "One", body: "b", figure: null }],
      deliverables: [{ label: "Report", path: "report/final.pdf" }],
      skills: ["R"], nextSteps: "n",
    });
    expect(ok.success).toBe(true);
  });
});

describe("outside courses in career content (v6.6.0)", () => {
  const doc = {
    v: 2, siteTitle: "A", headline: "h", about: "a", skillGroups: [], projects: [],
    courses: [], experience: [], education: [],
  };

  it("still parses stored content written before the field existed", () => {
    expect(careerContentSchema.safeParse(doc).success).toBe(true);
  });

  it("stores up to five outside courses and rejects a sixth", () => {
    const five = Array.from({ length: 5 }, (_, i) => ({ name: `Course ${i}`, why: "w" }));
    expect(careerContentSchema.safeParse({ ...doc, otherCourses: five }).success).toBe(true);
    expect(careerContentSchema.safeParse({ ...doc, otherCourses: [...five, { name: "x", why: "w" }] }).success).toBe(false);
  });

  it("requires the field in what the model must return", () => {
    expect(careerGenerationSchema.safeParse(doc).success).toBe(false);
    expect(careerGenerationSchema.safeParse({ ...doc, otherCourses: [] }).success).toBe(true);
  });
});
