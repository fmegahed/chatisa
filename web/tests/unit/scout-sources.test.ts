import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import { normalizeJsearch } from "@/lib/scout/sources/jsearch";
import { normalizeUsajobs } from "@/lib/scout/sources/usajobs";
import { fingerprintOf, toStateCode } from "@/lib/scout/sources/types";
import {
  HARVEST_QUERIES,
  USAJOBS_KEYWORDS,
  isRelevantTitle,
} from "@/lib/scout/queries";

const fixture = (name: string) =>
  JSON.parse(
    readFileSync(
      path.join(process.cwd(), "tests", "fixtures", "scout", name),
      "utf8",
    ),
  );

describe("normalizeJsearch (real /search-v2 envelope: data.jobs)", () => {
  const postings = normalizeJsearch(fixture("jsearch-page.json"), "fulltime");

  it("keeps well-formed postings, drops short descriptions and part-time-only rows", () => {
    // js-004's description is under the 100-char floor; js-005 is PARTTIME.
    expect(postings.map((p) => p.externalId)).toEqual([
      "js-001",
      "js-002",
      "js-003",
    ]);
  });

  it("normalizes fields including full state names (the real payload shape)", () => {
    const first = postings[0];
    expect(first).toMatchObject({
      source: "jsearch",
      title: "Data Analyst",
      company: "Acme Insurance",
      locationCity: "Cincinnati",
      locationState: "OH",
      remote: false,
      category: "fulltime",
      postedAt: "2026-07-22",
    });
    // js-003 arrives with lowercase "ohio" and must still fingerprint-match.
    expect(postings[2].locationState).toBe("OH");
    expect(fingerprintOf(postings[2])).toBe(fingerprintOf(first));
  });

  it("still accepts the legacy data:[...] array shape", () => {
    const legacy = {
      data: [
        {
          job_id: "legacy-1",
          job_title: "Data Analyst",
          employer_name: "Legacy Co",
          job_state: "OH",
          job_apply_link: "https://example.com/x",
          job_description: "A".repeat(120),
        },
      ],
    };
    expect(normalizeJsearch(legacy, "fulltime")).toHaveLength(1);
  });

  it("tolerates garbage payloads", () => {
    expect(normalizeJsearch(null, "fulltime")).toEqual([]);
    expect(normalizeJsearch({ data: "nope" }, "fulltime")).toEqual([]);
    expect(normalizeJsearch({ data: { jobs: "nope" } }, "fulltime")).toEqual([]);
  });
});

describe("cleanTitle", () => {
  it("strips the stuffed company/location suffix from aggregator titles", async () => {
    const { cleanTitle } = await import("@/lib/scout/sources/jsearch");
    expect(
      cleanTitle(
        "Cyber Security Intern - Summer 2025 at Park Place Technologies Cleveland, OH",
        "Park Place Technologies",
      ),
    ).toBe("Cyber Security Intern - Summer 2025");
    expect(cleanTitle("Data Analyst", "Acme")).toBe("Data Analyst");
    // Never cuts to nothing, and never cuts unrelated "at" phrases.
    expect(cleanTitle("Working at scale analyst", "Nomatch Co")).toBe(
      "Working at scale analyst",
    );
  });
});

describe("toStateCode", () => {
  it("maps full names, codes, and casings; rejects junk", () => {
    expect(toStateCode("Ohio")).toBe("OH");
    expect(toStateCode("kentucky")).toBe("KY");
    expect(toStateCode("Pennsylvania")).toBe("PA");
    expect(toStateCode(" District of Columbia ")).toBe("DC");
    expect(toStateCode("oh")).toBe("OH");
    expect(toStateCode("Not A State")).toBeNull();
    expect(toStateCode(null)).toBeNull();
  });
});

describe("normalizeUsajobs", () => {
  const postings = normalizeUsajobs(fixture("usajobs-page.json"));

  it("keeps complete items and drops those without an apply URL or text", () => {
    expect(postings).toHaveLength(1);
    expect(postings[0]).toMatchObject({
      source: "usajobs",
      externalId: "usaj-800100",
      title: "Management Analyst",
      company: "Department of the Treasury",
      locationCity: "Washington",
      locationState: "DC",
      category: "federal",
      applyUrl: "https://www.usajobs.gov/job/800100/apply",
      postedAt: "2026-07-20",
    });
  });

  it("joins summary, duties, and qualifications into the description", () => {
    expect(postings[0].description).toContain("performance dashboards");
    expect(postings[0].description).toContain("visualization tools");
  });
});

describe("query plan", () => {
  it("stays well under the weekly request budget", () => {
    const total = HARVEST_QUERIES.length + USAJOBS_KEYWORDS.length;
    expect(total).toBeLessThanOrEqual(600);
    expect(total).toBeGreaterThanOrEqual(100);
  });

  it("covers all three tracks", () => {
    expect(HARVEST_QUERIES.some((q) => q.category === "internship")).toBe(true);
    expect(HARVEST_QUERIES.some((q) => q.query.includes("remote"))).toBe(true);
    expect(USAJOBS_KEYWORDS.length).toBeGreaterThan(0);
  });
});

describe("isRelevantTitle", () => {
  it("keeps ISA-career titles", () => {
    for (const t of [
      "Data Analyst",
      "Business Intelligence Analyst",
      "IT Auditor",
      "Information Security Analyst",
      "Technology Consultant",
    ]) {
      expect(isRelevantTitle(t), t).toBe(true);
    }
  });

  it("drops senior and off-target titles", () => {
    for (const t of [
      "Senior Data Engineer",
      "Principal Architect",
      "Registered Nurse",
      "Warehouse Associate",
      "VP of Analytics",
    ]) {
      expect(isRelevantTitle(t), t).toBe(false);
    }
  });
});
