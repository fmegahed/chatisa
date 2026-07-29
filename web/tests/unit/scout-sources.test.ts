import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import { normalizeActiveJobs } from "@/lib/scout/sources/activejobs";
import { normalizeUsajobs } from "@/lib/scout/sources/usajobs";
import { fingerprintOf, toStateCode } from "@/lib/scout/sources/types";
import {
  ACTIVEJOBS_LOCATION,
  ACTIVEJOBS_QUERIES,
  ACTIVEJOBS_RUN_JOB_BUDGET,
  TARGET_STATE_CODES,
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

describe("normalizeActiveJobs (real /active-ats payload, captured 2026-07-29)", () => {
  const postings = normalizeActiveJobs(
    fixture("activejobs-page.json"),
    "fulltime",
    TARGET_STATE_CODES,
  );

  it("keeps entry-band postings and drops the clearly-senior one", () => {
    // Parsons (2283935218) carries ai_experience_level "5-10" and is out;
    // the two "2-5" rows stay for the tagging pass to judge.
    expect(postings.map((p) => p.externalId)).toEqual([
      "2283933056",
      "2282594299",
    ]);
  });

  it("maps fields, with apply URLs on the employer's own ATS domain", () => {
    expect(postings[0]).toMatchObject({
      source: "activejobs",
      title: "Sr Clinical Data Analyst",
      company: "University Hospitals",
      locationCity: "Cleveland",
      locationState: "OH",
      remote: false,
      category: "fulltime",
      postedAt: "2026-07-29",
    });
    for (const p of postings) {
      expect(p.applyUrl).not.toMatch(/linkedin|indeed|ziprecruiter/i);
    }
  });

  it("pins a multi-state posting to the first TARGET state, not the first listed", () => {
    // Metronet lists Nevada first; Indiana is the first state the board
    // serves, and its cities array is null so the city stays empty.
    const metronet = postings.find((p) => p.externalId === "2282594299")!;
    expect(metronet.locationState).toBe("IN");
    expect(metronet.locationCity).toBeNull();
    // Without a preference the first listed location wins.
    const unpinned = normalizeActiveJobs(
      fixture("activejobs-page.json"),
      "fulltime",
    ).find((p) => p.externalId === "2282594299")!;
    expect(unpinned.locationState).toBe("NV");
  });

  it("drops source duplicates, part-time-only rows, and short descriptions; interns recategorize", () => {
    const base = {
      id: 1,
      title: "Data Analyst",
      organization: "Acme",
      url: "https://acme.wd1.myworkdayjobs.com/job/1",
      description_text: "B".repeat(150),
      regions_derived: ["Ohio"],
      cities_derived: ["Cincinnati"],
    };
    const rows = [
      { ...base, id: 1 },
      { ...base, id: 2, ats_duplicate: true },
      { ...base, id: 3, employment_type: ["Part-time"] },
      { ...base, id: 4, description_text: "too short" },
      { ...base, id: 5, title: "Data Analytics Intern - Summer 2027" },
      { ...base, id: 6, ai_work_arrangement: "Remote Solely" },
    ];
    const out = normalizeActiveJobs(rows, "fulltime");
    expect(out.map((p) => p.externalId)).toEqual(["1", "5", "6"]);
    expect(out[1].category).toBe("internship");
    expect(out[2].remote).toBe(true);
  });

  it("tolerates garbage payloads", () => {
    expect(normalizeActiveJobs(null, "fulltime")).toEqual([]);
    expect(normalizeActiveJobs({ jobs: [] }, "fulltime")).toEqual([]);
    expect(normalizeActiveJobs([{ nonsense: true }], "fulltime")).toEqual([]);
  });

  it("fingerprints match across casing", () => {
    const a = { company: "Acme", title: "Data Analyst", locationState: "OH" };
    const b = { company: "ACME", title: "data analyst", locationState: "OH" };
    expect(fingerprintOf(a)).toBe(fingerprintOf(b));
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

describe("query plan (Active Jobs DB meters returned jobs, not requests)", () => {
  it("holds one run's job spend to a quarter of the monthly pool", () => {
    const jobs = ACTIVEJOBS_QUERIES.reduce((n, q) => n + q.limit, 0);
    expect(jobs).toBeLessThanOrEqual(ACTIVEJOBS_RUN_JOB_BUDGET);
    // Requests are the cheap unit now; the whole run is a handful.
    expect(ACTIVEJOBS_QUERIES.length + USAJOBS_KEYWORDS.length).toBeLessThan(20);
  });

  it("covers internships, security, and every board state", () => {
    expect(ACTIVEJOBS_QUERIES.some((q) => q.category === "internship")).toBe(true);
    expect(ACTIVEJOBS_QUERIES.some((q) => /security/i.test(q.title))).toBe(true);
    expect(USAJOBS_KEYWORDS.length).toBeGreaterThan(0);
    // The location expression names exactly the states the pin-picker uses.
    const quoted = [...ACTIVEJOBS_LOCATION.matchAll(/"([^"]+)"/g)].map((m) =>
      toStateCode(m[1]),
    );
    expect(new Set(quoted)).toEqual(TARGET_STATE_CODES);
  });

  it("filters every cluster to fresh grads or internships SERVER-SIDE", () => {
    // Quota strategy (user, 2026-07-29): a senior posting must never be
    // paid for. Full-time clusters carry the entry experience band; intern
    // clusters carry the INTERN employment type.
    for (const q of ACTIVEJOBS_QUERIES) {
      if (q.category === "fulltime") {
        expect(q.experienceLevels, q.title).toBe("0-2,2-5");
      } else {
        expect(q.employmentTypes, q.title).toBe("INTERN");
      }
    }
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

  it("drops senior, managerial, and off-target titles", () => {
    for (const t of [
      "Senior Data Engineer",
      "Principal Architect",
      "Registered Nurse",
      "Warehouse Associate",
      "VP of Analytics",
      // People-management roles are never fresh-grad (2026-07-29 smoke
      // surfaced "Business Intelligence Manager" twice).
      "Business Intelligence Manager",
      "Data Team Lead",
      "Analytics Supervisor",
    ]) {
      expect(isRelevantTitle(t), t).toBe(false);
    }
    // But leadership PROGRAMS for new grads stay.
    expect(isRelevantTitle("Technology Leadership Development Program Analyst")).toBe(
      true,
    );
  });
});
