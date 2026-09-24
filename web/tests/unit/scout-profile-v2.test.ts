import { beforeEach, describe, expect, it } from "vitest";
import { courseCodes, currentTerm, loadProfile, saveProfile, staleTakingNow } from "@/lib/scout/profile-store";

/**
 * Job Scout profile v2 (v6.7.0): programs, and courses with a status. A v1
 * profile (a plain list of course codes) migrates on read, so no student
 * loses what they entered or is asked to redo it.
 */
class MemoryStorage {
  private m = new Map<string, string>();
  getItem(k: string) { return this.m.get(k) ?? null; }
  setItem(k: string, v: string) { this.m.set(k, v); }
  removeItem(k: string) { this.m.delete(k); }
  clear() { this.m.clear(); }
}

beforeEach(() => {
  (globalThis as { localStorage?: unknown }).localStorage = new MemoryStorage();
});

describe("profile v2", () => {
  it("migrates a v1 profile on read: every course becomes Done, nothing is lost", () => {
    localStorage.setItem("js-profile-v1", JSON.stringify({
      v: 1, courses: ["ISA 125", "ISA 225", "ISA 401"], extras: [{ skillId: "sql", level: "applied", source: "resume" }],
      overrides: [{ skillId: "r", level: "strong" }],
    }));
    const p = loadProfile();
    expect(p).toEqual({
      v: 2, programs: [], removedPrereqs: [],
      courses: [{ code: "ISA 125", status: "done" }, { code: "ISA 225", status: "done" }, { code: "ISA 401", status: "done" }],
      extras: [{ skillId: "sql", level: "applied", source: "resume" }],
      overrides: [{ skillId: "r", level: "strong" }],
    });
  });

  it("saves v2 and reads it back", () => {
    saveProfile({
      v: 2, programs: ["business-analytics"], removedPrereqs: ["ISA 211"],
      courses: [{ code: "ISA 444", status: "now", term: "Fall 2026" }, { code: "ISA 345", status: "done", addedBecause: "ISA 401" }],
      extras: [], overrides: [],
    });
    expect(loadProfile()?.courses).toEqual([
      { code: "ISA 444", status: "now", term: "Fall 2026" },
      { code: "ISA 345", status: "done", addedBecause: "ISA 401" },
    ]);
  });

  it("courseCodes lists every course the student has, done or in progress", () => {
    expect(courseCodes({ courses: [{ code: "A", status: "done" }, { code: "B", status: "now" }] })).toEqual(["A", "B"]);
  });

  it("corrupt JSON degrades to no profile", () => {
    localStorage.setItem("js-profile-v1", "{not json");
    expect(loadProfile()).toBeNull();
  });
});

describe("terms", () => {
  it("names the term a date falls in", () => {
    expect(currentTerm(new Date("2026-09-24"))).toBe("Fall 2026");
    expect(currentTerm(new Date("2027-02-10"))).toBe("Spring 2027");
    expect(currentTerm(new Date("2027-06-15"))).toBe("Summer 2027");
    expect(currentTerm(new Date("2026-08-20"))).toBe("Fall 2026");
  });

  it("finds Taking-now courses from an earlier term, for the 'Did you finish it?' prompt", () => {
    const profile = { v: 2 as const, programs: [], removedPrereqs: [], extras: [], overrides: [],
      courses: [
        { code: "ISA 444", status: "now" as const, term: "Fall 2026" },
        { code: "ISA 491", status: "now" as const, term: "Spring 2027" },
        { code: "ISA 225", status: "done" as const },
      ] };
    expect(staleTakingNow(profile, new Date("2027-02-10")).map((c) => c.code)).toEqual(["ISA 444"]);
  });
});
