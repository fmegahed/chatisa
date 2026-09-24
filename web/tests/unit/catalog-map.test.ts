import { describe, expect, it } from "vitest";
import { fingerprint, mapCatalog, spendCap, type AskModel, type MapInput } from "@/scripts/catalog/map";
import type { CourseSkillLink } from "@/lib/scout/course-skills";

const course = (code: string, over: Record<string, unknown> = {}) => ({
  code, altCodes: [], title: `Title ${code}`, previousTitles: [], credits: 3, prereq: [], prereqUncertain: false, retired: false, ...over,
});

function input(over: Partial<MapInput> = {}): MapInput {
  return {
    courses: [course("FIN 301"), course("MKT 291")],
    descriptions: { "FIN 301": { description: "Corporate finance basics.", prereqText: "" }, "MKT 291": { description: "Marketing principles.", prereqText: "" } },
    syllabi: {},
    links: [],
    state: {},
    maxUsd: 10,
    ...over,
  };
}

/** Both models agree on corporate_finance (anchor) and disagree on the rest. */
const ask: AskModel = async (model, c) => {
  const common = { skillId: "corporate_finance", level: "anchor" as const, evidence: `${model} evidence for ${c.code}` };
  const extra = model === "a"
    ? [{ skillId: "financial_analysis", level: "applied" as const, evidence: "" }, { skillId: "made_up_skill", level: "exposure" as const, evidence: "" }]
    : [{ skillId: "financial_analysis", level: "exposure" as const, evidence: "" }, { skillId: "excel", level: "applied" as const, evidence: "" }];
  return { links: c.code === "MKT 291" ? [{ skillId: "marketing_strategy", level: model === "a" ? "anchor" : "applied", evidence: "x" }] : [common, ...extra], costUsd: 0.02 };
};

describe("mapCatalog", () => {
  it("writes the links both models agree on, with the first model's evidence", async () => {
    const out = await mapCatalog(input(), ask);
    const fin = out.links.filter((l) => l.course === "FIN 301");
    expect(fin[0]).toEqual({ course: "FIN 301", skillId: "corporate_finance", level: "anchor", evidence: "a evidence for FIN 301" });
  });

  it("resolves a depth disagreement to the lower level and drops single-model suggestions, without asking a person", async () => {
    const out = await mapCatalog(input(), ask);
    const fin = out.links.filter((l) => l.course === "FIN 301");
    // Both models link financial_analysis (applied vs exposure): the lower wins.
    expect(fin).toContainEqual({ course: "FIN 301", skillId: "financial_analysis", level: "exposure" });
    // Only one model suggested excel: not added, and not a decision for a person.
    expect(fin.some((l) => l.skillId === "excel")).toBe(false);
    expect(out.report.disputed).toEqual([]);
    expect(out.report.autoLowered).toContainEqual({ course: "FIN 301", skillId: "financial_analysis", a: "applied", b: "exposure" });
    expect(out.report.singleModelDropped).toContainEqual({ course: "FIN 301", skillId: "excel", level: "applied" });
    expect(JSON.stringify(out)).not.toContain("made_up_skill");
  });

  it("leaves a course's links alone when the agreed set has no anchor", async () => {
    const existing: CourseSkillLink[] = [{ course: "MKT 291", skillId: "consumer_behavior", level: "anchor", evidence: "approved earlier by the professor" }];
    const out = await mapCatalog(input({ links: existing }), ask);
    // marketing_strategy resolves to applied (lower of anchor/applied) and is
    // added beside the approved anchor; the approved link is untouched.
    expect(out.links.filter((l) => l.course === "MKT 291")).toContainEqual(existing[0]);
  });

  it("flags a course for a person when nothing agreed gives it an anchor", async () => {
    const out = await mapCatalog(input(), ask);
    // MKT 291 has no approved links and its only skill resolves to applied.
    expect(out.links.some((l) => l.course === "MKT 291")).toBe(false);
    expect(out.report.noAnchor).toEqual(["MKT 291"]);
  });

  it("skips a course whose fingerprint has not changed, at no cost", async () => {
    const i = input();
    const state = Object.fromEntries(i.courses.map((c) => [c.code, { fingerprint: fingerprint(c, i.descriptions[c.code], undefined), status: "agreed" as const }]));
    let calls = 0;
    const out = await mapCatalog({ ...i, state }, async (m, c) => { calls++; return ask(m, c); });
    expect(calls).toBe(0);
    expect(out.report.mapped).toEqual([]);
    expect(out.report.costUsd).toBe(0);
  });

  it("never maps freeform or retired courses", async () => {
    const out = await mapCatalog(input({ courses: [course("ISA 340", { special: "freeform" }), course("ISA 245", { retired: true })] }), ask);
    expect(out.report.mapped).toEqual([]);
  });

  it("keeps going when one course's model call fails, retries it next run, and still counts the spend (review fix)", async () => {
    const flaky: AskModel = async (model, c) => {
      if (c.code === "FIN 301" && model === "b") throw new Error("could not parse the response");
      return ask(model, c);
    };
    const out = await mapCatalog(input(), flaky);
    expect(out.report.mapped).toEqual(["MKT 291"]);
    expect(out.report.failed).toEqual([{ course: "FIN 301", error: "could not parse the response" }]);
    expect(out.state["FIN 301"]).toBeUndefined();
    // MKT 291's two calls, plus the FIN 301 call that did come back.
    expect(out.report.costUsd).toBeCloseTo(0.06, 5);
  });

  it("reports every link it wrote, so the reviewer sees new anchors and their evidence", async () => {
    const out = await mapCatalog(input(), ask);
    expect(out.report.written).toContainEqual({ course: "FIN 301", skillId: "corporate_finance", level: "anchor", evidence: "a evidence for FIN 301" });
    expect(out.report.written.every((l) => out.links.some((x) => x.course === l.course && x.skillId === l.skillId))).toBe(true);
  });

  describe("course level caps (professor's rule, 2026-09-24)", () => {
    const both = (skillId: string): AskModel => async () => ({
      links: [{ skillId, level: "anchor", evidence: "prepared full financial statements" }], costUsd: 0.01,
    });
    const low = (code: string) => input({
      courses: [course(code)],
      descriptions: { [code]: { description: "Intro.", prereqText: "" } },
    });

    it("writes a 200-level course's agreed anchor as applied, and needs no anchor from it", async () => {
      const out = await mapCatalog(low("ACC 221"), both("financial_accounting"));
      expect(out.links).toEqual([{ course: "ACC 221", skillId: "financial_accounting", level: "applied", evidence: "prepared full financial statements" }]);
      expect(out.report.noAnchor).toEqual([]);
      expect(out.state["ACC 221"].status).toBe("agreed");
    });

    it("writes a 100-level course's links as exposure", async () => {
      const out = await mapCatalog(low("BUS 101"), both("business_acumen"));
      expect(out.links.map((l) => l.level)).toEqual(["exposure"]);
    });

    it("keeps the professor's exception: Excel in CSE 148 stays an anchor", async () => {
      const out = await mapCatalog(low("CSE 148"), both("excel"));
      expect(out.links.map((l) => l.level)).toEqual(["anchor"]);
    });

    it("raises no dispute when the cap makes the models and the approved level agree", async () => {
      const approved: CourseSkillLink[] = [{ course: "ISA 125", skillId: "statistical_analysis", level: "exposure" }];
      const out = await mapCatalog({ ...low("ISA 125"), links: approved }, both("statistical_analysis"));
      expect(out.report.disputed).toEqual([]);
    });
  });

  it("stops at the spend cap, reports what it did not reach, and leaves that state untouched", async () => {
    const out = await mapCatalog(input({ maxUsd: 0.03 }), ask);
    expect(out.report.mapped).toEqual(["FIN 301"]);
    expect(out.report.skippedByCap).toEqual(["MKT 291"]);
    expect(out.state["MKT 291"]).toBeUndefined();
    expect(out.report.costUsd).toBeCloseTo(0.04, 5);
  });

  it("never removes or re-levels a link a person approved; adds agreed new skills; reports level changes", async () => {
    const approved: CourseSkillLink[] = [
      { course: "FIN 301", skillId: "corporate_finance", level: "applied", evidence: "approved earlier by the professor" },
      { course: "FIN 301", skillId: "tax", level: "exposure" },
    ];
    // Both models say corporate_finance is an anchor; neither proposes tax.
    const out = await mapCatalog(input({ links: approved }), ask);
    const fin = out.links.filter((l) => l.course === "FIN 301");
    expect(fin).toContainEqual(approved[0]);
    expect(fin).toContainEqual(approved[1]);
    expect(out.report.disputed).toContainEqual({ course: "FIN 301", skillId: "corporate_finance", a: "anchor", b: "anchor", approved: "applied" });
    expect(out.report.keptUnproposed).toContainEqual({ course: "FIN 301", skillId: "tax" });
  });

  it("adds an agreed skill that is new to a course with approved links", async () => {
    const approved: CourseSkillLink[] = [{ course: "FIN 301", skillId: "financial_accounting", level: "anchor", evidence: "read and prepared financial statements" }];
    const out = await mapCatalog(input({ links: approved }), ask);
    const fin = out.links.filter((l) => l.course === "FIN 301").map((l) => l.skillId);
    expect(fin).toEqual(expect.arrayContaining(["financial_accounting", "corporate_finance"]));
  });

  it("maps at most maxCourses per run, deferring the rest to the next run", async () => {
    const out = await mapCatalog(input({ maxCourses: 1 }), ask);
    expect(out.report.mapped).toEqual(["FIN 301"]);
    expect(out.report.deferred).toEqual(["MKT 291"]);
    expect(out.state["MKT 291"]).toBeUndefined();
  });

  it("maps courses in parallel yet reports them in a stable order", async () => {
    const slow: AskModel = async (m, c) => {
      await new Promise((r) => setTimeout(r, c.code === "FIN 301" ? 30 : 1));
      return ask(m, c);
    };
    const out = await mapCatalog(input({ concurrency: 2 }), slow);
    expect(out.report.mapped).toEqual(["FIN 301", "MKT 291"]);
  });

  it("keeps the whole file sorted by course so untouched courses never move", async () => {
    const existing: CourseSkillLink[] = [
      { course: "ACC 221", skillId: "financial_accounting", level: "anchor", evidence: "prepared financial statements" },
      { course: "ZZZ 100", skillId: "sales", level: "anchor", evidence: "sold things for a semester" },
    ];
    const out = await mapCatalog(input({ links: existing }), ask);
    expect([...new Set(out.links.map((l) => l.course))]).toEqual(["ACC 221", "FIN 301", "ZZZ 100"]);
  });
});

describe("spendCap", () => {
  it("defaults to $10 and reads a plain number", () => {
    expect(spendCap(undefined)).toBe(10);
    expect(spendCap("")).toBe(10);
    expect(spendCap(" 5.5 ")).toBe(5.5);
  });
  it("refuses anything else rather than run with no cap (review fix)", () => {
    for (const bad of ["$5", "5 dollars", "5,00", "0", "-1", "abc"]) expect(() => spendCap(bad)).toThrow(/CATALOG_MAX_RUN_USD/);
  });
});
