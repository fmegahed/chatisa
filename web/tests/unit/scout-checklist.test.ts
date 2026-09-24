import { describe, expect, it } from "vitest";
import {
  addedFor,
  finishCourses,
  keepTaking,
  markRequiredDone,
  removeAddedPrereq,
  setCourseStatus,
  statusOf,
  type ChecklistDeps,
  type ChecklistState,
} from "@/lib/scout/checklist";
import type { ProgramGroup } from "@/lib/scout/programs";

/**
 * The course checklist's state changes (v6.7.0). A student sets Done,
 * Taking now or Not yet on a course; prerequisites follow automatically and
 * are re-derived after every change, so they always match the student's own
 * courses and programs. A small synthetic catalog keeps these stable as the
 * bulletin changes.
 */
const course = (code: string, prereq: string[][] = []) => ({ code, prereq, prereqUncertain: false });
const catalog = new Map(
  [
    course("ISA 401", [["ISA 345"]]),
    course("ISA 345", [["ISA 235"]]),
    course("ISA 235"),
    course("ISA 125"),
    course("FIN 301"),
    course("FIN 311"),
    course("ACC 221"),
    course("ISA 495"),
  ].map((c) => [c.code, c]),
);
const group = (items: string[][], instruction: string | null = null, subtitle: string | null = null): ProgramGroup => ({
  title: "Required courses", subtitle, instruction, items: items.map((codes) => ({ codes })), notes: [],
});
const CORE: ProgramGroup[] = [
  group([["ISA 125"], ["ISA 235"], ["ACC 221"], ["FIN 301", "FIN 311"]]),
  group([["ISA 495"]], null, "FSB Senior Capstone Experience"),
];
const deps: ChecklistDeps = {
  getCourse: (code) => catalog.get(code),
  programCodes: (key) => (key === "business-core" ? ["ISA 125", "ISA 235", "ACC 221", "FIN 301", "FIN 311", "ISA 495"] : key === "finance" ? ["FIN 311"] : []),
  programGroups: (key) => (key === "business-core" ? CORE : []),
};
const empty: ChecklistState = { programs: [], courses: [], removedPrereqs: [] };
const FALL = new Date("2026-09-24");

describe("setCourseStatus", () => {
  it("records Taking now with the current term and brings in the prerequisites as Done", () => {
    const { state, added } = setCourseStatus(empty, "ISA 401", "now", deps, FALL);
    expect(state.courses).toEqual([
      { code: "ISA 401", status: "now", term: "Fall 2026" },
      { code: "ISA 345", status: "done", addedBecause: "ISA 401" },
      { code: "ISA 235", status: "done", addedBecause: "ISA 401" },
    ]);
    expect(added).toEqual(["ISA 345", "ISA 235"]);
    expect(statusOf(state, "ISA 235")).toBe("done");
    expect(statusOf(state, "ISA 125")).toBe("none");
  });

  it("Not yet on a course takes away the prerequisites it brought in", () => {
    const one = setCourseStatus(empty, "ISA 401", "now", deps, FALL).state;
    const { state } = setCourseStatus(one, "ISA 401", "none", deps, FALL);
    expect(state.courses).toEqual([]);
  });

  it("a prerequisite the student sets themselves becomes theirs and stays", () => {
    const one = setCourseStatus(empty, "ISA 401", "now", deps, FALL).state;
    const two = setCourseStatus(one, "ISA 345", "done", deps, FALL).state;
    const three = setCourseStatus(two, "ISA 401", "none", deps, FALL).state;
    expect(three.courses.map((c) => c.code)).toEqual(["ISA 345", "ISA 235"]);
    expect(three.courses[0]).toEqual({ code: "ISA 345", status: "done" });
    expect(addedFor(three, "ISA 345")).toEqual(["ISA 235"]);
  });

  it("Not yet on an added prerequisite removes it for good, with what came through it", () => {
    const one = setCourseStatus(empty, "ISA 401", "now", deps, FALL).state;
    const { state } = setCourseStatus(one, "ISA 345", "none", deps, FALL);
    expect(state.courses.map((c) => c.code)).toEqual(["ISA 401"]);
    expect(state.removedPrereqs).toEqual(["ISA 345"]);
  });

  it("setting a removed prerequisite yourself clears it from the removed list", () => {
    const one = removeAddedPrereq(setCourseStatus(empty, "ISA 401", "now", deps, FALL).state, "ISA 345", deps);
    const { state } = setCourseStatus(one, "ISA 345", "done", deps, FALL);
    expect(state.removedPrereqs).toEqual([]);
    expect(state.courses.map((c) => c.code)).toEqual(["ISA 401", "ISA 345", "ISA 235"]);
  });

  it("Not yet on a course the student set sticks, even when another course needs it (review fix)", () => {
    const one = setCourseStatus(empty, "ISA 345", "done", deps, FALL).state;
    const two = setCourseStatus(one, "ISA 401", "done", deps, FALL).state;
    const { state } = setCourseStatus(two, "ISA 345", "none", deps, FALL);
    expect(statusOf(state, "ISA 345")).toBe("none");
    expect(state.removedPrereqs).toEqual(["ISA 345"]);
  });

  it("changing Done to Taking now keeps one entry and stamps the term", () => {
    const one = setCourseStatus(empty, "ISA 125", "done", deps, FALL).state;
    const { state } = setCourseStatus(one, "ISA 125", "now", deps, FALL);
    expect(state.courses).toEqual([{ code: "ISA 125", status: "now", term: "Fall 2026" }]);
  });
});

describe("removeAddedPrereq", () => {
  it("removes the prerequisite and never adds it back", () => {
    const one = setCourseStatus(empty, "ISA 401", "now", deps, FALL).state;
    const state = removeAddedPrereq(one, "ISA 345", deps);
    expect(state.courses.map((c) => c.code)).toEqual(["ISA 401"]);
    const again = setCourseStatus(state, "ISA 401", "done", deps, FALL).state;
    expect(again.courses.map((c) => c.code)).toEqual(["ISA 401"]);
  });
});

describe("markRequiredDone", () => {
  it("marks every all-required core row Done, taking the student's own path on an or row, and leaves the capstone choice alone", () => {
    const { state, added } = markRequiredDone({ ...empty, programs: ["finance"] }, "business-core", deps);
    expect(state.courses.map((c) => c.code)).toEqual(["ISA 125", "ISA 235", "ACC 221", "FIN 311"]);
    expect(added).toEqual(["ISA 125", "ISA 235", "ACC 221", "FIN 311"]);
  });

  it("never changes a row the student already set", () => {
    const one = setCourseStatus(empty, "FIN 301", "now", deps, FALL).state;
    const { state } = markRequiredDone(one, "business-core", deps);
    expect(state.courses.find((c) => c.code === "FIN 301")?.status).toBe("now");
    expect(state.courses.some((c) => c.code === "FIN 311")).toBe(false);
  });
});

describe("next term", () => {
  const spring = new Date("2027-02-10");
  it("Finished turns Taking now into Done", () => {
    const one = setCourseStatus(empty, "ISA 125", "now", deps, FALL).state;
    expect(finishCourses(one, ["ISA 125"], deps).courses).toEqual([{ code: "ISA 125", status: "done" }]);
  });
  it("Still taking moves the course to the current term", () => {
    const one = setCourseStatus(empty, "ISA 125", "now", deps, FALL).state;
    expect(keepTaking(one, ["ISA 125"], spring).courses).toEqual([{ code: "ISA 125", status: "now", term: "Spring 2027" }]);
  });
});
