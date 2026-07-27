// tests/unit/project-scoping.test.ts
import { describe, expect, it } from "vitest";
import {
  scopingContentSchema,
  emptyScopingContent,
  applyScopingOp,
} from "@/lib/project/scoping";

describe("scoping content schema", () => {
  it("an empty deliverable validates", () => {
    expect(scopingContentSchema.safeParse(emptyScopingContent()).success).toBe(true);
  });

  it("caps the bounded tables at three rows", () => {
    const c = emptyScopingContent();
    c.goals = [
      { goal: "a", constraints: "" },
      { goal: "b", constraints: "" },
      { goal: "c", constraints: "" },
      { goal: "d", constraints: "" },
    ];
    expect(scopingContentSchema.safeParse(c).success).toBe(false);
  });
});

describe("applyScopingOp", () => {
  it("sets a top-level field", () => {
    const c = applyScopingOp(emptyScopingContent(), {
      kind: "setField",
      path: "organizationName",
      value: "Kroger",
    });
    expect(c.organizationName).toBe("Kroger");
  });

  it("sets a nested field", () => {
    const c = applyScopingOp(emptyScopingContent(), {
      kind: "setField",
      path: "problem.whatProblem",
      value: "Stockouts",
    });
    expect(c.problem.whatProblem).toBe("Stockouts");
  });

  it("ignores an unknown path and returns content unchanged", () => {
    const before = emptyScopingContent();
    const after = applyScopingOp(before, {
      kind: "setField",
      path: "problem.nonsense",
      value: "x",
    });
    expect(after).toEqual(before);
  });

  it("adds a row to a table and caps at three", () => {
    let c = emptyScopingContent();
    c = applyScopingOp(c, { kind: "addRow", table: "goals" });
    expect(c.goals).toHaveLength(1);
    c = applyScopingOp(c, { kind: "addRow", table: "goals" });
    c = applyScopingOp(c, { kind: "addRow", table: "goals" });
    c = applyScopingOp(c, { kind: "addRow", table: "goals" }); // 4th ignored
    expect(c.goals).toHaveLength(3);
  });

  it("sets a row's known keys and ignores unknown keys", () => {
    let c = emptyScopingContent();
    c = applyScopingOp(c, { kind: "addRow", table: "stakeholders" });
    c = applyScopingOp(c, {
      kind: "setRow",
      table: "stakeholders",
      index: 0,
      row: { orgDept: "Ops", involvement: "Owner", bogus: "drop me" },
    });
    expect(c.stakeholders[0]).toEqual({
      orgDept: "Ops",
      involvement: "Owner",
      counterpart: "",
    });
    expect("bogus" in c.stakeholders[0]).toBe(false);
  });

  it("leaves content unchanged for an out-of-range setRow", () => {
    const before = emptyScopingContent();
    const after = applyScopingOp(before, {
      kind: "setRow",
      table: "goals",
      index: 5,
      row: { goal: "x" },
    });
    expect(after).toEqual(before);
  });

  it("does not mutate the input", () => {
    const before = emptyScopingContent();
    const snapshot = JSON.stringify(before);
    applyScopingOp(before, { kind: "setField", path: "contacts", value: "Jo" });
    expect(JSON.stringify(before)).toBe(snapshot);
  });
});
