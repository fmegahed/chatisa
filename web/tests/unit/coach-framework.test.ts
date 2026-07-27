// tests/unit/coach-framework.test.ts
import { describe, expect, it } from "vitest";
import {
  buildEmptyContent,
  coachContentSchema,
  applyGenericOp,
  type CoachSpec,
} from "@/lib/project/coach-framework";

const SPEC: CoachSpec = {
  type: "demo",
  title: "Demo",
  fields: [{ key: "decision", label: "Decision" }],
  tables: [
    { key: "rows", label: "Rows", columns: [{ key: "a", label: "A" }, { key: "b", label: "B" }] },
  ],
  systemPrompt: "x",
};

describe("coach framework", () => {
  it("builds an empty content with every field and table present", () => {
    const c = buildEmptyContent(SPEC);
    expect(c).toEqual({ fields: { decision: "" }, tables: { rows: [] } });
    expect(coachContentSchema(SPEC).safeParse(c).success).toBe(true);
  });

  it("sets a known field and ignores an unknown one", () => {
    let c = buildEmptyContent(SPEC);
    c = applyGenericOp(SPEC, c, { kind: "setField", path: "decision", value: "Go" });
    expect(c.fields.decision).toBe("Go");
    const before = c;
    const after = applyGenericOp(SPEC, c, { kind: "setField", path: "nope", value: "x" });
    expect(after).toEqual(before);
  });

  it("adds and sets table rows, ignoring unknown tables and columns", () => {
    let c = buildEmptyContent(SPEC);
    c = applyGenericOp(SPEC, c, { kind: "addRow", table: "rows" });
    expect(c.tables.rows).toHaveLength(1);
    c = applyGenericOp(SPEC, c, { kind: "setRow", table: "rows", index: 0, row: { a: "1", z: "drop" } });
    expect(c.tables.rows[0]).toEqual({ a: "1", b: "" });
    const before = c;
    expect(applyGenericOp(SPEC, c, { kind: "addRow", table: "ghost" })).toEqual(before);
    expect(applyGenericOp(SPEC, c, { kind: "setRow", table: "rows", index: 9, row: {} })).toEqual(before);
  });

  it("does not mutate its input", () => {
    const c = buildEmptyContent(SPEC);
    const snap = JSON.stringify(c);
    applyGenericOp(SPEC, c, { kind: "setField", path: "decision", value: "x" });
    expect(JSON.stringify(c)).toBe(snap);
  });
});
