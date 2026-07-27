// tests/unit/coach-engine.test.ts
import { describe, expect, it } from "vitest";
import { getCoachEngine } from "@/lib/project/coach-engine";

describe("coach engine", () => {
  it("returns null for an unknown coach", () => {
    expect(getCoachEngine("banana")).toBeNull();
  });

  it("drives a generic coach (devils_advocate) through setField", () => {
    const engine = getCoachEngine("devils_advocate")!;
    const empty = engine.emptyContent();
    const next = engine.applyOp(empty, { kind: "setField", path: "decision", value: "Ship Friday" });
    const json = JSON.stringify(next);
    expect(json).toContain("Ship Friday");
    // Round-trips through parseContent.
    const reread = engine.parseContent(json) as { fields: { decision: string } };
    expect(reread.fields.decision).toBe("Ship Friday");
    expect(engine.parseUnknown({ nonsense: true })).toBeNull();
  });

  it("wraps scoping without changing its behavior", () => {
    const engine = getCoachEngine("scoping")!;
    const next = engine.applyOp(engine.emptyContent(), {
      kind: "setField",
      path: "organizationName",
      value: "Kroger",
    });
    expect(JSON.stringify(next)).toContain("Kroger");
    expect(engine.systemPrompt.length).toBeGreaterThan(50);
  });
});
