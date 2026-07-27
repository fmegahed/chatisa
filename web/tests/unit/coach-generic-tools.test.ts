// tests/unit/coach-generic-tools.test.ts
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-generic-coach-"));
process.env.CHATISA_DATA_DIR = dataDir;

const { closeDb, upsertUser } = await import("@/lib/db");
const { createProject, getDeliverable, getOrCreateDeliverable, saveDeliverableContent } =
  await import("@/lib/db/projects");
const { getCoachEngine } = await import("@/lib/project/coach-engine");

const LEAD = "lead@miamioh.edu";
beforeAll(() => upsertUser(LEAD, "Lead"));
afterAll(() => {
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
});

function applyAndSave(projectId: string, coachType: string, op: Parameters<NonNullable<ReturnType<typeof getCoachEngine>>["applyOp"]>[1]) {
  const engine = getCoachEngine(coachType)!;
  const row = getDeliverable(projectId, coachType);
  const content = row ? engine.parseContent(row.contentJson) : engine.emptyContent();
  const next = engine.applyOp(content, op);
  saveDeliverableContent({ projectId, coachType, contentJson: JSON.stringify(next), updatedBy: "Lead" });
}

describe("generic coach route wiring", () => {
  it("premortem: description plus a failure row accumulate", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Lead",
      courseCode: "496",
      name: "Premortem",
      organization: "",
      coachTypes: ["premortem"],
    });
    getOrCreateDeliverable(projectId, "premortem");
    applyAndSave(projectId, "premortem", { kind: "setField", path: "projectDescription", value: "A forecasting tool" });
    applyAndSave(projectId, "premortem", { kind: "addRow", table: "failures" });
    applyAndSave(projectId, "premortem", { kind: "setRow", table: "failures", index: 0, row: { failure: "No data access", howToAvoid: "Confirm access early" } });

    const saved = JSON.parse(getDeliverable(projectId, "premortem")!.contentJson);
    expect(saved.fields.projectDescription).toBe("A forecasting tool");
    expect(saved.tables.failures[0]).toEqual({ failure: "No data access", howToAvoid: "Confirm access early" });
  });
});
