// tests/unit/project-scoping-tools.test.ts
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-coach-tools-"));
process.env.CHATISA_DATA_DIR = dataDir;

const { closeDb, upsertUser } = await import("@/lib/db");
const { createProject, getDeliverable, getOrCreateDeliverable, saveDeliverableContent } =
  await import("@/lib/db/projects");
const { applyScopingOp, scopingContentSchema, emptyScopingContent } =
  await import("@/lib/project/scoping");

const LEAD = "lead@miamioh.edu";

beforeAll(() => upsertUser(LEAD, "Lead"));
afterAll(() => {
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
});

/** Mirrors the route's applyAndSave, the code a tool execute runs. */
function applyAndSave(projectId: string, op: Parameters<typeof applyScopingOp>[1]) {
  const row = getDeliverable(projectId, "scoping");
  const parsed = scopingContentSchema.safeParse(row ? JSON.parse(row.contentJson) : {});
  const content = parsed.success ? parsed.data : emptyScopingContent();
  const next = applyScopingOp(content, op);
  saveDeliverableContent({
    projectId,
    coachType: "scoping",
    contentJson: JSON.stringify(next),
    updatedBy: "Lead",
  });
}

describe("coach tool wiring", () => {
  it("setField then addRow then setRow accumulate in the saved deliverable", () => {
    const projectId = createProject({
      ownerEmail: LEAD,
      ownerName: "Lead",
      courseCode: "401/501",
      name: "Coach tools",
      organization: "",
      coachTypes: ["scoping"],
    });
    getOrCreateDeliverable(projectId, "scoping");

    applyAndSave(projectId, { kind: "setField", path: "organizationName", value: "Kroger" });
    applyAndSave(projectId, { kind: "addRow", table: "goals" });
    applyAndSave(projectId, { kind: "setRow", table: "goals", index: 0, row: { goal: "Cut stockouts" } });

    const saved = scopingContentSchema.parse(JSON.parse(getDeliverable(projectId, "scoping")!.contentJson));
    expect(saved.organizationName).toBe("Kroger");
    expect(saved.goals[0].goal).toBe("Cut stockouts");
    expect(getDeliverable(projectId, "scoping")!.lastUpdatedBy).toBe("Lead");
  });
});
