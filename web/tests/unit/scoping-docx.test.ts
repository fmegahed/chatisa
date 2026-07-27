import { describe, expect, it } from "vitest";
import { renderScopingDocx } from "@/lib/documents/scoping-docx";
import { emptyScopingContent } from "@/lib/project/scoping";

const header = {
  projectName: "Retail dashboard",
  courseLabel: "ISA 401/501: Business Intelligence and Data Visualization",
  organization: "Kroger",
  members: ["Team Lead", "Teammate"],
};

describe("renderScopingDocx", () => {
  it("produces a non-empty .docx buffer for an empty worksheet", async () => {
    const buf = await renderScopingDocx(emptyScopingContent(), header);
    expect(buf.byteLength).toBeGreaterThan(0);
    // .docx is a zip: it starts with the "PK" local-file signature.
    expect(buf.subarray(0, 2).toString("latin1")).toBe("PK");
  });

  it("produces a larger buffer once the worksheet has content", async () => {
    const empty = await renderScopingDocx(emptyScopingContent(), header);
    const filled = emptyScopingContent();
    filled.organizationName = "Kroger";
    filled.goals = [{ goal: "Cut stockouts", constraints: "One quarter" }];
    filled.stakeholders = [
      { orgDept: "Operations", involvement: "Owner", counterpart: "Analyst" },
    ];
    const withContent = await renderScopingDocx(filled, header);
    expect(withContent.byteLength).toBeGreaterThan(empty.byteLength);
  });
});
