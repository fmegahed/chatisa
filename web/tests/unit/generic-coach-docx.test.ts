// tests/unit/generic-coach-docx.test.ts
import { describe, expect, it } from "vitest";
import { renderGenericCoachDocx } from "@/lib/documents/generic-coach-docx";
import { COACH_SPECS } from "@/lib/project/coach-specs";
import { buildEmptyContent } from "@/lib/project/coach-framework";

const header = {
  projectName: "Retail dashboard",
  courseLabel: "ISA 496: Business Analytics Practicum",
  organization: "Kroger",
  members: ["Lead", "Mate"],
};

describe("renderGenericCoachDocx", () => {
  it("renders a valid .docx for a premortem deliverable", async () => {
    const spec = COACH_SPECS.premortem;
    const content = buildEmptyContent(spec);
    content.fields.projectDescription = "A forecasting tool";
    content.tables.failures = [{ failure: "No data", howToAvoid: "Confirm early" }];
    const buf = await renderGenericCoachDocx(spec, content, header);
    expect(buf.byteLength).toBeGreaterThan(0);
    expect(buf.subarray(0, 2).toString("latin1")).toBe("PK");
  });

  it("renders a valid .docx for a fields-only coach (reflection)", async () => {
    const spec = COACH_SPECS.reflection;
    const buf = await renderGenericCoachDocx(spec, buildEmptyContent(spec), header);
    expect(buf.subarray(0, 2).toString("latin1")).toBe("PK");
  });
});
