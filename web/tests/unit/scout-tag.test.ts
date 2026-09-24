import { describe, expect, it } from "vitest";
import { buildTagInstructions, keepSupportedDomainTags, vocabularyIds } from "@/lib/scout/tag";
import { SKILLS } from "@/lib/scout/taxonomy";

/**
 * The job tagger's instructions (v6.7.0). Business-domain skills are new in
 * taxonomy v2, and the professor's rules for them must reach the model:
 * tag them only when the role asks, never from the employer's industry,
 * and default them to "preferred". v1 is kept so the before/after tag
 * evaluation can compare the two vocabularies on the same postings.
 */
const business = SKILLS.filter((s) => s.category === "business").map((s) => s.id);

describe("buildTagInstructions", () => {
  it("v2 carries the domain rules and a one-line definition for every business skill", () => {
    const text = buildTagInstructions("v2");
    expect(text).toMatch(/never because of the employer's industry/i);
    expect(text).toMatch(/"preferred" unless the posting explicitly requires/i);
    for (const id of business) expect(text).toContain(`${id} (`);
  });

  it("v1 is the vocabulary before business skills existed, with no domain rules", () => {
    const text = buildTagInstructions("v1");
    expect(text).not.toMatch(/employer's industry/i);
    for (const id of business) expect(text).not.toContain(id);
  });
});

describe("keepSupportedDomainTags", () => {
  const posting = "Management Analyst. Leads the Lean Six Sigma section; performs cost analysis for three clinics.";

  it("keeps a business-domain tag the posting's own words support", () => {
    const kept = keepSupportedDomainTags([{ skillId: "process_improvement", importance: "preferred" }], posting);
    expect(kept).toEqual([{ skillId: "process_improvement", importance: "preferred" }]);
  });

  it("drops a business-domain tag with no supporting wording (the industry-guess mistake)", () => {
    const kept = keepSupportedDomainTags([{ skillId: "negotiation", importance: "required" }], posting);
    expect(kept).toEqual([]);
  });

  it("matches whole words only: 'hr' is not found inside 'three'", () => {
    expect(keepSupportedDomainTags([{ skillId: "human_capital_management", importance: "preferred" }], posting)).toEqual([]);
  });

  it("never filters the analytics and IS skills, which the model tags as before", () => {
    const tags = [{ skillId: "sql", importance: "required" as const }];
    expect(keepSupportedDomainTags(tags, "No mention at all.")).toEqual(tags);
  });
});

describe("vocabularyIds", () => {
  it("v1 excludes business skills; v2 is every skill", () => {
    expect(vocabularyIds("v1").some((id) => business.includes(id))).toBe(false);
    expect(vocabularyIds("v2")).toHaveLength(SKILLS.length);
  });
});
