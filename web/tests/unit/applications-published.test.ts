import { describe, expect, it } from "vitest";
import { parsePublishedWork, publishedWorkBlock } from "@/lib/jobs/published-work";

describe("published work for JobApp Drafter", () => {
  it("parses and caps the form field", () => {
    const items = parsePublishedWork(JSON.stringify([
      { title: "Churn", summary: "s", url: "https://x.github.io/churn/", skills: ["R"] },
      { title: "bad", summary: "s", url: "javascript:alert(1)", skills: [] },
    ]));
    expect(items).toEqual([{ title: "Churn", summary: "s", url: "https://x.github.io/churn/", skills: ["R"] }]);
    expect(parsePublishedWork("nope")).toEqual([]);
    expect(parsePublishedWork(null)).toEqual([]);
  });
  it("renders a block the drafts can cite", () => {
    const block = publishedWorkBlock([{ title: "Churn", summary: "A model.", url: "https://x.github.io/churn/", skills: ["R", "SQL"] }]);
    expect(block).toContain("Published work");
    expect(block).toContain("- Churn: A model. (https://x.github.io/churn/) Skills: R, SQL");
  });
});
