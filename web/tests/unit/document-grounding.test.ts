import { describe, expect, it } from "vitest";
import {
  checkClaim,
  checkClaims,
  describeGrounding,
  hasInventedNumbers,
  numbersIn,
  overlap,
} from "@/lib/documents/grounding";

/**
 * The point of these tests is adversarial: a model asked to tailor a resume
 * will add plausible experience the student never had, and those are exactly
 * the claims that collapse in an interview. Rewording must pass; invention must
 * not.
 */

const RESUME = `
Data Analytics Intern, Acme Logistics, Summer 2025
Built weekly reports in Excel and SQL for the operations team
Cleaned shipment data and flagged duplicate records
Presented findings to five managers at the end of the internship
Treasurer, Business Analytics Club, 2024 to 2025
Managed a budget of $4,000 across eight events
`;

describe("rewording is allowed", () => {
  it("accepts a bullet that rewords the student's own line", () => {
    const result = checkClaim(
      "Automated weekly operations reporting using SQL and Excel",
      "Built weekly reports in Excel and SQL for the operations team",
      RESUME,
    );
    expect(result.verdict).toBe("grounded");
  });

  it("accepts a stronger action verb over the same substance", () => {
    const result = checkClaim(
      "Reconciled shipment data and eliminated duplicate records",
      "Cleaned shipment data and flagged duplicate records",
      RESUME,
    );
    expect(result.verdict).toBe("grounded");
  });

  it("accepts a bullet drawing on the resume as a whole", () => {
    // A bullet may legitimately combine several lines, so a null source falls
    // back to the whole resume rather than being rejected outright.
    const result = checkClaim(
      "Presented analytics findings to managers at Acme Logistics",
      null,
      RESUME,
    );
    expect(result.verdict).toBe("grounded");
  });
});

describe("invention is caught", () => {
  it("rejects experience that appears nowhere in the resume", () => {
    const result = checkClaim(
      "Led migration of the company data warehouse to Snowflake using dbt and Airflow",
      null,
      RESUME,
    );
    expect(result.verdict).toBe("unsupported");
    expect(result.note).toMatch(/could not match/i);
  });

  it("rejects a fabricated percentage, the most damaging kind", () => {
    // Reads well, and is indefensible if the student never measured it.
    const result = checkClaim(
      "Built weekly reports in SQL, improving operations efficiency by 40%",
      "Built weekly reports in Excel and SQL for the operations team",
      RESUME,
    );
    expect(result.verdict).toBe("invented_numbers");
    expect(result.note).toMatch(/figure that is not in your resume/i);
  });

  it("allows a number the student actually wrote down", () => {
    const result = checkClaim(
      "Managed a $4,000 budget across eight club events",
      "Managed a budget of $4,000 across eight events",
      RESUME,
    );
    expect(result.verdict).toBe("grounded");
  });

  it("is not fooled by generic verbs shared with the resume", () => {
    // "managed" and "team" both appear in this resume, so a permissive
    // word-overlap check passes this while the entire substance is invented.
    const result = checkClaim("Managed a team of forty consultants", null, RESUME);
    expect(result.verdict).toBe("unsupported");
  });

  it("does not accept a claim just because it names a real source line", () => {
    // The model can cite any line it likes; the citation is verified, not
    // trusted, which is the same stance Exam Ally takes on question grounding.
    const result = checkClaim(
      "Directed a team of twelve engineers across three product lines",
      "Cleaned shipment data and flagged duplicate records",
      RESUME,
    );
    expect(result.verdict).toBe("unsupported");
  });
});

describe("number extraction", () => {
  it("finds numbers in the forms a resume uses", () => {
    expect(numbersIn("Managed $4,000 across 8 events, up 12%")).toEqual(
      expect.arrayContaining(["4000", "8", "12%"]),
    );
  });

  it("only flags numbers absent from the source", () => {
    expect(hasInventedNumbers("grew by 40%", "grew by 12%")).toBe(true);
    expect(hasInventedNumbers("grew by 12%", "grew by 12%")).toBe(false);
    expect(hasInventedNumbers("no numbers here", RESUME)).toBe(false);
  });
});

describe("overlap scoring", () => {
  it("ignores filler words when comparing", () => {
    expect(
      overlap("Presented findings to managers", "Presented the findings to five managers"),
    ).toBeGreaterThan(0.7);
  });

  it("scores unrelated text near zero", () => {
    expect(overlap("Wrote Kubernetes operators in Go", RESUME)).toBeLessThan(0.35);
  });
});

describe("reporting to the student", () => {
  it("summarises what needs attention and says why", () => {
    const report = checkClaims(
      [
        { text: "Built weekly reports in Excel and SQL", sourceLine: null },
        { text: "Managed a team of forty consultants", sourceLine: null },
      ],
      RESUME,
    );
    expect(report.checked).toBe(2);
    expect(report.grounded).toBe(1);
    expect(report.flagged).toHaveLength(1);

    const message = describeGrounding(report);
    expect(message).toMatch(/one line needs/i);
    expect(message).toMatch(/traced back to your resume/i);
  });

  it("says nothing when everything is grounded", () => {
    const report = checkClaims(
      [{ text: "Cleaned shipment data and flagged duplicates", sourceLine: null }],
      RESUME,
    );
    expect(report.flagged).toHaveLength(0);
    expect(describeGrounding(report)).toBeNull();
  });
});
