import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import path from "node:path";
import { parseCoursePage, parsePrereq, parseProgram } from "@/scripts/catalog/bulletin";

/**
 * The bulletin parsers run against saved copies of real pages (fixtures,
 * 2026-09-24), so a change in the bulletin's markup fails here, in CI,
 * instead of quietly dropping courses from a student's checklist.
 */
const fixture = (name: string) =>
  readFileSync(path.join(process.cwd(), "tests", "fixtures", "bulletin", `${name}.html`), "utf8");

describe("parseProgram", () => {
  it("reads the business core, merging or-rows into one item and keeping select-one groups", () => {
    const { groups, unparsed } = parseProgram(fixture("core"));
    expect(unparsed).toEqual([]);
    const all = groups.flatMap((g) => g.items);
    expect(all).toContainEqual({ codes: ["ACC 221"] });
    expect(all).toContainEqual({ codes: ["FIN 301", "FIN 311"] });
    const calc = groups.find((g) => g.instruction?.startsWith("Select one of the following"));
    expect(calc?.items.map((i) => i.codes[0])).toEqual(expect.arrayContaining(["MTH 141", "MTH 151"]));
  });

  it("splits concentrations into their own groups with the bulletin's wording", () => {
    const { groups } = parseProgram(fixture("iscm"));
    const titles = groups.map((g) => g.title);
    expect(titles).toContain("Information Systems Concentration");
    expect(titles).toContain("Cybersecurity Management Concentration");
    const electives = groups.find((g) => g.instruction?.includes("one must be an ISA 400 level course"));
    expect(electives?.items.length).toBeGreaterThan(1);
  });

  it("keeps the AI minor's required courses and its free-text option as a note", () => {
    const { groups } = parseProgram(fixture("ai-minor"));
    const codes = groups.flatMap((g) => g.items.flatMap((i) => i.codes));
    expect(codes).toEqual(expect.arrayContaining(["ISA 336", "ISA 381", "ISA 414", "ISA 211", "ISA 235", "MGT 490", "MKT 490"]));
    expect(groups.some((g) => g.notes.some((n) => n.includes("AI in context")))).toBe(true);
  });

  it("reports a row it does not understand instead of dropping it", () => {
    const html = fixture("business-analytics").replace(
      "</table>",
      '<tr class="odd"><td class="weird">Something new</td></tr></table>',
    );
    expect(parseProgram(html).unparsed).toEqual(["Something new"]);
  });
});

describe("parseCoursePage", () => {
  const isa = parseCoursePage(fixture("courses-isa"));
  const byCode = new Map(isa.map((c) => [c.code, c]));

  it("reads code, title, credits and the graduate equivalent of a 401/501 pair", () => {
    const c = byCode.get("ISA 401");
    expect(c?.title).toBe("Business Intelligence and Data Visualization");
    expect(c?.credits).toBe(3);
    expect(c?.altCodes).toContain("ISA 501");
  });

  it("parses every course block on the page", () => {
    expect(isa.length).toBeGreaterThanOrEqual(45);
    for (const c of isa) expect(c.code).toMatch(/^ISA \d{3}[A-Z]?$/);
  });

  it("parses a prerequisite chain", () => {
    expect(byCode.get("ISA 401")?.prereq).toEqual([["ISA 245", "ISA 345", "CSE 385"]]);
  });
});

describe("parsePrereq", () => {
  it("reads and-terms of or-options", () => {
    expect(parsePrereq("ACC 221 and ACC 222.")).toEqual({ groups: [["ACC 221"], ["ACC 222"]], uncertain: false });
    expect(parsePrereq("MKT 291 , ISA 125 or STA 125 or STA 261 or STA 301 .").groups)
      .toEqual([["MKT 291"], ["ISA 125", "STA 125", "STA 261", "STA 301"]]);
  });

  it("treats a cross-listed pair as one option", () => {
    expect(parsePrereq("ECO 311 or ISA 291 or ISA 391 or STA 463/STA 563 .").groups)
      .toEqual([["ECO 311", "ISA 291", "ISA 391", "STA 463"]]);
  });

  it("reads One of (...) as a single or-group", () => {
    expect(parsePrereq("One of (ISA 281, ISA 381, ISA 401/ISA 501); or permission of instructor.")).toEqual({
      groups: [["ISA 281", "ISA 381", "ISA 401"]],
      uncertain: true,
    });
  });

  it("flags permission and class standing as uncertain, keeping the course terms", () => {
    const r = parsePrereq(
      "earn a grade of at least a C in ECO 201 , ECO 202 , and ISA 125 or STA 125 , and MTH 151 or MTH 141 ; or permission of the instructor.",
    );
    expect(r.uncertain).toBe(true);
    expect(r.groups).toEqual([["ECO 201"], ["ECO 202"], ["ISA 125", "STA 125"], ["MTH 151", "MTH 141"]]);
    expect(parsePrereq("Junior standing.").uncertain).toBe(true);
  });

  it("treats prerequisite text with no course codes as uncertain, not as none", () => {
    expect(parsePrereq("determined by professor.")).toEqual({ groups: [], uncertain: true });
    expect(parsePrereq("Completion of the 12 hours of coursework in the Graduate Certificate in Analytics.").uncertain).toBe(true);
  });

  it("reads an unparenthesised 'one of A, B, C or D' as one group (FIN 401, review fix)", () => {
    expect(parsePrereq('FIN 301 and FIN 303 with a grade "C" or better and one of ISA 225, STA 261, STA 301 or STA 368.')).toEqual({
      groups: [["FIN 301"], ["FIN 303"], ["ISA 225", "STA 261", "STA 301", "STA 368"]],
      uncertain: false,
    });
  });

  it("marks 'A or B and C' uncertain rather than guess its grouping (FIN 381, review fix)", () => {
    const r = parsePrereq('FIN 311 with a grade of "C" or better OR FIN 301 and FIN 309 with a grade of "C" or better.');
    expect(r.uncertain).toBe(true);
  });

  it("trusts grouping the Bulletin makes explicit with parentheses (IMS 440, ISA 225)", () => {
    expect(parsePrereq("( IMS 355 or IMS 421 ) and ( IMS 422/IMS 522 or IMS 351 or CSE 252 ).")).toEqual({
      groups: [["IMS 355", "IMS 421"], ["IMS 422", "IMS 351", "CSE 252"]],
      uncertain: false,
    });
    expect(parsePrereq("( MTH 141 or MTH 151 ) and ISA/STA 125.").uncertain).toBe(false);
  });

  it("marks a list ending in ', or' uncertain, since a bare comma means and elsewhere", () => {
    expect(parsePrereq("ISA 235, ISA 245, or CSE 385.").uncertain).toBe(true);
  });

  it("returns no groups for empty text", () => {
    expect(parsePrereq("")).toEqual({ groups: [], uncertain: false });
  });
});
