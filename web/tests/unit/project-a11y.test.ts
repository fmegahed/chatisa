import { describe, expect, it } from "vitest";
import { coachPageTitle, projectPageTitle } from "@/lib/project/page-titles";
import { INVITE_EMAIL_HINT, inviteEmailError } from "@/lib/project/invite";

describe("project page titles (#28)", () => {
  const project = { name: "Team Alpha", courseCode: "401/501" };

  it("names the project and course for a member", () => {
    expect(projectPageTitle(project)).toBe("Team Alpha (ISA 401/501)");
  });

  it("stays generic without an accessible project", () => {
    expect(projectPageTitle(undefined)).toBe("Project");
    expect(projectPageTitle(null)).toBe("Project");
  });

  it("puts the coach type first on coach pages", () => {
    expect(coachPageTitle("scoping", project)).toBe(
      "Project Scoping Coach: Team Alpha (ISA 401/501)",
    );
    expect(coachPageTitle("premortem", undefined)).toBe("Premortem Coach");
    expect(coachPageTitle("not-a-coach", undefined)).toBe("Coach");
  });
});

describe("teammate invite rule (#26)", () => {
  it("states the domain requirement up front", () => {
    expect(INVITE_EMAIL_HINT).toContain("miamioh.edu");
  });

  it("accepts a Miami address in any case, with spaces trimmed", () => {
    expect(inviteEmailError(" Jane.Doe@MiamiOH.edu ")).toBeNull();
  });

  it("explains each kind of problem", () => {
    expect(inviteEmailError("")).toMatch(/Enter your teammate/);
    expect(inviteEmailError("not an email")).toMatch(/valid email/);
    expect(inviteEmailError("someone@gmail.com")).toMatch(/miamioh\.edu/);
    expect(inviteEmailError("someone@miamioh.edu.evil.com")).toMatch(/miamioh\.edu/);
    expect(inviteEmailError(`${"a".repeat(200)}@miamioh.edu`)).toMatch(/valid email/);
  });
});
