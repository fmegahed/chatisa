import { afterAll, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

/** Each run gets its own database file, so tests never touch dev data. */
const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-interview-link-"));
process.env.CHATISA_DATA_DIR = dataDir;

const {
  closeDb,
  createInterview,
  createJobApplication,
  getOwnedInterview,
  upsertUser,
} = await import("@/lib/db");

afterAll(() => {
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
});

describe("createInterview applicationId link (JobApp -> Interview handoff)", () => {
  it("stores the link when given, and stays null otherwise", () => {
    upsertUser("student@miamioh.edu", "Student");
    const appId = createJobApplication({
      userEmail: "student@miamioh.edu",
      company: "Acme",
      positionTitle: "Data Analyst",
      jobUrl: null,
      descriptionSource: "job_scout",
      postingText: "Posting text",
      resumeText: null,
      resumeFilename: null,
      roleBrief: null,
      candidateBrief: null,
    });

    const base = {
      userEmail: "student@miamioh.edu",
      modelId: "gpt-5.6-terra",
      interviewType: "mixed",
      jobTitle: "Data Analyst",
      roleBrief: null,
      candidateBrief: null,
      gradeLevel: null,
      major: null,
      plannedQuestions: 5,
    };
    const linked = createInterview({ ...base, applicationId: appId });
    expect(getOwnedInterview(linked, 'student@miamioh.edu')?.applicationId).toBe(appId);

    const unlinked = createInterview(base);
    expect(getOwnedInterview(unlinked, 'student@miamioh.edu')?.applicationId).toBeNull();
  });
});
