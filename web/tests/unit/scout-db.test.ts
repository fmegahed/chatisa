import { afterAll, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

/** Each run gets its own database file, so tests never touch dev data. */
const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-scout-db-"));
process.env.CHATISA_DATA_DIR = dataDir;

const {
  closeDb,
  getDb,
  countScoutPostings,
  createScoutRun,
  finishScoutRun,
  getScoutPosting,
  latestSuccessfulScoutRun,
  listScoutPostings,
  retireScoutPostings,
  scoutFingerprintExists,
  scoutRunInProgress,
  upsertScoutPosting,
} = await import("@/lib/db");
const dbSchema = await import("@/lib/db/schema");

afterAll(() => {
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
});

function makePosting(overrides: Record<string, unknown> = {}) {
  return {
    source: "jsearch" as const,
    externalId: "job-1",
    fingerprint: "acme corp|data analyst|OH",
    title: "Data Analyst",
    company: "Acme Corp",
    locationCity: "Cincinnati",
    locationState: "OH",
    remote: false,
    category: "fulltime" as const,
    applyUrl: "https://careers.acme.example/1",
    description: "Analyze data with SQL and Tableau.",
    postedAt: "2026-07-24",
    skillsJson: JSON.stringify([{ skillId: "sql", importance: "required" }]),
    visaSponsorship: "unknown",
    taxonomyVersion: 1,
    ...overrides,
  };
}

describe("scout postings", () => {
  it("upsert keeps the id stable across weekly re-harvests", () => {
    const first = upsertScoutPosting(makePosting());
    const second = upsertScoutPosting(
      makePosting({ description: "Updated text." }),
    );
    expect(second).toBe(first);
    expect(getScoutPosting(first)?.description).toBe("Updated text.");
    expect(countScoutPostings()).toBe(1);
  });

  it("detects cross-source duplicates by fingerprint", () => {
    expect(scoutFingerprintExists("acme corp|data analyst|OH")).toBe(true);
    expect(scoutFingerprintExists("nobody|nothing|XX")).toBe(false);
  });

  it("filters the feed by category and state without descriptions", () => {
    upsertScoutPosting(
      makePosting({
        externalId: "job-2",
        fingerprint: "fed|analyst|DC",
        category: "federal",
        locationState: "DC",
        title: "Management Analyst",
      }),
    );
    const federal = listScoutPostings({
      category: "federal",
      limit: 10,
      offset: 0,
    });
    expect(federal).toHaveLength(1);
    expect(federal[0].title).toBe("Management Analyst");
    expect("description" in federal[0]).toBe(false);
    expect(
      listScoutPostings({ state: "OH", limit: 10, offset: 0 }),
    ).toHaveLength(1);
  });

  it("retires postings older than a month by their own post date", () => {
    upsertScoutPosting(
      makePosting({
        externalId: "job-old",
        fingerprint: "old|analyst|OH",
        postedAt: "2026-06-01",
      }),
    );
    const { deactivated } = retireScoutPostings({
      unseenSinceIso: "2000-01-01T00:00:00.000Z",
      purgeBeforeIso: "2000-01-01T00:00:00.000Z",
      postedBeforeIso: "2026-06-28",
    });
    expect(deactivated).toBe(1);
  });

  it("retires unseen postings and purges old rows", () => {
    const future = new Date(Date.now() + 60_000).toISOString();
    const { deactivated } = retireScoutPostings({
      unseenSinceIso: future,
      purgeBeforeIso: "2000-01-01T00:00:00.000Z",
      postedBeforeIso: "2000-01-01",
    });
    expect(deactivated).toBe(2);
    expect(countScoutPostings()).toBe(0);
    const { purged } = retireScoutPostings({
      unseenSinceIso: future,
      purgeBeforeIso: future,
      postedBeforeIso: "2000-01-01",
    });
    expect(purged).toBe(3);
  });
});

describe("scout runs", () => {
  it("treats a running row older than two hours as dead, not in progress", () => {
    getDb()
      .insert(dbSchema.scoutRuns)
      .values({
        id: "stale-run",
        startedAt: new Date(Date.now() - 3 * 3_600_000).toISOString(),
        status: "running",
        trigger: "manual",
      })
      .run();
    expect(scoutRunInProgress()).toBe(false);
  });

  it("tracks the lifecycle and reports the latest success", () => {
    expect(latestSuccessfulScoutRun()).toBeNull();
    const id = createScoutRun("manual");
    expect(scoutRunInProgress()).toBe(true);
    finishScoutRun(id, { status: "completed", taggedCount: 5, costUsd: 0.2 });
    expect(scoutRunInProgress()).toBe(false);
    expect(latestSuccessfulScoutRun()?.id).toBe(id);
    // A later failed run does not displace the last success.
    const failed = createScoutRun("schedule");
    finishScoutRun(failed, { status: "failed", error: "network" });
    expect(latestSuccessfulScoutRun()?.id).toBe(id);
  });
});
