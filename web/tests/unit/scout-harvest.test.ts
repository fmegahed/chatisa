import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

/** Each run gets its own database file, so tests never touch dev data. */
const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-scout-harvest-"));
process.env.CHATISA_DATA_DIR = dataDir;
// Keys must look configured so the sources actually call the fake fetcher.
process.env.RAPIDAPI_KEY = "test-key";
process.env.USAJOBS_API_KEY = "test-key";
process.env.USAJOBS_EMAIL = "test@example.edu";

const { runHarvest } = await import("@/lib/scout/harvest");
const { closeDb, countScoutPostings, latestSuccessfulScoutRun } = await import(
  "@/lib/db"
);
const { getDb } = await import("@/lib/db");
const schema = await import("@/lib/db/schema");
import type { RawPosting } from "@/lib/scout/sources/types";
import type { TagResult } from "@/lib/scout/tag";

beforeAll(() => {
  // Freeze the calendar (Date only; real timers stay live for fetch/promises)
  // so the fixtures' postedAt values never age past the 30-day drop rule and
  // start failing these tests months from now.
  vi.useFakeTimers({ now: new Date("2026-07-28T12:00:00.000Z"), toFake: ["Date"] });
});

afterAll(() => {
  vi.useRealTimers();
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
});

beforeEach(() => {
  const db = getDb();
  db.delete(schema.scoutPostings).run();
  db.delete(schema.scoutRuns).run();
  delete process.env.CHATISA_SCOUT_MAX_RUN_USD;
});

const jsearchPage = readFileSync(
  path.join(process.cwd(), "tests", "fixtures", "scout", "jsearch-page.json"),
  "utf8",
);
const usajobsPage = readFileSync(
  path.join(process.cwd(), "tests", "fixtures", "scout", "usajobs-page.json"),
  "utf8",
);

const jsonResponse = (body: string) =>
  new Response(body, {
    status: 200,
    headers: { "content-type": "application/json" },
  });

function fakeFetcher(opts: { failUsajobs?: boolean } = {}) {
  return async (input: string | URL | Request) => {
    const url = String(input);
    if (url.includes("jsearch")) return jsonResponse(jsearchPage);
    if (opts.failUsajobs) return new Response("nope", { status: 500 });
    return jsonResponse(usajobsPage);
  };
}

const okTag = (skills: TagResult["skills"]): ((p: RawPosting) => Promise<TagResult>) =>
  async (p) => ({
    skills,
    category: p.category,
    seniorityOk: true,
    visaSponsorship: "unknown",
    costUsd: 0.001,
  });

describe("runHarvest", () => {
  it("dedupes across queries and sources, filters titles, stores tagged postings", async () => {
    const summary = await runHarvest(
      { trigger: "manual" },
      {
        fetcher: fakeFetcher() as typeof fetch,
        tagger: okTag([{ skillId: "sql", importance: "required" }]),
      },
    );
    if ("alreadyRunning" in summary) throw new Error("unexpected");
    // Fixture yields, after external-id + fingerprint dedupe and the title
    // gate: js-001 (Data Analyst) and usaj-800100 (Management Analyst).
    // js-002 is senior; js-003 is a cross-board duplicate of js-001.
    expect(summary.status).toBe("completed");
    expect(summary.tagged).toBe(2);
    expect(countScoutPostings()).toBe(2);
    expect(latestSuccessfulScoutRun()?.taggedCount).toBe(2);
  });

  it("degrades to one source with an honest error when the other fails", async () => {
    const summary = await runHarvest(
      { trigger: "manual" },
      {
        fetcher: fakeFetcher({ failUsajobs: true }) as typeof fetch,
        tagger: okTag([{ skillId: "sql", importance: "required" }]),
      },
    );
    if ("alreadyRunning" in summary) throw new Error("unexpected");
    expect(summary.status).toBe("completed");
    expect(summary.tagged).toBe(1);
    expect(summary.sourceErrors.usajobs).toContain("500");
  });

  it("stops tagging at the cost cap and marks the run partial", async () => {
    process.env.CHATISA_SCOUT_MAX_RUN_USD = "5";
    // Six unique relevant postings so the four concurrent workers pass the
    // pre-flight check and the remaining two hit the cap deterministically.
    const sixJobs = {
      data: Array.from({ length: 6 }, (_, i) => ({
        job_id: `bulk-${i}`,
        job_title: "Data Analyst",
        employer_name: `Employer ${i}`,
        job_city: "Columbus",
        job_state: "OH",
        job_is_remote: false,
        job_apply_link: `https://example.com/${i}`,
        job_description:
          "A full-length posting describing analytics duties, SQL reporting, dashboard development, and stakeholder communication for an entry-level analyst position.",
        job_posted_at_datetime_utc: "2026-07-22T09:00:00.000Z",
      })),
    };
    const summary = await runHarvest(
      { trigger: "manual" },
      {
        fetcher: (async (input: string | URL | Request) =>
          String(input).includes("jsearch")
            ? jsonResponse(JSON.stringify(sixJobs))
            : new Response("nope", { status: 500 })) as typeof fetch,
        tagger: async (p) => ({
          skills: [{ skillId: "sql", importance: "required" }],
          category: p.category,
          seniorityOk: true,
          visaSponsorship: "unknown" as const,
          costUsd: 6,
        }),
      },
    );
    if ("alreadyRunning" in summary) throw new Error("unexpected");
    expect(summary.status).toBe("partial");
    expect(summary.tagged).toBe(4);
    expect(summary.droppedByCap).toBe(2);
  });

  it("drops postings older than a month before any model spend", async () => {
    const twoJobs = {
      data: [
        {
          job_id: "fresh-1",
          job_title: "Data Analyst",
          employer_name: "Fresh Co",
          job_city: "Columbus",
          job_state: "OH",
          job_is_remote: false,
          job_apply_link: "https://example.com/fresh",
          job_description:
            "A full-length posting describing analytics duties, SQL reporting, dashboard development, and stakeholder communication for an entry-level analyst position.",
          job_posted_at_datetime_utc: "2026-07-20T09:00:00.000Z",
        },
        {
          job_id: "stale-1",
          job_title: "Data Analyst",
          employer_name: "Stale Co",
          job_city: "Columbus",
          job_state: "OH",
          job_is_remote: false,
          job_apply_link: "https://example.com/stale",
          job_description:
            "An equally full-length posting describing analytics duties, SQL reporting, dashboard development, and stakeholder communication, posted well over a month ago.",
          // Frozen "today" is 2026-07-28, so this is 57 days old.
          job_posted_at_datetime_utc: "2026-06-01T09:00:00.000Z",
        },
      ],
    };
    const summary = await runHarvest(
      { trigger: "manual" },
      {
        fetcher: (async (input: string | URL | Request) =>
          String(input).includes("jsearch")
            ? jsonResponse(JSON.stringify(twoJobs))
            : new Response("nope", { status: 500 })) as typeof fetch,
        tagger: okTag([{ skillId: "sql", importance: "required" }]),
      },
    );
    if ("alreadyRunning" in summary) throw new Error("unexpected");
    expect(summary.tagged).toBe(1);
    expect(countScoutPostings()).toBe(1);
  });

  it("stops querying JSearch after the first 429 instead of burning quota", async () => {
    let jsearchCalls = 0;
    const summary = await runHarvest(
      { trigger: "manual" },
      {
        fetcher: (async (input: string | URL | Request) => {
          if (String(input).includes("jsearch")) {
            jsearchCalls += 1;
            return new Response("quota", { status: 429 });
          }
          return jsonResponse(usajobsPage);
        }) as typeof fetch,
        tagger: okTag([{ skillId: "sql", importance: "required" }]),
      },
    );
    if ("alreadyRunning" in summary) throw new Error("unexpected");
    // The 2026-07-28 live run burned 160 requests against a dead quota;
    // this pins the fix: one request, then stop and degrade to USAJobs.
    expect(jsearchCalls).toBe(1);
    expect(summary.sourceErrors.jsearch).toContain("quota");
    expect(summary.status).toBe("completed");
    expect(summary.tagged).toBe(1);
  });

  it("fails the run only when both sources fail", async () => {
    const summary = await runHarvest(
      { trigger: "manual" },
      {
        fetcher: (async () => new Response("nope", { status: 500 })) as typeof fetch,
        tagger: okTag([]),
      },
    );
    if ("alreadyRunning" in summary) throw new Error("unexpected");
    expect(summary.status).toBe("failed");
    expect(summary.sourceErrors.jsearch).toBeDefined();
    expect(summary.sourceErrors.usajobs).toBeDefined();
  });
});
