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
const { getScoutDb } = await import("@/lib/db");
const schema = await import("@/lib/db/schema");
import type { RawPosting } from "@/lib/scout/sources/types";
import type { TagResult } from "@/lib/scout/tag";

beforeAll(() => {
  // Freeze the calendar (Date only; real timers stay live for fetch/promises)
  // so the fixtures' postedAt values never age past the 30-day drop rule and
  // start failing these tests months from now. The Active Jobs DB fixture was
  // captured 2026-07-29.
  vi.useFakeTimers({ now: new Date("2026-07-30T12:00:00.000Z"), toFake: ["Date"] });
});

afterAll(() => {
  vi.useRealTimers();
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
});

beforeEach(() => {
  const db = getScoutDb();
  db.delete(schema.scoutPostings).run();
  db.delete(schema.scoutRuns).run();
  delete process.env.CHATISA_SCOUT_MAX_RUN_USD;
});

const activejobsPage = readFileSync(
  path.join(process.cwd(), "tests", "fixtures", "scout", "activejobs-page.json"),
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
    if (url.includes("active-jobs-db")) return jsonResponse(activejobsPage);
    if (opts.failUsajobs) return new Response("nope", { status: 500 });
    return jsonResponse(usajobsPage);
  };
}

/** An /active-ats row in the captured payload's shape. */
const atsRow = (i: number, overrides: Record<string, unknown> = {}) => ({
  id: 9000 + i,
  title: "Data Analyst",
  organization: `Employer ${i}`,
  url: `https://employer${i}.wd1.myworkdayjobs.com/job/${i}`,
  description_text:
    "A full-length posting describing analytics duties, SQL reporting, dashboard development, and stakeholder communication for an entry-level analyst position.",
  regions_derived: ["Ohio"],
  cities_derived: ["Columbus"],
  date_posted: "2026-07-25T09:00:00.000",
  ...overrides,
});

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
    // The captured page normalizes to two rows (the "5-10" senior row is
    // dropped at the source), then the title gate removes "Sr Clinical Data
    // Analyst" (the \bsr\b rule). Every query pass returns the SAME page, so
    // external-id dedupe collapses the repeats. Stored: Metronet's
    // "Data Engineer - SQL & Analytics" plus the USAJobs Management Analyst.
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
    const sixJobs = Array.from({ length: 6 }, (_, i) => atsRow(i));
    const summary = await runHarvest(
      { trigger: "manual" },
      {
        fetcher: (async (input: string | URL | Request) =>
          String(input).includes("active-jobs-db")
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
    const twoJobs = [
      atsRow(0, { organization: "Fresh Co" }),
      // Frozen "today" is 2026-07-30, so this one is two months old.
      atsRow(1, { organization: "Stale Co", date_posted: "2026-06-01T09:00:00.000" }),
    ];
    const summary = await runHarvest(
      { trigger: "manual" },
      {
        fetcher: (async (input: string | URL | Request) =>
          String(input).includes("active-jobs-db")
            ? jsonResponse(JSON.stringify(twoJobs))
            : new Response("nope", { status: 500 })) as typeof fetch,
        tagger: okTag([{ skillId: "sql", importance: "required" }]),
      },
    );
    if ("alreadyRunning" in summary) throw new Error("unexpected");
    expect(summary.tagged).toBe(1);
    expect(countScoutPostings()).toBe(1);
  });

  it("stops querying Active Jobs DB after the first 429 instead of burning quota", async () => {
    let activejobsCalls = 0;
    const summary = await runHarvest(
      { trigger: "manual" },
      {
        fetcher: (async (input: string | URL | Request) => {
          if (String(input).includes("active-jobs-db")) {
            activejobsCalls += 1;
            return new Response("quota", { status: 429 });
          }
          return jsonResponse(usajobsPage);
        }) as typeof fetch,
        tagger: okTag([{ skillId: "sql", importance: "required" }]),
      },
    );
    if ("alreadyRunning" in summary) throw new Error("unexpected");
    // The 2026-07-28 JSearch run burned 160 requests against a dead quota;
    // this pins the same fix here: one request, then degrade to USAJobs.
    expect(activejobsCalls).toBe(1);
    expect(summary.sourceErrors.activejobs).toContain("quota");
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
    expect(summary.sourceErrors.activejobs).toBeDefined();
    expect(summary.sourceErrors.usajobs).toBeDefined();
  });
});
