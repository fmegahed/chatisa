import { afterAll, describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

/** Each run gets its own database file, so tests never touch dev data. */
const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-scout-routes-"));
process.env.CHATISA_DATA_DIR = dataDir;
process.env.CHATISA_MOCK_LLM = "1";

/** Mutable session so one mock serves the 401 and signed-in paths. */
let sessionEmail: string | null = "student@miamioh.edu";
vi.mock("@/lib/auth", () => ({
  auth: async () =>
    sessionEmail ? { user: { email: sessionEmail, name: "Test Student" } } : null,
}));

const { upsertScoutPosting, closeDb } = await import("@/lib/db");
const feed = await import("@/app/api/scout/feed/route");
const detail = await import("@/app/api/scout/postings/[id]/route");
const project = await import("@/app/api/scout/project/route");
const refresh = await import("@/app/api/scout/refresh/route");

afterAll(() => {
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
  delete process.env.CHATISA_MOCK_LLM;
});

const posting = (overrides: Record<string, unknown> = {}) => ({
  source: "activejobs" as const,
  externalId: `ext-${Math.random().toString(36).slice(2)}`,
  fingerprint: `fp-${Math.random().toString(36).slice(2)}`,
  title: "Data Analyst",
  company: "Acme",
  locationCity: "Cincinnati",
  locationState: "OH",
  remote: false,
  category: "fulltime" as const,
  applyUrl: "https://example.com/apply",
  description: "Long description of the analyst role and its requirements.",
  postedAt: "2026-07-24",
  skillsJson: JSON.stringify([{ skillId: "sql", importance: "required" }]),
  visaSponsorship: "unknown",
  taxonomyVersion: 1,
  ...overrides,
});

describe("GET /api/scout/feed", () => {
  it("requires sign-in", async () => {
    sessionEmail = null;
    const res = await feed.GET(new Request("http://x/api/scout/feed"));
    expect(res.status).toBe(401);
    sessionEmail = "student@miamioh.edu";
  });

  it("shape=index returns the whole active feed in one response", async () => {
    upsertScoutPosting(posting({ externalId: "idx-1", fingerprint: "fp-idx-1" }));
    upsertScoutPosting(posting({ externalId: "idx-2", fingerprint: "fp-idx-2" }));
    const res = await feed.GET(
      new Request("http://x/api/scout/feed?shape=index"),
    );
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.postings.length).toBeGreaterThanOrEqual(2);
    expect(body.postings[0].visaSponsorship).toBeDefined();
    expect(body.postings[0].description).toBeUndefined();
    expect(body.freshness).toBeDefined();
  });

  it("returns postings with freshness and honors filters", async () => {
    upsertScoutPosting(posting());
    upsertScoutPosting(posting({ category: "federal", locationState: "DC" }));
    const res = await feed.GET(
      new Request("http://x/api/scout/feed?category=federal"),
    );
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.postings).toHaveLength(1);
    expect(body.postings[0].category).toBe("federal");
    // Cards never carry full descriptions.
    expect(body.postings[0].description).toBeUndefined();
    // The index-shape test above seeded rows too; count only grows.
    expect(body.freshness.totalActive).toBeGreaterThanOrEqual(2);
  });
});

describe("GET /api/scout/postings/[id]", () => {
  it("serves the full posting and a plain 404", async () => {
    const id = upsertScoutPosting(posting());
    const ok = await detail.GET(new Request("http://x"), {
      params: Promise.resolve({ id }),
    });
    expect(ok.status).toBe(200);
    expect((await ok.json()).posting.description).toContain("analyst role");
    const missing = await detail.GET(new Request("http://x"), {
      params: Promise.resolve({ id: "nope" }),
    });
    expect(missing.status).toBe(404);
  });
});

describe("POST /api/scout/project", () => {
  const request = (body: unknown) =>
    new Request("http://x/api/scout/project", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });

  it("rejects unknown skill ids before any model call", async () => {
    const res = await project.POST(
      request({ modelId: "gpt-5.6-terra", skillIds: ["not_a_skill"] }),
    );
    expect(res.status).toBe(400);
  });

  it("generates a scaffold under mock mode", async () => {
    const res = await project.POST(
      request({
        modelId: "gpt-5.6-terra",
        skillIds: ["sql", "data_visualization"],
        evidence: ["wrote SQL to manage structured data"],
      }),
    );
    expect(res.status).toBe(200);
    const { scaffold } = await res.json();
    expect(scaffold.repoName).toMatch(/^[a-z0-9][a-z0-9-]+$/);
    expect(scaffold.files.length).toBeGreaterThanOrEqual(2);
    expect(scaffold.instructions.join("\n")).toContain("gh repo create");
  });
});

describe("POST /api/scout/refresh", () => {
  it("refuses non-admins with a plain message", async () => {
    delete process.env.CHATISA_SCOUT_ADMINS;
    const res = await refresh.POST();
    expect(res.status).toBe(403);
  });

  it("accepts a configured admin", async () => {
    process.env.CHATISA_SCOUT_ADMINS = "Student@miamioh.edu";
    const res = await refresh.POST();
    // 202 (started) or 409 (a run from another test is still open):
    // both prove the admin gate passed.
    expect([202, 409]).toContain(res.status);
    delete process.env.CHATISA_SCOUT_ADMINS;
  });
});
