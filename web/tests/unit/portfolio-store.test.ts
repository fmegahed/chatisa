import { beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The draft side of the store lives in IndexedDB (lib/scout/device-files),
 * which jsdom does not provide, so the four primitives are stubbed with a
 * map: the migration under test is about what it reads, writes, and deletes.
 */
const idb = new Map<string, unknown>();
vi.mock("@/lib/scout/device-files", () => ({
  getItem: async (id: string) => idb.get(id) ?? null,
  putItem: async (id: string, value: unknown) => {
    idb.set(id, value);
    return true;
  },
  removeItem: async (id: string) => void idb.delete(id),
}));

const { careerSite, getDraft, loadSites, migrateJobScoutPortfolio, removeSite, upsertSite } =
  await import("@/lib/portfolio/store");
const { loadPublished, removePublished, subscribePublished, upsertPublished } =
  await import("@/lib/portfolio/published");

/** Minimal in-memory Storage for tests (the ask-chat-store / scout-profile-store pattern). */
function memoryStorage(): Storage {
  const map = new Map<string, string>();
  return {
    get length() {
      return map.size;
    },
    clear: () => map.clear(),
    getItem: (k: string) => map.get(k) ?? null,
    key: (i: number) => [...map.keys()][i] ?? null,
    removeItem: (k: string) => void map.delete(k),
    setItem: (k: string, v: string) => void map.set(k, v),
  } as Storage;
}

beforeEach(() => {
  (globalThis as { localStorage?: Storage }).localStorage = memoryStorage();
  idb.clear();
});

describe("site records", () => {
  it("upserts by id and keeps one career site", () => {
    upsertSite({ v: 1, id: "a", kind: "career", title: "Me", repoName: "portfolio", repoUrl: null, pagesUrl: null, generatedAt: "t", publishedAt: null });
    upsertSite({ v: 1, id: "a", kind: "career", title: "Me 2", repoName: "portfolio", repoUrl: null, pagesUrl: null, generatedAt: "t", publishedAt: null });
    expect(loadSites()).toHaveLength(1);
    expect(careerSite()?.title).toBe("Me 2");
    removeSite("a");
    expect(loadSites()).toEqual([]);
  });
  it("degrades corrupt JSON to an empty list", () => {
    localStorage.setItem("pb-sites-v1", "{nope");
    expect(loadSites()).toEqual([]);
  });
  it("keeps one career site: a second career build replaces the first", () => {
    upsertSite({ v: 1, id: "a", kind: "career", title: "Me", repoName: "portfolio", repoUrl: null, pagesUrl: null, generatedAt: "t", publishedAt: null });
    upsertSite({ v: 1, id: "b", kind: "showcase", title: "Churn", repoName: "isa-401-churn", repoUrl: null, pagesUrl: null, generatedAt: "t", publishedAt: null });
    upsertSite({ v: 1, id: "c", kind: "career", title: "Me again", repoName: "portfolio", repoUrl: null, pagesUrl: null, generatedAt: "t", publishedAt: null });
    expect(loadSites().filter((s) => s.kind === "career")).toHaveLength(1);
    expect(careerSite()?.id).toBe("c");
    // Showcase sites still accumulate.
    expect(loadSites().some((s) => s.id === "b")).toBe(true);
  });
  it("migrates the Job Scout v6.3.0 portfolio record once, with its draft", async () => {
    localStorage.setItem("js-portfolio-v1", JSON.stringify({
      v: 1, repoName: "portfolio", repoUrl: "https://github.com/a/portfolio",
      pagesUrl: "https://a.github.io/portfolio/", generatedAt: "g", publishedAt: "p", jobIds: [],
    }));
    idb.set("portfolio", {
      html: "<!doctype html><p>old</p>",
      content: {
        siteTitle: "Ada", headline: "Analytics student", about: "Hello",
        skillGroups: [{ title: "Tools", skills: ["R"] }],
        projectCards: [{ repoName: "churn-model", title: "Churn", blurb: "b", skillLabels: ["R"], repoUrl: "https://github.com/a/churn-model" }],
        courseHighlights: [{ course: "ISA 401", why: "w" }],
      },
    });
    const id = await migrateJobScoutPortfolio();
    const site = careerSite();
    expect(site?.repoUrl).toBe("https://github.com/a/portfolio");
    expect(site?.pagesUrl).toBe("https://a.github.io/portfolio/");
    expect(id).toBe(site?.id);
    expect(localStorage.getItem("js-portfolio-v1")).toBeNull();
    // The v1 content came across as v2 and the legacy record is gone, so the
    // student can open and edit the site rather than only look at it.
    const draft = await getDraft(id as string);
    expect(draft?.content.kind).toBe("career");
    expect(draft?.content.content.v).toBe(2);
    expect(draft?.html).toBe("<!doctype html><p>old</p>");
    expect(idb.has("portfolio")).toBe(false);
    await migrateJobScoutPortfolio();
    expect(loadSites()).toHaveLength(1);
  });
  it("drops a legacy draft it cannot migrate rather than keeping it forever", async () => {
    localStorage.setItem("js-portfolio-v1", JSON.stringify({ repoName: "portfolio" }));
    idb.set("portfolio", { content: { nope: true } });
    const id = await migrateJobScoutPortfolio();
    expect(await getDraft(id as string)).toBeNull();
    expect(idb.has("portfolio")).toBe(false);
  });
});

describe("published work", () => {
  it("round-trips and notifies subscribers", () => {
    let calls = 0;
    const off = subscribePublished(() => { calls++; });
    upsertPublished({ id: "s1", kind: "showcase", title: "Churn", summary: "s", skillIds: ["r"], repoUrl: "https://github.com/a/isa-401-churn", pagesUrl: null, publishedAt: "p" });
    expect(loadPublished()).toHaveLength(1);
    expect(calls).toBe(1);
    removePublished("s1");
    expect(loadPublished()).toEqual([]);
    off();
  });
});
