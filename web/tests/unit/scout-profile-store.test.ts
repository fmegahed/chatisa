import { beforeEach, describe, expect, it } from "vitest";
import {
  addProject,
  hidePosting,
  loadProfile,
  loadProjects,
  loadSaved,
  projectExtras,
  removeProject,
  saveProfile,
  setProjectRepoUrl,
  toggleSaved,
} from "@/lib/scout/profile-store";
import { POPULAR_CODES, getCourse } from "@/lib/scout/courses";

/** Minimal in-memory Storage for tests (the ask-chat-store pattern). */
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
});

describe("profile store", () => {
  it("round-trips a profile", () => {
    expect(loadProfile()).toBeNull();
    saveProfile({
      v: 1,
      courses: ["ISA 241", "ISA 401"],
      extras: [{ skillId: "tableau", level: "applied", source: "resume" }],
    });
    const loaded = loadProfile();
    expect(loaded?.courses).toEqual(["ISA 241", "ISA 401"]);
    expect(loaded?.extras[0]?.skillId).toBe("tableau");
  });

  it("degrades corrupt JSON to no profile, never a crash", () => {
    localStorage.setItem("js-profile-v1", "{not json");
    expect(loadProfile()).toBeNull();
    localStorage.setItem("js-profile-v1", JSON.stringify({ v: 99 }));
    expect(loadProfile()).toBeNull();
  });
});

describe("saved store (v2 snapshots)", () => {
  const snap = {
    id: "a",
    title: "Data Analyst",
    company: "Acme",
    applyUrl: "https://x.example/a",
  };

  it("toggles saves with snapshots so retirement cannot orphan them", () => {
    const after = toggleSaved(snap);
    expect(after.saved[0]).toMatchObject(snap);
    expect(after.saved[0].savedAt.length).toBeGreaterThan(0);
    expect(toggleSaved(snap).saved).toEqual([]);
  });

  it("migrates v1 bare ids into placeholder snapshots", () => {
    localStorage.setItem(
      "js-saved-v1",
      JSON.stringify({ v: 1, savedIds: ["old-1"], hiddenIds: ["h-1"] }),
    );
    const state = loadSaved();
    expect(state.v).toBe(2);
    expect(state.saved[0]).toMatchObject({ id: "old-1", title: "Saved posting" });
    expect(state.hiddenIds).toEqual(["h-1"]);
  });

  it("hiding removes a save", () => {
    toggleSaved(snap);
    const after = hidePosting("a");
    expect(after.hiddenIds).toEqual(["a"]);
    expect(after.saved).toEqual([]);
  });
});

describe("projects store", () => {
  const record = {
    id: "p1",
    repoName: "retail-demand-analytics",
    summary: "SQL and viz on public retail data",
    skillIds: ["sql", "data_visualization"],
    createdAt: "2026-07-29T00:00:00.000Z",
    repoUrl: null,
  };

  it("adds, updates repo URL, and removes", () => {
    addProject(record);
    expect(loadProjects().projects).toHaveLength(1);
    setProjectRepoUrl("p1", "https://github.com/student/retail");
    expect(loadProjects().projects[0].repoUrl).toContain("github.com");
    removeProject("p1");
    expect(loadProjects().projects).toEqual([]);
  });

  it("polished projects count immediately: they are real, existing work", () => {
    addProject({
      ...record,
      id: "p2",
      mode: "polished",
    });
    const extras = projectExtras(loadProjects());
    expect(extras.map((e) => e.skillId).sort()).toEqual([
      "data_visualization",
      "sql",
    ]);
    removeProject("p2");
  });

  it("contributes skills to the profile ONLY once a repo URL exists", () => {
    addProject(record);
    expect(projectExtras(loadProjects())).toEqual([]);
    setProjectRepoUrl("p1", "https://github.com/student/retail");
    const extras = projectExtras(loadProjects());
    expect(extras.map((e) => e.skillId).sort()).toEqual([
      "data_visualization",
      "sql",
    ]);
    expect(extras[0].level).toBe("applied");
    expect(extras[0].evidence).toContain("retail-demand-analytics");
  });
});

describe("popular course subsets", () => {
  it("every popular code exists in the catalog (instructor list, 2026-07-29)", () => {
    for (const codes of Object.values(POPULAR_CODES)) {
      for (const code of codes) {
        expect(getCourse(code)?.code, code).toBe(code);
      }
    }
  });
});
