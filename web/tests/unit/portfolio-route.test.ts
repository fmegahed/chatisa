import { afterAll, describe, expect, it, vi } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const dataDir = mkdtempSync(path.join(tmpdir(), "chatisa-portfolio-route-"));
process.env.CHATISA_DATA_DIR = dataDir;
process.env.CHATISA_MOCK_LLM = "1";
// The route allows 4 generations a minute per student; this file makes more
// than that, and the limiter is not what these tests are about.
process.env.CHATISA_PORTFOLIO_LIMIT_PER_MINUTE = "50";

let sessionEmail: string | null = "student@miamioh.edu";
vi.mock("@/lib/auth", () => ({
  auth: async () => (sessionEmail ? { user: { email: sessionEmail, name: "Test Student" } } : null),
}));

const { closeDb } = await import("@/lib/db");
const { getPageModels } = await import("@/lib/config/models");
const route = await import("@/app/api/portfolio/generate/route");

afterAll(() => {
  closeDb();
  rmSync(dataDir, { recursive: true, force: true });
  delete process.env.CHATISA_MOCK_LLM;
  delete process.env.CHATISA_PORTFOLIO_LIMIT_PER_MINUTE;
});

function request(mode: string, payload: unknown, modelId = getPageModels("portfolio")[0]) {
  const form = new FormData();
  form.append("modelId", modelId);
  form.append("mode", mode);
  form.append("payload", JSON.stringify(payload));
  return new Request("http://localhost/api/portfolio/generate", { method: "POST", body: form });
}

describe("POST /api/portfolio/generate", () => {
  it("401s without a session", async () => {
    sessionEmail = null;
    const res = await route.POST(request("career", {}));
    expect(res.status).toBe(401);
    sessionEmail = "student@miamioh.edu";
  });

  it("generates career content and keeps only submitted project slugs", async () => {
    const res = await route.POST(request("career", {
      student: { name: "Ada", links: [] }, courses: ["ISA 401"],
      projects: [{ slug: "churn", title: "Churn", externalUrl: null, files: [{ kind: "text", name: "model.R", content: "lm(y~x)" }] }],
    }));
    expect(res.status).toBe(200);
    const body = (await res.json()) as { content: { v: number; projects: { slug: string }[] } };
    expect(body.content.v).toBe(2);
    for (const p of body.content.projects) expect(p.slug).toBe("churn");
    // The mock adds a project the student never submitted, so an empty array
    // here would mean the filter ate everything rather than only the extra.
    expect(body.content.projects).toHaveLength(1);
    expect(body.content.projects[0].slug).toBe("churn");
  });

  it("generates showcase content with figures and deliverables limited to published paths", async () => {
    const res = await route.POST(request("showcase", {
      course: "ISA 401", semester: "Spring 2026", team: [],
      prompts: { problem: "", hardest: "", next: "" },
      files: [
        { kind: "text", name: "model.R", role: "code", content: "lm(y~x)" },
        { kind: "binary", name: "roc.png", role: "figure", sizeBytes: 10 },
      ],
      publishedPaths: ["code/model.R", "figures/roc.png"],
    }));
    expect(res.status).toBe(200);
    const body = (await res.json()) as {
      content: { v: number; findings: { figure: string | null }[]; deliverables: { path: string }[] };
      readme: string; skillIds: string[];
    };
    expect(body.content.v).toBe(1);
    for (const f of body.content.findings) expect(f.figure === null || f.figure === "figures/roc.png").toBe(true);
    for (const d of body.content.deliverables) expect(["code/model.R", "figures/roc.png"]).toContain(d.path);
    expect(typeof body.readme).toBe("string");
    // The mock deliberately returns one unpublished deliverable, one finding
    // pointing at a figure nobody uploaded, and one skill outside the
    // taxonomy, so these assert the filters drop exactly those and no more.
    expect(body.content.deliverables.map((d) => d.path).sort()).toEqual(["code/model.R", "figures/roc.png"]);
    expect(body.content.findings.some((f) => f.figure === "figures/roc.png")).toBe(true);
    expect(body.content.findings.some((f) => f.figure === "figures/not-uploaded.png")).toBe(false);
    expect(body.skillIds).toEqual(["r", "sql"]);
  });

  it("keeps a typed link out of the way instead of failing the whole request", async () => {
    // A bare domain is what students type. It is not a URL, so it is dropped
    // (the project keeps its content) rather than 400ing everything they
    // filled in, and the link the page shows is the student's, not one the
    // model invented.
    const res = await route.POST(request("career", {
      student: { name: "Ada", links: [{ label: "LinkedIn", url: "linkedin.com/in/ada" }] },
      courses: [],
      projects: [{ slug: "churn", title: "Churn", externalUrl: "github.com/ada/churn", files: [] }],
    }));
    expect(res.status).toBe(200);
    const body = (await res.json()) as { content: { projects: { slug: string; externalUrl: string | null }[] } };
    expect(body.content.projects).toHaveLength(1);
    expect(body.content.projects[0].slug).toBe("churn");
    expect(body.content.projects[0].externalUrl).toBeNull();
  });

  it("pins a project link to the submitted url, never the model's", async () => {
    const res = await route.POST(request("career", {
      student: { name: "Ada", links: [] }, courses: [],
      projects: [{ slug: "churn", title: "Churn", externalUrl: "https://github.com/ada/churn", files: [] }],
    }));
    expect(res.status).toBe(200);
    const body = (await res.json()) as { content: { projects: { externalUrl: string | null }[] } };
    expect(body.content.projects[0].externalUrl).toBe("https://github.com/ada/churn");
  });

  it("400s on a malformed payload", async () => {
    const res = await route.POST(request("career", { student: 5 }));
    expect(res.status).toBe(400);
  });
});
