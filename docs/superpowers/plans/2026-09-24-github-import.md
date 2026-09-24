# Import from GitHub in the Portfolio Builder (part B, v6.9.0) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A student can start a career project or a showcase from one of their own public GitHub repositories: pick it, tick its files, import them as if uploaded, and the page links back to the repository, which is never written to.

**Architecture:** The browser lists the repository's files (`readRepoTree`) and downloads the ticked ones (`fetchRepoFile`, raw bytes) with the student's token, both in `lib/scout/github-read.ts` so the token rule holds. Each download becomes a browser `File` and goes through the existing `prepareFile`, so roles, size limits, notebooks, the size meter and publishing work unchanged. A shared `GithubImport` panel serves the career Projects step and the showcase Files step; the showcase gets a validated `sourceRepoUrl` rendered as "Original repository".

**Tech Stack:** Next.js 16, React 19, Vitest, Playwright + axe.

**Spec:** `docs/superpowers/specs/2026-09-24-github-reading-design.md` (part B; part A shipped as v6.8.0).

## Global Constraints

- The GitHub token never leaves the browser; `js-github-v1` is read only in `lib/scout/github*.ts` (v6.3.0 invariant). Import code calls `github-read.ts` functions; it never builds GitHub requests itself.
- The source repository is only read, never written to.
- Imported files enter the draft through `prepareFile(file, guessRole(name))` and are then indistinguishable from uploads: existing per-file (25 MB push), per-project (10 career files), per-showcase (40 files) and total (100 MB) limits apply unchanged.
- GitHub's contents API does not serve files over 100 MB; such files are listed but cannot be ticked.
- Links back: career projects set the existing `externalUrl` (only when empty); showcases set a new optional `sourceRepoUrl`. Both must match `https://github.com/<owner>/<repo>`; anything else is dropped, not rendered.
- Guests can import (the GitHub connection is available to them).
- Copy: no em dashes; plain words; errors never tell students to reload.
- Accessibility: the panel is an inline disclosure (not a modal) with `aria-expanded`; native radios and checkboxes with labels; progress in a polite live region; errors `role="alert"` and focused; focus returns to the toggle when the panel closes; axe A/AA at 1280 and 320 px, no sideways page scroll.
- Release: one commit `v6.9.0: ...`, annotated tag, `docs/releases/v6.9.0.md`, CHANGELOG, bundle (`rm -rf .next/dev .next/dev/types` first; a stale `.next/dev` cache caused false e2e failures on 2026-09-24).

## Review Focus

1. A repository with hundreds of files: the file list must stay usable (vendored folders hidden, README and main code pre-ticked, a filter), and the student must never exceed the project's remaining file room. Pinned in Task 1 (`importCandidates` room and vendored tests) and Task 3 (e2e counts).
2. Two files with the same name in different folders (`src/utils.py`, `tests/utils.py`): both must import under distinguishable names. Pinned in Task 1 (`importedName`).
3. One file failing to download (deleted since listing, network drop) must not lose the others; the student is told which. An expired token stops once and offers Connect. Pinned in Task 3 (e2e partial failure and expired token).
4. A binary file (PNG figure, PDF report) must import as binary, and a notebook must keep its plots for publishing (as uploads do today). Pinned in Task 2 (`fetchRepoFile` returns bytes) and Task 3 (e2e: a PNG becomes a figure).
5. A crafted or edited `sourceRepoUrl` (not a GitHub repository address, or containing markup) must never reach the published page. Pinned in Task 1 (`githubRepoUrl`) and Task 4 (render test).

---

### Task 1: Import helpers (pure)

**Files:**
- Create: `web/lib/portfolio/github-import.ts`
- Modify: `web/lib/scout/github-summary.ts` (export `VENDORED`)
- Test: `web/tests/unit/portfolio-github-import.test.ts`

**Interfaces:**
- Consumes: `VENDORED` from `lib/scout/github-summary.ts`; `guessRole`, `FileRole` from `lib/portfolio/files.ts`; `PUSH_LIMITS` from `lib/scout/github.ts`.
- Produces:
  - `GITHUB_CONTENTS_MAX_BYTES = 100_000_000`
  - `interface ImportCandidate { path: string; size: number; role: FileRole; selectable: boolean; preselected: boolean; note: string | null }`
  - `importCandidates(tree: { path: string; size: number }[], room: number): ImportCandidate[]`
  - `importedName(path: string, chosen: string[]): string`
  - `githubRepoUrl(raw: string | null | undefined): string | null`

- [ ] **Step 1: Export `VENDORED`** in `lib/scout/github-summary.ts`: change `const VENDORED =` to `export const VENDORED =`.

- [ ] **Step 2: Write the failing tests**

```ts
import { describe, expect, it } from "vitest";
import { GITHUB_CONTENTS_MAX_BYTES, githubRepoUrl, importCandidates, importedName } from "@/lib/portfolio/github-import";

const tree = [
  { path: "README.md", size: 900 },
  { path: "src/model.py", size: 9_000 },
  { path: "src/utils.py", size: 2_000 },
  { path: "tests/utils.py", size: 1_000 },
  { path: "notebooks/eda.ipynb", size: 400_000 },
  { path: "figures/roc.png", size: 50_000 },
  { path: "report/final.pdf", size: 2_000_000 },
  { path: "data/raw.csv", size: 30_000_000 },
  { path: "data/huge.parquet", size: 150_000_000 },
  { path: "node_modules/x/index.js", size: 10 },
  { path: ".git/config", size: 10 },
  { path: ".gitignore", size: 10 },
];

describe("importCandidates", () => {
  it("hides vendored and hidden paths, and pre-ticks the README and code within the room", () => {
    const c = importCandidates(tree, 4);
    expect(c.map((x) => x.path)).not.toContain("node_modules/x/index.js");
    expect(c.map((x) => x.path)).not.toContain(".git/config");
    expect(c.filter((x) => x.preselected).map((x) => x.path)).toEqual(["README.md", "notebooks/eda.ipynb", "src/model.py", "src/utils.py"]);
  });

  it("never pre-ticks data, and never more than the room left", () => {
    const c = importCandidates(tree, 2);
    expect(c.filter((x) => x.preselected)).toHaveLength(2);
    expect(c.find((x) => x.path === "data/raw.csv")?.preselected).toBe(false);
    expect(importCandidates(tree, 0).some((x) => x.preselected)).toBe(false);
  });

  it("lists files GitHub cannot serve as not selectable, with a reason", () => {
    const huge = importCandidates(tree, 10).find((x) => x.path === "data/huge.parquet");
    expect(huge).toMatchObject({ selectable: false, preselected: false });
    expect(huge?.note).toMatch(/over 100 MB/);
    expect(GITHUB_CONTENTS_MAX_BYTES).toBe(100_000_000);
  });

  it("notes files too large to publish; only their size reaches the page", () => {
    const raw = importCandidates(tree, 10).find((x) => x.path === "data/raw.csv");
    expect(raw).toMatchObject({ selectable: true });
    expect(raw?.note).toMatch(/too large to publish/);
  });

  it("guesses roles from names like an upload does", () => {
    const c = importCandidates(tree, 10);
    expect(c.find((x) => x.path === "figures/roc.png")?.role).toBe("figure");
    expect(c.find((x) => x.path === "notebooks/eda.ipynb")?.role).toBe("notebook");
  });
});

describe("importedName", () => {
  it("uses the file name, and the folder too when two chosen files share a name", () => {
    const chosen = ["src/utils.py", "tests/utils.py", "src/model.py"];
    expect(importedName("src/model.py", chosen)).toBe("model.py");
    expect(importedName("src/utils.py", chosen)).toBe("src_utils.py");
    expect(importedName("tests/utils.py", chosen)).toBe("tests_utils.py");
  });
});

describe("githubRepoUrl", () => {
  it("accepts a GitHub repository address and normalizes it", () => {
    expect(githubRepoUrl("https://github.com/ada/churn-model")).toBe("https://github.com/ada/churn-model");
    expect(githubRepoUrl("https://github.com/ada/churn.model/")).toBe("https://github.com/ada/churn.model");
  });
  it("rejects anything else", () => {
    for (const bad of ["http://github.com/ada/x", "https://evil.com/ada/x", "https://github.com/ada", "javascript:alert(1)", "https://github.com/ada/x\"><script>", "", null, undefined]) {
      expect(githubRepoUrl(bad)).toBeNull();
    }
  });
});
```

- [ ] **Step 3: Run to verify it fails**

Run: `cd web && npx vitest run tests/unit/portfolio-github-import.test.ts`
Expected: FAIL, module not found.

- [ ] **Step 4: Implement**

```ts
/**
 * Import from GitHub (v6.9.0): which files of a student's repository to
 * offer, how imported files are named, and the only repository links a
 * page may carry. Pure; the network lives in lib/scout/github-read.ts.
 */

import { guessRole, type FileRole } from "./files";
import { PUSH_LIMITS } from "@/lib/scout/github";
import { VENDORED } from "@/lib/scout/github-summary";

/** GitHub's contents API does not serve larger files (docs.github.com/en/rest/repos/contents). */
export const GITHUB_CONTENTS_MAX_BYTES = 100_000_000;

export interface ImportCandidate {
  path: string;
  size: number;
  role: FileRole;
  /** False when GitHub cannot serve the file at all. */
  selectable: boolean;
  preselected: boolean;
  /** Why a file is limited, in plain words, or null. */
  note: string | null;
}

const HIDDEN = /(^|\/)\.[^/]+/; // .git, .github, .gitignore, .DS_Store ...
const README = /(^|\/)readme(\.[a-z]+)?$/i;
const PRESELECT_ROLES = new Set<FileRole>(["code", "notebook"]);

const mb = (n: number) => `${Math.round(n / 1_000_000)} MB`;

/**
 * The repository's files as import choices: vendored and hidden paths left
 * out; the top-level README, then notebooks and code (largest first)
 * pre-ticked up to `room`; data never pre-ticked (it may be licensed).
 */
export function importCandidates(tree: { path: string; size: number }[], room: number): ImportCandidate[] {
  const listed = tree
    .filter((f) => !VENDORED.test(f.path) && !HIDDEN.test(f.path))
    .map((f): ImportCandidate => {
      const name = f.path.split("/").pop() ?? f.path;
      const role = guessRole(name);
      const selectable = f.size <= GITHUB_CONTENTS_MAX_BYTES;
      const note = !selectable
        ? `${mb(f.size)}: over 100 MB, which GitHub cannot send`
        : f.size > PUSH_LIMITS.fileBytes
          ? `${mb(f.size)}: too large to publish; only its size is used for the page`
          : null;
      return { path: f.path, size: f.size, role, selectable, preselected: false, note };
    });
  const order = [
    ...listed.filter((c) => README.test(c.path) && !c.path.includes("/")),
    ...listed
      .filter((c) => PRESELECT_ROLES.has(c.role))
      .sort((a, b) => (a.role === b.role ? b.size - a.size : a.role === "notebook" ? -1 : 1)),
  ].filter((c) => c.selectable && c.size <= PUSH_LIMITS.fileBytes);
  const pick = new Set(order.slice(0, Math.max(0, room)).map((c) => c.path));
  return listed.map((c) => ({ ...c, preselected: pick.has(c.path) }));
}

/**
 * The draft file name for an imported path: the file name, or folder and
 * name joined when two chosen files share a name, so neither hides the
 * other.
 */
export function importedName(path: string, chosen: string[]): string {
  const base = path.split("/").pop() ?? path;
  const clash = chosen.filter((p) => (p.split("/").pop() ?? p) === base).length > 1;
  return clash ? path.replace(/\//g, "_") : base;
}

/** `https://github.com/<owner>/<repo>` exactly, or null. */
export function githubRepoUrl(raw: string | null | undefined): string | null {
  const m = /^https:\/\/github\.com\/([A-Za-z0-9-]{1,39})\/([A-Za-z0-9._-]{1,100})\/?$/.exec(String(raw ?? "").trim());
  return m ? `https://github.com/${m[1]}/${m[2]}` : null;
}
```

Check `guessRole`'s mapping for `.ipynb` (notebook), `.png` (figure), `.py` (code) and `.pdf` (report) before running; if a name maps differently, adjust the test's expected role to what uploads produce today, since imports must match uploads.

- [ ] **Step 5: Run to verify it passes**

Run: `cd web && npx vitest run tests/unit/portfolio-github-import.test.ts tests/unit/github-summary.test.ts`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add web/lib/portfolio/github-import.ts web/lib/scout/github-summary.ts web/tests/unit/portfolio-github-import.test.ts
git commit -m "feat(portfolio): import helpers for GitHub repositories"
```

### Task 2: Reading a repository's files in the browser

**Files:**
- Modify: `web/lib/scout/github-read.ts` (add `readRepoTree`, `fetchRepoFile`)
- Test: `web/tests/unit/github-read.test.ts` (extend)

**Interfaces:**
- Consumes: `ReadError`, `client`, `asError`, `enc` inside `github-read.ts`.
- Produces:
  - `readRepoTree(conn, listing, fetchImpl?): Promise<{ ok: true; tree: { path: string; size: number }[]; truncated: boolean } | { ok: false; error: ReadError }>`
  - `fetchRepoFile(conn, fullName, path, fetchImpl?): Promise<{ ok: true; bytes: ArrayBuffer } | { ok: false; error: ReadError }>`

- [ ] **Step 1: Write the failing tests** (append to `tests/unit/github-read.test.ts`; `fake`, `json`, `conn`, `listing` already exist there)

```ts
import { fetchRepoFile, readRepoTree } from "@/lib/scout/github-read";

describe("readRepoTree", () => {
  it("lists blobs with sizes on the default branch", async () => {
    const { f, calls } = fake({
      "/repos/ada/churn/git/trees/main": () => json(200, { truncated: true, tree: [{ path: "a.py", type: "blob", size: 5 }, { path: "src", type: "tree" }] }),
    });
    expect(await readRepoTree(conn, listing, f)).toEqual({ ok: true, tree: [{ path: "a.py", size: 5 }], truncated: true });
    expect(calls[0]).toContain("recursive=1");
  });
  it("treats an empty repository as no files, and reports a revoked token", async () => {
    expect(await readRepoTree(conn, listing, fake({ "/repos/ada/churn/git/trees/main": () => json(409, {}) }).f)).toEqual({ ok: true, tree: [], truncated: false });
    expect(await readRepoTree(conn, listing, fake({ "/repos/ada/churn/git/trees/main": () => json(401, {}) }).f)).toEqual({ ok: false, error: { kind: "auth" } });
  });
});

describe("fetchRepoFile", () => {
  it("returns the raw bytes, binary included", async () => {
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47]);
    const { f, calls } = fake({ "/repos/ada/churn/contents/figures/roc%20curve.png": () => new Response(png) });
    const out = await fetchRepoFile(conn, "ada/churn", "figures/roc curve.png", f);
    expect(out.ok && new Uint8Array(out.bytes)).toEqual(png);
    expect(calls[0]).toBe("/repos/ada/churn/contents/figures/roc%20curve.png");
  });
  it("reports a file that has gone as not-found", async () => {
    expect(await fetchRepoFile(conn, "ada/churn", "gone.py", fake({}).f)).toEqual({ ok: false, error: { kind: "not-found" } });
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd web && npx vitest run tests/unit/github-read.test.ts`
Expected: FAIL, `readRepoTree` is not exported.

- [ ] **Step 3: Implement** (append to `lib/scout/github-read.ts`)

```ts
/** A repository's files (blobs) with sizes, for Import from GitHub (v6.9.0). */
export async function readRepoTree(
  conn: GithubConnection,
  listing: RepoListing,
  fetchImpl: typeof fetch = fetch,
): Promise<Result<{ tree: { path: string; size: number }[]; truncated: boolean }>> {
  try {
    const res = await client(conn, fetchImpl).get(
      `/repos/${enc(listing.fullName)}/git/trees/${encodeURIComponent(listing.defaultBranch)}?recursive=1`,
    );
    // 409 is GitHub's answer for an empty repository.
    if (res.status === 409) return { ok: true, tree: [], truncated: false };
    if (!res.ok) return { ok: false, error: await asError(res) };
    const body = (await res.json()) as { truncated?: boolean; tree?: { path: string; type: string; size?: number }[] };
    return {
      ok: true,
      tree: (body.tree ?? []).filter((t) => t.type === "blob").map((t) => ({ path: t.path, size: t.size ?? 0 })),
      truncated: Boolean(body.truncated),
    };
  } catch {
    return { ok: false, error: { kind: "network" } };
  }
}

/** One file's raw bytes (text or binary), read-only. */
export async function fetchRepoFile(
  conn: GithubConnection,
  fullName: string,
  path: string,
  fetchImpl: typeof fetch = fetch,
): Promise<Result<{ bytes: ArrayBuffer }>> {
  try {
    const res = await client(conn, fetchImpl).get(`/repos/${enc(fullName)}/contents/${enc(path)}`, true);
    if (!res.ok) return { ok: false, error: await asError(res) };
    return { ok: true, bytes: await res.arrayBuffer() };
  } catch {
    return { ok: false, error: { kind: "network" } };
  }
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd web && npx vitest run tests/unit/github-read.test.ts`
Expected: PASS (existing tests plus 4 new).

- [ ] **Step 5: Commit**

```bash
git add web/lib/scout/github-read.ts web/tests/unit/github-read.test.ts
git commit -m "feat(scout): read a repository's file list and raw files for import"
```

### Task 3: The import panel, in the career and showcase steps

**Files:**
- Create: `web/components/portfolio/GithubImport.tsx`
- Modify: `web/components/portfolio/career/ProjectsStep.tsx`, `web/components/portfolio/showcase/FilesStep.tsx`
- Modify: `web/tests/e2e/support/fake-github.ts` (contents for README, a PNG, a duplicate name, a vanishing file)
- Create: `web/tests/e2e/github-import.spec.ts`

**Interfaces:**
- Consumes: `useGithubConnection` (`lib/scout/use-scout-store.ts`), `GithubConnect` (`components/scout/GithubConnect.tsx`), `listOwnRepos`, `readRepoTree`, `fetchRepoFile`, `ReadError` (Task 2), `importCandidates`, `importedName`, `githubRepoUrl` (Task 1), `prepareFile` (`lib/portfolio/intake.ts`), `guessRole` (`lib/portfolio/files.ts`).
- Produces: `GithubImport(props: { label: string; room: number; disabled?: boolean; onImport: (files: PreparedFile[], repo: { fullName: string; url: string }) => void })`.

- [ ] **Step 1: Extend the fake GitHub API.** In the read block for `mockstudent` repositories, add contents for `README.md` (text), `figures/roc.png` (4 PNG bytes), `tests/model.py` (duplicate name) and `src/vanishing.py` (404 even though listed), and list them in `churn-model`'s tree:

```ts
      if (rest.startsWith("/git/trees/")) return reply(200, { truncated: false, tree: [
        { path: "README.md", type: "blob", size: 30 },
        { path: "src/model.py", type: "blob", size: 60 },
        { path: "tests/model.py", type: "blob", size: 40 },
        { path: "src/vanishing.py", type: "blob", size: 20 },
        { path: "figures/roc.png", type: "blob", size: 4 },
        ...(name === "churn-model" ? [{ path: "notebooks/eda.ipynb", type: "blob", size: 14_000_000 }] : []),
      ] });
      if (rest === "/contents/README.md") return route.fulfill({ status: 200, body: "# Churn model\nPredicts churn." });
      if (rest === "/contents/tests/model.py") return route.fulfill({ status: 200, body: "def test_fit():\n    assert True" });
      if (rest === "/contents/figures/roc.png") return route.fulfill({ status: 200, contentType: "image/png", body: Buffer.from([0x89, 0x50, 0x4e, 0x47]) });
      if (rest === "/contents/src/vanishing.py") return reply(404, {});
```

(Replace the existing single-line `/git/trees/` answer; keep the existing `/contents/src/model.py` and notebook answers. Check that `tests/e2e/github-skills.spec.ts` still passes: its skill suggestions come from the mock model, not from the tree, so the extra files do not change its expectations.)

- [ ] **Step 2: Write the failing e2e spec** `tests/e2e/github-import.spec.ts`

```ts
import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { fakeGithubApi } from "./support/fake-github";
import { makeTextPdf } from "../helpers/make-pdf";

async function connect(page: Page) {
  await page.evaluate(() => localStorage.setItem("js-github-v1", JSON.stringify({ v: 1, token: "t", login: "mockstudent", connectedAt: "" })));
  await page.reload();
}

async function openCareerProjects(page: Page) {
  await page.goto("/portfolio?mode=career");
  await connect(page);
  // Resume step: a resume is required; Classes step: mark one course.
  await page.locator('input[type="file"]').first().setInputFiles({
    name: "ada-resume.pdf", mimeType: "application/pdf", buffer: Buffer.from(makeTextPdf([["Ada Lovelace, analytics student"]])),
  });
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await page.getByRole("group", { name: /^ISA 225 / }).first().getByRole("radio", { name: "Done" }).check();
  await page.getByRole("button", { name: "Next", exact: true }).click();
  await page.getByRole("button", { name: "Add a project" }).click();
}

test.describe("Import from GitHub", () => {
  test("a career project imports chosen files and links back to the repository", async ({ page }) => {
    await fakeGithubApi(page);
    await openCareerProjects(page);
    const toggle = page.getByRole("button", { name: "Import from GitHub into project 1" });
    await toggle.click();
    await expect(toggle).toHaveAttribute("aria-expanded", "true");
    await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).check();
    // README and code are pre-ticked; the 14 MB notebook is listed.
    await expect(page.getByRole("checkbox", { name: /README\.md/ })).toBeChecked();
    await expect(page.getByRole("checkbox", { name: /src\/model\.py/ })).toBeChecked();
    await page.getByRole("checkbox", { name: /figures\/roc\.png/ }).check();
    await page.getByRole("checkbox", { name: /tests\/model\.py/ }).check();
    await page.getByRole("button", { name: /^Import \d+ files$/ }).click();
    // Both model.py files arrive under distinguishable names; the PNG is a figure.
    await expect(page.getByText("src_model.py")).toBeVisible();
    await expect(page.getByText("tests_model.py")).toBeVisible();
    await expect(page.getByText("roc.png")).toBeVisible();
    await expect(page.getByLabel("Title (optional)").first()).toHaveValue("churn-model");
    await expect(page.getByLabel("Link (repo or demo, optional)").first()).toHaveValue("https://github.com/mockstudent/churn-model");
    await expect(toggle).toBeFocused();
  });

  test("a file that fails to download is named, and the rest still import", async ({ page }) => {
    await fakeGithubApi(page);
    await openCareerProjects(page);
    await page.getByRole("button", { name: "Import from GitHub into project 1" }).click();
    await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).check();
    await page.getByRole("checkbox", { name: /src\/vanishing\.py/ }).check();
    await page.getByRole("button", { name: /^Import \d+ files$/ }).click();
    await expect(page.getByRole("alert")).toContainText("src/vanishing.py could not be read");
    await expect(page.getByText("README.md").first()).toBeVisible();
  });

  test("an expired connection stops and offers Connect GitHub", async ({ page }) => {
    await fakeGithubApi(page, { expireAfterList: true });
    await openCareerProjects(page);
    await page.getByRole("button", { name: "Import from GitHub into project 1" }).click();
    await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).check();
    await expect(page.getByRole("alert")).toContainText("Your GitHub connection has expired. Connect again to continue.");
    await expect(page.getByRole("button", { name: "Connect GitHub" })).toBeVisible();
  });

  test("a showcase imports files and the preview links the original repository", async ({ page }) => {
    await fakeGithubApi(page);
    await page.goto("/portfolio?mode=project");
    await connect(page);
    await page.getByTitle("Principles of Business Analytics").click();
    await page.getByRole("button", { name: "Next", exact: true }).click();
    await page.getByRole("button", { name: "Import from GitHub" }).click();
    await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).check();
    await page.getByRole("button", { name: /^Import \d+ files$/ }).click();
    await expect(page.getByText("README.md").first()).toBeVisible();
    await expect(page.getByRole("link", { name: "https://github.com/mockstudent/churn-model" })).toBeVisible();
  });

  for (const width of [1280, 320]) {
    test(`meets WCAG A and AA with the panel open at ${width}px`, async ({ page }) => {
      await page.setViewportSize({ width, height: 900 });
      await fakeGithubApi(page);
      await openCareerProjects(page);
      await page.getByRole("button", { name: "Import from GitHub into project 1" }).click();
      await page.getByRole("radio", { name: /mockstudent\/churn-model/ }).check();
      await expect(page.getByRole("checkbox", { name: /README\.md/ })).toBeVisible();
      const scan = await new AxeBuilder({ page }).withTags(["wcag2a", "wcag2aa"]).analyze();
      expect(scan.violations).toEqual([]);
      expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
    });
  }
});
```

Before running, check the career wizard's first steps against `tests/e2e/portfolio.spec.ts` (how it reaches the Projects step: the resume step's Next, the Classes checklist, "Add a project") and match `openCareerProjects` to it exactly.

- [ ] **Step 3: Run to verify it fails**

Run: `cd web && rm -rf .next/dev && timeout 590 npx playwright test tests/e2e/github-import.spec.ts --workers=2 --project=desktop`
Expected: FAIL (no "Import from GitHub" button).

- [ ] **Step 4: Implement `GithubImport.tsx`**

```tsx
"use client";

import { useEffect, useId, useRef, useState } from "react";
import { useGithubConnection } from "@/lib/scout/use-scout-store";
import { fetchRepoFile, listOwnRepos, readRepoTree, type ReadError } from "@/lib/scout/github-read";
import type { RepoListing } from "@/lib/scout/github-summary";
import { githubRepoUrl, importCandidates, importedName, type ImportCandidate } from "@/lib/portfolio/github-import";
import { guessRole } from "@/lib/portfolio/files";
import type { PreparedFile } from "@/lib/portfolio/files";
import { prepareFile } from "@/lib/portfolio/intake";
import { GithubConnect } from "@/components/scout/GithubConnect";

/**
 * Import from GitHub (v6.9.0): pick one of your public repositories, tick
 * its files, and they arrive as if uploaded. The repository is only read;
 * the page links back to it. An inline disclosure, not a modal.
 */

const EXPIRED = "Your GitHub connection has expired. Connect again to continue.";
const SHOWN = 10;

function problem(e: ReadError, what: string): string {
  if (e.kind === "not-found") return `${what} could not be read. It may be private, renamed or deleted.`;
  if (e.kind === "rate-limit") return "GitHub asked us to slow down. Try again in a few minutes.";
  if (e.kind === "network") return `${what} could not be reached. Check your connection and try again.`;
  return `${what} could not be read. Try again later.`;
}

const size = (n: number) => (n >= 1_000_000 ? `${(n / 1_000_000).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1000))} KB`);

export function GithubImport(props: {
  label: string;
  room: number;
  disabled?: boolean;
  onImport: (files: PreparedFile[], repo: { fullName: string; url: string }) => void;
}) {
  const uid = useId();
  const { connection, clear } = useGithubConnection();
  const [open, setOpen] = useState(false);
  const [repos, setRepos] = useState<RepoListing[] | null>(null);
  const [filter, setFilter] = useState("");
  const [repo, setRepo] = useState<RepoListing | null>(null);
  const [candidates, setCandidates] = useState<ImportCandidate[] | null>(null);
  const [ticked, setTicked] = useState<string[]>([]);
  const [status, setStatus] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const toggleRef = useRef<HTMLButtonElement>(null);
  const errorRef = useRef<HTMLParagraphElement>(null);

  const fail = (message: string) => {
    setError(message);
    setTimeout(() => errorRef.current?.focus(), 0);
  };
  const expire = () => {
    clear();
    setRepos(null);
    setRepo(null);
    setCandidates(null);
    fail(EXPIRED);
  };

  useEffect(() => {
    if (!open || !connection) return;
    let live = true;
    void (async () => {
      const out = await listOwnRepos(connection);
      if (!live) return;
      setError(null);
      if (out.ok) setRepos(out.repos);
      else if (out.error.kind === "auth") expire();
      else fail("Your repositories could not be listed. Try again in a few minutes.");
    })();
    return () => { live = false; };
    // expire and fail only use state setters and the store's stable clear.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, connection]);

  async function choose(r: RepoListing) {
    if (!connection) return;
    setRepo(r);
    setCandidates(null);
    setError(null);
    setStatus(`Listing the files in ${r.fullName}.`);
    const out = await readRepoTree(connection, r);
    if (!out.ok) {
      if (out.error.kind === "auth") return expire();
      return fail(problem(out.error, r.fullName));
    }
    const list = importCandidates(out.tree, props.room);
    setCandidates(list);
    setTicked(list.filter((c) => c.preselected).map((c) => c.path));
    setStatus(`${list.length} files listed${out.truncated ? " (GitHub shortened the list for this large repository)" : ""}.`);
  }

  function close() {
    setOpen(false);
    setTimeout(() => toggleRef.current?.focus(), 0);
  }

  async function doImport() {
    if (!connection || !repo || ticked.length === 0) return;
    setBusy(true);
    setError(null);
    const files: PreparedFile[] = [];
    const failed: string[] = [];
    for (const [i, path] of ticked.entries()) {
      setStatus(`Reading ${i + 1} of ${ticked.length}: ${path}`);
      const out = await fetchRepoFile(connection, repo.fullName, path);
      if (!out.ok) {
        if (out.error.kind === "auth") { setBusy(false); return expire(); }
        failed.push(path);
        continue;
      }
      const name = importedName(path, ticked);
      try {
        files.push(await prepareFile(new File([out.bytes], name), guessRole(name)));
      } catch {
        failed.push(path);
      }
    }
    setBusy(false);
    const url = githubRepoUrl(repo.htmlUrl) ?? `https://github.com/${repo.fullName}`;
    if (files.length) props.onImport(files, { fullName: repo.fullName, url });
    if (failed.length) {
      fail(`${failed.join(", ")} could not be read. The other files were imported.`);
      return;
    }
    setStatus(`Imported ${files.length} files from ${repo.fullName}.`);
    close();
  }

  const q = filter.trim().toLowerCase();
  const visible = (repos ?? []).filter((r) => !q || r.fullName.toLowerCase().includes(q));
  const panelId = `${uid}-panel`;
  const room = Math.max(0, props.room);

  return (
    <div className="mt-3">
      <button
        ref={toggleRef} type="button" aria-expanded={open} aria-controls={panelId}
        disabled={props.disabled}
        onClick={() => (open ? close() : setOpen(true))}
        className="rounded-card border-2 border-miami-red px-4 py-2 font-bold text-miami-red hover:bg-light-tan disabled:border-medium-gray disabled:text-medium-gray"
      >
        {props.label}
      </button>
      <div id={panelId} hidden={!open} className="mt-2 min-w-0 rounded-card border border-medium-tan bg-paper p-3">
        {error ? (
          <p ref={errorRef} role="alert" tabIndex={-1} className="mb-2 rounded-card border-2 border-miami-red p-2 font-bold text-miami-red">{error}</p>
        ) : null}
        {!connection ? (
          <GithubConnect returnPath="/portfolio" />
        ) : (
          <>
            <p className="text-dark-tan">Your repository is only read, never changed. The page links back to it.</p>
            {repos === null && !error ? <p role="status" className="mt-2">Loading your repositories...</p> : null}
            {repos && repos.length === 0 ? <p className="mt-2">You have no public repositories of your own yet.</p> : null}
            {repos && repos.length > SHOWN ? (
              <label className="mt-2 block max-w-md">
                <span className="block">Filter repositories</span>
                <input type="search" value={filter} onChange={(e) => setFilter(e.target.value)} className="mt-1 w-full rounded-card border border-medium-tan p-2" />
              </label>
            ) : null}
            {repos && repos.length > 0 ? (
              <fieldset className="mt-2 min-w-0">
                <legend className="font-bold">Repository</legend>
                <ul className="mt-1 space-y-1">
                  {(q ? visible : visible.slice(0, 50)).map((r) => (
                    <li key={r.fullName}>
                      <label className="flex items-start gap-2">
                        <input type="radio" name={`${uid}-repo`} className="mt-1 accent-miami-red" checked={repo?.fullName === r.fullName} disabled={busy} onChange={() => void choose(r)} />
                        <span className="min-w-0 break-all"><strong>{r.fullName}</strong>{r.language ? ` (${r.language})` : ""}</span>
                      </label>
                    </li>
                  ))}
                </ul>
              </fieldset>
            ) : null}
            {candidates ? (
              <fieldset className="mt-3 min-w-0">
                <legend className="font-bold">Files to import ({ticked.length} of up to {room})</legend>
                {candidates.length === 0 ? <p>This repository has no files yet.</p> : (
                  <ul className="mt-1 max-h-80 space-y-1 overflow-auto">
                    {candidates.map((c) => {
                      const on = ticked.includes(c.path);
                      return (
                        <li key={c.path}>
                          <label className="flex items-start gap-2">
                            <input
                              type="checkbox" className="mt-1 accent-miami-red" checked={on}
                              disabled={busy || !c.selectable || (!on && ticked.length >= room)}
                              onChange={() => setTicked(on ? ticked.filter((p) => p !== c.path) : [...ticked, c.path])}
                            />
                            <span className="min-w-0 break-all">
                              {c.path} <span className="text-dark-tan">({size(c.size)}, {c.role})</span>
                              {c.note ? <span className="block text-dark-tan">{c.note}</span> : null}
                            </span>
                          </label>
                        </li>
                      );
                    })}
                  </ul>
                )}
                {room === 0 ? <p className="mt-2">This project already has as many files as it can hold. Remove some to import more.</p> : null}
                <button
                  type="button" disabled={busy || ticked.length === 0} onClick={() => void doImport()}
                  className="mt-3 rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:bg-medium-gray"
                >
                  {busy ? "Importing..." : `Import ${ticked.length} ${ticked.length === 1 ? "file" : "files"}`}
                </button>
              </fieldset>
            ) : null}
          </>
        )}
        <p aria-live="polite" className="sr-only">{status}</p>
      </div>
    </div>
  );
}
```

(The e2e matcher `/^Import \d+ files$/` assumes more than one file is ticked; the pre-ticked README plus code makes that true in the fake repository.)

In `ProjectsStep.tsx`, directly after the "Add files" `</label>` of each project, add:

```tsx
            <GithubImport
              label={`Import from GitHub into project ${i + 1}`}
              room={MAX_PROJECT_FILES - p.files.length}
              disabled={busy !== null}
              onImport={(files, repo) =>
                update(i, {
                  files: [...p.files, ...files].slice(0, MAX_PROJECT_FILES),
                  title: p.title.trim() ? p.title : repo.fullName.split("/")[1],
                  externalUrl: p.externalUrl.trim() ? p.externalUrl : repo.url,
                })
              }
            />
```

In `FilesStep.tsx`, directly after the "Add files" `</label>`, add:

```tsx
      <GithubImport
        label="Import from GitHub"
        room={MAX_SHOWCASE_FILES - draft.files.length}
        disabled={busy}
        onImport={(files, repo) =>
          patch({ files: [...draft.files, ...files].slice(0, MAX_SHOWCASE_FILES), sourceRepoUrl: repo.url })
        }
      />
      {draft.sourceRepoUrl ? (
        <p className="mt-2">
          Original repository:{" "}
          <a href={draft.sourceRepoUrl} className="underline" target="_blank" rel="noreferrer">{draft.sourceRepoUrl}</a>{" "}
          <button type="button" className="underline" onClick={() => patch({ sourceRepoUrl: undefined })}>Remove link</button>
        </p>
      ) : null}
```

Import `GithubImport` from `../GithubImport` in both files. Add the optional `sourceRepoUrl?: string` field to `Draft` in `lib/portfolio/draft.ts` in this task (Task 4 carries it through rendering and storage).

- [ ] **Step 5: Run to verify it passes**

Run: `cd web && npx tsc --noEmit -p . && npx eslint components/portfolio && timeout 590 npx playwright test tests/e2e/github-import.spec.ts tests/e2e/github-skills.spec.ts --workers=2`
Expected: all pass on desktop and mobile-320.

- [ ] **Step 6: Commit**

```bash
git add web/components/portfolio web/tests/e2e/github-import.spec.ts web/tests/e2e/support/fake-github.ts
git commit -m "feat(portfolio): Import from GitHub for career projects and showcases"
```

### Task 4: The showcase links back to its repository

**Files:**
- Modify: `web/lib/portfolio/draft.ts` (`sourceRepoUrl?: string`), `web/lib/portfolio/store.ts` (`ShowcaseMeta.sourceRepoUrl?: string`), `web/lib/portfolio/html.ts` (`renderShowcase` meta), `web/components/portfolio/ReviewStep.tsx`, `web/components/portfolio/showcase/StoryStep.tsx`, `web/lib/portfolio/publish-plan.ts`, `web/components/portfolio/Publish.tsx`, `web/components/portfolio/PortfolioBuilder.tsx`
- Test: `web/tests/unit/portfolio-html.test.ts` (extend)

**Interfaces:**
- Consumes: `githubRepoUrl` (Task 1).
- Produces: `renderShowcase(content, meta & { sourceRepoUrl?: string | null })`.

- [ ] **Step 1: Write the failing render tests** (append inside the `renderShowcase` describe in `tests/unit/portfolio-html.test.ts`; reuse that file's `content` fixture)

```ts
  it("links the original repository when the showcase was imported from GitHub (v6.9.0)", () => {
    const html = renderShowcase(content, { course: "ISA 401", semester: "", team: [], repoUrl: null, figures: [], deliverablePaths: [], sourceRepoUrl: "https://github.com/ada/churn-model" });
    expect(html).toContain('<a href="https://github.com/ada/churn-model"');
    expect(html).toContain(">Original repository</a>");
  });

  it("never renders a source link that is not a GitHub repository address", () => {
    for (const bad of ["javascript:alert(1)", "https://evil.com/x/y", "https://github.com/ada/x\"><script>alert(1)</script>"]) {
      const html = renderShowcase(content, { course: "ISA 401", semester: "", team: [], repoUrl: null, figures: [], deliverablePaths: [], sourceRepoUrl: bad });
      expect(html).not.toContain("Original repository");
      expect(html).not.toContain("<script>alert");
    }
  });
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd web && npx vitest run tests/unit/portfolio-html.test.ts`
Expected: FAIL (no "Original repository").

- [ ] **Step 3: Implement.** In `renderShowcase`'s meta type add `/** The GitHub repository the showcase was imported from (v6.9.0). */ sourceRepoUrl?: string | null;`, and in `metaParts` after the `Repository` link:

```ts
    githubRepoUrl(meta.sourceRepoUrl) ? link(githubRepoUrl(meta.sourceRepoUrl)!, "Original repository") : "",
```

importing `githubRepoUrl` from `./github-import`. (A ruling against the spec's wording "under the files": the header's meta line already carries the "Repository" link, and "Original repository" beside it is where a reader looks; the render test pins it.)

Then carry the field everywhere a showcase is rendered or stored:
- `lib/portfolio/draft.ts`: `sourceRepoUrl?: string;` on `Draft` with a one-line comment ("Absent before v6.9.0").
- `lib/portfolio/store.ts`: `sourceRepoUrl?: string` on `ShowcaseMeta`.
- `ReviewStep.tsx`, `StoryStep.tsx`, `publish-plan.ts`: pass `sourceRepoUrl: draft.sourceRepoUrl ?? null` to `renderShowcase`; add `draft.sourceRepoUrl` to the `ReviewStep` `useMemo` dependency list.
- `Publish.tsx`: include `sourceRepoUrl: props.draft.sourceRepoUrl` in `showcaseMeta`.
- `PortfolioBuilder.tsx`: restore `sourceRepoUrl: stored.showcaseMeta?.sourceRepoUrl` when reopening a site.

- [ ] **Step 4: Run to verify it passes**

Run: `cd web && npx vitest run tests/unit/portfolio-*.test.ts && npx tsc --noEmit -p . && npx eslint components/portfolio lib/portfolio`
Expected: PASS; tsc and lint clean.

- [ ] **Step 5: Commit**

```bash
git add web/lib/portfolio web/components/portfolio web/tests/unit/portfolio-html.test.ts
git commit -m "feat(portfolio): a showcase imported from GitHub links its original repository"
```

### Task 5: Review and release v6.9.0

**Files:**
- Create: `docs/releases/v6.9.0.md`
- Modify: `docs/CHANGELOG.md`, `web/package.json`, `web/package-lock.json`

- [ ] **Step 1: Whole-branch review.** Build the review package and dispatch a fresh reviewer on the most capable model with this plan's Review Focus. Fix Critical and Important findings test-first (prove each new e2e assertion red against the pre-fix component, then clear `.next/dev` before the green run); ledger minors.
- [ ] **Step 2: Full suites.** `npx vitest run tests/unit` (speech-probe and indent flakes rerun alone), then e2e in foreground groups per the test-gotchas memory, desktop and mobile-320, clearing `.next/dev` first.
- [ ] **Step 3: Live read-only check.** Import `fmegahed/chatsqc` into a career project and a showcase from the dev server using the professor's approved repositories: in a local Playwright script, seed `js-github-v1` with `gh auth token` (read in-process only, never written to disk) and use the real api.github.com; confirm the files arrive and no request carries the token to a ChatISA route.
- [ ] **Step 4: Browser check.** Screenshots of the open panel at 1280 and 320 px; read them.
- [ ] **Step 5: Release.** Bump to 6.9.0, write `docs/releases/v6.9.0.md` and the CHANGELOG entry, `rm -rf .next/dev && node scripts/make-deploy-bundle.mjs`, verify the footer, squash onto main as `v6.9.0: Import from GitHub in the Portfolio Builder`, annotated tag, `git push --follow-tags origin main`.
