import type { Page } from "@playwright/test";

/**
 * An in-test GitHub API. Both push engines (Job Scout's project scaffold and
 * the Portfolio Builder's publish) run in the browser against
 * api.github.com, so the e2e suite intercepts that origin and answers like
 * GitHub would. No test traffic ever reaches the real GitHub.
 *
 * The returned handle records every tree that was pushed, so a test can
 * assert the exact file set a publish sent rather than only that it
 * succeeded. `trees.at(-1)` is the most recent push.
 */
export async function fakeGithubApi(
  page: Page,
  opts: { expireAfterList?: boolean } = {},
): Promise<{ trees: { path: string }[][]; setExpired: (value: boolean) => void }> {
  // Mutable, so a test can expire the token and then "reconnect".
  let expired = Boolean(opts.expireAfterList);
  const repos = new Set<string>();
  const trees: { path: string }[][] = [];
  await page.route("https://api.github.com/**", async (route) => {
    const req = route.request();
    const path = new URL(req.url()).pathname;
    const method = req.method();
    const reply = (status: number, body: unknown) =>
      route.fulfill({
        status,
        contentType: "application/json",
        body: JSON.stringify(body),
      });

    if (method === "POST" && path === "/user/repos") {
      const name = (JSON.parse(req.postData() ?? "{}") as { name: string }).name;
      repos.add(name);
      return reply(201, { default_branch: "main" });
    }
    if (method === "GET" && path === "/user/repos") {
      return reply(200, [
        { full_name: "mockstudent/churn-model", description: "Predicting customer churn", language: "Python", pushed_at: "2026-09-20T00:00:00Z", default_branch: "main", html_url: "https://github.com/mockstudent/churn-model", fork: false, archived: false, private: false },
        { full_name: "mockstudent/class-notes", description: "Notes", language: "R", pushed_at: "2026-09-10T00:00:00Z", default_branch: "main", html_url: "https://github.com/mockstudent/class-notes", fork: false, archived: false, private: false },
        { full_name: "mockstudent/gone", description: null, language: null, pushed_at: "2026-01-01T00:00:00Z", default_branch: "main", html_url: "https://github.com/mockstudent/gone", fork: false, archived: false, private: false },
        { full_name: "mockstudent/forked-lib", fork: true, archived: false, private: false },
      ]);
    }
    const read = /^\/repos\/mockstudent\/(churn-model|class-notes|gone)(\/.*)?$/.exec(path);
    if (method === "GET" && read) {
      if (expired) return reply(401, {});
      const [, name, rest = ""] = read;
      if (name === "gone") return reply(404, {});
      const commits = name === "churn-model" ? 20 : 3;
      if (rest === "") return reply(200, { description: name, topics: [], fork: false, archived: false });
      if (rest === "/languages") return reply(200, name === "churn-model" ? { Python: 9000 } : { R: 500 });
      if (rest.startsWith("/contributors")) return reply(200, [{ login: "mockstudent", type: "User", contributions: commits }, ...(name === "class-notes" ? [{ login: "classmate", type: "User", contributions: 30 }] : [])]);
      if (rest.startsWith("/git/trees/")) return reply(200, { truncated: false, tree: [{ path: "README.md", type: "blob", size: 30 }, { path: "src/model.py", type: "blob", size: 60 }, ...(name === "churn-model" ? [{ path: "notebooks/eda.ipynb", type: "blob", size: 14_000_000 }] : [])] });
      if (rest === "/contents/notebooks/eda.ipynb") return route.fulfill({ status: 200, body: JSON.stringify({ cells: [{ cell_type: "code", source: ["import seaborn as sns"], outputs: [] }] }) });
      if (rest === "/readme") return route.fulfill({ status: 200, body: "# Churn model" });
      if (rest === "/contents/src/model.py") return route.fulfill({ status: 200, body: "import pandas as pd\nfrom sklearn.ensemble import GradientBoostingClassifier" });
      return reply(404, {});
    }
    const repoMatch = /^\/repos\/mockstudent\/([^/]+)(\/.*)?$/.exec(path);
    if (repoMatch) {
      const [, name, rest] = repoMatch;
      if (!rest) {
        return repos.has(name)
          ? reply(200, {
              html_url: `https://github.com/mockstudent/${name}`,
              default_branch: "main",
            })
          : reply(404, {});
      }
      if (rest.startsWith("/git/ref/")) return reply(200, { object: { sha: "p" } });
      if (rest.startsWith("/git/commits/")) return reply(200, { tree: { sha: "b" } });
      if (rest === "/git/trees") {
        trees.push((JSON.parse(req.postData() ?? "{}") as { tree: { path: string }[] }).tree);
        return reply(201, { sha: "t" });
      }
      if (rest === "/git/blobs") return reply(201, { sha: "blob" });
      if (rest === "/git/commits") return reply(201, { sha: "c" });
      if (rest.startsWith("/git/refs/")) return reply(200, {});
      if (rest === "/pages") return reply(201, {});
    }
    return reply(500, { unexpected: path });
  });
  return { trees, setExpired: (value: boolean) => { expired = value; } };
}
