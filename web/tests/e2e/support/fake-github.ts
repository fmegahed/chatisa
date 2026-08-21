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
export async function fakeGithubApi(page: Page): Promise<{ trees: { path: string }[][] }> {
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
  return { trees };
}
