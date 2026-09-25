/**
 * A size-capped summary of one GitHub repository (v6.8.0), built in the
 * browser from the student's own public repository and sent to
 * /api/scout/repo-skills. Pure: no network, no storage. The server clips
 * with the same function, so the caps hold whatever a browser sends.
 */

export const SUMMARY_LIMITS = {
  totalChars: 150_000,
  readmeChars: 8_000,
  fileChars: 45_000,
  codeFiles: 6,
  treeEntries: 500,
  /** Files larger than this are skipped unless the student asks for them. */
  maxFetchBytes: 6_000_000,
  /** GitHub's contents API does not serve files above this (raw media type). */
  githubMaxBytes: 100_000_000,
  reposPerRequest: 5,
} as const;

export interface RepoListing {
  fullName: string;
  description: string | null;
  language: string | null;
  pushedAt: string;
  defaultBranch: string;
  htmlUrl: string;
}

export interface RepoFile { path: string; text: string }

export interface RepoSummary {
  fullName: string;
  description: string;
  topics: string[];
  fork: boolean;
  archived: boolean;
  languages: Record<string, number>;
  authorship: { studentCommits: number; totalCommits: number };
  tree: { path: string; size: number }[];
  treeTruncated: boolean;
  readme: string;
  dependencyFiles: RepoFile[];
  codeFiles: RepoFile[];
  /** Code files over 6 MB that were not read; the student may include them. */
  skippedLarge: { path: string; size: number }[];
}

const DEPENDENCY_NAMES = new Set([
  "requirements.txt", "pyproject.toml", "environment.yml", "renv.lock", "DESCRIPTION", "package.json",
]);
// HTML is left out on purpose: in student repositories it is mostly
// rendered reports, not code they wrote (seen in the v6.8.0 eval).
const CODE_EXT = /\.(py|r|rmd|qmd|ipynb|sql|js|jsx|ts|tsx|jl|java|go|cpp|cc|c|cs|rb|php|sas|scala|kt|swift)$/i;
// renv/ is generated as a whole (activate.R is bootstrap boilerplate, found
// taking a code slot in the v6.8.0 eval), not only renv/library/.
export const VENDORED = /(^|\/)(node_modules|dist|build|\.venv|venv|site-packages|renv|\.git)\//;

const base = (p: string) => p.split("/").pop() ?? p;

export function pickDependencyPaths(tree: { path: string; size: number }[]): string[] {
  return tree.filter((f) => DEPENDENCY_NAMES.has(base(f.path)) && !VENDORED.test(f.path)).map((f) => f.path);
}

/**
 * The code files to read, largest first. Files over 6 MB are skipped and
 * reported unless the student chose them (`include`), in which case they go
 * first, up to GitHub's 100 MB ceiling. Only the skipped files that would
 * otherwise have made the cut are reported.
 */
export function pickCodePaths(
  tree: { path: string; size: number }[],
  include: string[] = [],
): { paths: string[]; skippedLarge: { path: string; size: number }[] } {
  const code = tree.filter((f) => CODE_EXT.test(f.path) && !VENDORED.test(f.path) && f.size > 0 && f.size <= SUMMARY_LIMITS.githubMaxBytes);
  const chosen = code.filter((f) => include.includes(f.path));
  const normal = code.filter((f) => !include.includes(f.path) && f.size <= SUMMARY_LIMITS.maxFetchBytes);
  const large = code.filter((f) => !include.includes(f.path) && f.size > SUMMARY_LIMITS.maxFetchBytes);
  const paths = [...chosen, ...normal.sort((a, b) => b.size - a.size)].slice(0, SUMMARY_LIMITS.codeFiles).map((f) => f.path);
  return { paths, skippedLarge: large.sort((a, b) => b.size - a.size).slice(0, SUMMARY_LIMITS.codeFiles) };
}

/** A notebook's code and markdown cells as plain text; outputs dropped. */
export function stripNotebook(raw: string): string {
  try {
    const nb = JSON.parse(raw) as { cells?: { cell_type?: string; source?: string | string[] }[] };
    if (!Array.isArray(nb.cells)) return raw;
    return nb.cells
      .filter((c) => c.cell_type === "code" || c.cell_type === "markdown")
      .map((c) => (Array.isArray(c.source) ? c.source.join("") : c.source ?? ""))
      .join("\n\n");
  } catch {
    return raw;
  }
}

/**
 * The student's commits and all human commits, from GitHub's contributor
 * list. Bots are left out of both; anonymous authors (commits whose email
 * is not linked to an account) count toward the total only, which is why
 * the card explains a low share rather than hiding it.
 */
export function authorshipFrom(
  contributors: { login?: string; type?: string; contributions: number }[],
  login: string,
): { studentCommits: number; totalCommits: number } {
  let student = 0;
  let total = 0;
  for (const c of contributors) {
    if (c.type === "Bot") continue;
    total += c.contributions;
    if (c.login && c.login.toLowerCase() === login.toLowerCase()) student += c.contributions;
  }
  return { studentCommits: student, totalCommits: total };
}

/** Not a fork, at least 60% and at least 10 of the commits by the student. */
export function isSubstantial(s: RepoSummary): boolean {
  const { studentCommits, totalCommits } = s.authorship;
  return !s.fork && studentCommits >= 10 && totalCommits > 0 && studentCommits / totalCommits >= 0.6;
}

/** Every cap, and the total, applied in a fixed order: README, deps, code. */
export function clipSummary(s: RepoSummary): RepoSummary {
  let budget = SUMMARY_LIMITS.totalChars;
  const take = (text: string, cap: number) => {
    const out = text.slice(0, Math.max(0, Math.min(cap, budget)));
    budget -= out.length;
    return out;
  };
  const readme = take(typeof s.readme === "string" ? s.readme : "", SUMMARY_LIMITS.readmeChars);
  // Anything a browser sends is untrusted: non-objects are dropped and
  // every field is coerced and clipped (review fix).
  const files = (list: RepoFile[], max: number) =>
    (Array.isArray(list) ? list : [])
      .filter((f): f is RepoFile => typeof f === "object" && f !== null)
      .slice(0, max)
      .map((f) => ({ path: String(f.path ?? "").slice(0, 300), text: take(String(f.text ?? ""), SUMMARY_LIMITS.fileChars) }))
      .filter((f) => f.path.length > 0 && f.text.length > 0);
  const dependencyFiles = files(s.dependencyFiles ?? [], DEPENDENCY_NAMES.size);
  const codeFiles = files(s.codeFiles ?? [], SUMMARY_LIMITS.codeFiles);
  const rawTree = (Array.isArray(s.tree) ? s.tree : []).filter((t) => typeof t === "object" && t !== null);
  const tree = rawTree
    .slice(0, SUMMARY_LIMITS.treeEntries)
    .map((t) => ({ path: String(t.path ?? "").slice(0, 300), size: Math.max(0, Number(t.size) || 0) }))
    .filter((t) => t.path.length > 0);
  const count = (n: unknown) => Math.max(0, Math.floor(Number(n) || 0));
  const languages = Object.fromEntries(
    Object.entries(typeof s.languages === "object" && s.languages !== null ? s.languages : {})
      .slice(0, 30)
      .map(([k, v]) => [String(k).slice(0, 40), count(v)]),
  );
  return {
    fullName: String(s.fullName).slice(0, 200),
    description: typeof s.description === "string" ? s.description.slice(0, 500) : "",
    topics: (Array.isArray(s.topics) ? s.topics : []).slice(0, 20).map((t) => String(t).slice(0, 50)),
    fork: Boolean(s.fork),
    archived: Boolean(s.archived),
    languages,
    authorship: {
      studentCommits: count(s.authorship?.studentCommits),
      totalCommits: count(s.authorship?.totalCommits),
    },
    tree,
    treeTruncated: Boolean(s.treeTruncated) || rawTree.length > SUMMARY_LIMITS.treeEntries,
    readme,
    dependencyFiles,
    codeFiles,
    skippedLarge: (Array.isArray(s.skippedLarge) ? s.skippedLarge : [])
      .filter((f) => typeof f === "object" && f !== null)
      .slice(0, SUMMARY_LIMITS.codeFiles)
      .map((f) => ({ path: String(f.path ?? "").slice(0, 300), size: count(f.size) })),
  };
}
