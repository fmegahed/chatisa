/**
 * Import from GitHub (v6.9.0): which files of a student's repository to
 * offer, how imported files are named, and the only repository links a
 * page may carry. Pure; the network lives in lib/scout/github-read.ts.
 */

import { formatSize, guessRole, type FileRole } from "./files";
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

const mb = formatSize;

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
      // One limit for every file and every model (professor, 2026-09-24):
      // 25 MB, the publishing limit for one file. No model ever receives a
      // file's bytes (text files give their first 30,000 characters, others
      // only their name and size), so the model chosen does not change it.
      // GitHub's own 100 MB ceiling is named when it is the tighter reason.
      const selectable = f.size <= PUSH_LIMITS.fileBytes;
      const note = f.size > GITHUB_CONTENTS_MAX_BYTES
        ? `${mb(f.size)}: over 100 MB, which GitHub cannot send`
        : !selectable
          ? `${mb(f.size)}: over the 25 MB limit for one file on a published page, so it cannot be imported`
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
export function importedName(path: string, chosen: string[], existing: string[] = []): string {
  const base = path.split("/").pop() ?? path;
  const taken = new Set(existing);
  // Files already in the project count too (review fix): two rows with the
  // same name are indistinguishable in the builder and in the prompt.
  const clash = chosen.filter((p) => (p.split("/").pop() ?? p) === base).length > 1 || taken.has(base);
  let name = clash ? path.replace(/\//g, "_") : base;
  if (taken.has(name)) {
    const dot = name.lastIndexOf(".");
    const stem = dot > 0 ? name.slice(0, dot) : name;
    const ext = dot > 0 ? name.slice(dot) : "";
    let n = 2;
    while (taken.has(`${stem} (${n})${ext}`)) n++;
    name = `${stem} (${n})${ext}`;
  }
  return name;
}

/** `https://github.com/<owner>/<repo>` exactly, or null. */
export function githubRepoUrl(raw: string | null | undefined): string | null {
  const m = /^https:\/\/github\.com\/([A-Za-z0-9-]{1,39})\/([A-Za-z0-9._-]{1,100})\/?$/.exec(String(raw ?? "").trim());
  return m ? `https://github.com/${m[1]}/${m[2]}` : null;
}
