import { PUSH_LIMITS, pushFileBytes, type PushFile } from "@/lib/scout/github";

export type FileRole = "data" | "code" | "notebook" | "report" | "slides" | "figure" | "other";

export const ROLE_LABELS: Record<FileRole, string> = {
  data: "Data", code: "Code", notebook: "Notebook", report: "Report",
  slides: "Slides", figure: "Figure", other: "Other",
};

export const CAREER_REPO = "portfolio";
export const PHOTO_PATH = "assets/photo.jpg";
export const RESUME_PATH = "resume.pdf";
export const MAX_PROJECT_FILES = 10;
export const MAX_SHOWCASE_FILES = 40;

export interface PreparedFile {
  name: string;
  role: FileRole;
  publish: boolean;
  bytes: number;
  text: string | null;
  base64: string | null;
}

export function guessRole(name: string): FileRole {
  const ext = name.toLowerCase().split(".").pop() ?? "";
  if (["csv", "tsv", "xlsx", "xls", "json", "parquet", "rds", "rdata", "sav", "dta", "db", "sqlite"].includes(ext)) return "data";
  if (ext === "ipynb") return "notebook";
  if (["py", "r", "rmd", "qmd", "sql", "js", "ts", "sas", "do", "m", "jl", "sh"].includes(ext)) return "code";
  if (["pdf", "docx", "doc", "md", "txt", "html"].includes(ext)) return "report";
  if (["pptx", "ppt", "key"].includes(ext)) return "slides";
  if (["png", "jpg", "jpeg", "gif", "svg", "webp"].includes(ext)) return "figure";
  return "other";
}

export function slugify(s: string): string {
  const out = s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 60);
  return out.length >= 3 ? out : "project";
}

export function safeFileName(name: string): string {
  const base = name.split(/[\\/]/).pop() ?? "file";
  const cleaned = base.replaceAll(" ", "-").replace(/[^\w.-]/g, "").replace(/^\.+/, "");
  return cleaned || "file";
}

export function rolePath(role: FileRole, name: string): string {
  const file = safeFileName(name);
  const folder: Record<FileRole, string> = {
    data: "data", code: "code", notebook: "code", report: "report",
    slides: "slides", figure: "figures", other: "other",
  };
  return `${folder[role]}/${file}`;
}

export function showcaseRepoName(courseCode: string, title: string): string {
  return slugify(`${courseCode} ${title}`);
}

export function measure(files: PushFile[]): {
  count: number; totalBytes: number; over: { path: string; bytes: number }[]; ok: boolean;
} {
  let totalBytes = 0;
  const over: { path: string; bytes: number }[] = [];
  for (const f of files) {
    const bytes = pushFileBytes(f);
    totalBytes += bytes;
    if (bytes > PUSH_LIMITS.fileBytes) over.push({ path: f.path, bytes });
  }
  const ok = over.length === 0 && files.length <= PUSH_LIMITS.files && totalBytes <= PUSH_LIMITS.totalBytes;
  return { count: files.length, totalBytes, over, ok };
}

/**
 * Two files can land on the same repository path: "Final Report.pdf" and
 * "final-report.pdf" both become report/final-report.pdf once names are
 * cleaned, and a GitHub tree with a repeated path silently keeps one of
 * them. The second and later copies get a numeric suffix before the
 * extension instead, so nothing a student ticked disappears.
 */
function uniquePath(path: string, seen: Set<string>): string {
  if (!seen.has(path)) {
    seen.add(path);
    return path;
  }
  const dot = path.lastIndexOf(".");
  const slash = path.lastIndexOf("/");
  const cut = dot > slash + 1 ? dot : path.length;
  const stem = path.slice(0, cut);
  const ext = path.slice(cut);
  let n = 2;
  while (seen.has(`${stem}-${n}${ext}`)) n++;
  const out = `${stem}-${n}${ext}`;
  seen.add(out);
  return out;
}

/** The paths a file list will actually occupy, collisions suffixed. */
export function dedupePaths(paths: string[]): string[] {
  const seen = new Set<string>();
  return paths.map((p) => uniquePath(p, seen));
}

function toPush(path: string, f: PreparedFile): PushFile {
  return f.base64 !== null && f.base64.length > 0
    ? { path, contents: f.base64, encoding: "base64" }
    : { path, contents: f.text ?? "" };
}

const CAREER_README =
  "# Portfolio\n\nThis site was built with ChatISA's Portfolio Builder and is published with GitHub Pages. Edit index.html to make it yours. Project files live under projects/.\n";

export function careerFileSet(args: {
  html: string;
  photoBase64: string | null;
  resumeBase64: string | null;
  projects: { slug: string; files: PreparedFile[] }[];
}): PushFile[] {
  const files: PushFile[] = [
    { path: "index.html", contents: args.html },
    { path: ".nojekyll", contents: "" },
    { path: "README.md", contents: CAREER_README },
  ];
  if (args.photoBase64) files.push({ path: PHOTO_PATH, contents: args.photoBase64, encoding: "base64" });
  if (args.resumeBase64) files.push({ path: RESUME_PATH, contents: args.resumeBase64, encoding: "base64" });
  const seen = new Set(files.map((f) => f.path));
  for (const p of args.projects) {
    for (const f of p.files) {
      if (!f.publish) continue;
      files.push(toPush(uniquePath(`projects/${p.slug}/${safeFileName(f.name)}`, seen), f));
    }
  }
  return files;
}

export function showcaseFileSet(args: {
  html: string; readme: string; gitignore: string; files: PreparedFile[];
}): PushFile[] {
  const files: PushFile[] = [
    { path: "index.html", contents: args.html },
    { path: ".nojekyll", contents: "" },
    { path: "README.md", contents: args.readme },
    { path: ".gitignore", contents: args.gitignore },
  ];
  const seen = new Set(files.map((f) => f.path));
  for (const f of args.files) {
    if (!f.publish) continue;
    files.push(toPush(uniquePath(rolePath(f.role, f.name), seen), f));
  }
  return files;
}

export const DEFAULT_GITIGNORE = [
  ".Rproj.user/", ".Rhistory", ".RData", "renv/library/", ".ipynb_checkpoints/",
  "__pycache__/", ".venv/", ".env", ".DS_Store", "Thumbs.db", "",
].join("\n");
