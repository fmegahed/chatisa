/**
 * Deterministic HTML renderers for Portfolio Builder (2026-08-20). The model
 * supplies CONTENT ONLY (validated by lib/portfolio/content.ts); every byte
 * of markup and style comes from this file, and every interpolated string
 * passes through escapeHtml. That split is the injection defense: employer
 * job descriptions and student uploads feed the model, so model output must
 * never reach the published page as markup.
 *
 * Pages are self-contained: inline CSS, system fonts, no JavaScript, no
 * external requests. This is the only site renderer: Job Scout's v6.3.0
 * portfolio-html.ts was removed with its tab (2026-08-20).
 */

import { SAFE_PATH, type CareerContent, type ShowcaseContent } from "./content";
import { getCourse } from "@/lib/scout/courses";

export function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/** Only http(s) URLs may become hrefs; anything else renders as text. */
export function safeHref(url: string): string | null {
  try {
    const parsed = new URL(url);
    return parsed.protocol === "https:" || parsed.protocol === "http:"
      ? parsed.toString()
      : null;
  } catch {
    return null;
  }
}

const CSS = `
:root { --ink: #1f1a17; --paper: #fdfbf7; --accent: #c3142d; --tan: #e8e0d2; --muted: #6b6156; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink);
  font: 17px/1.6 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
main { max-width: 46rem; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }
header { border-bottom: 3px solid var(--accent); padding-bottom: 1.5rem; }
h1 { font-size: 2.2rem; margin: 0; line-height: 1.2; }
.lede { font-size: 1.25rem; color: var(--muted); margin: 0.25rem 0 0; }
h2 { font-size: 1.35rem; margin: 2.2rem 0 0.6rem; border-bottom: 1px solid var(--tan); padding-bottom: 0.3rem; }
h3 { margin: 1.25rem 0 0.25rem; font-size: 1.1rem; }
.hero { display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap; }
.hero img { width: 9rem; height: 9rem; border-radius: 50%; object-fit: cover; border: 3px solid var(--accent); }
.figure { margin: 1rem 0; } .figure img { max-width: 100%; height: auto; border: 1px solid var(--tan); }
.meta { color: var(--muted); font-size: 0.95rem; }
.chips { display: flex; flex-wrap: wrap; gap: 0.4rem; padding: 0; list-style: none; }
.chips li { background: var(--tan); border-radius: 999px; padding: 0.15rem 0.7rem; font-size: 0.9rem; }
`;

const PHOTO_PATH = "assets/photo.jpg";
const RESUME_PATH = "resume.pdf";

function section(title: string, body: string): string {
  return body ? `<section><h2>${escapeHtml(title)}</h2>${body}</section>` : "";
}
/**
 * An empty (or whitespace-only) field renders as nothing rather than as an
 * empty paragraph, so section() drops its heading with it: a student who
 * clears "About" loses the About heading too instead of keeping a bare rule.
 */
function para(text: string): string {
  return text
    .split(/\n{2,}/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0)
    .map((p) => `<p>${escapeHtml(p)}</p>`)
    .join("");
}
function chips(items: string[]): string {
  return items.length ? `<ul class="chips">${items.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul>` : "";
}
function link(url: string, label: string): string {
  const href = safeHref(url);
  return href ? `<a href="${escapeHtml(href)}" rel="noopener">${escapeHtml(label)}</a>` : escapeHtml(label);
}
function page(title: string, body: string): string {
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(title)}</title>
<style>${CSS}</style>
</head>
<body><main>${body}</main></body>
</html>`;
}

/**
 * Skips a relative path (deliverable or figure) that could carry a URL
 * scheme, an absolute path, or a ".." traversal segment. Must match
 * SAFE_PATH (word chars, dots, hyphens, "/" separators only, so no ":" or
 * "//" scheme prefix) AND contain no literal ".." segment, since SAFE_PATH
 * alone admits ".." as a segment.
 */
function relativeSafe(path: string): boolean {
  return SAFE_PATH.test(path) && !path.split("/").some((seg) => seg === "..");
}

/**
 * `folders` is the set of project slugs that actually have files pushed under
 * projects/<slug>/. A project the student typed into the editor has content
 * but no folder, so linking "Files" for it would publish a 404; only slugs in
 * this set get the link. `repoName` is the repository the site is actually
 * published to (usually "portfolio", but a student whose account already had
 * that name publishes elsewhere), so the Files links point at the real repo.
 */
export function renderCareer(
  content: CareerContent,
  student: {
    name: string;
    links: { label: string; url: string }[];
    hasPhoto: boolean;
    resumeLink: boolean;
    login: string | null;
    folders: string[];
    repoName: string;
  },
): string {
  const photo = student.hasPhoto ? `<img src="${PHOTO_PATH}" alt="Photo of ${escapeHtml(student.name)}">` : "";
  const links = [
    ...student.links.map((l) => link(l.url, l.label)),
    ...(student.resumeLink ? [`<a href="${RESUME_PATH}">Resume (PDF)</a>`] : []),
  ];
  const hero = `<header class="hero">${photo}<div><h1>${escapeHtml(content.siteTitle || student.name)}</h1><p class="lede">${escapeHtml(content.headline)}</p>${links.length ? `<p class="meta">${links.join(" · ")}</p>` : ""}</div></header>`;
  const skills = content.skillGroups.map((g) => `<h3>${escapeHtml(g.title)}</h3>${chips(g.skills)}`).join("");
  const hasFolder = new Set(student.folders);
  const projects = content.projects.map((p) => {
    const folder = student.login && hasFolder.has(p.slug)
      ? `https://github.com/${encodeURIComponent(student.login)}/${encodeURIComponent(student.repoName)}/tree/main/projects/${p.slug}`
      : null;
    const refs = [
      ...(folder ? [link(folder, "Files")] : []),
      ...(p.externalUrl ? [link(p.externalUrl, "Project link")] : []),
    ];
    return `<article><h3>${escapeHtml(p.title)}</h3>${para(p.blurb)}${chips(p.skills)}${refs.length ? `<p class="meta">${refs.join(" · ")}</p>` : ""}</article>`;
  }).join("");
  // The catalog supplies the title; the model supplies only the code and
  // the reason, so a line reads "ISA 444 - Business Forecasting: why" with
  // code and title in bold (professor's call, 2026-08-23).
  const courses = content.courses.length
    ? `<ul>${content.courses.map((c) => {
        const title = getCourse(c.code)?.title;
        return `<li><strong>${escapeHtml(c.code)}${title ? ` - ${escapeHtml(title)}` : ""}</strong>: ${escapeHtml(c.why)}</li>`;
      }).join("")}</ul>` : "";
  const experience = content.experience.map((e) =>
    `<article><h3>${escapeHtml(e.role)}, ${escapeHtml(e.org)}</h3><p class="meta">${escapeHtml(e.dates)}</p><ul>${e.bullets.map((b) => `<li>${escapeHtml(b)}</li>`).join("")}</ul></article>`).join("");
  const education = content.education.map((e) =>
    `<p><strong>${escapeHtml(e.school)}</strong>${e.degree ? `, ${escapeHtml(e.degree)}` : ""}${e.dates ? ` <span class="meta">${escapeHtml(e.dates)}</span>` : ""}</p>`).join("");
  return page(content.siteTitle || student.name, [
    hero,
    section("About", para(content.about)),
    section("Skills", skills),
    section("Projects", projects),
    section("Coursework", courses),
    section("Experience", experience),
    section("Education", education),
  ].join(""));
}

export function renderShowcase(
  content: ShowcaseContent,
  /**
   * `figures` and `deliverablePaths` are the paths that will actually exist in
   * the repository. The model can name a file the student later unticked, so
   * both lists are allow-lists: a link to a file that was never pushed is a
   * 404 on the published page.
   */
  meta: {
    course: string; semester: string; team: string[]; repoUrl: string | null;
    figures: string[]; deliverablePaths: string[];
  },
): string {
  const allowed = new Set(meta.figures);
  const publishable = new Set(meta.deliverablePaths);
  const head = `<header><h1>${escapeHtml(content.title)}</h1><p class="lede">${escapeHtml(content.tagline)}</p><p class="meta">${escapeHtml(meta.course)}${meta.semester ? `, ${escapeHtml(meta.semester)}` : ""}${meta.team.length ? ` · ${meta.team.map(escapeHtml).join(", ")}` : ""}${meta.repoUrl ? ` · ${link(meta.repoUrl, "Repository")}` : ""}</p></header>`;
  const findings = content.findings.map((f) => {
    const fig = f.figure && allowed.has(f.figure) && relativeSafe(f.figure)
      ? `<figure class="figure"><img src="${escapeHtml(f.figure)}" alt="${escapeHtml(f.heading)}"></figure>` : "";
    return `<article><h3>${escapeHtml(f.heading)}</h3>${fig}${para(f.body)}</article>`;
  }).join("");
  const deliverables = content.deliverables.filter(
    (d) => relativeSafe(d.path) && publishable.has(d.path),
  );
  const deliverablesHtml = deliverables.length
    ? `<ul>${deliverables.map((d) => `<li><a href="${escapeHtml(d.path)}">${escapeHtml(d.label)}</a></li>`).join("")}</ul>` : "";
  return page(content.title, [
    head,
    section("The problem", para(content.problem)),
    section("The data", para(content.data)),
    section("Approach", para(content.approach)),
    section("Findings", findings),
    section("Deliverables", deliverablesHtml),
    section("Skills demonstrated", chips(content.skills)),
    section("What I would do next", para(content.nextSteps)),
  ].join(""));
}
