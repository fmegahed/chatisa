/**
 * Deterministic portfolio site renderer (v6.3.0). The model supplies
 * CONTENT ONLY (validated JSON); every byte of markup and style comes from
 * this template, and every interpolated string passes through escapeHtml.
 * That split is the injection defense: employer job descriptions feed the
 * model, so model output must never reach the published page as markup.
 *
 * The page is deliberately self-contained: inline CSS, system fonts, no
 * JavaScript, no external requests. It renders anywhere GitHub Pages can
 * serve a file, and a student can open index.html locally and understand
 * every line they are publishing under their own name.
 */

export interface PortfolioContent {
  siteTitle: string;
  headline: string;
  about: string;
  skillGroups: { title: string; skills: string[] }[];
  projectCards: {
    repoName: string;
    title: string;
    blurb: string;
    skillLabels: string[];
    repoUrl: string;
  }[];
  courseHighlights: { course: string; why: string }[];
}

export interface PortfolioStudent {
  name: string;
  links: { label: string; url: string }[];
}

export function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

/** Only http(s) URLs may become hrefs; anything else renders as text. */
function safeHref(url: string): string | null {
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
.headline { font-size: 1.15rem; color: var(--muted); margin: 0.5rem 0 0; }
nav { margin-top: 0.9rem; }
nav a { color: var(--accent); font-weight: 700; margin-right: 1rem; }
h2 { font-size: 1.35rem; margin: 2.2rem 0 0.6rem; border-bottom: 1px solid var(--tan); padding-bottom: 0.3rem; }
.card { border: 1px solid var(--tan); border-radius: 10px; padding: 1rem 1.1rem; margin-top: 0.9rem; }
.card h3 { margin: 0; font-size: 1.1rem; }
.card a { color: var(--accent); }
.tags { margin: 0.5rem 0 0; padding: 0; list-style: none; display: flex; flex-wrap: wrap; gap: 0.4rem; }
.tags li { background: var(--tan); border-radius: 6px; padding: 0.1rem 0.55rem; font-size: 0.85rem; }
dl { margin: 0.5rem 0 0; }
dt { font-weight: 700; margin-top: 0.6rem; }
dd { margin: 0; color: var(--muted); }
footer { margin-top: 3rem; color: var(--muted); font-size: 0.85rem; }
@media print { nav a { text-decoration: none; } }
`;

export function renderPortfolioHtml(
  content: PortfolioContent,
  student: PortfolioStudent,
): string {
  const links = student.links
    .map((l) => {
      const href = safeHref(l.url);
      return href
        ? `<a href="${escapeHtml(href)}" rel="me noopener">${escapeHtml(l.label)}</a>`
        : "";
    })
    .filter(Boolean)
    .join("\n      ");

  const skillGroups = content.skillGroups
    .map(
      (g) => `<h3>${escapeHtml(g.title)}</h3>
    <ul class="tags">${g.skills.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul>`,
    )
    .join("\n    ");

  const projects = content.projectCards
    .map((p) => {
      const href = safeHref(p.repoUrl);
      const title = href
        ? `<a href="${escapeHtml(href)}" rel="noopener">${escapeHtml(p.title)}</a>`
        : escapeHtml(p.title);
      return `<article class="card">
      <h3>${title}</h3>
      <p>${escapeHtml(p.blurb)}</p>
      <ul class="tags">${p.skillLabels.map((s) => `<li>${escapeHtml(s)}</li>`).join("")}</ul>
    </article>`;
    })
    .join("\n    ");

  const courses = content.courseHighlights
    .map(
      (c) => `<dt>${escapeHtml(c.course)}</dt>
      <dd>${escapeHtml(c.why)}</dd>`,
    )
    .join("\n      ");

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${escapeHtml(content.siteTitle)}</title>
<meta name="description" content="${escapeHtml(content.headline)}">
<style>${CSS}</style>
</head>
<body>
<main>
  <header>
    <h1>${escapeHtml(student.name)}</h1>
    <p class="headline">${escapeHtml(content.headline)}</p>
    ${links ? `<nav aria-label="Profiles">\n      ${links}\n    </nav>` : ""}
  </header>

  <section aria-labelledby="about-h">
    <h2 id="about-h">About</h2>
    <p>${escapeHtml(content.about)}</p>
  </section>

  ${
    content.projectCards.length > 0
      ? `<section aria-labelledby="projects-h">
    <h2 id="projects-h">Projects</h2>
    ${projects}
  </section>`
      : ""
  }

  <section aria-labelledby="skills-h">
    <h2 id="skills-h">Skills</h2>
    ${skillGroups}
  </section>

  ${
    content.courseHighlights.length > 0
      ? `<section aria-labelledby="courses-h">
    <h2 id="courses-h">Relevant coursework</h2>
    <dl>
      ${courses}
    </dl>
  </section>`
      : ""
  }

  <footer>Built by ${escapeHtml(student.name)}. Hosted on GitHub Pages.</footer>
</main>
</body>
</html>
`;
}
