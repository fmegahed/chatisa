# Portfolio Builder: design

Date: 2026-08-20. Status: approved; implemented on branch portfolio-builder (v6.4.0). Amended 2026-08-21 to record two implementation cuts.
Supersedes the Portfolio Site tab shipped inside Job Scout in v6.3.0.

## 1. Purpose

A standalone module, listed first under "For your job search", that turns a
student's work into a published GitHub Pages site. Two modes, toggled on the
first step:

| | Career portfolio | Project showcase |
|---|---|---|
| Purpose | "Who I am and what I can do" | "One project, told as a story" |
| Inputs | resume PDF (required), classes taken, 1 to 5 projects, optional photo, name and links | one course, optional team members, the project's files: data, code or notebooks, analysis, written report, slides, figures. Any subset. |
| Page | hero with photo, about, skills, projects, courses, experience, education, contact | landing page: problem, data, approach, findings (figures when images were uploaded), deliverables, team |
| Repository | `portfolio`, one per student, republished in place | `<course>-<project-slug>` (for example `isa-401-churn-model`), one per project, many per student |
| Resume | source material; the PDF is published only when the student ticks "Include a resume download link" (with a warning that phone and address become public) | none |

Both modes share intake, generation, the deterministic renderer, the live
preview editor, and the push to GitHub. A published showcase becomes a
one-click project card in the career portfolio.

Job Scout's "Polish a project I already built" flow is absorbed by showcase
mode and removed from Job Scout. Job Scout keeps "Generate project scaffold".

## 2. Locked decisions

- Browser-only persistence. Records in localStorage, blobs (files, photo,
  generated content) in IndexedDB. The server reads uploads transiently for
  parsing and stores nothing. Usage events are content-free
  (`portfolio_generated`, `portfolio_published`, with `mode`).
- GitHub token handling is unchanged from v6.3.0: browser-only token,
  `public_repo` scope, hand-rolled OAuth under `/api/scout/github/*`, pushes
  go straight from the browser to api.github.com. The connect landing page
  becomes module-neutral (`/portfolio/github-connected`, old path redirects).
- No zip download anywhere. GitHub is the only destination.
- Single-page static HTML in the Miami house style, rendered by a
  deterministic escaping template. The model emits content JSON only.
- Data files default to excluded from the public repo, with a per-file
  "publish" checkbox and a reminder about licensed or instructor-provided
  data. Code, reports, slides, and figures default to included.
- Size caps come from the push engine (60 files, 400 KB per file, 2 MB
  total). The wizard shows a running size meter and refuses before
  generation, never at push.

## 3. The wizard

Step 0, Mode: two cards, "Career portfolio" and "Project showcase". A
previously published site of either kind is listed here with "Update" and
"View" actions. Deep links `/portfolio?mode=career|project` preselect.

### Career portfolio

1. Resume. PDF only, via `readResumePdf`. The device-resume offer lets Job
   Scout and JobApp Drafter users reuse their saved resume.
2. Classes. Checklist over `COURSES` with the popular-code shortcuts and a
   search box; preselected from the Job Scout profile when one exists.
3. Projects (1 to 5). Each card: optional title, files (drag and drop), and
   optional external URL (repo or demo). Published showcases and Job Scout
   projects with a repo URL are offered as link-only cards. Up to 10 files
   per project. The model writes title, blurb, and skills from the content.
4. Details. Optional photo (JPEG or PNG, resized in the browser to 512 px
   on the long side, re-encoded under 150 KB), name, up to 4 links, the
   resume-link checkbox. Then Generate.
5. Review and publish (see below).

### Project showcase

1. Course. One pick from `COURSES`, semester, optional team members (names
   only). The model titles the page, so no separate freeform title field is
   needed for ISA 340/480/481.
2. Files. Drag and drop everything the project has. Each file gets a role
   the student can change: data, code, notebook, report, slides, figure,
   other. Roles drive the repo layout (`data/`, `code/`, `report/`,
   `slides/`, `figures/`). Data files start unchecked for publishing.
3. Story prompts. Three optional short answers: what problem, what was the
   hardest part, what would you do next. Then Generate.
4. Review and publish.

### Review and publish (both modes)

Left: a structured editor over every generated field (text inputs and
textareas, reorder and delete for lists). Regenerating is whole-page: go Back
and press Generate again (a per-section regenerate was cut during
implementation, 2026-08-21, as unneeded while every field is editable). Right:
a live preview in a sandboxed `iframe` (`srcdoc`, `sandbox=""`) re-rendered
on every edit. Below: Connect GitHub if needed, repository name (editable,
validated against GitHub's rules), Publish. On success: Pages URL, repo URL,
and the record is saved. Republishing updates the same repository.

## 4. Generation

One route, `POST /api/portfolio/generate` (multipart), same recipe as the
Job Scout routes: `auth()`, rate limit, model gating via `PAGE_MODELS.portfolio`
(structured output required, 64k context), zod input, `generateObject`,
`recordUsageEvent`. Uploaded text files are extracted with the existing
office, notebook, and PDF readers and sliced to a per-request budget;
binary files are described by name and size only. All student and file text
is nonce-fenced.

Output schemas (`lib/portfolio/content.ts`):

```ts
interface CareerContent {
  v: 2;
  siteTitle: string; headline: string; about: string;
  skillGroups: { title: string; skills: string[] }[];
  projects: { slug: string; title: string; blurb: string; skills: string[];
              externalUrl: string | null }[];
  courses: { code: string; why: string }[];
  experience: { org: string; role: string; dates: string; bullets: string[] }[];
  education: { school: string; degree: string; dates: string }[];
}

interface ShowcaseContent {
  v: 1;
  title: string; tagline: string;
  problem: string; data: string; approach: string;
  findings: { heading: string; body: string; figure: string | null }[];
  deliverables: { label: string; path: string }[];
  skills: string[];
  nextSteps: string;
}
```

`figure` and `path` are validated against the uploaded file set on the
server and again in the renderer; the model can only reference files that
exist. Skills are mapped to taxonomy IDs by label with a fallback to free
text for display only. The v1 `PortfolioContent` from v6.3.0 migrates to
`CareerContent` v2 (projects gain `slug`, `externalUrl`; new sections empty).

## 5. Rendering

`lib/portfolio/html.ts` replaces `lib/scout/portfolio-html.ts`. Two layouts,
`renderCareer` and `renderShowcase`, share the palette, CSS, `escapeHtml`,
and `safeHref`. No JavaScript, no external requests, inline CSS, system
fonts. Image paths are fixed (`assets/photo.jpg`, `figures/<name>`), never
model-supplied. Project cards in the career layout link to
`https://github.com/<login>/portfolio/tree/main/projects/<slug>` and the
external URL when present.

Repository file sets (`lib/portfolio/files.ts`):

- Career: `index.html`, `.nojekyll`, `README.md`, `assets/photo.jpg`
  (optional), `resume.pdf` (opt-in), `projects/<slug>/<files>`.
- Showcase: `index.html`, `.nojekyll`, `README.md` (the story as markdown),
  `.gitignore`, `data/`, `code/`, `report/`, `slides/`, `figures/`, `other/`
  as present, with file names slugified (spaces to hyphens).

## 6. Feeding the other modules

`lib/portfolio/published.ts` holds the bridge record:

```ts
interface PublishedWork {
  id: string; kind: "career" | "showcase";
  title: string; summary: string; skillIds: string[];
  repoUrl: string; pagesUrl: string | null; publishedAt: string;
}
```

Written on every successful publish, with a `subscribe` hook like the GitHub
connection store.

- Job Scout reads it and counts a showcase's `skillIds` toward the profile
  exactly as a built project does today (repo URL present, so the
  "built, not scaffolded" gate is satisfied). Job Scout's own project
  records are unchanged.
- JobApp Drafter gains an opt-in "Include published work" toggle per draft.
  When on, the request carries the portfolio URL and each showcase's title,
  summary, skills, and URL; the resume gets a Projects or Portfolio line with
  links and the cover letter may cite a linkable deliverable. Off by default.
- Interview Mentor is out of scope; the record shape supports it later.

## 7. Files

New:
- `lib/portfolio/{content.ts, store.ts, published.ts, files.ts, image.ts, html.ts}`
- `app/api/portfolio/generate/route.ts`
- `app/(app)/portfolio/page.tsx`, `app/(app)/portfolio/github-connected/page.tsx`
- `components/portfolio/{PortfolioBuilder, ModeStep, ResumeStep, ClassesStep, ProjectsStep, DetailsStep, CourseStep, FilesStep, StoryStep, ReviewStep, Preview, Publish, SizeMeter}.tsx`

Changed:
- `lib/modules.ts`: `portfolio` first in `jobs`.
- `lib/config/models.ts`: `portfolio` page key and default model.
- `next.config.ts`: redirect `/job-scout/github-connected` to `/portfolio/github-connected`; `/job-scout?tab=portfolio` to `/portfolio`.
- `lib/scout/github-state.ts`: `safeReturnPath` default becomes `/portfolio`.
- `components/scout/JobScout.tsx`: three tabs; link cards to `/portfolio?mode=career` (Profile tab) and `/portfolio?mode=project` (Projects tab).
- `components/scout/ProjectsTab.tsx`: Polish pane removed; scaffold pane kept.
- `components/scout/ProfileTab.tsx` or `profile-store.ts`: merge `PublishedWork` skills.
- `components/jobs/JobAppAssistant.tsx` and `app/api/applications/route.ts`: opt-in published-work context.
- `lib/scout/profile-store.ts`: `PortfolioRecord` v1 migrates into the new store.

Removed:
- `components/scout/PortfolioTab.tsx`, `lib/scout/portfolio-html.ts`,
  `app/api/scout/portfolio/route.ts`, `app/api/scout/polish/route.ts`.

## 8. Error handling

- Upload caps and size meter refuse before generation with the exact file
  names over the limit.
- Generation failures show the existing plain-language copy and keep the
  student's inputs.
- Push errors reuse `PushError` copy; `name-taken` offers a rename field
  inline. Pages enablement failure falls back to the settings link as today.
- A lost IndexedDB (cleared storage) degrades to "re-add these files", never
  to a crash; records without blobs stay listed with their URLs.

## 9. Testing

- Unit: content schemas and v1 migration; size meter and file-role layout;
  both renderers (escaping, `safeHref`, fixed image paths, figure and path
  validation); photo resize (jsdom canvas stub); published-work store and
  subscribe; `safeReturnPath` default.
- Route: `/api/portfolio/generate` under `CHATISA_MOCK_LLM=1` for both modes,
  including oversize and role validation.
- E2E (`tests/e2e/portfolio.spec.ts`): full career wizard and full showcase
  wizard with `CHATISA_MOCK_GITHUB=1` and the intercepted api.github.com,
  asserting the pushed file set and that a showcase appears in Job Scout's
  profile skills and in JobApp Drafter's opt-in toggle. Axe checks on each
  step at desktop and 320 px.
- Existing Job Scout e2e updated for the removed tab and Polish pane.

## 10. Out of scope

Multiple themes, custom domains, Jekyll, DOCX resumes, pushing career
projects as separate repositories, server-side storage of any content,
Interview Mentor integration.
