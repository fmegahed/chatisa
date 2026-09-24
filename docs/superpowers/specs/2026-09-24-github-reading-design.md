# Reading a student's GitHub: skills in Job Scout, import in the Portfolio Builder (design)

**Date:** 2026-09-24. **Status:** approved in conversation, section by section, by the professor (lead-engineer roadmap, project 4).
**Releases:** part A (skills in Job Scout) as v6.8.0; part B (Portfolio Builder import) as v6.9.0.

## Goal

Let a student's own public GitHub work count, honestly. Job Scout suggests
skills from repositories the student picks, and the student confirms each
one; the Portfolio Builder can start a project or showcase from an existing
repository instead of uploads. This matters most for guests, who have no
Miami courses, and for any student whose best evidence is code they wrote
outside class.

## Constraints (professor)

- Don't break the running app; keep it slick and usable; pass the
  accessibility requirements (axe A/AA, desktop and 320 px); be useful.
- No overselling: repository evidence is weighed with the same care as the
  course-level rule of v6.7.0.
- "The student should have the right to overwrite a level; they know what
  they know." (2026-09-24)

## Decisions made with the professor

1. **Scope:** both parts, sharing one repository reader; built and released
   as two parts, A first.
2. **Depth:** a repository skill defaults to applied (tops out at Working,
   like built projects and published sites today). It can be an anchor
   (can reach Strong) only when the repository is substantially the
   student's own work and the evidence is in real code (rules below).
3. **The student decides:** the guards set the suggested level; the student
   may choose any level. A level raised above the suggestion is recorded as
   the student's own call and labelled "set by you", like a level override.
4. **Which repositories:** the student picks up to 5 of their own public,
   non-fork repositories per analysis.
5. **Portfolio import:** read the repository in as if uploaded, and link
   back to it. The student's repository is never written to.
6. **Architecture:** the browser reads GitHub with the student's own token;
   only a size-capped repository summary reaches our server. The v6.3.0
   invariant stands: the token never leaves the browser, and the token key
   is read only in `lib/scout/github*.ts`.

## Architecture

### Shared repository reader (`lib/scout/github-read.ts`, browser)

Lives beside the existing GitHub code so the token invariant holds.

- `listOwnRepos(conn)`: the student's public repositories they own
  (`GET /user/repos?visibility=public&affiliation=owner&sort=pushed`,
  paginated to 100), forks and archived repositories removed. Each entry:
  full name, description, main language, pushed date, default branch.
- `readRepo(conn, fullName, opts)`: builds a `RepoSummary`, capped at
  150,000 characters in total (professor, 2026-09-24):
  - basics: name, description, topics, default branch, fork / template /
    archived flags;
  - languages: `GET /repos/{o}/{r}/languages` (bytes per language);
  - authorship: `GET /repos/{o}/{r}/contributors` gives the student's
    commit count and the total (bots excluded);
  - file list: `GET /repos/{o}/{r}/git/trees/{branch}?recursive=1`, paths
    and sizes, first 500 entries (GitHub truncates very large trees; the
    summary says so);
  - README: first 8,000 characters;
  - dependency files, read in full when present: `requirements.txt`,
    `pyproject.toml`, `environment.yml`, `renv.lock`, `DESCRIPTION`,
    `package.json`;
  - up to 6 analysis or code files by extension (`.py .R .Rmd .qmd .ipynb
    .sql .js .ts .jl`), largest first, skipping vendored and generated
    paths (`node_modules/`, `dist/`, `.venv/`, `renv/library/`), 45,000
    characters each; `.ipynb` reduced to its code and markdown cells,
    outputs removed.
- Download size (professor, 2026-09-24):
  - files up to 6 MB are read automatically;
  - larger code files are skipped and listed per repository with "Read it
    anyway", which re-reads that repository including the chosen file, up
    to GitHub's 100 MB limit for the contents API
    (https://docs.github.com/en/rest/repos/contents);
  - the model still sees at most 45,000 characters of each file's code, so
    this mainly helps notebooks whose size is embedded plots.
- File reads use the contents API with the raw media type. All requests
  carry the student's token; each student spends their own GitHub quota
  (5,000 requests an hour when authenticated).

### Part A: skills in Job Scout

**Route `POST /api/scout/repo-skills`** (new; same shape as
`/api/scout/resume-skills`): signed-in users, guests included; body is
`{ modelId, repos: RepoSummary[] }` with at most 5 repositories. The
server clips every field to the caps above (clip, never reject), fences
the README and file contents as data with a per-request nonce (as for
resumes and syllabi), and makes one model call per repository.

**The model's task:** map the repository to taxonomy skills, at most 8,
each with a level and a short student-voice evidence phrase that names the
file it comes from ("trained a gradient-boosted churn model in
`src/model.py`").

**Fixed guards, applied on the server after the model** (lowering, not
rejecting):

1. The skill must resolve to a taxonomy id (`resolveSkillId`), or it is
   dropped.
2. Tool skills (a language, library or product) need proof beyond the
   README: the tool appears in a dependency file, in an import or library
   call in a read file, or in the language byte counts. Otherwise dropped.
3. **Anchor rule:** a suggested anchor stays an anchor only when both hold:
   - the repository is substantial: not a fork, the student authored at
     least 60% of the commits, and at least 10 commits;
   - the evidence names a code or data file present in the summary (not
     the README).
   Otherwise the suggestion becomes applied.
4. At most 3 anchors per repository; any beyond that become applied, the
   strongest kept by the model's order.
5. A repository with no code or data files read gives exposure only, and
   the response says why.

The response carries, per repository: the suggestions (skill, suggested
level, evidence), the authorship share and commit count, whether anchors
were possible and why, and any per-repository error. Nothing is stored
server-side; a content-free usage event is recorded (event type, model,
token counts, repository count; never names or content).

**Profile storage:** `ProfileExtra` gains:
- `source: "github"`;
- `repo?: string` (the `owner/name`);
- `setByStudent?: boolean` (true when the confirmed level is above the
  suggestion).

Re-analyzing a repository replaces that repository's earlier extras
instead of adding duplicates. Extras need no migration: older profiles
simply have none with this source.

**Counting:** a confirmed repository extra counts like any extra today: an
anchor counts as an anchor (can reach Strong), applied tops out at Working
(`CAP_WITHOUT_ANCHOR`), exposure at Introduced. The skills panel names the
source: "from ada/churn-model", plus ", set by you" when `setByStudent`
(`SkillsPanel` currently shows unknown sources as "added by you"; the new
source gets its own label).

### Part B: import in the Portfolio Builder

- **Career, Projects step:** each project card gets "Import from GitHub"
  beside "Add files".
- **Showcase, Files step:** the same.

The import panel:
1. pick one repository;
2. see its files with sizes, the README and main code files pre-ticked up
   to the existing per-file and total limits, with the existing size meter;
3. "Import N files".

Imported files enter the draft through the existing intake
(`prepareFile`), so file roles, generation, preview and publishing work
unchanged; binaries keep the existing binary handling.

The page links back to the original repository:
- career: the project's existing `externalUrl` field is set to the
  repository's address;
- showcase: a new optional draft field `sourceRepoUrl`, rendered as
  "Original repository" under the files and stored with the site record
  (absent on older drafts).

The source repository is only read, never written.

## Screens

### Job Scout, My Profile: "Your GitHub (optional)"

A new block under the resume section.

- **Not connected:** one line ("Suggest skills from your public
  repositories. Nothing about them is stored on our server.") and the
  existing `GithubConnect` button, returning to this block.
- **Connected:**
  - "Connected as ada";
  - a checklist of their repositories, 10 shown, then "Show all" and a
    filter box. Each row shows name, main language, description and
    updated date;
  - after 5 are ticked the rest are disabled, with a text note saying why;
  - the button "Suggest skills from N repositories".
- **Progress:** per repository ("Reading ada/churn-model", "Suggesting
  skills"), in a polite live region.
- **Results:** grouped by repository, using the resume suggestion cards.
  - Each group states its rule in words: "You wrote 84% of 57 commits here,
    so this repository can support an anchor", or "You wrote 30% of the
    commits here, so its skills are suggested as applied".
  - Each card starts at the suggested level; the student may pick any
    level, then confirm or dismiss.
  - "Add all as suggested" confirms at the suggested levels.
  - Focus moves to the results heading when they arrive.
- **Per-repository problems, inline, never failing the rest:**
  - not readable (private, deleted or renamed);
  - GitHub rate limit, with the reset time;
  - expired or revoked token, which offers the existing reconnect;
  - model failure, with "Try again" for that repository only.

### Portfolio Builder

The import panel above is an inline disclosure, not a modal:
- the toggle button exposes `aria-expanded`;
- focus returns to the button when the panel closes;
- files over the limit stay unticked, with their size shown.

### Accessibility (both parts)

- native checkboxes and radios with visible labels;
- headings for each block;
- live regions for progress;
- errors in `role="alert"` and focused, as elsewhere in the app;
- no colour-only state;
- axe A/AA at 1280 and 320 px with no sideways page scroll, as for the
  course checklist.

## Limits and cost

- **Per request:** 5 repositories and at most 150,000 characters of summary
  per repository, enforced in the browser and clipped again on the server.
- **Rate limit:** a per-student limit on the new route, env-overridable
  like the other scout limits (`CHATISA_SCOUT_REPO_LIMIT_PER_MINUTE`);
  raised in the e2e config.
- **Cost:** one model call per repository. Measured with the app's price
  table for a full-size summary (about 37,500 input tokens at four
  characters a token) and 1,500 output tokens per repository:
  - GPT-6 Luna: $0.004;
  - DeepSeek V4.1 Flash: $0.013;
  - Gemini 3.8 Flash: $0.034;
  - GPT-6 Sol (Job Scout's default): $0.09;
  - Claude Sonnet 5: $0.09;
  - Claude Opus 5.5: $0.18.

  Most repositories are smaller than the cap and cost less. Five
  full-size repositories with the default model: about $0.45. The student
  chooses the model, as for resumes.

## Privacy and security

- **Token:** stays in the browser.
- **Summaries:** sent to the chosen model provider for that request only;
  nothing stored on our server; the usage log is content-free.
- **Untrusted input:** repository text is fenced as data and cannot
  instruct the model; a repository is untrusted input exactly like a
  resume.
- **Portfolio page:** the page is rendered by the existing escaping
  templates; the repository link is validated as an `https://github.com/`
  address before use.

## Testing

### Unit (test first)

- **Summary builder,** against captured GitHub API responses (fixtures,
  scrubbed):
  - caps;
  - notebook stripping;
  - file choice and vendored-path skipping;
  - fork exclusion;
  - authorship share with bots excluded;
  - truncated trees.
- **Guards:**
  - unknown skills;
  - tool proof (README-only mention dropped; dependency or import kept);
  - anchor rule both ways (not substantial; README-only evidence);
  - the 3-anchor cap;
  - exposure-only for repositories without code;
  - clipping oversize input.
- **Route,** with the mock model misbehaving the way real models do:
  - invented skills;
  - README-only anchors;
  - file paths not in the summary;
  - more than 8 skills.
- **Profile:**
  - re-analysis replaces a repository's extras;
  - `setByStudent` is set only when the level is raised;
  - the skills panel labels.

### End to end

Against the existing fake GitHub API (Playwright-intercepted; no traffic
reaches GitHub):
- connect, list, pick 3, confirm cards, raise one level, see "set by you";
- the error cases: private repository, rate limit, expired token;
- part B: import into a career project and a showcase, and check the page
  links back;
- axe at 1280 and 320 px.

### Live check before each release

One real, read-only run against one of the professor's public
repositories, with the professor's OK.

## Out of scope

- private repositories (the connection's `public_repo` scope and the
  professor's v6.3.0 decision);
- repositories the student contributed to but does not own;
- organisation repositories;
- storing anything about a student's GitHub on the server;
- writing to an imported repository;
- re-checking later whether a confirmed repository still exists (the
  student removes a skill themselves).
