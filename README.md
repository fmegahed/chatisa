# ChatISA

AI tools for Miami University students, built by the Farmer School of
Business: coding help, an in-browser data workbench (Python, R, SQL via
WebAssembly), exam prep, project coaching, interview practice, resume
drafting, research-capable general chat, and model comparison. Free to
students, sponsored by the university.

## Layout

| Folder | What it is |
| --- | --- |
| `web/` | The application: Next.js 16, React 19. Start at `web/README.md`. |
| `docs/` | Design docs, decision log, migration log, `operations.md` (running it). |
| `assets/` | Source brand assets (deck template, TikZ figures). |
| `legacy/` | The retired Streamlit app (2023-2026), kept for reference. |

## Running it

- Development: see `web/README.md`.
- Production (the Windows server): see the Deployment section of
  `docs/operations.md`. The short version: run
  `node scripts/make-deploy-bundle.mjs` in `web/` on a dev machine, then
  follow `INSTALL.txt` inside the bundle it produces. The server needs
  Node.js only: no npm, no Python, no R (students' code runs in their own
  browsers).

Educational use. See `LICENSE`.
