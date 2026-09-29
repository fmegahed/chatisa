# Video to-do: ChatISA executive overview, fall 2026

Status on September 29, 2026: everything is filmed except the Job Scout job
board. We are waiting for the automatic harvest on Sunday, October 4, the
first one with the new RapidAPI key, so the board on camera includes
employer postings again, not only USAJobs.

The video files live outside this repository, in
`C:\Users\megahefm\Documents\claude\chatisa\videos\chatisa\`:

- the narration: `executive-fall2026/script.md`, approved: conversational,
  no marketing terms, no guest features;
- the filming scenarios: `capture_f26.py`;
- the footage: `captures/f26_*.webm`.

## Monday, October 5

### You (professor)

- [ ] Copy **all three** job files from the production server into
      `videos\chatisa\.capture-data\`: `scout.db`, `scout.db-wal` and
      `scout.db-shm`. The `-wal` file holds the newest records; a copy
      without it looks out of date.
- [ ] Copy **only** those three files. Never copy `chatisa.db`: it is the
      real user database and does not belong near a camera.

### Claude

- [ ] Confirm the Sunday run in `scout_runs`: status "completed" and
      `activejobs_found` above 0. If employer postings are still missing,
      stop and report why before filming.
- [ ] Write and film `f26_board`: the This Week's Jobs tab, this week's
      count, match cards, a saved job, and the handoff to JobApp Drafter.
- [ ] Fill in the numbers the script leaves in brackets:
  - the posting count, from the board as filmed;
  - the model count (19 on September 25), from
    `web/lib/config/models.ts`.
- [ ] Keep the job-board line ("from employers' own hiring pages and from
      USAJobs") only if the board shows both sources.
- [ ] Record the narration in one take with **ElevenLabs v4**
      (`model_id: eleven_v4`), same voice as before (Rachel). It was checked
      on September 29: plain speech and the with-timestamps endpoint both
      work.
- [ ] Make the title and closing cards, build the timeline, assemble the
      video, and QC it:
  - frames under every caption;
  - audio levels;
  - 1920x1080 at 30 fps.
- [ ] Send the video for review. Expect two or three rounds.

## Filmed so far (local v6.9.2 build, September 25)

- Home page.
- Job Scout profile: Finance and Business Analytics majors; the core
  shortcut; FIN 301 shows as Strong and 200-level courses as Working.
- Skills from GitHub: "You wrote 100% of 24 commits", with each suggestion
  citing its file.
- Portfolio Builder: Import from GitHub, generate, publish (to a fake
  GitHub), and a site tour.
- JobApp Drafter, including the check that flags a resume line it could
  not match.
- Interview Mentor: a spoken answer and the feedback report.
- Montage: the Coding Studio chart, Coding Tutor, Exam Prep, Project
  Assistant (cut at the top of the workspace), Ask Anything reading a
  notebook, and AI Comparison.
- Keeping it current: all 19 models (Coding Tutor's list), and the catalog
  review packet on GitHub.

## Capture notes

- Start the capture server with the harvest keys **blank**:
  `RAPIDAPI_KEY= USAJOBS_API_KEY=`. Otherwise it runs its own harvest and
  spends quota.
- Only one Next.js dev server can run per folder. Stop the capture server
  (port 3200) before running the end-to-end tests.
