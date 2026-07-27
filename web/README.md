# ChatISA web app

Next.js 16 / React 19. Students' Python, R, and SQL run in their own
browsers on self-hosted WebAssembly runtimes; the server holds the model API
keys, authentication, and a content-free usage database (conversation
content is never stored server-side, ADR-022).

## First-time setup (dev machine)

```bash
npm install
npm run setup:runtimes   # downloads the WASM runtimes + package mirrors into public/ (~215 MB, gitignored)
# create .env.local with the variables below
npm run dev
```

Required env for a full-featured dev server (`lib/config/env.ts` validates):
`AUTH_SECRET`, `AUTH_URL=http://localhost:3000`, `AUTH_GOOGLE_ID`,
`AUTH_GOOGLE_SECRET`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`GOOGLE_API_KEY`, `HF_TOKEN`, `DEEPGRAM_TOKEN`, `CHATISA_DATA_DIR`.
Optional: guest passes (`../docs/operations.md`), rate-limit overrides
(`lib/ratelimit.ts`). Test-only, never in production: `AUTH_TEST_MODE`,
`CHATISA_MOCK_LLM`, `CHATISA_PROXY_ALLOW_LOCAL`.

## Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | Dev server (Turbopack). |
| `npm test` | Unit tests (vitest). |
| `npm run test:e2e` | Playwright suite against a mock-LLM dev server (no spend). `CHATISA_LIVE_NET=1` adds the live-network tests. |
| `npm run lint` / `npm run typecheck` | eslint / tsc. |
| `npm run setup:runtimes` | Mirror Pyodide, webR, SQLite Wasm and the package sets into `public/runtimes/`. |
| `npm run verify:models -- <filter>` | LIVE calls to catalog models (costs pennies; required by ADR-018 before shipping a model swap). |
| `node scripts/make-guest-passes.mjs` | Mint collaborator invite links (`../docs/operations.md`). |
| `node scripts/make-deploy-bundle.mjs` | Build + assemble the production folder for the Windows server. |

## Conventions that bite

- **Read `AGENTS.md`**: this Next.js version differs from what you (or your
  tooling) may assume; the bundled docs in `node_modules/next/dist/docs/` are
  the authority.
- Every change lands with the gate green: `typecheck`, `lint`, `npm test`,
  and `CI=1 npx playwright test`; a migration-log entry
  (`../docs/development/migration-log.md`) records each slice.
- No em dashes in student-facing copy. Miami brand tokens in
  `app/globals.css`. WCAG 2.1 AA; axe runs in the e2e suite.
- Model catalog changes need live verification (`verify:models`) and are
  governed by ADR-005/ADR-018 (see `../docs/development/decision-log.md`).

## Deployment

`node scripts/make-deploy-bundle.mjs`, then follow `INSTALL.txt` inside
`deploy/chatisa-app/`. Full operator docs: `../docs/operations.md`.
