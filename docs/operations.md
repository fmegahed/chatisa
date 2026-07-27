# ChatISA operations

Operator procedures that are not build slices. Grows as deployment nears.

## Guest magic passes (external collaborators)

Guests sign in through invite links (`/guest?pass=<token>`) instead of a
Miami Google account. The server stores only SHA-256 hashes of the tokens
plus a REQUIRED expiry date; without an expiry, guest access is off.

### Minting

```bash
cd webapp/web
node scripts/make-guest-passes.mjs 10 2026-09-30 https://your-domain
```

Prints the shareable links and the two env lines. **Save the links
immediately**: the server keeps only hashes and cannot regenerate them.
Add the env lines to the deployment (or `.env.local` in development):

> **The mistake that broke guest login on 2026-07-27:** only the env line
> (the hashes) was saved, and links were later reconstructed by pasting the
> hashes after `?pass=`. Hashed links can never work, and the real tokens
> cannot be recovered, so the whole set had to be re-minted. Check before
> sharing: a working link has a 32-character code after `pass=`; a
> 64-character code is a hash and is dead on arrival.

```
CHATISA_GUEST_PASS_HASHES=<64-hex-hash>,<64-hex-hash>,...
CHATISA_GUEST_EXPIRES=2026-09-30
```

### Identity and auditing

Link position is identity: the third link signs in as
`guest-3@guest.chatisa`, and its activity and spend appear under that email
in the usage events. Hand out ONE link per collaborator and note who got
which number. Per-user rate limits apply to guests exactly as to students.

### Revoking and rotating

- Revoke one guest: replace their hash in `CHATISA_GUEST_PASS_HASHES` with
  64 zeros. Positions must be preserved (position = identity), so never
  delete an entry outright.
- End the trial: remove the variable, or simply let the expiry pass.
- Rotate: mint a fresh set and replace both variables. Old links die with
  their hashes.

Env changes take effect on the next server restart (or immediately on
platforms that hot-reload environment variables).

### Security shape (for reviewers)

Authentication has three independent layers, and guest access changes none
of them: `proxy.ts` middleware walls every non-public path; the `(app)`
layout re-verifies the session server-side; every privileged API route calls
`auth()` itself. The public surface added for guests is exactly one page,
`/guest` (exact path segment in the middleware matcher), which renders an
invitation; entering still requires a token that hashes into the configured
list before its expiry, verified server-side in the `guest-pass` credentials
provider. `tests/e2e/guest.spec.ts` includes a probe test asserting module
paths and API routes stay walled for unauthenticated visitors.

## Relaxed per-user rate limits (2026-07-24)

Defaults are set to never touch a fast human, only runaway clients:
chat 60/min, editor completions 120/min, exam uploads 30/min, exam
generation 20/min, speech token 20/min, speech synthesis 30/min (kept low
deliberately: Deepgram concurrency is a shared resource), hosted file
downloads 60/min. Every limit is overridable per deployment via the
`CHATISA_*_LIMIT_PER_MINUTE` variables in `lib/ratelimit.ts`.

## Deployment (Windows server)

The server needs ONE program: Node.js LTS (20.9+). No npm, no Python, no R,
no Conda: students' code runs in their browsers on WASM runtimes the app
serves as static files. TLS is terminated by our own launcher on port 443
using the SAME certificate files the old Streamlit command used, and the
existing Task Scheduler job keeps working; only the bat it points at changes.

### Build the bundle (dev machine)

```bash
cd webapp/web
node scripts/make-deploy-bundle.mjs
```

Produces `deploy/chatisa-app/` (~350 MB): the standalone Next server with
its pruned node_modules (native better-sqlite3 for Windows x64 included),
`public/` (WASM runtimes), `assets/`, the launcher `chatisa-server.mjs`, a
`chatisa.bat`, `chatisa.env.example`, and `INSTALL.txt` with the full
step-by-step server instructions (install Node, copy folder, fill
chatisa.env, double-click test, repoint the scheduled task, verify
/api/health).

### How it runs

`chatisa.bat` -> `node chatisa-server.mjs`, which loads `chatisa.env`,
starts the Next standalone server on 127.0.0.1:3000 (restarting it on
crash), and terminates HTTPS on 443, streaming requests through (chat
streaming is unbuffered). The bat loops, so even a launcher exit comes back
in 5 seconds. Logs append to `chatisa.log` beside the bat; delete when
large.

### The three files an operator touches

1. `chatisa.env` - all secrets and paths (from `chatisa.env.example`).
   `CHATISA_DATA_DIR` names the folder holding the usage database: back it
   up. Never set the test flags there.
2. `chatisa.bat` - only if the install folder differs from the default.
3. The Task Scheduler action - points at `chatisa-app\chatisa.bat`.

### Updating

Rebuild the bundle, End the task, swap the folder (keep `chatisa.env`),
Run the task, check `https://chatisa.fsb.miamioh.edu/api/health`.
