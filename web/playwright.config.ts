import { defineConfig, devices } from "@playwright/test";

const STORAGE_STATE = "tests/e2e/.auth/user.json";

// The e2e web server is `next dev`. A production build cannot be used for e2e by
// design: the env validator refuses to boot a production server with the test-only
// AUTH_TEST_MODE / CHATISA_MOCK_LLM (a deliberate safety guard, verified 2026-07-24),
// so the suite must run against dev with those test vars. See the timeout note.
const IS_CI = !!process.env.CI;

export default defineConfig({
  testDir: "tests/e2e",
  fullyParallel: true,
  // Locally, no retries: a flake surfaces immediately so it is not ignored. In CI,
  // two retries absorb the one residual flake source under `next dev` at full
  // parallelism (Turbopack's cold per-route / lazy-chunk compile occasionally
  // exceeding even the 15s assertion timeout). Playwright still reports a retried
  // test as "flaky", so nothing is masked; the warmup below makes it rare.
  retries: IS_CI ? 2 : 0,
  // CI also caps parallelism: the suite now includes tests that cold-boot WASM
  // runtimes (Ask Anything's Pyodide loop, the sandbox), and at full fan-out on
  // one box the CPU saturation pushes streaming assertions past their timeouts
  // faster than retries can absorb. Four workers trades a longer wall clock for
  // a deterministic pass.
  workers: IS_CI ? 4 : undefined,
  // Under the full parallel run against `next dev`, the first hit to a route (and
  // the first client import of a heavy lazy chunk such as CodeMirror) compiles it,
  // and that latency can exceed Playwright's 5s default assertion timeout, so a
  // `toBeVisible` right after navigating flaked while passing in isolation. A 15s
  // assertion timeout absorbs first-compile; per-test timeout is widened to match.
  // The auth setup (a dependency of every project) also warms the heaviest routes
  // once, up front, so a cold compile no longer happens under parallel contention.
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:3100",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "npm run dev -- --port 3100",
    url: "http://localhost:3100/api/health",
    // Never adopt a server this run did not start, and always tear down the
    // one it did. A lingering mock-mode server was picked up by a later
    // "npm run dev" and served canned answers that looked real.
    reuseExistingServer: false,
    // Raised from 120s to 600s on 2026-07-26 (professor's instruction), for the
    // same reason as the 600s in tests/e2e/auth.setup.ts: this budget covers
    // `next dev` booting AND compiling /api/health, which on a cold Turbopack
    // cache is the slow part. If it trips, the suite dies before a single test
    // runs, so the whole run reports as broken for a reason that is not a defect.
    // On a warm machine the server is ready in seconds and this number never
    // matters; it only decides how patient a cold run is allowed to be.
    timeout: 600_000,
    env: {
      // Test-only auth: fake login provider, fixed non-secret signing key.
      // Real .env.local values (if present) are overridden where needed.
      AUTH_TEST_MODE: "1",
      AUTH_URL: "http://localhost:3100",
      AUTH_SECRET:
        process.env.AUTH_SECRET ??
        "chatisa-e2e-only-not-a-real-secret-0123456789",
      // Deterministic stand-in model: no provider calls, no spend.
      CHATISA_MOCK_LLM: "1",
      // Deterministic stand-in GitHub OAuth: the start route skips github.com
      // and redirects straight to the callback with a mock code.
      CHATISA_MOCK_GITHUB: "1",
      // Keep test data out of the development database.
      CHATISA_DATA_DIR: "tests/e2e/.data",
      // The suite uploads more often than a person would; the limiter itself
      // is covered by its own unit tests.
      CHATISA_UPLOAD_LIMIT_PER_MINUTE: "200",
      CHATISA_SCOUT_PROJECT_LIMIT_PER_MINUTE: "200",
      CHATISA_EXAM_LIMIT_PER_MINUTE: "200",
      // Every chat module runs in parallel through one shared account, and the
      // Ask Anything loop sends a second request per tool turn; a production
      // per-student ceiling (20/min) legitimately trips under that fan-out.
      CHATISA_CHAT_LIMIT_PER_MINUTE: "300",
      // Three specs now read PDFs (exam, interview, JobApp), so the worker
      // pool queue is given headroom to prevent flaky "server is busy" 503s
      // at peak parallelism. Queue length adds no extra child processes.
      CHATISA_PDF_QUEUE: "80",
      // Guest magic-pass flow (tests/e2e/guest.spec.ts): the hash of the
      // fixed test token "e2e-guest-pass-1234567890abcdef".
      CHATISA_GUEST_PASS_HASHES:
        "81caea1ffe5b339a783279eeb4451ba7eb234eee27c0f92c835f556ca2e9588c",
      CHATISA_GUEST_EXPIRES: "2099-01-01",
      // Lets the Python web-proxy e2e fetch THIS test server through the
      // proxy (127.0.0.1 target vs localhost origin), keeping the test
      // offline. Never honored in production (see proxyAllowsLocal).
      CHATISA_PROXY_ALLOW_LOCAL: "1",
    },
  },
  projects: [
    { name: "setup", testMatch: /auth\.setup\.ts/ },
    {
      name: "desktop",
      use: { ...devices["Desktop Chrome"], storageState: STORAGE_STATE },
      dependencies: ["setup"],
    },
    {
      name: "mobile-320",
      use: {
        ...devices["Desktop Chrome"],
        viewport: { width: 320, height: 720 },
        storageState: STORAGE_STATE,
      },
      dependencies: ["setup"],
      // HTTP contract tests carry no layout meaning, so they run once in the
      // desktop project. Running them twice against one account would also
      // trip the upload rate limit, which is the limiter working correctly.
      testIgnore: /exam-(upload|generate)\.spec\.ts/,
    },
  ],
});
