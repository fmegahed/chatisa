import { defineConfig, devices } from "@playwright/test";

/**
 * LIVE hardening runs: the real models, the real providers, real money.
 *
 * This is a second, opt-in configuration and deliberately NOT part of
 * `npm run test:e2e`. The ordinary e2e suite runs with CHATISA_MOCK_LLM=1 and
 * proves the app's own behaviour deterministically; nothing there ever calls a
 * provider. That is the right default and this file does not change it.
 *
 * What it cannot prove is the thing that actually breaks in front of students:
 * whether a real model, given a real task, drives our tools to a correct
 * artifact. A canned answer never fails to install a package, never emits R that
 * does not parse, never loops on a hosted sandbox. So these runs exist to find
 * defects, and they are expected to be slow, noisy, and occasionally to fail for
 * reasons outside the app. Treat a failure here as a lead to investigate, not as
 * a red build.
 *
 *   npm run test:live                      # everything
 *   npm run test:live -- --grep "tutor"    # one area
 *
 * The server is NOT started here, on purpose. Real runs take many minutes and
 * are usually driven against a server the operator is already watching the logs
 * of. Start one yourself, with real keys and WITHOUT the mock flag:
 *
 *   AUTH_TEST_MODE=1 AUTH_URL=http://localhost:3200 \
 *     CHATISA_DATA_DIR=tests/e2e/.data-live \
 *     npm run dev -- --port 3200
 *
 * Every spec asserts the target is not in mock mode before it trusts anything
 * (see assertLiveServer), because a live suite quietly pointed at a mock server
 * produces confident, meaningless passes. That has happened here before.
 */

const BASE_URL = process.env.CHATISA_LIVE_BASE_URL ?? "http://localhost:3200";
const STORAGE_STATE = "tests/live/.auth/user.json";

export default defineConfig({
  testDir: "tests/live",
  // One at a time. These runs cold-boot WASM runtimes, drive hosted provider
  // sandboxes, and share one account's rate limits; in parallel they mostly
  // measure contention. Serial also keeps the transcript readable.
  fullyParallel: false,
  workers: 1,
  // Never retry. A retry on a live run spends money again and, worse, can hide
  // exactly the intermittent provider behaviour these runs exist to surface.
  retries: 0,
  // A hosted deck build plus a browser analysis loop legitimately runs for
  // minutes. Individual waits carry their own tighter deadlines.
  timeout: 15 * 60_000,
  expect: { timeout: 60_000 },
  reporter: [["list"], ["json", { outputFile: "tests/live/.artifacts/report.json" }]],
  outputDir: "tests/live/.artifacts/trace",
  use: {
    baseURL: BASE_URL,
    // Always keep the trace: the point of the run is the evidence.
    trace: "on",
    video: "off",
    screenshot: "only-on-failure",
    ...devices["Desktop Chrome"],
  },
  projects: [
    { name: "live-setup", testMatch: /live\.setup\.ts/ },
    {
      name: "live",
      testIgnore: /live\.setup\.ts/,
      use: { storageState: STORAGE_STATE },
      dependencies: ["live-setup"],
    },
  ],
});
