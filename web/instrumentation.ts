/**
 * Next.js instrumentation hook — runs once at server start.
 * Fails fast on malformed configuration; reports (names only) which
 * provider features are unavailable.
 */
export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    const [{ assertBootEnv }, { logger }] = await Promise.all([
      import("./lib/config/env"),
      import("./lib/log"),
    ]);
    const report = assertBootEnv();
    // Open the database and apply pending migrations before serving traffic.
    const { getDb } = await import("./lib/db");
    getDb();

    // Job Scout's weekly harvest (design 2026-07-28). Self-guarding: no-op
    // under test/mock mode or when no source key is configured.
    const { startScoutScheduler } = await import("./lib/scout/scheduler");
    startScoutScheduler();

    if (process.env.CHATISA_MOCK_LLM === "1") {
      logger.error(
        {},
        "MOCK MODEL ENABLED: every AI response on this server is canned test output, not a real model. Unset CHATISA_MOCK_LLM before using this server for anything real.",
      );
      // A deterministic Job Scout feed, so e2e has postings without any
      // harvest. Mock mode only; the real feed comes from the Sunday run.
      const { seedScoutFixtures } = await import("./lib/scout/mock-fixtures");
      seedScoutFixtures();
    }

    if (report.missingProviders.length > 0) {
      // Not a failure: the app runs with whatever providers are configured
      // and hides the rest. Says which models students lose, so the operator
      // can decide whether that is intended.
      const { describeProviderAvailability } = await import("./lib/providers");
      const availability = describeProviderAvailability(
        report.missingProviders,
      );
      logger.warn(
        {
          missingProviderKeys: report.missingProviders,
          hiddenModelCount: availability.hiddenModelCount,
          availableModelCount: availability.availableModelCount,
        },
        "Configuration notice: some providers have no key, so their models are hidden from students. Set the listed variables in .env.local to enable them.",
      );
    } else {
      logger.info("Environment validated; all provider keys present.");
    }
  }
}
