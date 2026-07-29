import "server-only";
import { latestSuccessfulScoutRun } from "@/lib/db";
import { logger } from "@/lib/log";
import { runHarvest } from "./harvest";

/**
 * In-process weekly scheduler (design §4.1; the app is one long-lived Node
 * process on the Windows VM, same single-process reasoning as
 * lib/ratelimit.ts). Timers are derived from persisted state — the launcher
 * restarts the child on crash, so an in-memory "already ran" flag would
 * either never fire or double-fire after a 2 AM restart.
 */

const CHECK_INTERVAL_MS = 60 * 60 * 1000;
/** Target: Sunday 02:00 in America/New_York (user decision, 2026-07-28). */
const TARGET_DOW = 0;
const TARGET_HOUR = 2;

/** America/New_York wall-clock parts for an instant, DST handled by Intl. */
function easternParts(d: Date): { dow: number; hour: number; ymd: string } {
  const fmt = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    weekday: "short",
    hour: "numeric",
    hourCycle: "h23",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  });
  const parts = Object.fromEntries(
    fmt.formatToParts(d).map((p) => [p.type, p.value]),
  );
  const dows = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  return {
    dow: dows.indexOf(parts.weekday),
    hour: Number(parts.hour),
    ymd: `${parts.year}-${parts.month}-${parts.day}`,
  };
}

/**
 * The most recent Sunday-02:00-Eastern boundary at or before `now`,
 * identified by its Eastern calendar date (a wall-clock label, deliberately
 * not an instant: comparing instants across a DST shift is how a 2 AM job
 * fires twice or never).
 */
export function lastDueLabel(now: Date): string | null {
  // Walk back hour by hour until we cross the target wall-clock moment.
  // ">= TARGET_HOUR" rather than "===" because on spring-forward Sundays
  // (e.g. 2026-03-08) 2 AM Eastern does not exist: clocks jump 1:59 to
  // 3:00, and an equality check would skip that week's harvest entirely.
  // 8 days of hours bounds the loop even across DST transitions.
  for (let back = 0; back < 8 * 24; back++) {
    const probe = new Date(now.getTime() - back * 3_600_000);
    const parts = easternParts(probe);
    if (parts.dow === TARGET_DOW && parts.hour >= TARGET_HOUR) {
      return parts.ymd;
    }
  }
  return null;
}

/**
 * Due when no successful run has happened since the last Sunday 2 AM
 * boundary. Catch-up at boot falls out for free: if the server was down at
 * 2 AM, the first hourly check after boot sees a stale lastSuccess.
 */
export function isHarvestDue(
  now: Date,
  lastSuccessIso: string | null,
): boolean {
  const due = lastDueLabel(now);
  if (!due) return false;
  if (!lastSuccessIso) return true;
  const lastSuccess = new Date(lastSuccessIso);
  // A future timestamp means clock skew; do nothing rather than double-run.
  if (lastSuccess > now) return false;
  // A success at-or-after the boundary carries the boundary's label.
  return lastDueLabel(lastSuccess) !== due;
}

let started = false;

/** Registered once from instrumentation.ts. */
export function startScoutScheduler(): void {
  if (started) return;
  if (process.env.NODE_ENV === "test") return;
  if (process.env.CHATISA_MOCK_LLM === "1") return;
  if (!process.env.RAPIDAPI_KEY && !process.env.USAJOBS_API_KEY) {
    logger.warn(
      {},
      "Job Scout scheduler not started: neither RAPIDAPI_KEY nor USAJOBS_API_KEY is set, so a harvest could find nothing.",
    );
    return;
  }
  started = true;

  const tick = async () => {
    try {
      const last = latestSuccessfulScoutRun();
      if (isHarvestDue(new Date(), last?.startedAt ?? null)) {
        logger.info({}, "Job Scout weekly harvest starting");
        await runHarvest({ trigger: "schedule" });
      }
    } catch (err) {
      // The next hourly tick retries; a scheduler must never take the
      // server down with it.
      logger.error(
        { err: err instanceof Error ? err.message : String(err) },
        "Job Scout scheduler tick failed",
      );
    } finally {
      const timer = setTimeout(tick, CHECK_INTERVAL_MS);
      // Never hold the process open just for the scheduler.
      timer.unref();
    }
  };
  void tick();
}
