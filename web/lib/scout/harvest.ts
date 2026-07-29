import "server-only";
import {
  createScoutRun,
  finishScoutRun,
  retireScoutPostings,
  scoutRunInProgress,
  upsertScoutPosting,
} from "@/lib/db";
import { logger } from "@/lib/log";
import {
  ACTIVEJOBS_LOCATION,
  ACTIVEJOBS_QUERIES,
  TARGET_STATE_CODES,
  USAJOBS_KEYWORDS,
  isRelevantTitle,
} from "./queries";
import { searchActiveJobs } from "./sources/activejobs";
import { searchUsajobs } from "./sources/usajobs";
import { fingerprintOf, type Fetcher, type RawPosting } from "./sources/types";
import { maxRunUsd, tagPosting, TAXONOMY_VERSION, type TagResult } from "./tag";

/**
 * The weekly harvest orchestrator (design §4.3):
 * normalize → dedupe → relevance filter → tag → store → retire.
 * One source failing degrades the run to the other and is recorded honestly;
 * only both failing (or an unexpected throw) fails the run.
 */

const TAG_CONCURRENCY = 4;
/** Unseen for two weekly runs (plus slack) → inactive. */
const UNSEEN_DAYS = 13;
/** Rows unseen this long are deleted outright. Public content; no hoarding. */
const PURGE_DAYS = 90;
/** Nothing older than a month is stored or stays listed (user, 2026-07-28). */
const MAX_POSTING_AGE_DAYS = 30;

export interface ScoutRunSummary {
  runId: string;
  status: "completed" | "partial" | "failed";
  activejobsFound: number;
  usajobsFound: number;
  tagged: number;
  /** Postings dropped because the cost cap ended tagging early. */
  droppedByCap: number;
  costUsd: number;
  sourceErrors: { activejobs?: string; usajobs?: string; tagging?: string };
}

/** Injectable for tests; production callers pass nothing. */
export interface HarvestDeps {
  fetcher?: Fetcher;
  tagger?: (p: RawPosting) => Promise<TagResult>;
}

async function mapWithConcurrency<T, R>(
  items: T[],
  limit: number,
  fn: (item: T) => Promise<R>,
): Promise<R[]> {
  const results: R[] = new Array(items.length);
  let next = 0;
  const workers = Array.from(
    { length: Math.min(limit, items.length) },
    async () => {
      while (next < items.length) {
        const i = next++;
        results[i] = await fn(items[i]);
      }
    },
  );
  await Promise.all(workers);
  return results;
}

export async function runHarvest(
  opts: { trigger: "schedule" | "manual" },
  deps: HarvestDeps = {},
): Promise<ScoutRunSummary | { alreadyRunning: true }> {
  if (scoutRunInProgress()) return { alreadyRunning: true };
  const fetcher = deps.fetcher ?? fetch;
  const tagger = deps.tagger ?? tagPosting;
  const runId = createScoutRun(opts.trigger);
  const sourceErrors: ScoutRunSummary["sourceErrors"] = {};

  try {
    // --- collect ---------------------------------------------------------
    // Errors keep the LAST message plus a failure count: the 2026-07-28 run
    // recorded only the first error, which hid that the other 159 queries
    // were failing for a different reason.
    let activejobsRequests = 0;
    let activejobsFailures = 0;
    let activejobsError: string | undefined;
    const activejobsPostings: RawPosting[] = [];
    for (const q of ACTIVEJOBS_QUERIES) {
      const result = await searchActiveJobs(
        {
          title: q.title,
          location: ACTIVEJOBS_LOCATION,
          timeFrame: "7d",
          limit: q.limit,
          category: q.category,
          preferStates: TARGET_STATE_CODES,
          experienceLevels: q.experienceLevels,
          employmentTypes: q.employmentTypes,
        },
        fetcher,
      );
      activejobsRequests += result.requests;
      activejobsPostings.push(...result.postings);
      if (result.error) {
        activejobsFailures += 1;
        activejobsError = result.error;
        if (result.requests === 0) break; // not configured: stop looping
        // Quota exhausted: every further request burns the same dead pool.
        if (result.quotaExhausted) break;
      }
    }
    if (activejobsError) {
      sourceErrors.activejobs =
        activejobsFailures > 1
          ? `${activejobsError} (${activejobsFailures} of ${activejobsRequests} queries failed)`
          : activejobsError;
    }

    let usajobsRequests = 0;
    let usajobsFailures = 0;
    let usajobsError: string | undefined;
    const usajobsPostings: RawPosting[] = [];
    for (const keyword of USAJOBS_KEYWORDS) {
      const result = await searchUsajobs(keyword, fetcher);
      usajobsRequests += result.requests;
      usajobsPostings.push(...result.postings);
      if (result.error) {
        usajobsFailures += 1;
        usajobsError = result.error;
        if (result.requests === 0) break;
      }
    }
    if (usajobsError) {
      sourceErrors.usajobs =
        usajobsFailures > 1
          ? `${usajobsError} (${usajobsFailures} of ${usajobsRequests} queries failed)`
          : usajobsError;
    }

    // --- dedupe + relevance ---------------------------------------------
    const seenExternal = new Set<string>();
    const seenFingerprint = new Set<string>();
    const oldestAllowed = new Date(
      Date.now() - MAX_POSTING_AGE_DAYS * 86_400_000,
    )
      .toISOString()
      .slice(0, 10);
    const candidates: RawPosting[] = [];
    for (const p of [...activejobsPostings, ...usajobsPostings]) {
      const externalKey = `${p.source}|${p.externalId}`;
      const fp = fingerprintOf(p);
      if (seenExternal.has(externalKey) || seenFingerprint.has(fp)) continue;
      seenExternal.add(externalKey);
      seenFingerprint.add(fp);
      if (!isRelevantTitle(p.title)) continue;
      // A dated posting older than a month never enters the feed; undated
      // ones pass, since both sources are queried for recent listings.
      if (p.postedAt && p.postedAt < oldestAllowed) continue;
      candidates.push(p);
    }

    // --- tag + store -----------------------------------------------------
    const cap = maxRunUsd();
    let costUsd = 0;
    let tagged = 0;
    let droppedByCap = 0;
    let taggingFailures = 0;
    let lastTagError = "";
    await mapWithConcurrency(candidates, TAG_CONCURRENCY, async (posting) => {
      if (costUsd >= cap) {
        droppedByCap += 1;
        return;
      }
      try {
        const tags = await tagger(posting);
        costUsd += tags.costUsd;
        if (!tags.seniorityOk) return;
        upsertScoutPosting({
          source: posting.source,
          externalId: posting.externalId,
          fingerprint: fingerprintOf(posting),
          title: posting.title,
          company: posting.company,
          locationCity: posting.locationCity,
          locationState: posting.locationState,
          remote: posting.remote,
          // The model's category call wins: the ad itself says intern/federal
          // more reliably than which query surfaced it.
          category: tags.category,
          applyUrl: posting.applyUrl,
          description: posting.description,
          postedAt: posting.postedAt,
          skillsJson: JSON.stringify(tags.skills),
          visaSponsorship: tags.visaSponsorship,
          taxonomyVersion: TAXONOMY_VERSION,
        });
        tagged += 1;
      } catch (err) {
        // One bad posting (model refusal, schema retry exhaustion) must not
        // sink the run; it is simply not stored this week. But the FAILURES
        // ARE COUNTED and reported: the 2026-07-29 run tagged 0 of 424 with
        // zero explanation because this catch used to swallow silently.
        taggingFailures += 1;
        lastTagError = err instanceof Error ? err.message : String(err);
      }
    });
    if (taggingFailures > 0) {
      sourceErrors.tagging = `${taggingFailures} of ${candidates.length} postings failed tagging (last: ${lastTagError.slice(0, 160)})`;
    }

    // --- retire ----------------------------------------------------------
    const now = Date.now();
    retireScoutPostings({
      unseenSinceIso: new Date(now - UNSEEN_DAYS * 86_400_000).toISOString(),
      purgeBeforeIso: new Date(now - PURGE_DAYS * 86_400_000).toISOString(),
      // Date-only, matching the sources' YYYY-MM-DD postedAt values.
      postedBeforeIso: oldestAllowed,
    });

    const bothFailed =
      activejobsPostings.length === 0 &&
      usajobsPostings.length === 0 &&
      Boolean(sourceErrors.activejobs) &&
      Boolean(sourceErrors.usajobs);
    const status: ScoutRunSummary["status"] = bothFailed
      ? "failed"
      : droppedByCap > 0 || (candidates.length > 0 && tagged === 0)
        ? "partial"
        : "completed";

    finishScoutRun(runId, {
      status,
      activejobsRequests,
      activejobsFound: activejobsPostings.length,
      usajobsRequests,
      usajobsFound: usajobsPostings.length,
      dedupedCount: candidates.length,
      taggedCount: tagged,
      costUsd: Number(costUsd.toFixed(4)),
      sourceErrorsJson: JSON.stringify(sourceErrors),
      error: bothFailed ? "both sources failed" : null,
    });
    logger.info(
      { runId, status, tagged, droppedByCap, costUsd },
      "job scout harvest finished",
    );
    return {
      runId,
      status,
      activejobsFound: activejobsPostings.length,
      usajobsFound: usajobsPostings.length,
      tagged,
      droppedByCap,
      costUsd,
      sourceErrors,
    };
  } catch (err) {
    finishScoutRun(runId, {
      status: "failed",
      error: err instanceof Error ? err.message : "unexpected failure",
      sourceErrorsJson: JSON.stringify(sourceErrors),
    });
    logger.error({ runId }, "job scout harvest failed");
    return {
      runId,
      status: "failed",
      activejobsFound: 0,
      usajobsFound: 0,
      tagged: 0,
      droppedByCap: 0,
      costUsd: 0,
      sourceErrors,
    };
  }
}
