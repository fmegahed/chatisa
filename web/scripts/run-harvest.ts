/**
 * Runs one Job Scout harvest from the command line, against the database
 * that .env.local's CHATISA_DATA_DIR points at (or web/data by default).
 *
 * Run: `npm run scout:harvest`  (dev machine only; the production server
 * harvests on its own Sunday schedule, or via POST /api/scout/refresh).
 *
 * First used 2026-07-28 to seed the feed ahead of the first scheduled run
 * (user instruction: "a manual run populate everything today as an
 * exception to seed this").
 */

import { readFileSync } from "node:fs";
import path from "node:path";

// .env.local first, before any lib import touches keys or the database.
try {
  const envFile = readFileSync(path.join(process.cwd(), ".env.local"), "utf8");
  for (const line of envFile.split("\n")) {
    const m = /^([A-Z_][A-Z0-9_]*)=(.*)$/.exec(line.trim());
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2];
  }
} catch {
  // Fine: the environment may already be configured.
}

async function main() {
  if (!process.env.RAPIDAPI_KEY && !process.env.USAJOBS_API_KEY) {
    console.error(
      "Neither RAPIDAPI_KEY nor USAJOBS_API_KEY is set; a harvest would find nothing.",
    );
    process.exit(1);
  }
  const { runHarvest } = await import("../lib/scout/harvest");
  console.log("Harvest starting; a full run takes several minutes...");
  const summary = await runHarvest({ trigger: "manual" });
  if ("alreadyRunning" in summary) {
    console.error("A harvest is already running (scout_runs has a 'running' row).");
    process.exit(1);
  }
  console.log(JSON.stringify(summary, null, 2));
  process.exit(summary.status === "failed" ? 1 : 0);
}

void main();
