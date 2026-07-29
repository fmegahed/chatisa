import { NextResponse } from "next/server";
import { validateEnv } from "@/lib/config/env";
import { checkRateLimit } from "@/lib/ratelimit";

/**
 * Liveness + readiness. Reports variable NAMES only: never values.
 * Public by design (excluded from the auth proxy) so the Task Scheduler /
 * monitoring can probe it.
 *
 * ?deep=1 additionally EXERCISES the features that have historically broken
 * only in production (2026-07-25: the PDF worker's dependency was missing
 * from a deploy bundle and nobody noticed until a student uploaded a PDF):
 * a real PDF parse through the worker pool, a database write with read-back,
 * the brand assets, and Deepgram speech. The deploy pipeline refuses to ship a
 * bundle that fails this, and the production launcher runs it at startup.
 *
 * The speech check has different semantics from the others, on purpose. An
 * absent DEEPGRAM_TOKEN is a legitimate configuration (Interview Mentor
 * degrades to typed answers), so it reports "not-configured" WITHOUT failing
 * the server. A token that is present but refused is a real fault and does
 * fail, because that is the state which is invisible from outside: the
 * shallow check has always reported the variable as present, and presence is
 * not validity. That gap is why "the interviewer has no voice" could not be
 * diagnosed remotely.
 */

/**
 * A minimal but STRUCTURALLY VALID one-page PDF, built with correct xref
 * offsets at module load. Hand-written fixtures with guessed offsets open in
 * lenient viewers but not in pdf.js, so it is constructed, not pasted.
 */
function buildTinyPdf(): Uint8Array {
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
    "<< /Length 44 >>\nstream\nBT /F1 12 Tf 10 100 Td (deep health) Tj ET\nendstream",
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
  ];
  let body = "%PDF-1.4\n";
  const offsets: number[] = [];
  objects.forEach((content, i) => {
    offsets.push(body.length);
    body += `${i + 1} 0 obj\n${content}\nendobj\n`;
  });
  const xrefAt = body.length;
  let xref = `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  for (const at of offsets) {
    xref += `${String(at).padStart(10, "0")} 00000 n \n`;
  }
  body += `${xref}trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xrefAt}\n%%EOF`;
  return new TextEncoder().encode(body);
}

const failText = (err: unknown) =>
  `failed: ${err instanceof Error ? err.message : String(err)}`.slice(0, 200);

async function deepChecks(): Promise<{ deep: Record<string, string>; ok: boolean }> {
  const deep: Record<string, string> = {};

  // The PDF worker: a real child process parsing a real document, proving the
  // worker file, its packages (unpdf), and the spawn path all work HERE.
  try {
    const { processPdfInWorker } = await import("@/lib/exam/pdf-pool");
    const result = await processPdfInWorker({
      bytes: buildTinyPdf(),
      maxVisionPages: 0,
      deadlineMs: 20_000,
    });
    deep.pdfWorker =
      result.pageCount === 1 ? "ok" : `unexpected pageCount ${result.pageCount}`;
  } catch (err) {
    deep.pdfWorker = failText(err);
  }

  // The database: an actual write with read-back through the native driver.
  try {
    const { dbWriteProbe } = await import("@/lib/db");
    deep.dbRoundtrip = dbWriteProbe() ? "ok" : "write not visible on read-back";
  } catch (err) {
    deep.dbRoundtrip = failText(err);
  }

  // Brand assets on disk (get_miami_style and slice E's deck template).
  try {
    const { getMiamiStyle } = await import("@/lib/ask/miami-style");
    const style = await getMiamiStyle("colors");
    deep.brandAssets = "error" in style ? `failed: ${style.error}` : "ok";
  } catch (err) {
    deep.brandAssets = failText(err);
  }

  // Interview Mentor's voice. Reported separately from the pass/fail set below,
  // because "no speech configured" is a legitimate way to run this app and must
  // not turn a healthy server red, while "configured but refused" must.
  let speech = "unknown";
  try {
    const { probeSpeech } = await import("@/lib/speech/deepgram");
    const probe = await probeSpeech();
    speech = probe.state === "ok" ? "ok" : `${probe.state}: ${probe.detail}`;
  } catch (err) {
    speech = failText(err);
  }

  // Job Scout feed freshness. Informational like speech's not-configured
  // state: an empty or stale feed is visible here for the operator but never
  // turns the server red, because the module itself tells students honestly
  // and every other feature is unaffected (design 2026-07-28).
  let scout = "unknown";
  try {
    const { latestSuccessfulScoutRun, countScoutPostings } = await import(
      "@/lib/db"
    );
    const last = latestSuccessfulScoutRun();
    if (!last) {
      scout = "no harvest yet";
    } else {
      const ageDays =
        (Date.now() - new Date(last.startedAt).getTime()) / 86_400_000;
      const freshness = ageDays > 8 ? `stale (${Math.floor(ageDays)}d)` : "ok";
      scout = `${freshness}, ${countScoutPostings()} active postings`;
    }
  } catch (err) {
    scout = failText(err);
  }

  const ok =
    Object.values(deep).every((v) => v === "ok") &&
    !speech.startsWith("broken");
  return { deep: { ...deep, speech, scout }, ok };
}

export async function GET(req: Request) {
  const { report } = validateEnv();

  let db = "ok";
  try {
    const { dbReady } = await import("@/lib/db");
    db = dbReady() ? "ok" : "error";
  } catch {
    db = "error";
  }

  const wantsDeep = new URL(req.url).searchParams.get("deep") === "1";
  let deepResult: { deep: Record<string, string>; ok: boolean } | null = null;
  if (wantsDeep) {
    // Deep checks spawn a worker process; bound how often anyone triggers that.
    const limit = checkRateLimit("deep-health", { limit: 6, windowMs: 60_000 });
    if (!limit.allowed) {
      return NextResponse.json(
        { status: "rate-limited", retryAfterSeconds: limit.retryAfterSeconds },
        { status: 429 },
      );
    }
    deepResult = await deepChecks();
  }

  const ready = report.ok && db === "ok" && (deepResult === null || deepResult.ok);
  return NextResponse.json(
    {
      status: ready ? "ok" : "degraded",
      checks: {
        env: report.ok ? "ok" : "invalid",
        invalidEnv: report.invalid,
        missingProviderKeys: report.missingProviders,
        authConfigured: report.authConfigured,
        db,
        ...(deepResult ? { deep: deepResult.deep } : {}),
      },
      timestamp: new Date().toISOString(),
    },
    { status: ready ? 200 : 503 },
  );
}
