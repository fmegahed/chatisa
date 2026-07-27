import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync } from "node:fs";
import { join, relative, resolve, sep } from "node:path";

const ROOT = resolve(__dirname, "..", "..");
const SCRIPT = join(ROOT, "scripts", "make-deploy-bundle.mjs");

/**
 * The deploy bundle must not carry development data.
 *
 * Found on 2026-07-26: `.next/standalone` mirrors the project directory, and the
 * bundler copied it wholesale. So every bundle shipped
 *
 *   data/chatisa.db   3 user rows (email addresses), 150 usage events,
 *                     126 exam document page excerpts, 2 tailored resumes
 *   deploy/           the PREVIOUS bundle, nested three levels deep, 612 MB
 *
 * Neither belongs on the production server: it reads CHATISA_DATA_DIR for its
 * database, so the shipped copy had no purpose and was pure leakage.
 *
 * The bundler is not executed here. It runs `next build`, stamps a source file,
 * and takes minutes, which is not a unit test. (Learned the hard way: importing
 * it to "syntax check" it started a real build and rewrote the footer date.) So
 * these tests re-implement the filter from the script's own constant and check the
 * guard exists, and the script's runtime assertion is what protects a real run.
 */
const source = readFileSync(SCRIPT, "utf8");

/** The exclusion list, read out of the script so the two cannot drift. */
function neverBundle(): string[] {
  const match = /const NEVER_BUNDLE = \[([^\]]*)\]/.exec(source);
  expect(match, "NEVER_BUNDLE not found in make-deploy-bundle.mjs").not.toBeNull();
  return (match?.[1] ?? "")
    .split(",")
    .map((s) => s.trim().replace(/^["']|["']$/g, ""))
    .filter(Boolean);
}

/** The same predicate the script applies to the standalone copy. */
function wouldCopy(relPath: string, excluded: string[]): boolean {
  if (relPath === "") return true;
  return !excluded.includes(relPath.split(sep)[0]);
}

describe("deploy bundle exclusions", () => {
  it("excludes the development database and any previous bundle", () => {
    expect(neverBundle().sort()).toEqual(["data", "deploy"]);
  });

  it("drops exactly the offending paths and keeps everything else", () => {
    const excluded = neverBundle();
    for (const p of [
      "data",
      join("data", "chatisa.db"),
      join("data", "chatisa.db-wal"),
      "deploy",
      join("deploy", "chatisa-app", "data", "chatisa.db"),
    ]) {
      expect(wouldCopy(p, excluded), `${p} must not be bundled`).toBe(false);
    }
    for (const p of [
      "server.js",
      "package.json",
      join("node_modules", "next", "package.json"),
      join(".next", "server", "app", "page.js"),
      join("workers", "pdf-worker.mjs"),
      join("drizzle", "0000_needy_lady_deathstrike.sql"),
      // A file merely CONTAINING the word is not a top-level match.
      join("lib", "data-helpers.js"),
    ]) {
      expect(wouldCopy(p, excluded), `${p} must be bundled`).toBe(true);
    }
  });

  it("applies the filter to the standalone copy, not just declaring it", () => {
    // The constant existing is worthless if the cp() call ignores it.
    expect(source).toMatch(/cp\(\s*standalone,\s*bundle,\s*\{[^}]*filter:/);
  });

  it("verifies the finished artifact as well as filtering the copy", () => {
    // A filter guards one known path; the assertion guards the outcome. The leak
    // happened because a source directory quietly gained contents, and the next
    // surprise will come from elsewhere.
    expect(source).toMatch(/\.db\(-wal\|-shm\)\?\$/);
    expect(source).toMatch(/FATAL: the bundle contains development data/);
    // Fatal, not a warning: shipping a student's email is not a log line.
    const guard = source.slice(source.indexOf("FATAL: the bundle contains"));
    expect(guard).toMatch(/process\.exit\(1\)/);
  });

  it("says what the operator should do about it", () => {
    // A fatal error that only names the file leaves the reader guessing why it
    // matters and where it came from.
    expect(source).toMatch(/user emails and document excerpts/i);
    expect(source).toMatch(/NEVER_BUNDLE/);
  });
});

describe("what the bundle actually contains on disk", () => {
  it("has no database or nested bundle, if a bundle is present", () => {
    // The bundle is git-ignored, so this only runs where one has been built.
    // Assembled before the fix it will fail, which is the correct signal: the
    // stale artifact still holds the data and should be rebuilt.
    const bundle = join(ROOT, "deploy", "chatisa-app");
    let entries: string[];
    try {
      entries = readdirRecursive(bundle, bundle);
    } catch {
      return; // no bundle on this machine
    }
    const offenders = entries.filter(
      (rel) => /\.db(-wal|-shm)?$/i.test(rel) || rel.split(sep)[0] === "deploy",
    );
    expect(
      offenders.slice(0, 10),
      "stale bundle: re-run node scripts/make-deploy-bundle.mjs",
    ).toEqual([]);
  });
});

function readdirRecursive(dir: string, base: string, out: string[] = []): string[] {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    out.push(relative(base, p));
    // Do not descend into a nested bundle: reporting it once is enough, and the
    // tree can be very deep.
    if (entry.isDirectory() && entry.name !== "deploy") readdirRecursive(p, base, out);
  }
  return out;
}

describe("bundle self-test tolerances", () => {
  /** The rule the script applies to each deep-health value. */
  function isBad(key: string, value: string): boolean {
    const optional: Record<string, RegExp> = { speech: /^not-configured[-:\s]/ };
    return value !== "ok" && !optional[key]?.test(value);
  }

  it("accepts speech being unconfigured, which it always is in the self-test", () => {
    // The self-test boots the bundle with AUTH placeholders and no
    // DEEPGRAM_TOKEN, so "not-configured" is the correct answer there. Adding
    // speech to the deep block without this failed a bundle that was fine.
    expect(isBad("speech", "not-configured: DEEPGRAM_TOKEN is not set, ...")).toBe(
      false,
    );
  });

  it("still fails a bundle whose speech credential is refused", () => {
    // The state that actually matters: a token IS set and Deepgram rejects it.
    expect(isBad("speech", "broken: DEEPGRAM_TOKEN is set but Deepgram refused it")).toBe(
      true,
    );
    expect(isBad("speech", "failed: something else")).toBe(true);
  });

  it("keeps every other check strict", () => {
    for (const key of ["pdfWorker", "dbRoundtrip", "brandAssets"]) {
      expect(isBad(key, "ok")).toBe(false);
      expect(isBad(key, "not-configured: whatever"), `${key} must stay strict`).toBe(
        true,
      );
      expect(isBad(key, "failed: boom")).toBe(true);
    }
  });

  it("matches the tolerance the script actually encodes", () => {
    // Re-implemented above for clarity; pinned here so the two cannot drift.
    expect(source).toMatch(/OPTIONAL_DEEP\s*=\s*\{\s*speech:\s*\/\^not-configured/);
  });
});
