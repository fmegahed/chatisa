import { describe, expect, it } from "vitest";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import { MODELS, temperatureFor } from "@/lib/config/models";

const ROOT = resolve(__dirname, "..", "..");

/**
 * Claude Opus 5 rejects the `temperature` parameter outright:
 *
 *   AI_APICallError: `temperature` is deprecated for this model.
 *
 * Found on 2026-07-26 by driving Ask Anything on it live. Every request carrying
 * a temperature failed, so the model was offered in five modules and answered in
 * none of them. Claude Sonnet 5 is the same generation and only WARNS, which is
 * exactly why this survived: the default model degraded silently while the
 * premium one broke loudly, and nothing exercised the premium one.
 *
 * The lesson these tests encode is that a per-model incompatibility must be
 * handled in ONE place and used EVERYWHERE, because the cost of missing a single
 * call site is a whole module that cannot answer.
 */
describe("temperatureFor", () => {
  it("omits the parameter for models that reject it", () => {
    expect(temperatureFor("claude-opus-5", 0.7)).toBeUndefined();
    expect(temperatureFor("claude-sonnet-5", 0)).toBeUndefined();
  });

  it("passes it through for models that accept it", () => {
    expect(temperatureFor("gpt-5.6-sol", 0.7)).toBe(0.7);
    expect(temperatureFor("gpt-5.6-luna", 0)).toBe(0);
    expect(temperatureFor("gemini-3.6-flash", 0.25)).toBe(0.25);
  });

  it("treats an unknown model id as accepting it", () => {
    // The pre-existing behaviour. A freshly added id must not silently lose its
    // temperature just because the catalog entry is not there yet.
    expect(temperatureFor("some-new-model", 0.4)).toBe(0.4);
  });

  it("keeps the published temperature range, which is a different fact", () => {
    // The range describes what the model's sampling would do; the flag describes
    // whether we are allowed to ask. Zeroing the range would corrupt the catalog.
    expect(MODELS["claude-opus-5"].temperatureRange).toEqual([0.0, 2.0]);
    expect(MODELS["claude-opus-5"].supportsTemperature).toBe(false);
  });
});

describe("every provider call routes temperature through the helper", () => {
  const SKIP = new Set(["node_modules", ".next", "test-results", "playwright-report"]);

  function sourceFiles(dir: string, found: string[] = []): string[] {
    for (const entry of readdirSync(dir)) {
      if (SKIP.has(entry)) continue;
      const path = join(dir, entry);
      if (statSync(path).isDirectory()) sourceFiles(path, found);
      else if (/\.tsx?$/.test(entry)) found.push(path);
    }
    return found;
  }

  it("has no raw temperature literal left in a provider call", () => {
    // A static check, because this is precisely the failure mode: seven call
    // sites existed, and missing any one of them leaves a module broken on
    // Anthropic's models while the others look fine.
    const offenders: string[] = [];
    for (const file of [
      ...sourceFiles(join(ROOT, "lib")),
      ...sourceFiles(join(ROOT, "app")),
    ]) {
      // The catalog itself declares ranges and defaults; it is not a call site.
      if (file.endsWith(join("config", "models.ts"))) continue;
      // The per-module config is data, read by the routes and passed to the
      // helper there.
      if (file.endsWith(join("chat", "config.ts"))) continue;

      const text = readFileSync(file, "utf8");
      for (const match of text.matchAll(/^\s*temperature:\s*(.+?),\s*$/gm)) {
        const value = match[1].trim();
        if (value.startsWith("temperatureFor(")) continue;
        offenders.push(`${relative(ROOT, file).replace(/\\/g, "/")}: ${value}`);
      }
    }
    expect(
      offenders,
      "these pass a temperature straight to a provider; wrap them in temperatureFor(modelId, ...)",
    ).toEqual([]);
  });

  it("finds the call sites it is meant to be guarding", () => {
    // Canary: if the scan matched nothing, the test above would pass vacuously.
    let wrapped = 0;
    for (const file of [
      ...sourceFiles(join(ROOT, "lib")),
      ...sourceFiles(join(ROOT, "app")),
    ]) {
      wrapped += (readFileSync(file, "utf8").match(/temperature: temperatureFor\(/g) ?? [])
        .length;
    }
    // Chat, Ask Anything, the coach, completions, exam generation, grading, and
    // vision transcription.
    expect(wrapped).toBeGreaterThanOrEqual(7);
  });
});
