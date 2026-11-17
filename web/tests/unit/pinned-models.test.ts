import { afterEach, describe, expect, it } from "vitest";
import { MODELS, DEFAULT_MODELS, MODEL_CATEGORIES } from "@/lib/config/models";
import { TAG_MODEL_ID, maxRunUsd } from "@/lib/scout/tag";
import { COMPLETION_DEFAULT_MODEL, COMPLETION_MODELS } from "@/lib/sandbox/completion";

/**
 * Model ids pinned outside the catalog must still be in it.
 *
 * A catalog refresh replaces ids in lib/config/models.ts, but several modules
 * pin a model of their own. Nothing tied those pins to the catalog, so a
 * refresh that removed a pinned id compiled and passed every test while the
 * module behind it broke. The sharpest case is the Job Scout tagger: it reads
 * its cost from the catalog, so an id missing there would make every weekly
 * harvest record $0 against its spending cap. v6.10.1 removed the tagger's
 * model (gemini-3.8-flash) from the catalog, which is how this gap was found.
 */
describe("models pinned outside the catalog", () => {
  it("the weekly tagger's model is in the catalog, with structured output", () => {
    expect(MODELS[TAG_MODEL_ID], TAG_MODEL_ID).toBeDefined();
    expect(MODELS[TAG_MODEL_ID].supportsStructuredOutput).toBe(true);
  });

  it("every Sandbox completion model is in the catalog", () => {
    for (const id of [COMPLETION_DEFAULT_MODEL, ...COMPLETION_MODELS]) {
      expect(MODELS[id], id).toBeDefined();
    }
  });

  it("every module default is in the catalog and in a picker group", () => {
    const grouped = new Set(Object.values(MODEL_CATEGORIES).flatMap((c) => c.models));
    for (const [module, id] of Object.entries(DEFAULT_MODELS)) {
      expect(MODELS[id!], `${module} -> ${id}`).toBeDefined();
      expect(grouped.has(id!), `${module} -> ${id} grouped`).toBe(true);
    }
  });
});

describe("weekly tagging cost cap", () => {
  const saved = process.env.CHATISA_SCOUT_MAX_RUN_USD;
  afterEach(() => {
    if (saved === undefined) delete process.env.CHATISA_SCOUT_MAX_RUN_USD;
    else process.env.CHATISA_SCOUT_MAX_RUN_USD = saved;
  });

  it("defaults to $20, enough for a full week on Gemini 4 Hydrogen", () => {
    // v6.10.1 doubled the default with the tagger's per-token price. A full
    // week measured at about $12 on Hydrogen, so $10 would cut the last
    // postings of a full week.
    delete process.env.CHATISA_SCOUT_MAX_RUN_USD;
    expect(maxRunUsd()).toBe(20);
  });

  it("still honours the server override", () => {
    process.env.CHATISA_SCOUT_MAX_RUN_USD = "12.5";
    expect(maxRunUsd()).toBe(12.5);
  });
});
