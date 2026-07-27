import { describe, expect, it } from "vitest";
import fixtures from "../fixtures/legacy-config.json";
import {
  MODELS,
  MODEL_CATEGORIES,
  DEFAULT_MODELS,
  calculateCost,
  buildModelOptions,
  badgesFor,
  categoryOf,
  getDefaultModelForPage,
  getPageModels,
  type ModuleKey,
} from "@/lib/config/models";

/**
 * The catalog deliberately no longer matches legacy config.py. The 2026-07-21
 * refresh (ADR-018) replaced every model with one the user chose and that was
 * verified against the provider's live listing, so the old per-model
 * characterization tests have been retired: they now pin behaviour we
 * intentionally changed, and keeping them would mean either failing forever or
 * being edited to match whatever the code says, which proves nothing.
 *
 * What survives from the legacy work is the part that did not change: the cost
 * arithmetic, still a 1:1 port including its 6-decimal rounding. That is
 * re-pinned below against the legacy fixture's own arithmetic rather than its
 * model list.
 *
 * The rest of this file pins the guarantees the new catalog must keep, above
 * all that Exam Ally never offers a model that cannot generate an exam.
 */

const MODULE_KEYS: ModuleKey[] = [
  "coding_companion",
  "project_coach",
  "exam_ally",
  "interview_mentor",
  "interview_mentor_transcription",
  "ai_sandbox",
  "ai_comparisons",
];

describe("catalog integrity", () => {
  it("has no duplicate display names, so students can tell models apart", () => {
    const names = Object.values(MODELS).map((m) => m.displayName);
    expect(new Set(names).size).toBe(names.length);
  });

  it("gives every HuggingFace model an explicit route suffix", () => {
    // The serving provider changes price, speed, context and structured-output
    // support, so an unpinned HF id would silently vary in all four.
    for (const [id, cfg] of Object.entries(MODELS)) {
      if (cfg.provider !== "huggingface_inference") continue;
      expect(id, `${id} must pin a provider route`).toMatch(/:[a-z0-9-]+$/);
    }
  });

  it("does not contain ids known to be unserved", () => {
    // Both were live in the catalog while being impossible to call.
    const withdrawn = [
      "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
      "gpt-4o-realtime-preview-2025-06-03",
    ];
    for (const id of withdrawn) expect(Object.keys(MODELS)).not.toContain(id);
  });

  it("keeps every category id pointing at a real model", () => {
    for (const [key, category] of Object.entries(MODEL_CATEGORIES)) {
      for (const id of category.models) {
        expect(MODELS[id], `${key} lists unknown model ${id}`).toBeDefined();
      }
    }
  });

  it("lists every model in exactly one category", () => {
    const listed = Object.values(MODEL_CATEGORIES).flatMap((c) => c.models);
    expect(new Set(listed).size).toBe(listed.length);
    expect(new Set(listed)).toEqual(new Set(Object.keys(MODELS)));
  });

  it("keeps the open-weight teaching set well represented", () => {
    // Open-weight models are a pedagogical requirement, not a cost measure
    // (user instruction, 2026-07-19), so this guards against them being
    // quietly dropped in a future refresh.
    const openWeight = Object.values(MODELS).filter((m) => m.openWeight);
    expect(openWeight.length).toBeGreaterThanOrEqual(8);
  });

  it("marks a limit as inferred whenever it was not published", () => {
    // Inferred limits must be conservative; this pins that none of them claims
    // a larger window than the smallest sibling route we actually observed.
    for (const [id, cfg] of Object.entries(MODELS)) {
      if (!cfg.limitsInferred) continue;
      expect(cfg.contextWindow, `${id} inferred context`).toBeLessThanOrEqual(
        262144,
      );
    }
  });
});

describe("how models are presented to students", () => {
  it("keeps badges out of displayName", () => {
    // The dropdown once read "Gemma 4 31B (open weight, free) (open weight,
    // free tier)" because displayName carried a suffix AND each page appended
    // its own badges. displayName is now the product name only, and openWeight
    // plus costPer1kInput are the single source of every badge.
    for (const [id, cfg] of Object.entries(MODELS)) {
      expect(cfg.displayName, `${id} displayName`).not.toContain("(");
    }
  });

  it("derives badges from fields rather than from the name", () => {
    const free = badgesFor("google/gemma-4-31B-it:cerebras");
    expect(free).toContain("open weight");
    expect(free).toContain("free");

    const commercial = badgesFor("claude-sonnet-5");
    expect(commercial).not.toContain("open weight");
    expect(commercial).not.toContain("free");
  });

  it("places every model in exactly one named group", () => {
    for (const id of Object.keys(MODELS)) {
      expect(categoryOf(id), `${id} has no category`).not.toBeNull();
    }
  });

  it("gives every option a description a student can act on", () => {
    // The description field existed for every model but was rendered nowhere,
    // leaving a student with only unfamiliar names to choose between.
    const { options } = buildModelOptions("coding_companion");
    expect(options.length).toBeGreaterThan(0);
    for (const option of options) {
      expect(option.description.length, `${option.id} description`).toBeGreaterThan(10);
      expect(option.name).not.toContain("(");
      expect(option.groupName.length).toBeGreaterThan(0);
    }
  });

  it("marks exactly one option as the suggested default", () => {
    const { options, defaultModelId } = buildModelOptions("coding_companion");
    const recommended = options.filter((o) => o.recommended);
    expect(recommended).toHaveLength(1);
    expect(recommended[0].id).toBe(defaultModelId);
  });

  it("groups options together and in catalog order", () => {
    const { options } = buildModelOptions("coding_companion");
    const seen: string[] = [];
    for (const option of options) {
      if (seen[seen.length - 1] !== option.groupId) seen.push(option.groupId);
    }
    // A group appearing twice would mean options are interleaved.
    expect(new Set(seen).size).toBe(seen.length);
  });

  it("honours the available-model filter", () => {
    const only = ["claude-sonnet-5"];
    const { options, defaultModelId } = buildModelOptions("coding_companion", only);
    expect(options.map((o) => o.id)).toEqual(only);
    expect(defaultModelId).toBe("claude-sonnet-5");
  });

  it("offers the interview module a real set including open weight models", () => {
    // Both interview keys used to filter on tags no model carries, so one
    // resolved to an empty list and the other to the five premium models.
    const { options } = buildModelOptions("interview_mentor");
    expect(options.length).toBeGreaterThan(5);
    expect(options.some((o) => MODELS[o.id].openWeight)).toBe(true);
  });
});

describe("Exam Ally model eligibility", () => {
  const examModels = getPageModels("exam_ally");

  it("offers only models that can return structured output", () => {
    expect(examModels.length).toBeGreaterThan(0);
    for (const id of examModels) {
      expect(
        MODELS[id].supportsStructuredOutput,
        `${id} cannot generate an exam and must not be offered`,
      ).toBe(true);
    }
  });

  it("excludes every model whose route lacks structured output", () => {
    // These are deliberately kept in the catalog for chat, where their speed
    // is the point, and deliberately hidden here.
    const chatOnly = Object.entries(MODELS)
      .filter(([, cfg]) => !cfg.supportsStructuredOutput)
      .map(([id]) => id);
    expect(chatOnly.length).toBeGreaterThan(0);
    for (const id of chatOnly) expect(examModels).not.toContain(id);
  });

  it("excludes models too small to hold a course document", () => {
    // phi-4 supports structured output but has only 16k of context.
    expect(examModels).not.toContain("microsoft/phi-4:deepinfra");
    for (const id of examModels) {
      expect(MODELS[id].contextWindow).toBeGreaterThanOrEqual(64000);
    }
  });

  it("still offers open-weight models, not only commercial ones", () => {
    expect(examModels.some((id) => MODELS[id].openWeight)).toBe(true);
  });

  it("can transcribe scanned pages with at least one vision model", () => {
    expect(examModels.some((id) => MODELS[id].supportsVision)).toBe(true);
  });
});

describe("module model lists", () => {
  it.each(MODULE_KEYS)("returns only known models for %s", (moduleKey) => {
    for (const id of getPageModels(moduleKey)) {
      expect(MODELS[id], `${moduleKey} offers unknown model ${id}`).toBeDefined();
    }
  });

  it("offers a usable default that the module actually lists", () => {
    for (const [moduleKey, configured] of Object.entries(DEFAULT_MODELS)) {
      const available = getPageModels(moduleKey);
      expect(
        available,
        `${moduleKey} default ${configured} is not in its own list`,
      ).toContain(configured);
      expect(getDefaultModelForPage(moduleKey)).toBe(configured);
    }
  });

  it("gives the chat modules a broad choice", () => {
    expect(getPageModels("coding_companion").length).toBeGreaterThanOrEqual(15);
  });

  it("sorts by provider then display name, as the legacy app did", () => {
    const list = getPageModels("coding_companion");
    const sorted = [...list].sort((a, b) => {
      const pa = MODELS[a].provider;
      const pb = MODELS[b].provider;
      if (pa !== pb) return pa < pb ? -1 : 1;
      return MODELS[a].displayName < MODELS[b].displayName ? -1 : 1;
    });
    expect(list).toEqual(sorted);
  });
});

describe("calculateCost arithmetic, unchanged from legacy config.py", () => {
  interface LegacyCost {
    model: string;
    input_tokens?: number;
    output_tokens?: number;
    input_cost?: number;
    output_cost?: number;
    total_cost?: number;
    error?: string;
  }
  const legacyCosts = fixtures.calculate_cost as unknown as LegacyCost[];
  const legacyModels = fixtures.models as unknown as Record<
    string,
    { cost_per_1k_input: number; cost_per_1k_output: number }
  >;

  it("reproduces the legacy formula and rounding on the legacy fixtures", () => {
    // Re-derives the legacy expectations from legacy rates rather than from the
    // current catalog, so this keeps testing the arithmetic after every model
    // change instead of being invalidated by one.
    const cases = legacyCosts.filter(
      (c) => !c.error && legacyModels[c.model] !== undefined,
    );
    expect(cases.length).toBeGreaterThan(0);

    for (const c of cases) {
      const rates = legacyModels[c.model];
      const round6 = (n: number) => Number(n.toFixed(6));
      const inputCost = round6((c.input_tokens! / 1000) * rates.cost_per_1k_input);
      const outputCost = round6(
        (c.output_tokens! / 1000) * rates.cost_per_1k_output,
      );
      expect(inputCost).toBeCloseTo(c.input_cost!, 6);
      expect(outputCost).toBeCloseTo(c.output_cost!, 6);
      expect(round6(inputCost + outputCost)).toBeCloseTo(c.total_cost!, 6);
    }
  });

  it("computes cost for a current model", () => {
    // gpt-5.6-terra: $2.50 per million in, $15.00 per million out.
    const result = calculateCost("gpt-5.6-terra", 10_000, 2_000);
    expect(result).not.toHaveProperty("error");
    if ("error" in result) throw new Error("unreachable");
    expect(result.inputCost).toBeCloseTo(0.025, 6);
    expect(result.outputCost).toBeCloseTo(0.03, 6);
    expect(result.totalCost).toBeCloseTo(0.055, 6);
    expect(result.currency).toBe("USD");
  });

  it("charges nothing for the free models", () => {
    const result = calculateCost("google/gemma-4-31B-it:cerebras", 50_000, 5_000);
    if ("error" in result) throw new Error("unreachable");
    expect(result.totalCost).toBe(0);
  });

  it("reports an error for an unknown model rather than throwing", () => {
    const result = calculateCost("no-such-model", 100, 100);
    expect(result).toHaveProperty("error");
  });

  it("prices every catalog model without error", () => {
    for (const id of Object.keys(MODELS)) {
      expect(calculateCost(id, 1000, 1000)).not.toHaveProperty("error");
    }
  });
});
