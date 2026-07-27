# AI Comparison Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `ai-comparisons` module: a blind, side-by-side comparison where a student asks two AI models the same prompt over one to five trials, votes left or right each time, and sees a final report that reveals both models, the winner, and each model's vote count.

**Architecture:** A server page (`app/(app)/ai-comparisons/page.tsx`) mirrors the existing module pages: it runs the auth guard, resolves the module's available models through the existing catalog helpers, records a content-free `module_open` event, and renders a client component. The client component `AiComparison` is a three-phase state machine (setup, trials, report). Each trial runs two independent `useChat` instances that both POST to the existing `/api/chat` route, one per model, in parallel. All pairing, side-assignment, and vote-tallying logic lives in one pure module (`lib/comparison/pairing.ts`) so it is unit-testable without a browser. Nothing about a comparison is persisted server-side; the only server write is the content-free usage event the shared route already records.

**Tech Stack:** Next.js 16 (App Router, React Server Components), React 19, `@ai-sdk/react` `useChat` with `DefaultChatTransport`, the `ai` SDK v7 streaming route, Tailwind v4 with Miami brand tokens, Vitest (node env) for pure logic, Playwright + `@axe-core/playwright` for end-to-end and accessibility.

## Global Constraints

Copied verbatim from the spec and the project's standing rules. Every task's requirements implicitly include this section.

- **No git commits.** The working tree stays uncommitted. Every task ends by running its tests and confirming they pass, not by committing. Do not run `git add` or `git commit` at any point.
- **Ephemeral, no persistence.** No server-side persistence of prompts or answers. The only server write is the content-free `usage_events` row that `/api/chat` already records (module, event type, model, provider, token counts, cost, latency, prompt and response character *lengths*, outcome; never text). The report is computed entirely in the browser session.
- **No secrets in the client.** The browser sends a model *id* only. Provider keys stay on the server, inside the existing `/api/chat` route. No new route handles credentials.
- **Models hidden until the report.** Model display names must not appear anywhere in the setup answers, the trial panes, or the DOM during trials. They are revealed only in the final report.
- **WCAG 2.1 AA.** Native controls (radios in a real `fieldset`, buttons, labelled textareas). State announced with `role="status"` / `role="alert"`, never colour alone. Winner marked with text ("Winner"), not only a colour. Must pass `@axe-core/playwright` at desktop and at 320px width (the two panes stack on narrow screens).
- **Miami brand tokens only.** Use the existing token classes: `miami-red`, `accent-red`, `light-tan`, `medium-tan`, `dark-tan`, `medium-gray`, `paper`, `rounded-card`, `ribbon`. No raw hex, no new colours.
- **No em dashes in any user-facing copy.** Use a period, a comma, or "to" for ranges. This applies to every string a student can read.
- **n trials:** default 1, minimum 1, maximum 5. One prompt at a time.
- **This is not the Next.js you know** (`web/AGENTS.md`): before writing any framework-level code, consult `node_modules/next/dist/docs/` and heed deprecation notices. The page pattern in this plan is copied from a working module page, so follow it exactly rather than from memory.

---

## Design Decisions (spec left these open)

The spec explicitly asks the plan to settle these. Each is decided here with a rationale; the tasks implement the decision.

### D1. Streaming two models: reuse `/api/chat` twice, do not add a route

**Decision:** Run two independent `useChat` instances in the trial component, each with its own `DefaultChatTransport` pointed at the existing `/api/chat`, and call `sendMessage` on both with the same prompt text and each side's `modelId`. Add one small config entry so the shared route accepts the new module.

**Why:** `/api/chat` already does everything a comparison needs per side: server-authoritative model policy (`getPageModels(module).includes(modelId)`), provider-key resolution, rate limiting, empty-response and truncation notices, error classification with student-safe messages, and the content-free usage event. A dedicated `/api/compare` route would duplicate roughly 200 lines of that and would have to multiplex two independent token streams over one HTTP response, which is strictly harder than two parallel streams. The only server change required is registering the module (see D2). Cost of the decision: two rate-limit tokens are consumed per trial against the `chat:${email}` bucket (limit 20 per minute); at the maximum of 5 trials that is 10 tokens, well within budget.

**Rejected alternative:** A single server route that fans out to both models and merges the streams. Rejected as duplicative and more complex, with no reuse benefit.

### D2. Register `ai_comparisons` in `CHAT_MODULES`

**Decision:** Add an `ai_comparisons` entry to `CHAT_MODULES` in `lib/chat/config.ts`, with a neutral, identical system prompt for both models.

**Why:** `/api/chat` rejects any module not present in `CHAT_MODULES` (`if (!moduleConfig) return errorResponse(400, "Unknown module.")`). The model allow-list side (`getPageModels("ai_comparisons")`) already exists in `lib/config/models.ts` (confirmed: `PAGE_MODELS.ai_comparisons = { includeAll: true, excludeTags: ["realtime", "speech"] }`). Both models receive the *same* neutral prompt so the student is comparing the models, not two different instructions. The prompt is deliberately minimal (no tutor persona, no coding-style block) for fairness.

### D3. Time-seed mechanism: a seeded PRNG, seeded from `Date.now()` at session start

**Decision:** A pure function `pickPair(available, seed)` uses a small deterministic PRNG (mulberry32) to choose two distinct model ids. In anonymous mode the client passes `seed = Date.now()` captured once when the student starts, and holds it in state so re-renders never re-pick.

**Why:** The spec calls for a "time-based seed". A seeded PRNG honours that wording and, unlike a bare `Math.random()`, is reproducible: the same seed and model list always yield the same pair, which makes the logic unit-testable. mulberry32 is not cryptographic and does not need to be; it only shuffles a public list of model ids.

### D4. Left/right assignment: deterministic per-trial alternation, seeded per session

**Decision:** `leftSlotForTrial(seed, trialIndex)` returns which slot (0 or 1) sits on the left for that trial, computed as `(seedParity(seed) + trialIndex) % 2`. The starting side is chosen by a seed-derived coin flip, then the two models swap sides every trial.

**Why:** This satisfies blind fairness two ways. First, because the starting side flips unpredictably per session, a student cannot infer identity from "the left one is always X". Second, strict alternation guarantees each model spends an equal number of trials on each side (for an even trial count), which pure per-trial randomness does not: with n=2, random assignment can land the same model on the left both times and reintroduce the position bias we are trying to remove. It is also fully deterministic given the seed, so both the pairing logic and the end-to-end flows are testable. The models stay hidden until the report regardless of side.

### D5. A fixed pair for the whole session; votes tally per model

**Decision:** The two models are chosen once (randomly in anonymous mode, or picked once by the student) and the same pair is used for every trial. Votes accumulate per underlying model (per slot), not per screen side. `resolveVote(side, leftSlot)` maps a left/right vote back to the slot that received it.

**Why:** The report must "reveal both models and the number of votes each received", which requires a stable identity for each model across all trials. Slots (0 and 1) are that stable identity; sides (left/right) are presentation only and swap per D4.

### D6. Report layout and tie handling

**Decision:** After the last trial, a report section reveals slot 0 and slot 1 by display name, shows each one's vote count, and highlights the winner with a bordered card plus the text "Winner" (not colour alone). A tie (equal vote counts, including 0 to 0) shows the heading "It is a tie", no winner highlight, and a sentence stating both models received the same number of votes.

**Why:** WCAG requires the winner cue to be more than colour, hence the "Winner" text and heavier border. Ties are a real outcome at even trial counts and must read unambiguously.

---

## File Structure

**New files**

- `web/lib/comparison/pairing.ts` — pure logic: seeded PRNG, `pickPair`, `leftSlotForTrial`, `resolveVote`, `decideOutcome`, the `ComparisonPair` and `Outcome` types, and `NotEnoughModelsError`. No React, no imports from the app. This is the unit-tested core.
- `web/lib/comparison/config.ts` — constants: `DEFAULT_TRIALS`, `MAX_TRIALS`, `COMPARISON_MODULE_KEY`, `COMPARISON_MODULE_SLUG`.
- `web/lib/prompts/ai-comparison.ts` — `AI_COMPARISON_SYSTEM_PROMPT`, the identical neutral prompt both models receive.
- `web/app/(app)/ai-comparisons/page.tsx` — server component: auth guard, model resolution, `module_open` event, renders `AiComparison`. Overrides the `[module]` placeholder simply by existing.
- `web/components/comparison/AiComparison.tsx` — client root; the three-phase state machine.
- `web/components/comparison/ComparisonSetup.tsx` — mode radios (anonymous vs pick), trial-count input, two model pickers for pick mode.
- `web/components/comparison/ComparisonTrial.tsx` — prompt input, two `useChat` instances, two panes, vote buttons.
- `web/components/comparison/ComparisonPane.tsx` — one blind answer pane (presentational).
- `web/components/comparison/ComparisonReport.tsx` — reveal, winner highlight, vote counts, tie copy, restart.
- `web/tests/unit/comparison-pairing.test.ts` — Vitest for `pairing.ts`.
- `web/tests/unit/comparison-config.test.ts` — Vitest asserting the module wiring (config entry, prompt, allowed-model list).
- `web/tests/e2e/ai-comparison.spec.ts` — Playwright flow, blindness, report, and axe checks.

**Modified files**

- `web/lib/chat/config.ts` — add the `ai_comparisons` entry to `CHAT_MODULES` and import the neutral prompt.

**Unchanged but relied upon (do not edit)**

- `web/app/api/chat/route.ts` — reused as-is for streaming both sides.
- `web/lib/config/models.ts` — `ai_comparisons` already in `PAGE_MODELS`; `buildModelOptions`, `getPageModels`, `filterAvailableModels`, `getModelDisplayName` reused.
- `web/lib/modules.ts` — `ai-comparisons` slug already registered.
- `web/components/ModelChooser.tsx`, `web/components/chat/Markdown.tsx` — reused.

---

## Task 1: Pairing and outcome logic (pure, unit-tested)

**Files:**
- Create: `web/lib/comparison/pairing.ts`
- Test: `web/tests/unit/comparison-pairing.test.ts`

**Interfaces:**
- Consumes: nothing (no app imports).
- Produces:
  - `type ComparisonPair = readonly [string, string]`
  - `class NotEnoughModelsError extends Error`
  - `pickPair(available: string[], seed: number): ComparisonPair`
  - `leftSlotForTrial(seed: number, trialIndex: number): 0 | 1`
  - `resolveVote(side: "left" | "right", leftSlot: 0 | 1): 0 | 1`
  - `interface Outcome { votesSlot0: number; votesSlot1: number; winner: 0 | 1 | null }`
  - `decideOutcome(slotVotes: (0 | 1)[]): Outcome`

- [ ] **Step 1: Write the failing test**

Create `web/tests/unit/comparison-pairing.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import {
  pickPair,
  leftSlotForTrial,
  resolveVote,
  decideOutcome,
  NotEnoughModelsError,
} from "@/lib/comparison/pairing";

const MODELS = ["a", "b", "c", "d", "e"];

describe("pickPair", () => {
  it("is deterministic for a given seed and list", () => {
    expect(pickPair(MODELS, 12345)).toEqual(pickPair(MODELS, 12345));
  });

  it("returns two distinct ids, both from the list", () => {
    for (let seed = 0; seed < 200; seed++) {
      const [x, y] = pickPair(MODELS, seed);
      expect(x).not.toBe(y);
      expect(MODELS).toContain(x);
      expect(MODELS).toContain(y);
    }
  });

  it("can reach every model across seeds, so no id is unpickable", () => {
    const seen = new Set<string>();
    for (let seed = 0; seed < 500; seed++) {
      const [x, y] = pickPair(MODELS, seed);
      seen.add(x);
      seen.add(y);
    }
    expect(seen).toEqual(new Set(MODELS));
  });

  it("ignores duplicate ids in the input", () => {
    const [x, y] = pickPair(["a", "a", "b"], 7);
    expect(x).not.toBe(y);
  });

  it("throws when fewer than two distinct models are available", () => {
    expect(() => pickPair(["a"], 1)).toThrow(NotEnoughModelsError);
    expect(() => pickPair(["a", "a"], 1)).toThrow(NotEnoughModelsError);
  });
});

describe("leftSlotForTrial", () => {
  it("is deterministic for a given seed and trial", () => {
    expect(leftSlotForTrial(999, 3)).toBe(leftSlotForTrial(999, 3));
  });

  it("alternates sides on consecutive trials", () => {
    for (const seed of [0, 1, 42, 100000]) {
      expect(leftSlotForTrial(seed, 0)).not.toBe(leftSlotForTrial(seed, 1));
      expect(leftSlotForTrial(seed, 1)).not.toBe(leftSlotForTrial(seed, 2));
    }
  });

  it("gives each slot the left side equally over an even trial count", () => {
    const seed = 555;
    const lefts = [0, 1, 2, 3].map((t) => leftSlotForTrial(seed, t));
    expect(lefts.filter((s) => s === 0).length).toBe(2);
    expect(lefts.filter((s) => s === 1).length).toBe(2);
  });
});

describe("resolveVote", () => {
  it("maps a left vote to the slot on the left", () => {
    expect(resolveVote("left", 0)).toBe(0);
    expect(resolveVote("left", 1)).toBe(1);
  });
  it("maps a right vote to the other slot", () => {
    expect(resolveVote("right", 0)).toBe(1);
    expect(resolveVote("right", 1)).toBe(0);
  });
});

describe("decideOutcome", () => {
  it("names the slot with more votes", () => {
    expect(decideOutcome([0, 0, 1]).winner).toBe(0);
    expect(decideOutcome([1, 1, 0]).winner).toBe(1);
  });
  it("returns a tie for equal votes, including zero to zero", () => {
    expect(decideOutcome([0, 1]).winner).toBeNull();
    expect(decideOutcome([]).winner).toBeNull();
  });
  it("counts votes per slot", () => {
    const out = decideOutcome([0, 0, 1]);
    expect(out.votesSlot0).toBe(2);
    expect(out.votesSlot1).toBe(1);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd web && npx vitest run tests/unit/comparison-pairing.test.ts`
Expected: FAIL. The module does not exist yet, so Vitest reports "Failed to resolve import \"@/lib/comparison/pairing\"".

- [ ] **Step 3: Write the implementation**

Create `web/lib/comparison/pairing.ts`:

```ts
/**
 * Pure logic for the AI Comparison module: choosing the pair, deciding which
 * model sits on which side per trial, mapping a vote back to a model, and
 * tallying the result. No React and no app imports, so it is unit-testable in
 * isolation and carries the parts of the module that must be provably correct.
 */

/**
 * mulberry32: a small, well-distributed 32-bit PRNG. Deterministic given its
 * seed, which is the whole point here: an anonymous pairing seeded from the
 * clock must be reproducible within a session and testable. It is not
 * cryptographic and does not need to be, since it only shuffles a public list
 * of model ids.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The two model ids under comparison, in fixed slot order [slot0, slot1]. */
export type ComparisonPair = readonly [string, string];

export class NotEnoughModelsError extends Error {
  constructor() {
    super("At least two different models are needed to run a comparison.");
    this.name = "NotEnoughModelsError";
  }
}

/**
 * Two distinct model ids drawn from `available` using `seed`. The same seed and
 * list always yield the same pair, which is what makes the anonymous
 * time-based seed reproducible and testable. Duplicates in the input are
 * collapsed first, so a short list padded with repeats cannot pick the same
 * model twice.
 */
export function pickPair(available: string[], seed: number): ComparisonPair {
  const unique = [...new Set(available)];
  if (unique.length < 2) throw new NotEnoughModelsError();
  const rand = mulberry32(seed);
  const first = Math.floor(rand() * unique.length);
  // Draw the second from the remaining n-1 positions, then map back around the
  // hole at `first`, so the two indices are always distinct.
  let second = Math.floor(rand() * (unique.length - 1));
  if (second >= first) second += 1;
  return [unique[first], unique[second]] as const;
}

/** A stable per-seed coin flip, deciding which slot starts on the left. */
function seedParity(seed: number): 0 | 1 {
  return mulberry32(seed)() < 0.5 ? 0 : 1;
}

/**
 * Which slot (0 or 1) is shown on the LEFT for a given trial. The starting side
 * is a seed-derived coin flip, then the two models swap sides every trial. This
 * blinds the student (a consistent left position never reveals identity because
 * the starting side flips per session) and balances position bias (each slot
 * spends an equal number of trials on the left for an even trial count).
 */
export function leftSlotForTrial(seed: number, trialIndex: number): 0 | 1 {
  return (((seedParity(seed) + trialIndex) % 2) as 0 | 1);
}

/**
 * Maps a vote for a screen side back to the slot that actually received it,
 * given which slot was on the left for that trial.
 */
export function resolveVote(side: "left" | "right", leftSlot: 0 | 1): 0 | 1 {
  const rightSlot: 0 | 1 = leftSlot === 0 ? 1 : 0;
  return side === "left" ? leftSlot : rightSlot;
}

export interface Outcome {
  votesSlot0: number;
  votesSlot1: number;
  /** Winning slot, or null for a tie (including zero to zero). */
  winner: 0 | 1 | null;
}

/** Tallies a list of per-slot votes into a final outcome. */
export function decideOutcome(slotVotes: (0 | 1)[]): Outcome {
  const votesSlot0 = slotVotes.filter((s) => s === 0).length;
  const votesSlot1 = slotVotes.filter((s) => s === 1).length;
  const winner =
    votesSlot0 === votesSlot1 ? null : votesSlot0 > votesSlot1 ? 0 : 1;
  return { votesSlot0, votesSlot1, winner };
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd web && npx vitest run tests/unit/comparison-pairing.test.ts`
Expected: PASS. All cases green.

- [ ] **Step 5: Verify (no commit)**

Run: `cd web && npm run typecheck`
Expected: exits 0 with no errors. Do not commit; leave the working tree uncommitted.

---

## Task 2: Module wiring and constants (config + prompt, unit-tested)

**Files:**
- Create: `web/lib/comparison/config.ts`
- Create: `web/lib/prompts/ai-comparison.ts`
- Modify: `web/lib/chat/config.ts` (add the `ai_comparisons` entry to `CHAT_MODULES`, add the import)
- Test: `web/tests/unit/comparison-config.test.ts`

**Interfaces:**
- Consumes: `CHAT_OUTPUT_TOKENS` from `@/lib/chat/budget` (already imported in `lib/chat/config.ts`), `getPageModels` from `@/lib/config/models`.
- Produces:
  - `DEFAULT_TRIALS = 1`, `MAX_TRIALS = 5`, `COMPARISON_MODULE_KEY = "ai_comparisons"`, `COMPARISON_MODULE_SLUG = "ai-comparisons"` from `@/lib/comparison/config`.
  - `AI_COMPARISON_SYSTEM_PROMPT` from `@/lib/prompts/ai-comparison`.
  - A `CHAT_MODULES.ai_comparisons` entry so `/api/chat` accepts the module.

- [ ] **Step 1: Write the failing test**

Create `web/tests/unit/comparison-config.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { CHAT_MODULES } from "@/lib/chat/config";
import { getPageModels, MODELS } from "@/lib/config/models";
import { AI_COMPARISON_SYSTEM_PROMPT } from "@/lib/prompts/ai-comparison";
import {
  DEFAULT_TRIALS,
  MAX_TRIALS,
  COMPARISON_MODULE_KEY,
  COMPARISON_MODULE_SLUG,
} from "@/lib/comparison/config";

describe("comparison constants", () => {
  it("defaults to one trial and caps at five", () => {
    expect(DEFAULT_TRIALS).toBe(1);
    expect(MAX_TRIALS).toBe(5);
  });
  it("keys and slug match the catalog and the route segment", () => {
    expect(COMPARISON_MODULE_KEY).toBe("ai_comparisons");
    expect(COMPARISON_MODULE_SLUG).toBe("ai-comparisons");
  });
});

describe("the chat route accepts the comparison module", () => {
  it("registers ai_comparisons in CHAT_MODULES", () => {
    const cfg = CHAT_MODULES[COMPARISON_MODULE_KEY];
    expect(cfg).toBeDefined();
    expect(cfg.slug).toBe(COMPARISON_MODULE_SLUG);
    expect(cfg.systemPrompt).toBe(AI_COMPARISON_SYSTEM_PROMPT);
    expect(cfg.temperature).toBeGreaterThanOrEqual(0);
    expect(cfg.maxOutputTokens).toBeGreaterThan(0);
  });

  it("uses a neutral prompt with no em dashes and no tutor persona", () => {
    expect(AI_COMPARISON_SYSTEM_PROMPT.length).toBeGreaterThan(20);
    expect(AI_COMPARISON_SYSTEM_PROMPT).not.toContain("—"); // em dash
    expect(AI_COMPARISON_SYSTEM_PROMPT.toLowerCase()).not.toContain("tutor");
  });
});

describe("the comparison offers at least two real models", () => {
  it("lists two or more known models", () => {
    const list = getPageModels(COMPARISON_MODULE_KEY);
    expect(list.length).toBeGreaterThanOrEqual(2);
    for (const id of list) expect(MODELS[id]).toBeDefined();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd web && npx vitest run tests/unit/comparison-config.test.ts`
Expected: FAIL. Imports `@/lib/prompts/ai-comparison` and `@/lib/comparison/config` do not resolve, and `CHAT_MODULES.ai_comparisons` is undefined.

- [ ] **Step 3: Write the implementation**

Create `web/lib/comparison/config.ts`:

```ts
/** Trial bounds and identifiers for the AI Comparison module. */
export const DEFAULT_TRIALS = 1;
export const MAX_TRIALS = 5;

/** Catalog and analytics key (matches PAGE_MODELS in lib/config/models.ts). */
export const COMPARISON_MODULE_KEY = "ai_comparisons";
/** Route segment (matches lib/modules.ts). */
export const COMPARISON_MODULE_SLUG = "ai-comparisons";
```

Create `web/lib/prompts/ai-comparison.ts`:

```ts
/**
 * Both models in a comparison receive this identical, neutral instruction, so
 * the student compares the models themselves and not two different prompts.
 * Deliberately minimal: no tutor persona and no coding-style block, both of
 * which would shape the answers and blunt the comparison.
 */
export const AI_COMPARISON_SYSTEM_PROMPT =
  "You are a helpful assistant for an undergraduate business analytics student. Answer the question clearly and directly. If code helps, keep it short and correct.";
```

Modify `web/lib/chat/config.ts`. Add the import near the other prompt imports at the top of the file:

```ts
import { AI_COMPARISON_SYSTEM_PROMPT } from "@/lib/prompts/ai-comparison";
```

Then add this entry to the `CHAT_MODULES` object (after the `sandbox_chat` entry, before the closing brace):

```ts
  ai_comparisons: {
    key: "ai_comparisons",
    slug: "ai-comparisons",
    name: "AI Comparison",
    systemPrompt: AI_COMPARISON_SYSTEM_PROMPT,
    // No fixed opening turn: each trial is a single, self-contained question.
    // A moderate temperature so answers are natural rather than clipped; both
    // models get the same value, so the comparison stays fair.
    temperature: 0.7,
    maxOutputTokens: CHAT_OUTPUT_TOKENS,
    placeholder: "Ask both models the same question.",
  },
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd web && npx vitest run tests/unit/comparison-config.test.ts`
Expected: PASS.

- [ ] **Step 5: Verify (no commit)**

Run: `cd web && npm run typecheck && npx vitest run tests/unit/models.test.ts tests/unit/chat-config.test.ts`
Expected: exits 0. `models.test.ts` already includes `ai_comparisons` in its module-key sweep, so this confirms the module still resolves to known models. Do not commit.

---

## Task 3: Page route and setup screen render

**Files:**
- Create: `web/app/(app)/ai-comparisons/page.tsx`
- Create: `web/components/comparison/AiComparison.tsx`
- Create: `web/components/comparison/ComparisonSetup.tsx`
- Test: `web/tests/e2e/ai-comparison.spec.ts` (first test only; later tasks extend it)

**Interfaces:**
- Consumes: `auth` from `@/lib/auth`; `buildModelOptions`, `getPageModels`, `type ModelOption` from `@/lib/config/models`; `filterAvailableModels` from `@/lib/providers`; `recordUsageEvent` from `@/lib/db`; `DEFAULT_TRIALS`, `MAX_TRIALS` from `@/lib/comparison/config`; `ModelChooser` from `@/components/ModelChooser`.
- Produces:
  - `AiComparison({ models }: { models: ModelOption[] })` React component.
  - `ComparisonSetup` with `type SetupMode = "anonymous" | "pick"` and an `onStart(config: { mode: SetupMode; trials: number; leftPick: string; rightPick: string }) => void` prop.

**Note on UI testing:** This project has no jsdom/RTL setup (Vitest runs in the `node` environment). UI is verified end-to-end with Playwright against the mock model (`CHATISA_MOCK_LLM=1`, set by `playwright.config.ts`), which also provides test-mode auth via stored session state, so the page loads without a manual login. Each UI task writes a failing Playwright test first, then implements to green.

- [ ] **Step 1: Write the failing test**

Create `web/tests/e2e/ai-comparison.spec.ts`:

```ts
import { test, expect } from "@playwright/test";

test.describe("AI Comparison setup", () => {
  test("shows the module and its two setup modes", async ({ page }) => {
    await page.goto("/ai-comparisons");
    await expect(
      page.getByRole("heading", { level: 1, name: "AI Comparison" }),
    ).toBeVisible();

    // Anonymous is the default and is a real radio.
    const surprise = page.getByRole("radio", { name: /Surprise me/i });
    await expect(surprise).toBeVisible();
    await expect(surprise).toBeChecked();
    await expect(
      page.getByRole("radio", { name: /Pick the two models/i }),
    ).toBeVisible();

    // Trial count control is present and bounded.
    const trials = page.getByLabel(/How many questions/i);
    await expect(trials).toBeVisible();
    await expect(trials).toHaveValue("1");

    // No model names are shown yet in the default anonymous mode.
    await expect(page.getByText("Choose a different model")).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: "Start comparing" }),
    ).toBeVisible();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd web && npm run test:e2e -- ai-comparison.spec.ts`
Expected: FAIL. Navigating to `/ai-comparisons` renders the module placeholder (the `[module]` dynamic route), so the level-1 heading "AI Comparison" from this new page is not present, or the setup radios are missing.

- [ ] **Step 3: Write the page and the root component**

Create `web/app/(app)/ai-comparisons/page.tsx`:

```tsx
import type { Metadata } from "next";
import { redirect } from "next/navigation";
import { auth } from "@/lib/auth";
import { buildModelOptions, getPageModels } from "@/lib/config/models";
import { filterAvailableModels } from "@/lib/providers";
import { recordUsageEvent } from "@/lib/db";
import { AiComparison } from "@/components/comparison/AiComparison";

export const metadata: Metadata = { title: "AI Comparison" };

export default async function AiComparisonsPage() {
  const session = await auth();
  if (!session?.user?.email) redirect("/login");

  const available = filterAvailableModels(getPageModels("ai_comparisons"));
  const { options } = buildModelOptions("ai_comparisons", available);

  recordUsageEvent({
    userEmail: session.user.email,
    module: "ai_comparisons",
    eventType: "module_open",
  });

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <p className="ribbon">Module</p>
      <h1 className="mt-5 text-4xl">AI Comparison</h1>
      <p className="mt-3 max-w-2xl text-lg leading-relaxed">
        Put two AI models side by side on the same question and vote for the
        answer you prefer. The models stay hidden until the end.
      </p>

      {options.length < 2 ? (
        <div
          role="status"
          className="mt-8 rounded-card border-2 border-miami-red bg-paper p-5"
        >
          <h2 className="font-bold text-miami-red">Not enough models available</h2>
          <p className="mt-1">
            A comparison needs at least two models, and this server does not
            have two configured right now. Contact the ChatISA maintainers.
          </p>
        </div>
      ) : (
        <div className="mt-8">
          <AiComparison models={options} />
        </div>
      )}
    </div>
  );
}
```

Create `web/components/comparison/AiComparison.tsx` (setup phase wired; trial and report phases are placeholders filled in by later tasks, but the imports and state machine are complete so nothing dangles):

```tsx
"use client";

import { useState } from "react";
import { ComparisonSetup, type SetupMode } from "./ComparisonSetup";
import {
  pickPair,
  type ComparisonPair,
} from "@/lib/comparison/pairing";
import type { ModelOption } from "@/lib/config/models";

type Phase =
  | { name: "setup" }
  | {
      name: "trials";
      pair: ComparisonPair;
      seed: number;
      trials: number;
      index: number;
      votes: (0 | 1)[];
    }
  | { name: "report"; pair: ComparisonPair; votes: (0 | 1)[] };

export function AiComparison({ models }: { models: ModelOption[] }) {
  const [phase, setPhase] = useState<Phase>({ name: "setup" });
  const ids = models.map((m) => m.id);

  function start(config: {
    mode: SetupMode;
    trials: number;
    leftPick: string;
    rightPick: string;
  }) {
    const seed = Date.now();
    const pair: ComparisonPair =
      config.mode === "anonymous"
        ? pickPair(ids, seed)
        : [config.leftPick, config.rightPick];
    setPhase({
      name: "trials",
      pair,
      seed,
      trials: config.trials,
      index: 0,
      votes: [],
    });
  }

  if (phase.name === "setup") {
    return <ComparisonSetup models={models} onStart={start} />;
  }

  // Trials and report phases are implemented in Tasks 5 to 7. Until then this
  // branch is unreachable in normal use because no other phase is set; it is a
  // placeholder that keeps the state machine total.
  return null;
}
```

Create `web/components/comparison/ComparisonSetup.tsx`:

```tsx
"use client";

import { useState } from "react";
import { ModelChooser } from "@/components/ModelChooser";
import { DEFAULT_TRIALS, MAX_TRIALS } from "@/lib/comparison/config";
import type { ModelOption } from "@/lib/config/models";

export type SetupMode = "anonymous" | "pick";

export function ComparisonSetup({
  models,
  onStart,
}: {
  models: ModelOption[];
  onStart: (config: {
    mode: SetupMode;
    trials: number;
    leftPick: string;
    rightPick: string;
  }) => void;
}) {
  const [mode, setMode] = useState<SetupMode>("anonymous");
  const [trials, setTrials] = useState(DEFAULT_TRIALS);
  const [leftPick, setLeftPick] = useState(models[0].id);
  const [rightPick, setRightPick] = useState(models[1].id);

  const samePick = mode === "pick" && leftPick === rightPick;

  function start(event: React.FormEvent) {
    event.preventDefault();
    if (samePick) return;
    onStart({ mode, trials, leftPick, rightPick });
  }

  return (
    <form onSubmit={start} className="flex flex-col gap-6">
      <fieldset>
        <legend className="text-sm font-bold">
          How should the two models be chosen?
        </legend>
        <div className="mt-2 flex flex-col gap-2">
          <label className="flex items-start gap-2">
            <input
              type="radio"
              name="comparison-mode"
              checked={mode === "anonymous"}
              onChange={() => setMode("anonymous")}
              className="mt-1.5"
            />
            <span>
              <strong>Surprise me (blind)</strong>
              <span className="block text-sm text-dark-tan">
                Two models are chosen at random and stay hidden until the end.
              </span>
            </span>
          </label>
          <label className="flex items-start gap-2">
            <input
              type="radio"
              name="comparison-mode"
              checked={mode === "pick"}
              onChange={() => setMode("pick")}
              className="mt-1.5"
            />
            <span>
              <strong>Pick the two models</strong>
              <span className="block text-sm text-dark-tan">
                Choose both models yourself. Their answers are still shown left
                and right without labels.
              </span>
            </span>
          </label>
        </div>
      </fieldset>

      {mode === "pick" ? (
        <div className="grid gap-4 md:grid-cols-2">
          <div>
            <p className="text-sm font-bold">First model</p>
            <ModelChooser options={models} value={leftPick} onChange={setLeftPick} />
          </div>
          <div>
            <p className="text-sm font-bold">Second model</p>
            <ModelChooser
              options={models}
              value={rightPick}
              onChange={setRightPick}
            />
          </div>
        </div>
      ) : null}

      {samePick ? (
        <p role="alert" className="text-miami-red">
          Choose two different models.
        </p>
      ) : null}

      <div>
        <label htmlFor="comparison-trials" className="text-sm font-bold">
          How many questions? (1 to {MAX_TRIALS})
        </label>
        <input
          id="comparison-trials"
          type="number"
          min={1}
          max={MAX_TRIALS}
          value={trials}
          onChange={(e) =>
            setTrials(Math.max(1, Math.min(MAX_TRIALS, Number(e.target.value) || 1)))
          }
          className="mt-1 block w-24 rounded-card border border-medium-tan bg-paper p-2"
        />
      </div>

      <button
        type="submit"
        disabled={samePick}
        className="self-start rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
      >
        Start comparing
      </button>
    </form>
  );
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd web && npm run test:e2e -- ai-comparison.spec.ts`
Expected: PASS. The new static route overrides the placeholder, and the setup screen renders with both mode radios and the trial-count input.

- [ ] **Step 5: Verify (no commit)**

Run: `cd web && npm run typecheck && npm run lint`
Expected: exits 0. Do not commit.

---

## Task 4: Setup interactions (mode switch, trial count, pick mode)

**Files:**
- Modify: `web/tests/e2e/ai-comparison.spec.ts` (add a test)

No component changes are expected; this task proves the setup behaviours the previous task built. If a test fails, fix `ComparisonSetup.tsx`.

**Interfaces:**
- Consumes: `ComparisonSetup` from Task 3.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

Append to `web/tests/e2e/ai-comparison.spec.ts`, inside a new describe block:

```ts
test.describe("AI Comparison setup interactions", () => {
  test("reveals two model pickers only in pick mode and enforces distinctness", async ({
    page,
  }) => {
    await page.goto("/ai-comparisons");

    // Anonymous mode: no model pickers.
    await expect(page.getByText("Choose a different model")).toHaveCount(0);

    // Switching to pick mode reveals two pickers.
    await page.getByRole("radio", { name: /Pick the two models/i }).check();
    await expect(
      page.getByRole("button", { name: "Choose a different model" }),
    ).toHaveCount(2);

    // The trial count clamps to the maximum of five.
    const trials = page.getByLabel(/How many questions/i);
    await trials.fill("9");
    await trials.blur();
    await expect(trials).toHaveValue("5");

    // Start is available with two distinct default picks.
    await expect(
      page.getByRole("button", { name: "Start comparing" }),
    ).toBeEnabled();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `cd web && npm run test:e2e -- ai-comparison.spec.ts`
Expected: This should PASS immediately if Task 3 was implemented correctly (the behaviours are already built). If it FAILS, treat the failure as the red bar: fix `ComparisonSetup.tsx` (for example, the clamp in the `onChange` handler or the `md:grid-cols-2` picker block) until it passes. The clamp on `fill("9")` relies on the `onChange` handler capping at `MAX_TRIALS`; confirm that handler runs on input.

- [ ] **Step 3: Confirm green**

Run: `cd web && npm run test:e2e -- ai-comparison.spec.ts`
Expected: PASS.

- [ ] **Step 4: Verify (no commit)**

Run: `cd web && npm run typecheck`
Expected: exits 0. Do not commit.

---

## Task 5: A single trial: two streams, blind panes, and a vote

**Files:**
- Create: `web/components/comparison/ComparisonPane.tsx`
- Create: `web/components/comparison/ComparisonTrial.tsx`
- Modify: `web/components/comparison/AiComparison.tsx` (render the trial phase, add `vote`)
- Modify: `web/tests/e2e/ai-comparison.spec.ts` (add a test)

**Interfaces:**
- Consumes: `useChat` from `@ai-sdk/react`; `DefaultChatTransport`, `type UIMessage` from `ai`; `Markdown` from `@/components/chat/Markdown`; `leftSlotForTrial`, `resolveVote`, `type ComparisonPair` from `@/lib/comparison/pairing`.
- Produces:
  - `ComparisonPane({ side, text, status, error })` where `side: "left" | "right"`, `text: string`, `status: string`, `error?: string`.
  - `ComparisonTrial({ pair, seed, trialIndex, trialCount, onVote })` where `pair: ComparisonPair`, `seed: number`, `trialIndex: number`, `trialCount: number`, `onVote: (slot: 0 | 1) => void`.
  - `AiComparison` now handles the `"trials"` phase and defines `vote(slot: 0 | 1)`.

- [ ] **Step 1: Write the failing test**

Append to `web/tests/e2e/ai-comparison.spec.ts`:

```ts
test.describe("AI Comparison trial", () => {
  test("asks both models, streams two blind answers, and takes a vote", async ({
    page,
  }) => {
    await page.goto("/ai-comparisons");

    // Default anonymous mode, one trial.
    await page.getByRole("button", { name: "Start comparing" }).click();

    // The prompt goes to both models.
    await page
      .getByLabel(/Your question for both models/i)
      .fill("How do I read a CSV?");
    await page.getByRole("button", { name: "Ask both models" }).click();

    // Two blind panes appear, labelled by side, never by model name.
    const left = page.getByRole("article", { name: "Answer on the left" });
    const right = page.getByRole("article", { name: "Answer on the right" });
    await expect(left).toContainText("read a CSV in both languages", {
      timeout: 15_000,
    });
    await expect(right).toContainText("read a CSV in both languages", {
      timeout: 15_000,
    });

    // The models are still hidden: no result section during the trial.
    await expect(page.getByText("Result", { exact: true })).toHaveCount(0);

    // Voting is offered once both answers are ready.
    await expect(
      page.getByText(/Both answers are ready/i),
    ).toBeVisible({ timeout: 15_000 });
    await expect(
      page.getByRole("button", { name: "Prefer the left answer" }),
    ).toBeEnabled();
    await page.getByRole("button", { name: "Prefer the left answer" }).click();
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd web && npm run test:e2e -- ai-comparison.spec.ts`
Expected: FAIL. `Start comparing` sets the trials phase, but `AiComparison` returns `null` for it, so the prompt textarea never appears.

- [ ] **Step 3: Implement the pane, the trial, and the trials phase**

Create `web/components/comparison/ComparisonPane.tsx`:

```tsx
"use client";

import { Markdown } from "@/components/chat/Markdown";

/**
 * One answer, kept blind. The heading names the side only ("left" or "right"),
 * never the model, so a student cannot learn which model they are reading until
 * the report reveals it.
 */
export function ComparisonPane({
  side,
  text,
  status,
  error,
}: {
  side: "left" | "right";
  text: string;
  status: string;
  error?: string;
}) {
  const heading = side === "left" ? "Answer on the left" : "Answer on the right";
  const headingId = `pane-${side}`;
  const streaming = status === "submitted" || status === "streaming";
  return (
    <article
      aria-labelledby={headingId}
      className="rounded-card border border-medium-tan bg-paper p-4"
    >
      <h2 id={headingId} className="mb-1 text-sm font-bold text-dark-tan">
        {heading}
      </h2>
      {error ? (
        <p role="alert" className="text-miami-red">
          This model could not answer. {error}
        </p>
      ) : text ? (
        <Markdown>{text}</Markdown>
      ) : (
        <p className="text-dark-tan">{streaming ? "Thinking." : "Waiting."}</p>
      )}
    </article>
  );
}
```

Create `web/components/comparison/ComparisonTrial.tsx`:

```tsx
"use client";

import { useMemo, useState } from "react";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { ComparisonPane } from "./ComparisonPane";
import {
  leftSlotForTrial,
  resolveVote,
  type ComparisonPair,
} from "@/lib/comparison/pairing";

/** Concatenated text of the most recent assistant message. */
function assistantText(messages: UIMessage[]): string {
  const last = [...messages].reverse().find((m) => m.role === "assistant");
  if (!last) return "";
  return last.parts
    .filter((p) => p.type === "text")
    .map((p) => ("text" in p ? p.text : ""))
    .join("");
}

export function ComparisonTrial({
  pair,
  seed,
  trialIndex,
  trialCount,
  onVote,
}: {
  pair: ComparisonPair;
  seed: number;
  trialIndex: number;
  trialCount: number;
  onVote: (slot: 0 | 1) => void;
}) {
  const [input, setInput] = useState("");
  const [submitted, setSubmitted] = useState(false);

  const leftSlot = useMemo(
    () => leftSlotForTrial(seed, trialIndex),
    [seed, trialIndex],
  );
  const rightSlot: 0 | 1 = leftSlot === 0 ? 1 : 0;
  const leftModelId = pair[leftSlot];
  const rightModelId = pair[rightSlot];

  // Two transports, two conversations, both aimed at the shared chat route.
  const [leftTransport] = useState(
    () => new DefaultChatTransport({ api: "/api/chat" }),
  );
  const [rightTransport] = useState(
    () => new DefaultChatTransport({ api: "/api/chat" }),
  );
  const left = useChat({ transport: leftTransport });
  const right = useChat({ transport: rightTransport });

  const busy =
    left.status === "submitted" ||
    left.status === "streaming" ||
    right.status === "submitted" ||
    right.status === "streaming";

  const leftText = assistantText(left.messages);
  const rightText = assistantText(right.messages);
  const bothReady =
    submitted &&
    !busy &&
    left.status === "ready" &&
    right.status === "ready" &&
    leftText.length > 0 &&
    rightText.length > 0;

  function ask(event: React.FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || busy || submitted) return;
    setSubmitted(true);
    left.sendMessage(
      { text },
      { body: { module: "ai_comparisons", modelId: leftModelId } },
    );
    right.sendMessage(
      { text },
      { body: { module: "ai_comparisons", modelId: rightModelId } },
    );
  }

  return (
    <section
      aria-label={`Trial ${trialIndex + 1} of ${trialCount}`}
      className="flex flex-col gap-4"
    >
      <p className="ribbon">
        Trial {trialIndex + 1} of {trialCount}
      </p>

      {!submitted ? (
        <form onSubmit={ask} className="flex flex-col gap-2">
          <label htmlFor="comparison-prompt" className="text-sm font-bold">
            Your question for both models
          </label>
          <textarea
            id="comparison-prompt"
            rows={3}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="w-full rounded-card border border-medium-tan bg-paper p-3"
          />
          <button
            type="submit"
            disabled={input.trim().length === 0}
            className="self-start rounded-card bg-miami-red px-4 py-2 font-bold text-paper hover:bg-accent-red disabled:cursor-not-allowed disabled:bg-medium-gray"
          >
            Ask both models
          </button>
        </form>
      ) : (
        <>
          <div className="grid gap-4 md:grid-cols-2">
            <ComparisonPane
              side="left"
              text={leftText}
              status={left.status}
              error={left.error?.message}
            />
            <ComparisonPane
              side="right"
              text={rightText}
              status={right.status}
              error={right.error?.message}
            />
          </div>

          <p role="status" className="text-sm text-dark-tan">
            {busy
              ? "Both models are answering."
              : bothReady
                ? "Both answers are ready. Choose the one you prefer."
                : ""}
          </p>

          <fieldset disabled={!bothReady} className="flex flex-col gap-2">
            <legend className="text-sm font-bold">
              Which answer do you prefer?
            </legend>
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={() => onVote(resolveVote("left", leftSlot))}
                className="rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red disabled:cursor-not-allowed disabled:text-medium-gray"
              >
                Prefer the left answer
              </button>
              <button
                type="button"
                onClick={() => onVote(resolveVote("right", leftSlot))}
                className="rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red disabled:cursor-not-allowed disabled:text-medium-gray"
              >
                Prefer the right answer
              </button>
            </div>
          </fieldset>
        </>
      )}
    </section>
  );
}
```

Modify `web/components/comparison/AiComparison.tsx`. Add the import for `ComparisonTrial`, add the `vote` function, and replace the `return null` placeholder with the trials-phase render. The `decideOutcome` import and report render arrive in Task 7; for now the last trial's vote moves to the report phase, which still renders nothing until Task 7. Full updated file:

```tsx
"use client";

import { useState } from "react";
import { ComparisonSetup, type SetupMode } from "./ComparisonSetup";
import { ComparisonTrial } from "./ComparisonTrial";
import { pickPair, type ComparisonPair } from "@/lib/comparison/pairing";
import type { ModelOption } from "@/lib/config/models";

type Phase =
  | { name: "setup" }
  | {
      name: "trials";
      pair: ComparisonPair;
      seed: number;
      trials: number;
      index: number;
      votes: (0 | 1)[];
    }
  | { name: "report"; pair: ComparisonPair; votes: (0 | 1)[] };

export function AiComparison({ models }: { models: ModelOption[] }) {
  const [phase, setPhase] = useState<Phase>({ name: "setup" });
  const ids = models.map((m) => m.id);

  function start(config: {
    mode: SetupMode;
    trials: number;
    leftPick: string;
    rightPick: string;
  }) {
    const seed = Date.now();
    const pair: ComparisonPair =
      config.mode === "anonymous"
        ? pickPair(ids, seed)
        : [config.leftPick, config.rightPick];
    setPhase({
      name: "trials",
      pair,
      seed,
      trials: config.trials,
      index: 0,
      votes: [],
    });
  }

  function vote(slot: 0 | 1) {
    setPhase((prev) => {
      if (prev.name !== "trials") return prev;
      const votes = [...prev.votes, slot];
      if (prev.index + 1 >= prev.trials) {
        return { name: "report", pair: prev.pair, votes };
      }
      return { ...prev, index: prev.index + 1, votes };
    });
  }

  if (phase.name === "setup") {
    return <ComparisonSetup models={models} onStart={start} />;
  }

  if (phase.name === "trials") {
    return (
      <ComparisonTrial
        // A fresh key per trial resets both useChat instances between prompts.
        key={phase.index}
        pair={phase.pair}
        seed={phase.seed}
        trialIndex={phase.index}
        trialCount={phase.trials}
        onVote={vote}
      />
    );
  }

  // Report phase is implemented in Task 7.
  return null;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd web && npm run test:e2e -- ai-comparison.spec.ts`
Expected: PASS. Both panes stream the mock reply, stay blind, and the left vote is accepted. (With n=1 the vote moves to the report phase, which renders nothing yet; the test does not assert on the report, so it passes.)

- [ ] **Step 5: Verify (no commit)**

Run: `cd web && npm run typecheck`
Expected: exits 0. Do not commit.

---

## Task 6: Multiple trials advance one prompt at a time

**Files:**
- Modify: `web/tests/e2e/ai-comparison.spec.ts` (add a test)

No component changes are expected; the state machine from Task 5 already advances. If the test fails, fix `AiComparison.tsx`.

**Interfaces:**
- Consumes: the `AiComparison` state machine and `ComparisonTrial` from Task 5.
- Produces: nothing new.

- [ ] **Step 1: Write the failing test**

Append to `web/tests/e2e/ai-comparison.spec.ts`:

```ts
test.describe("AI Comparison multiple trials", () => {
  test("runs one prompt per trial and advances after each vote", async ({
    page,
  }) => {
    await page.goto("/ai-comparisons");

    // Two trials.
    await page.getByLabel(/How many questions/i).fill("2");
    await page.getByRole("button", { name: "Start comparing" }).click();

    // Trial 1.
    await expect(page.getByText("Trial 1 of 2")).toBeVisible();
    await page.getByLabel(/Your question for both models/i).fill("First question");
    await page.getByRole("button", { name: "Ask both models" }).click();
    await expect(page.getByText(/Both answers are ready/i)).toBeVisible({
      timeout: 15_000,
    });
    await page.getByRole("button", { name: "Prefer the left answer" }).click();

    // Trial 2 starts with a fresh, empty prompt (one prompt at a time).
    await expect(page.getByText("Trial 2 of 2")).toBeVisible();
    const secondPrompt = page.getByLabel(/Your question for both models/i);
    await expect(secondPrompt).toHaveValue("");
    await secondPrompt.fill("Second question");
    await page.getByRole("button", { name: "Ask both models" }).click();
    await expect(page.getByText(/Both answers are ready/i)).toBeVisible({
      timeout: 15_000,
    });
    await page.getByRole("button", { name: "Prefer the right answer" }).click();

    // After the last vote the trial screen is gone.
    await expect(page.getByText("Trial 2 of 2")).toHaveCount(0);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails or passes**

Run: `cd web && npm run test:e2e -- ai-comparison.spec.ts`
Expected: The first two trial steps pass. The final assertion (`Trial 2 of 2` gone) passes only because the report phase renders `null` for now, so the trial disappears. If any earlier assertion FAILS (for example the fresh empty prompt), fix the `key={phase.index}` reset in `AiComparison.tsx`, which is what clears `useChat` and the local prompt state between trials.

- [ ] **Step 3: Confirm green**

Run: `cd web && npm run test:e2e -- ai-comparison.spec.ts`
Expected: PASS.

- [ ] **Step 4: Verify (no commit)**

Run: `cd web && npm run typecheck`
Expected: exits 0. Do not commit.

---

## Task 7: The report: reveal, winner, vote counts, ties, and accessibility

**Files:**
- Create: `web/components/comparison/ComparisonReport.tsx`
- Modify: `web/components/comparison/AiComparison.tsx` (render the report phase, add `restart`)
- Modify: `web/tests/e2e/ai-comparison.spec.ts` (add tests, including axe)

**Interfaces:**
- Consumes: `getModelDisplayName` from `@/lib/config/models`; `decideOutcome`, `type ComparisonPair`, `type Outcome` from `@/lib/comparison/pairing`.
- Produces:
  - `ComparisonReport({ pair, outcome, onRestart })` where `pair: ComparisonPair`, `outcome: Outcome`, `onRestart: () => void`.
  - `AiComparison` now renders the `"report"` phase using `decideOutcome(phase.votes)` and defines `restart()` which returns to setup.

- [ ] **Step 1: Write the failing test**

Append to `web/tests/e2e/ai-comparison.spec.ts`:

```ts
import AxeBuilder from "@axe-core/playwright";

test.describe("AI Comparison report", () => {
  test("reveals both models and highlights a winner after one trial", async ({
    page,
  }) => {
    await page.goto("/ai-comparisons");
    await page.getByRole("button", { name: "Start comparing" }).click();

    await page.getByLabel(/Your question for both models/i).fill("A question");
    await page.getByRole("button", { name: "Ask both models" }).click();
    await expect(page.getByText(/Both answers are ready/i)).toBeVisible({
      timeout: 15_000,
    });
    await page.getByRole("button", { name: "Prefer the left answer" }).click();

    // Now the models are revealed.
    const result = page.getByRole("region", { name: "Comparison result" });
    await expect(result).toBeVisible();
    // Exactly one winner marker for a single-vote trial (no tie).
    await expect(result.getByText("Winner", { exact: true })).toHaveCount(1);
    // Both vote tallies are shown.
    await expect(result.getByText(/vote/)).toHaveCount(2);

    // The report is accessible.
    const axe = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"])
      .analyze();
    expect(axe.violations).toEqual([]);

    // Restarting returns to setup.
    await page.getByRole("button", { name: "Run another comparison" }).click();
    await expect(
      page.getByRole("radio", { name: /Surprise me/i }),
    ).toBeVisible();
  });

  test("declares a tie when votes are split across two trials", async ({
    page,
  }) => {
    await page.goto("/ai-comparisons");
    await page.getByLabel(/How many questions/i).fill("2");
    await page.getByRole("button", { name: "Start comparing" }).click();

    // Sides alternate every trial (D4), so voting the same screen side both
    // times sends one vote to each model: a guaranteed tie.
    for (let trial = 0; trial < 2; trial++) {
      await page
        .getByLabel(/Your question for both models/i)
        .fill(`Question ${trial + 1}`);
      await page.getByRole("button", { name: "Ask both models" }).click();
      await expect(page.getByText(/Both answers are ready/i)).toBeVisible({
        timeout: 15_000,
      });
      await page
        .getByRole("button", { name: "Prefer the left answer" })
        .click();
    }

    const result = page.getByRole("region", { name: "Comparison result" });
    await expect(result.getByRole("heading", { name: "It is a tie" })).toBeVisible();
    await expect(result.getByText("Winner", { exact: true })).toHaveCount(0);
  });
});
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd web && npm run test:e2e -- ai-comparison.spec.ts`
Expected: FAIL. The report phase still renders `null`, so no `region` named "Comparison result" appears.

- [ ] **Step 3: Implement the report and wire the report phase**

Create `web/components/comparison/ComparisonReport.tsx`:

```tsx
"use client";

import { getModelDisplayName } from "@/lib/config/models";
import type { ComparisonPair, Outcome } from "@/lib/comparison/pairing";

export function ComparisonReport({
  pair,
  outcome,
  onRestart,
}: {
  pair: ComparisonPair;
  outcome: Outcome;
  onRestart: () => void;
}) {
  const names = [getModelDisplayName(pair[0]), getModelDisplayName(pair[1])];
  const votes = [outcome.votesSlot0, outcome.votesSlot1];
  const tie = outcome.winner === null;
  const heading = tie ? "It is a tie" : `${names[outcome.winner as 0 | 1]} won`;

  return (
    <section aria-label="Comparison result" className="flex flex-col gap-4">
      <p className="ribbon">Result</p>
      <h2 className="text-2xl">{heading}</h2>

      <ul className="flex flex-col gap-3">
        {[0, 1].map((slot) => {
          const isWinner = outcome.winner === slot;
          return (
            <li
              key={slot}
              className={
                isWinner
                  ? "rounded-card border-2 border-miami-red bg-light-tan p-4"
                  : "rounded-card border border-medium-tan bg-paper p-4"
              }
            >
              <p className="font-bold">
                {names[slot]}
                {isWinner ? (
                  <span className="ml-2 text-miami-red">Winner</span>
                ) : null}
              </p>
              <p className="text-sm text-dark-tan">
                {votes[slot]} {votes[slot] === 1 ? "vote" : "votes"}
              </p>
            </li>
          );
        })}
      </ul>

      {tie ? (
        <p>Both models received the same number of votes.</p>
      ) : null}

      <button
        type="button"
        onClick={onRestart}
        className="self-start rounded-card border border-medium-tan bg-paper px-4 py-2 font-bold hover:border-miami-red hover:text-miami-red"
      >
        Run another comparison
      </button>
    </section>
  );
}
```

Modify `web/components/comparison/AiComparison.tsx`. Add imports for `ComparisonReport` and `decideOutcome`, replace the final `return null` with the report render, and add a `restart` handler. The relevant changes:

Add to the imports:

```tsx
import { ComparisonReport } from "./ComparisonReport";
import { pickPair, decideOutcome, type ComparisonPair } from "@/lib/comparison/pairing";
```

(Replace the existing `import { pickPair, type ComparisonPair } ...` line with the line above so `decideOutcome` is included.)

Replace the closing `// Report phase is implemented in Task 7.` and `return null;` with:

```tsx
  return (
    <ComparisonReport
      pair={phase.pair}
      outcome={decideOutcome(phase.votes)}
      onRestart={() => setPhase({ name: "setup" })}
    />
  );
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd web && npm run test:e2e -- ai-comparison.spec.ts`
Expected: PASS. The single-trial run reveals a winner with exactly one "Winner" marker and two vote tallies, passes axe, and restarts. The two-trial same-side run declares a tie with no winner marker.

- [ ] **Step 5: Run the full suite and verify (no commit)**

Run: `cd web && npx vitest run && npm run typecheck && npm run lint`
Expected: all unit tests pass, typecheck exits 0, lint clean. Then optionally run the full e2e suite to confirm no regression in the shared chat route or model tests:

Run: `cd web && npm run test:e2e -- ai-comparison.spec.ts chat.spec.ts`
Expected: PASS for both. Do not commit; leave the working tree uncommitted.

---

## Self-Review

**1. Spec coverage** (each spec bullet mapped to a task):

- Two models side by side, left and right, same prompt: Task 5 (`ComparisonTrial`, two `useChat`, two panes).
- Anonymous mode with a time-based seed picking two available models, blind: Task 1 (`pickPair` seeded), Task 3/5 (`AiComparison` seeds from `Date.now()`, panes blind), decision D3.
- Pick-models mode: Task 3 (`ComparisonSetup` pick mode, two `ModelChooser`s).
- n trials, default 1, max 5, one prompt at a time: Task 2 (`DEFAULT_TRIALS`, `MAX_TRIALS`), Task 3 (trial-count input with clamp), Task 6 (advance one prompt per trial).
- Vote left or right: Task 5 (vote buttons, `resolveVote`).
- Report reveals both models, highlights the winner, shows each model's vote count: Task 7 (`ComparisonReport`).
- Tie handling: Task 1 (`decideOutcome` returns `winner: null`), Task 7 (tie heading and copy, tie e2e).
- Models hidden until the report: Task 5 (blind pane headings, "no Result during trial" assertion), Task 7 (reveal only in the report region), decision D4.
- Randomize left/right per trial: Task 1 (`leftSlotForTrial` alternation), decision D4.
- Reuse model catalog and `ai_comparisons` key: Task 2, Task 3 (`getPageModels`, `buildModelOptions`, `filterAvailableModels`).
- Reuse streaming and error handling; two concurrent calls: Task 5 (two `useChat` on the existing `/api/chat`), decisions D1 and D2.
- Ephemeral, no server persistence, content-free usage events only: relies on the unchanged `/api/chat` (no new persistence) plus the page's `module_open` event; called out in Global Constraints.
- WCAG 2.1 AA: native controls throughout, `role="status"`/`role="alert"`, winner marked by text not colour, axe check in Task 7.
- Miami brand tokens, no em dashes, no secrets in the client: Global Constraints; enforced in Task 2 (prompt em-dash assertion) and by sending model ids only.
- Non-goals (no leaderboard, no cross-session persistence): honoured by never persisting comparison outcomes; nothing in the plan adds a store.

No uncovered spec requirements found.

**2. Placeholder scan:** No "TBD", "add error handling", or "write tests for the above" without code. The `return null` branches in `AiComparison` during Tasks 3 and 5 are deliberate, staged scaffolding that later tasks replace, and each is called out as such with the completing task named; the final file has no `return null` report branch.

**3. Type consistency:** `ComparisonPair`, `Outcome`, and the slot type `0 | 1` are defined in Task 1 and used unchanged in Tasks 5 and 7. `SetupMode` is defined in Task 3 and consumed in Task 5's `start` config. The `onStart` config shape (`{ mode, trials, leftPick, rightPick }`) matches between `ComparisonSetup` (Task 3) and `AiComparison.start` (Task 3, unchanged in later tasks). `onVote: (slot: 0 | 1) => void` matches between `ComparisonTrial` (Task 5) and `AiComparison.vote` (Task 5). Function names are stable across tasks (`pickPair`, `leftSlotForTrial`, `resolveVote`, `decideOutcome`, `getModelDisplayName`).

**One risk to watch during execution:** the AI SDK `useChat`/`UIMessage` types in this project are pinned to `@ai-sdk/react` ^4 and `ai` ^7. The `assistantText` helper reads `message.parts` exactly as `components/chat/Chat.tsx` already does, so it should compile against the same types. If the `useChat` return type differs (for example `status` values), align with `components/chat/Chat.tsx`, which is the working reference for this exact API in this repo.

---

## Execution Handoff

Plan complete and saved to `docs/development/2026-07-23-ai-comparison-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration. REQUIRED SUB-SKILL: superpowers:subagent-driven-development.

**2. Inline Execution** - Execute tasks in this session with checkpoints for review. REQUIRED SUB-SKILL: superpowers:executing-plans.

Reminder for either path: this project makes NO git commits. Each task ends by running its tests and confirming they pass, leaving the working tree uncommitted.

Which approach?
