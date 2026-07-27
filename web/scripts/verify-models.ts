/**
 * Live catalog verification. Calls every chat model in the catalog for real and
 * reports which ones actually work.
 *
 * This exists because a model can sit in the catalog, pass every test, appear
 * in the student's dropdown, and still be impossible to call. That happened on
 * 2026-07-21: Llama-4-Maverick was offered to students while the router served
 * no provider for it. Unit tests cannot catch that, because the fault is in the
 * provider's fleet rather than in our code. Only a real call can.
 *
 * Two things are checked per model, because both are load-bearing:
 *   1. Streaming chat, which every module depends on.
 *   2. Structured output, which Exam Ally depends on. A model that chats but
 *      cannot emit conforming JSON breaks exam generation only, so checking
 *      chat alone would give false confidence.
 *
 * Not checked: vision. Verifying it needs a fixture image per call and costs
 * meaningfully more, so `supportsVision` in the catalog stays a claim this
 * script does not confirm. It is reported as unverified rather than passed.
 *
 * Usage:  npm run verify:models            (whole catalog)
 *         npm run verify:models -- gpt-5   (substring filter)
 *
 * Costs real money, though very little: roughly 30 tokens per model.
 * Exits non-zero if any reachable model fails, so a model update can be gated
 * on it.
 */
import { config as loadEnv } from "dotenv";
// Next reads .env.local automatically; a standalone script does not, and
// silently running with no keys would report the whole catalog as "no key".
loadEnv({ path: ".env.local" });
loadEnv({ path: ".env" });

import { generateObject, streamText } from "ai";
import { z } from "zod";
import { MODELS, calculateCost } from "../lib/config/models";
import { getLanguageModel, isModelAvailable, PROVIDER_ENV_KEY } from "../lib/providers";
import { classifyProviderFailure } from "../lib/providers/errors";

type Outcome = "pass" | "fail" | "skipped" | "unproven";

/**
 * Providers go busy. An "Overloaded" 529 from Anthropic says nothing about
 * whether a model works, and reporting it as a failure would condemn a healthy
 * model on the strength of one bad second. Transient failures are retried, and
 * if they persist they are reported as `unproven` rather than `fail`: not
 * verified, but not evidence of a broken model either.
 */
const RETRIES = 3;

async function withRetry<T>(label: string, fn: () => Promise<T>): Promise<T> {
  let last: unknown;
  for (let attempt = 1; attempt <= RETRIES; attempt++) {
    try {
      return await fn();
    } catch (err) {
      last = err;
      const failure = classifyProviderFailure(err);
      if (!failure.retryable || attempt === RETRIES) throw err;
      process.stdout.write(`(${label} retry ${attempt}) `);
      await new Promise((r) => setTimeout(r, 1500 * attempt));
    }
  }
  throw last;
}

type Result = {
  modelId: string;
  provider: string;
  chat: Outcome;
  structured: Outcome;
  latencyMs: number;
  costUsd: number;
  note: string;
};

const PROBE_SCHEMA = z.object({
  color: z.string().describe("any color name"),
  count: z.number().describe("any whole number from 1 to 5"),
});

async function checkChat(modelId: string) {
  const started = Date.now();
  const result = streamText({
    model: getLanguageModel(modelId),
    instructions: "You are a test probe. Answer in one short word.",
    messages: [{ role: "user", content: "Say ok." }],
    // Generous on purpose. An earlier 16-token cap produced 14 false failures:
    // reasoning models spend their output budget on hidden reasoning tokens
    // and emit no visible text before hitting the ceiling, which looked
    // identical to a dead model. A probe that cannot tell those apart is worse
    // than no probe, and this still costs a fraction of a cent per model.
    maxOutputTokens: 1024,
  });

  let text = "";
  // Consuming the stream is the point: some providers only fail once the
  // first chunk is requested, which is exactly what a student would hit.
  for await (const chunk of result.textStream) text += chunk;

  const usage = await result.usage;
  return {
    latencyMs: Date.now() - started,
    text: text.trim(),
    inputTokens: usage.inputTokens ?? 0,
    outputTokens: usage.outputTokens ?? 0,
  };
}

async function checkStructured(modelId: string) {
  const result = await generateObject({
    model: getLanguageModel(modelId),
    schema: PROBE_SCHEMA,
    prompt: "Pick a color and a count.",
  });
  return {
    inputTokens: result.usage.inputTokens ?? 0,
    outputTokens: result.usage.outputTokens ?? 0,
    object: result.object,
  };
}

/** calculateCost also returns an error shape for unknown ids, which cannot
 * happen here because we iterate the catalog itself. */
function costOf(modelId: string, input: number, output: number): number {
  const breakdown = calculateCost(modelId, input, output);
  return "error" in breakdown ? 0 : breakdown.totalCost;
}

async function verify(modelId: string): Promise<Result> {
  const config = MODELS[modelId];
  const base: Result = {
    modelId,
    provider: config.provider,
    chat: "skipped",
    structured: "skipped",
    latencyMs: 0,
    costUsd: 0,
    note: "",
  };

  if (config.realtimeOnly) {
    return { ...base, note: "realtime only, not a chat model" };
  }
  if (!isModelAvailable(modelId)) {
    return {
      ...base,
      note: `no key: set ${PROVIDER_ENV_KEY[config.provider]}`,
    };
  }

  let inputTokens = 0;
  let outputTokens = 0;

  try {
    const chat = await withRetry("chat", () => checkChat(modelId));
    base.latencyMs = chat.latencyMs;
    inputTokens += chat.inputTokens;
    outputTokens += chat.outputTokens;
    base.chat = chat.text.length > 0 ? "pass" : "fail";
    if (chat.text.length === 0) base.note = "streamed an empty response";
  } catch (err) {
    const failure = classifyProviderFailure(err);
    base.chat = failure.retryable ? "unproven" : "fail";
    // The raw provider text is what an operator needs here, and this output
    // goes to an operator's terminal rather than to a student.
    base.note = `${failure.kind}: ${err instanceof Error ? err.message : String(err)}`;
    base.costUsd = costOf(modelId, inputTokens, outputTokens);
    return base;
  }

  if (!config.supportsStructuredOutput) {
    base.note ||= "route has no structured output (chat only, hidden from Exam Ally)";
  } else {
    try {
      const structured = await withRetry("struct", () => checkStructured(modelId));
      inputTokens += structured.inputTokens;
      outputTokens += structured.outputTokens;
      base.structured = "pass";
    } catch (err) {
      const failure = classifyProviderFailure(err);
      base.structured = failure.retryable ? "unproven" : "fail";
      base.note ||= failure.retryable
        ? `structured output could not be checked, the provider stayed busy: ${
            err instanceof Error ? err.message : String(err)
          }`
        : `structured output failed, Exam Ally will not work with this model: ${
            err instanceof Error ? err.message : String(err)
          }`;
    }
  }

  base.costUsd = costOf(modelId, inputTokens, outputTokens);
  return base;
}

function mark(outcome: Outcome): string {
  if (outcome === "pass") return "pass";
  if (outcome === "fail") return "FAIL";
  if (outcome === "unproven") return "busy";
  return "-";
}

async function main() {
  const filter = process.argv.slice(2).find((a) => !a.startsWith("-"));
  const modelIds = Object.keys(MODELS).filter(
    (id) => !filter || id.toLowerCase().includes(filter.toLowerCase()),
  );

  if (modelIds.length === 0) {
    console.error(`No catalog model matches "${filter}".`);
    process.exit(2);
  }

  console.log(`Verifying ${modelIds.length} model(s) against live providers.\n`);

  const results: Result[] = [];
  // Sequential on purpose: parallel probes trip per-provider rate limits and
  // turn a clean report into noise that looks like real failures.
  for (const modelId of modelIds) {
    process.stdout.write(`  ${modelId} ... `);
    const result = await verify(modelId);
    results.push(result);
    console.log(
      result.chat === "skipped"
        ? "skipped"
        : `chat ${mark(result.chat)}, structured ${mark(result.structured)}, ${result.latencyMs}ms`,
    );
  }

  const width = Math.max(...results.map((r) => r.modelId.length));
  console.log("\n" + "-".repeat(width + 46));
  console.log(
    `${"model".padEnd(width)}  ${"chat".padEnd(6)}${"struct".padEnd(8)}${"latency".padEnd(10)}cost`,
  );
  console.log("-".repeat(width + 46));
  for (const r of results) {
    console.log(
      `${r.modelId.padEnd(width)}  ${mark(r.chat).padEnd(6)}${mark(r.structured).padEnd(8)}${
        r.latencyMs ? `${r.latencyMs}ms`.padEnd(10) : "".padEnd(10)
      }$${r.costUsd.toFixed(5)}`,
    );
    if (r.note) console.log(`${" ".repeat(width + 2)}  ${r.note}`);
  }
  console.log("-".repeat(width + 46));

  const failed = results.filter((r) => r.chat === "fail" || r.structured === "fail");
  const unproven = results.filter(
    (r) => r.chat === "unproven" || r.structured === "unproven",
  );
  const skipped = results.filter((r) => r.chat === "skipped");
  const totalCost = results.reduce((sum, r) => sum + r.costUsd, 0);
  const visionClaims = results.filter(
    (r) => r.chat === "pass" && MODELS[r.modelId].supportsVision,
  );

  console.log(`\nTotal cost of this run: $${totalCost.toFixed(4)}`);
  console.log(
    `${results.length - failed.length - skipped.length} passed, ${failed.length} failed, ${skipped.length} skipped.`,
  );
  if (visionClaims.length > 0) {
    console.log(
      `Not verified by this script: ${visionClaims.length} model(s) claim vision support in the catalog.`,
    );
  }
  if (unproven.length > 0) {
    console.log(
      `${unproven.length} model(s) could not be checked because the provider stayed busy. ` +
        `That is not a failure and not a pass: re-run to settle it.`,
    );
    console.log(`  ${unproven.map((r) => r.modelId).join(", ")}`);
  }
  if (skipped.length > 0) {
    console.log(
      "Skipped models were not tested at all. That is not a pass; set the missing keys to check them.",
    );
  }

  if (failed.length > 0) {
    console.log(`\nFailing: ${failed.map((r) => r.modelId).join(", ")}`);
    console.log("These are in the catalog, so students can currently pick them.");
  }
  // Unproven exits non-zero too. "Could not be checked" must never read as
  // "fine" to a gate: a model that fails every retry looks identical to a
  // healthy one that happened to be busy, and only one of those is safe.
  if (failed.length > 0 || unproven.length > 0) {
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
