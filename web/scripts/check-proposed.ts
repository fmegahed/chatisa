/**
 * Verifies a proposed model list against what providers actually serve, before
 * anything is written into the catalog.
 *
 * Checks, per model:
 *   - the id exists at that provider (the Llama-4-Maverick failure was a
 *     one-suffix mismatch that no test could have caught)
 *   - for HuggingFace routes pinned as `model:provider`, that the pinned
 *     provider is live for that model
 *   - the provider's own published price against the price we intend to store,
 *     since cost figures are shown to students
 *   - structured-output support, which Exam Ally requires and which varies
 *     between inference providers serving identical weights
 *
 * Read-only. Prints identifiers, limits and prices only, never credentials.
 */
import { config as loadEnv } from "dotenv";
loadEnv({ path: ".env.local" });
loadEnv({ path: ".env" });

/** Prices are per 1M tokens, matching how providers publish them. */
type Proposed = {
  id: string;
  provider: "openai" | "anthropic" | "google" | "huggingface";
  inputPerM: number;
  outputPerM: number;
};

// v6.10.1 list (2026-11-16). Prices are the providers' published rates that
// day; where a route publishes none through the router the stored figure
// comes from the provider's own docs or is zero, and the price check simply
// has nothing to compare against.
const PROPOSED: Proposed[] = [
  { id: "gpt-6.1-sol", provider: "openai", inputPerM: 2, outputPerM: 10 },
  { id: "gpt-6.1-terra", provider: "openai", inputPerM: 1, outputPerM: 5 },
  { id: "gpt-6-luna", provider: "openai", inputPerM: 0.1, outputPerM: 0.5 },
  { id: "claude-sonnet-5-5", provider: "anthropic", inputPerM: 2, outputPerM: 10 },
  // Up to 100K-token prompts; $0.5/$2.5 above.
  { id: "claude-haiku-5-5", provider: "anthropic", inputPerM: 0.1, outputPerM: 0.5 },
  { id: "claude-opus-5-5", provider: "anthropic", inputPerM: 4, outputPerM: 20 },
  { id: "zai-org/GLM-5.3:novita", provider: "huggingface", inputPerM: 0.63, outputPerM: 1.98 },
  { id: "thinkingmachines/Inkling:deepinfra", provider: "huggingface", inputPerM: 0.95, outputPerM: 4.05 },
  { id: "deepseek-ai/DeepSeek-V4-Pro-0813:deepinfra", provider: "huggingface", inputPerM: 1.3, outputPerM: 2.6 },
  { id: "deepseek-ai/DeepSeek-V4.1-Flash:baseten", provider: "huggingface", inputPerM: 0.3, outputPerM: 1.2 },
  { id: "moonshotai/Kimi-K3.1-Pro:inference-net", provider: "huggingface", inputPerM: 1.8, outputPerM: 9 },
  { id: "meta-models/Muse-Spark-Open-816B-A42B:together", provider: "huggingface", inputPerM: 1.25, outputPerM: 4.25 },
  { id: "mistralai/Mistral-Large-4-1T-A52B:mistral", provider: "huggingface", inputPerM: 0.7, outputPerM: 2.1 },
  { id: "Qwen/Qwen3.8-2.4T-A95B:deepinfra", provider: "huggingface", inputPerM: 2, outputPerM: 6 },
  { id: "Qwen/Qwen3.8-27B:ovhcloud", provider: "huggingface", inputPerM: 0.47, outputPerM: 3.19 },
  { id: "meta-models/Muse-Glimmer-30B:together", provider: "huggingface", inputPerM: 0.35, outputPerM: 1.5 },
  { id: "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:deepinfra", provider: "huggingface", inputPerM: 0.5, outputPerM: 2.2 },
  { id: "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16:deepinfra", provider: "huggingface", inputPerM: 0.15, outputPerM: 0.5 },
  { id: "MiniMaxAI/MiniMax-M3.1-Flash:wafer", provider: "huggingface", inputPerM: 0.135, outputPerM: 0.54 },
  { id: "openai/gpt-oss-120b:cerebras", provider: "huggingface", inputPerM: 0.35, outputPerM: 0.75 },
  { id: "openai/gpt-oss-20b:groq", provider: "huggingface", inputPerM: 0.075, outputPerM: 0.3 },
  { id: "microsoft/phi-4:deepinfra", provider: "huggingface", inputPerM: 0.07, outputPerM: 0.14 },
  // Promotional $2/$10 through 2026-11-30; the standard $4/$20 is recorded.
  { id: "gemini-4-carbon", provider: "google", inputPerM: 4, outputPerM: 20 },
  { id: "gemini-4-argon", provider: "google", inputPerM: 4, outputPerM: 20 },
  { id: "gemini-4-hydrogen", provider: "google", inputPerM: 1.5, outputPerM: 7.5 },
];

type HfProvider = {
  provider: string;
  status?: string;
  context_length?: number;
  pricing?: { input: number; output: number };
  supports_tools?: boolean;
  supports_structured_output?: boolean;
};

async function getJson(url: string, headers: Record<string, string>) {
  const res = await fetch(url, { headers });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

/**
 * Google and Anthropic both paginate. An earlier version of this script read
 * only the first page and silently under-reported, which made it claim a model
 * was withdrawn when it was merely on page two. A discovery tool that quietly
 * truncates is worse than no tool, so both are now followed to the end.
 */
async function getAllGoogleModels(key: string) {
  const models: Array<Record<string, unknown>> = [];
  let pageToken = "";
  do {
    const url = new URL("https://generativelanguage.googleapis.com/v1beta/models");
    url.searchParams.set("pageSize", "1000");
    if (pageToken) url.searchParams.set("pageToken", pageToken);
    const page = await getJson(url.toString(), { "x-goog-api-key": key });
    models.push(...(page.models ?? []));
    pageToken = page.nextPageToken ?? "";
  } while (pageToken);
  return models;
}

async function getAllAnthropicModels(key: string) {
  const models: Array<{ id: string }> = [];
  let afterId = "";
  for (;;) {
    const url = new URL("https://api.anthropic.com/v1/models");
    url.searchParams.set("limit", "100");
    if (afterId) url.searchParams.set("after_id", afterId);
    const page = await getJson(url.toString(), {
      "x-api-key": key,
      "anthropic-version": "2023-06-01",
    });
    models.push(...page.data);
    if (!page.has_more || !page.last_id) break;
    afterId = page.last_id;
  }
  return models;
}

function priceNote(stated: number, actual: number | undefined, label: string): string {
  if (actual === undefined) return "";
  // Providers report floating point noise, so compare at cent-per-million.
  if (Math.abs(stated - actual) < 0.011) return "";
  return `  PRICE MISMATCH ${label}: you said $${stated}/M, provider says $${actual}/M`;
}

async function main() {
  const problems: string[] = [];

  const openaiIds = new Set<string>(
    (
      await getJson("https://api.openai.com/v1/models", {
        authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
      })
    ).data.map((m: { id: string }) => m.id),
  );

  const anthropicIds = new Set<string>(
    (await getAllAnthropicModels(process.env.ANTHROPIC_API_KEY!)).map((m) => m.id),
  );

  const googleModels = new Map<string, { ctx?: number; out?: number }>();
  for (const m of await getAllGoogleModels(process.env.GOOGLE_API_KEY!)) {
    const raw = m as {
      name: string;
      inputTokenLimit?: number;
      outputTokenLimit?: number;
      supportedGenerationMethods?: string[];
    };
    // Only chat-capable models belong here: a model that cannot do
    // generateContent is not usable by any of our modules.
    if (!(raw.supportedGenerationMethods ?? []).includes("generateContent")) continue;
    googleModels.set(raw.name.replace(/^models\//, ""), {
      ctx: raw.inputTokenLimit,
      out: raw.outputTokenLimit,
    });
  }

  const hfRaw = await getJson("https://router.huggingface.co/v1/models", {
    authorization: `Bearer ${process.env.HF_TOKEN}`,
  });
  const hfModels = new Map<string, HfProvider[]>();
  for (const m of hfRaw.data ?? []) {
    hfModels.set(m.id, m.providers ?? []);
  }

  for (const p of PROPOSED) {
    console.log(`\n${p.id}`);

    if (p.provider === "openai" || p.provider === "anthropic") {
      const set = p.provider === "openai" ? openaiIds : anthropicIds;
      if (set.has(p.id)) {
        console.log(`  served by ${p.provider}`);
        console.log(`  price not published via API; your figures used as given`);
      } else {
        console.log(`  NOT SERVED by ${p.provider}`);
        const near = [...set].filter((s) => s.startsWith(p.id.slice(0, 9)));
        if (near.length) console.log(`  closest served: ${near.join(", ")}`);
        problems.push(`${p.id} is not served by ${p.provider}`);
      }
      continue;
    }

    if (p.provider === "google") {
      const found = googleModels.get(p.id);
      if (found) {
        console.log(`  served by google  ctx ${found.ctx}, out ${found.out}`);
      } else {
        console.log(`  NOT SERVED by google`);
        const near = [...googleModels.keys()].filter((s) =>
          s.startsWith(p.id.split("-").slice(0, 2).join("-")),
        );
        if (near.length) console.log(`  closest served: ${near.join(", ")}`);
        problems.push(`${p.id} is not served by google`);
      }
      continue;
    }

    const [baseId, pinned] = p.id.split(":");
    let providers = hfModels.get(baseId);
    if (!providers) {
      // The bulk /v1/models listing omits some models the router does serve
      // (observed 2026-08-20 with Qwen/Qwen3.8-27B, served only by
      // featherless-ai). The per-model endpoint is authoritative, so ask it
      // before declaring a model unserved.
      try {
        const single = await getJson(
          `https://router.huggingface.co/v1/models/${baseId}`,
          { authorization: `Bearer ${process.env.HF_TOKEN}` },
        );
        providers = single.data?.providers ?? undefined;
        if (providers && providers.length > 0) {
          console.log(`  absent from the bulk listing; per-model endpoint has it`);
        }
      } catch {
        // Fall through to the unserved report.
      }
    }
    if (!providers || providers.length === 0) {
      console.log(`  NOT SERVED: the router does not list ${baseId}`);
      problems.push(`${baseId} is not on the HF router`);
      continue;
    }
    const route = providers.find((r) => r.provider === pinned);
    if (!route) {
      console.log(`  PINNED PROVIDER "${pinned}" DOES NOT SERVE THIS MODEL`);
      console.log(`  available: ${providers.map((r) => r.provider).join(", ")}`);
      problems.push(`${baseId} is not served by ${pinned}`);
      continue;
    }

    console.log(
      `  served by ${pinned}  status ${route.status}  ctx ${route.context_length ?? "unpublished"}`,
    );
    console.log(
      `  tools ${route.supports_tools ? "yes" : "no"}, structured output ${
        route.supports_structured_output ? "yes" : "NO"
      }`,
    );
    if (route.status !== "live") {
      problems.push(`${p.id} status is "${route.status}", not live`);
    }
    if (route.supports_structured_output === false) {
      problems.push(
        `${p.id} does not support structured output, so Exam Ally cannot use it`,
      );
      // Name a route that would work, since the weights are usually fine and
      // only the serving provider differs.
      const alt = providers.filter(
        (r) => r.supports_structured_output && r.status === "live",
      );
      if (alt.length) {
        console.log(
          `  routes that DO support structured output: ${alt
            .map(
              (r) =>
                `${r.provider} ($${r.pricing?.input ?? "?"}/$${r.pricing?.output ?? "?"} per M)`,
            )
            .join(", ")}`,
        );
      }
    }
    const inNote = priceNote(p.inputPerM, route.pricing?.input, "input");
    const outNote = priceNote(p.outputPerM, route.pricing?.output, "output");
    if (inNote) {
      console.log(inNote);
      problems.push(`${p.id} input price differs from the provider's`);
    }
    if (outNote) {
      console.log(outNote);
      problems.push(`${p.id} output price differs from the provider's`);
    }
  }

  console.log("\n" + "=".repeat(72));
  if (problems.length === 0) {
    console.log("All proposed models verified.");
  } else {
    console.log(`${problems.length} problem(s) to resolve before adopting this list:\n`);
    for (const p of problems) console.log(`  - ${p}`);
    process.exitCode = 1;
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(2);
});
