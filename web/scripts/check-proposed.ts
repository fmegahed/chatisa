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

const PROPOSED: Proposed[] = [
  { id: "gpt-5.6-sol", provider: "openai", inputPerM: 5, outputPerM: 30 },
  { id: "gpt-5.6-terra", provider: "openai", inputPerM: 2.5, outputPerM: 15 },
  { id: "gpt-5.6-luna", provider: "openai", inputPerM: 1, outputPerM: 6 },
  { id: "claude-sonnet-5", provider: "anthropic", inputPerM: 3, outputPerM: 15 },
  { id: "claude-opus-5", provider: "anthropic", inputPerM: 5, outputPerM: 25 },
  { id: "zai-org/GLM-5.2:fireworks-ai", provider: "huggingface", inputPerM: 1.4, outputPerM: 4.4 },
  { id: "thinkingmachines/Inkling:together", provider: "huggingface", inputPerM: 1.0, outputPerM: 4.05 },
  { id: "deepseek-ai/DeepSeek-V4-Pro:fireworks-ai", provider: "huggingface", inputPerM: 1.74, outputPerM: 3.48 },
  { id: "deepseek-ai/DeepSeek-V4-Flash:fireworks-ai", provider: "huggingface", inputPerM: 0.14, outputPerM: 0.28 },
  { id: "prism-ml/Ternary-Bonsai-27B-gguf:together", provider: "huggingface", inputPerM: 0, outputPerM: 0 },
  { id: "google/gemma-4-31B-it:cerebras", provider: "huggingface", inputPerM: 0, outputPerM: 0 },
  { id: "moonshotai/Kimi-K2.7-Code:fireworks-ai", provider: "huggingface", inputPerM: 0.95, outputPerM: 4.0 },
  { id: "Qwen/Qwen3.6-35B-A3B:scaleway", provider: "huggingface", inputPerM: 0.29, outputPerM: 1.71 },
  { id: "openai/gpt-oss-120b:cerebras", provider: "huggingface", inputPerM: 0.25, outputPerM: 0.69 },
  { id: "openai/gpt-oss-20b:groq", provider: "huggingface", inputPerM: 0.1, outputPerM: 0.5 },
  { id: "microsoft/phi-4:deepinfra", provider: "huggingface", inputPerM: 0.07, outputPerM: 0.14 },
  { id: "gemini-3.1-pro-preview-customtools", provider: "google", inputPerM: 2, outputPerM: 12 },
  { id: "gemini-3.6-flash", provider: "google", inputPerM: 1.5, outputPerM: 7.5 },
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
    const providers = hfModels.get(baseId);
    if (!providers) {
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
