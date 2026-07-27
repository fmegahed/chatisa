/**
 * Checks every id in our catalog against what each provider actually serves,
 * and pulls context/output limits where the provider exposes them.
 * Read-only. Prints model identifiers and limits only, never credentials.
 */
import { config as loadEnv } from "dotenv";
loadEnv({ path: ".env.local" });
loadEnv({ path: ".env" });

import { MODELS } from "../lib/config/models";

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

async function main() {
  const served = new Map<string, Set<string>>();

  const openai = await getJson("https://api.openai.com/v1/models", {
    authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
  });
  served.set("openai", new Set(openai.data.map((m: { id: string }) => m.id)));

  const anthropicModels = await getAllAnthropicModels(process.env.ANTHROPIC_API_KEY!);
  served.set("anthropic", new Set(anthropicModels.map((m) => m.id)));

  const google = { models: await getAllGoogleModels(process.env.GOOGLE_API_KEY!) };
  const googleLimits = new Map<string, string>();
  served.set(
    "google",
    new Set(
      (google.models ?? []).map((raw) => {
          const m = raw as {
            name: string;
            inputTokenLimit?: number;
            outputTokenLimit?: number;
          };
          const id = m.name.replace(/^models\//, "");
          googleLimits.set(id, `ctx ${m.inputTokenLimit}, out ${m.outputTokenLimit}`);
          return id;
        }),
    ),
  );

  const cohere = await getJson(
    "https://api.cohere.com/v1/models?page_size=100&endpoint=chat",
    { authorization: `Bearer ${process.env.COHERE_API_KEY}` },
  );
  const cohereLimits = new Map<string, string>();
  served.set(
    "cohere",
    new Set(
      (cohere.models ?? []).map(
        (m: { name: string; context_length?: number }) => {
          cohereLimits.set(m.name, `ctx ${m.context_length}`);
          return m.name;
        },
      ),
    ),
  );

  const groq = await getJson("https://api.groq.com/openai/v1/models", {
    authorization: `Bearer ${process.env.GROQ_API_KEY}`,
  });
  const groqLimits = new Map<string, string>();
  served.set(
    "meta (via Groq)",
    new Set(
      groq.data.map(
        (m: { id: string; context_window?: number; max_completion_tokens?: number }) => {
          groqLimits.set(
            m.id,
            `ctx ${m.context_window}, out ${m.max_completion_tokens}`,
          );
          return m.id;
        },
      ),
    ),
  );

  const hf = await getJson("https://router.huggingface.co/v1/models", {
    authorization: `Bearer ${process.env.HF_TOKEN}`,
  });
  served.set(
    "huggingface_inference",
    new Set(hf.data.map((m: { id: string }) => m.id)),
  );

  console.log("CATALOG AUDIT: is each id we ship actually served today?\n");
  let dead = 0;
  for (const [id, cfg] of Object.entries(MODELS)) {
    const set = served.get(cfg.provider);
    const ok = set?.has(id);
    if (!ok) dead++;
    const limits =
      googleLimits.get(id) ?? cohereLimits.get(id) ?? groqLimits.get(id) ?? "";
    console.log(
      `${ok ? "  served " : "  DEAD   "} ${id.padEnd(48)} ${cfg.provider}${limits ? "  [" + limits + "]" : ""}`,
    );
    if (!ok && set) {
      // Near-miss suggestions catch the case where only a suffix changed.
      const stem = id.split("/").pop()!.slice(0, 18).toLowerCase();
      const near = [...set].filter((s) => s.toLowerCase().includes(stem));
      if (near.length) console.log(`            served instead: ${near.join(", ")}`);
    }
  }
  console.log(`\n${dead} of ${Object.keys(MODELS).length} catalog ids are not served.`);

  console.log("\n\nGOOGLE limits for current chat candidates:");
  for (const [id, l] of googleLimits) {
    if (/^gemini-3(\.\d)?-(pro|flash)/.test(id)) console.log(`  ${id.padEnd(40)} ${l}`);
  }
  console.log("\nCOHERE limits:");
  for (const [id, l] of cohereLimits) {
    if (/^command-a/.test(id)) console.log(`  ${id.padEnd(40)} ${l}`);
  }
  console.log("\nGROQ limits:");
  for (const [id, l] of groqLimits) console.log(`  ${id.padEnd(40)} ${l}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
