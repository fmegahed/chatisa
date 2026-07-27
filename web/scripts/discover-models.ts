/**
 * Read-only discovery: asks each provider what models it actually serves today.
 *
 * Exists because a catalog drafted from memory goes stale silently, which is
 * how an unservable model reached students. This asks the providers instead.
 *
 * Prints model identifiers only. Credentials are read from the environment and
 * never echoed, logged, or written to any file.
 */
import { config as loadEnv } from "dotenv";
loadEnv({ path: ".env.local" });
loadEnv({ path: ".env" });

type Listing = {
  provider: string;
  envKey: string;
  models: string[];
  error?: string;
};

async function getJson(url: string, headers: Record<string, string>) {
  const res = await fetch(url, { headers });
  if (!res.ok) {
    // Body may echo the key back in some providers' error shapes, so only the
    // status is surfaced.
    throw new Error(`HTTP ${res.status} ${res.statusText}`);
  }
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

async function list(
  provider: string,
  envKey: string,
  fetcher: (key: string) => Promise<string[]>,
): Promise<Listing> {
  const key = process.env[envKey];
  if (!key) return { provider, envKey, models: [], error: "no key set" };
  try {
    const models = await fetcher(key);
    return { provider, envKey, models: models.sort() };
  } catch (err) {
    return {
      provider,
      envKey,
      models: [],
      error: err instanceof Error ? err.message : String(err),
    };
  }
}

async function main() {
  const listings = await Promise.all([
    list("openai", "OPENAI_API_KEY", async (key) => {
      const data = await getJson("https://api.openai.com/v1/models", {
        authorization: `Bearer ${key}`,
      });
      return data.data.map((m: { id: string }) => m.id);
    }),

    list("anthropic", "ANTHROPIC_API_KEY", async (key) =>
      (await getAllAnthropicModels(key)).map((m) => m.id),
    ),

    list("google", "GOOGLE_API_KEY", async (key) =>
      (await getAllGoogleModels(key))
        .filter((m) =>
          (
            (m as { supportedGenerationMethods?: string[] })
              .supportedGenerationMethods ?? []
          ).includes("generateContent"),
        )
        .map((m) => (m as { name: string }).name.replace(/^models\//, "")),
    ),

    list("cohere", "COHERE_API_KEY", async (key) => {
      const data = await getJson(
        "https://api.cohere.com/v1/models?page_size=100&endpoint=chat",
        { authorization: `Bearer ${key}` },
      );
      return (data.models ?? []).map((m: { name: string }) => m.name);
    }),

    list("groq", "GROQ_API_KEY", async (key) => {
      const data = await getJson("https://api.groq.com/openai/v1/models", {
        authorization: `Bearer ${key}`,
      });
      return data.data.map((m: { id: string }) => m.id);
    }),

    list("huggingface router", "HF_TOKEN", async (key) => {
      const data = await getJson("https://router.huggingface.co/v1/models", {
        authorization: `Bearer ${key}`,
      });
      return data.data.map((m: { id: string }) => m.id);
    }),
  ]);

  for (const l of listings) {
    console.log(`\n=== ${l.provider} (${l.envKey}) ===`);
    if (l.error) {
      console.log(`  unavailable: ${l.error}`);
      continue;
    }
    console.log(`  ${l.models.length} models served`);
    for (const m of l.models) console.log(`    ${m}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
