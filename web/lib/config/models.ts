/**
 * Model catalog.
 *
 * Every id here was verified against the provider's own live listing on
 * 2026-11-16 with `npm run models:discover`, and the HuggingFace routes were
 * verified per serving provider with `scripts/check-proposed.ts`. That process
 * exists because the previous catalog shipped
 * `meta-llama/Llama-4-Maverick-17B-128E-Instruct` for months while the router
 * only ever served the `-FP8` suffixed id, so students were offered a model
 * that could never answer. No unit test can catch that; only asking the
 * provider can.
 *
 * HuggingFace ids carry an explicit `:provider` route suffix. The route is
 * part of the identity, not a detail: providers serving identical weights
 * differ in price, speed, context, and crucially in whether they support
 * structured output.
 *
 * Replacing any id requires explicit approval (ADR-005), and the replacement
 * must be verified live before it ships (ADR-018).
 */

export type ProviderId =
  | "openai"
  | "anthropic"
  | "google"
  | "huggingface_inference";

export interface ModelConfig {
  provider: ProviderId;
  displayName: string;
  description: string;
  costPer1kInput: number;
  costPer1kOutput: number;
  maxTokens: number;
  contextWindow: number;
  supportsVision: boolean;
  supportsFunctionCalling: boolean;
  /**
   * Whether this route can return schema-conforming JSON. Distinct from
   * `supportsFunctionCalling`: on the HuggingFace router the same weights
   * support tools on one provider and structured output only on another.
   * Exam Ally cannot generate an exam without this, so it filters on it.
   */
  supportsStructuredOutput: boolean;
  /**
   * True where a limit is inferred rather than published by the provider.
   * Inferred limits are deliberately conservative, so the cost of being wrong
   * is a model being offered for slightly smaller documents than it could
   * actually handle, never a failed request.
   */
  limitsInferred?: boolean;
  temperatureRange: [number, number];
  defaultTemperature: number;
  /**
   * Whether the provider still accepts a `temperature` parameter for this model.
   * Absent means yes, so only models that reject it need saying.
   *
   * Added 2026-07-26 after Ask Anything failed outright on Claude Opus 5:
   * `AI_APICallError: temperature is deprecated for this model`. Every request
   * carrying a temperature was rejected, so the model was offered in five
   * modules and worked in none of them. Claude Sonnet 5 is the same generation
   * and only WARNS ("temperature is not supported ... and will be ignored"),
   * which is why this went unnoticed: the default model degraded silently while
   * the premium one broke loudly, and nothing exercised the premium one.
   *
   * temperatureRange is left as published rather than zeroed, because it
   * describes what the model's sampling would do; this flag describes whether we
   * are allowed to ask.
   */
  supportsTemperature?: boolean;
  openWeight: boolean;
  realtimeOnly?: boolean;
  tags: string[];
}

export const MODELS: Record<string, ModelConfig> = {
  // Replaced GPT-6 Sol on 2026-11-16 (v6.10.1); GPT-6 Sol had replaced GPT-5.6
  // Sol in v6.5.0. Same published standard rate, $2/$10 per million, so the
  // six module defaults that were on GPT-6 Sol move here unchanged in cost.
  // OpenAI bills prompts over 272K input tokens at $4/$15; like every tier
  // threshold in this catalog, that is not modelled, so cost shown for a very
  // long prompt is an underestimate. GPT-6 Astra ($10/$50) is still not
  // offered, on cost (professor's decision, v6.5.0).
  // https://developers.openai.com/api/docs/models/gpt-6.1-sol
  "gpt-6.1-sol": {
    provider: "openai",
    displayName: "GPT-6.1 Sol",
    description: "OpenAI's strong general purpose model. Best for hard reasoning and detailed feedback.",
    costPer1kInput: 0.002,
    costPer1kOutput: 0.01,
    maxTokens: 128000,
    contextWindow: 1050000,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    // GPT-6.x are reasoning models: @ai-sdk/openai drops temperature for them
    // with a warning ("temperature is not supported for reasoning models"),
    // measured 2026-11-16 on all three. Recording it here stops sending a
    // setting that has no effect; what the provider receives is unchanged.
    supportsTemperature: false,
    openWeight: false,
    tags: ["premium", "reasoning", "coding", "large_context", "vision", "sandbox"],
  },
  // Added 2026-11-16 (v6.10.1) on the professor's direction. GPT-6 shipped
  // without a Terra tier (GPT-5.6 Terra was retired in v6.5.0); 6.1 brings it
  // back. $1/$5 per million ($2/$7.50 over 272K input tokens, not modelled),
  // between Luna and Sol. Not a module default: Sol keeps those.
  // https://developers.openai.com/api/docs/models/gpt-6.1-terra
  "gpt-6.1-terra": {
    provider: "openai",
    displayName: "GPT-6.1 Terra",
    description: "OpenAI's middle tier. Quicker and cheaper than Sol, stronger than Luna.",
    costPer1kInput: 0.001,
    costPer1kOutput: 0.005,
    maxTokens: 128000,
    contextWindow: 1050000,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    // Dropped by the SDK for reasoning models: see gpt-6.1-sol.
    supportsTemperature: false,
    openWeight: false,
    tags: ["reasoning", "coding", "large_context", "vision"],
  },
  // Replaced GPT-5.6 Luna on 2026-09-22 (v6.5.0). $0.10/$0.50 per million,
  // OpenAI's published standard rate.
  // https://developers.openai.com/api/docs/models/gpt-6-luna
  "gpt-6-luna": {
    provider: "openai",
    displayName: "GPT-6 Luna",
    description: "OpenAI's fast, low cost model. A good default for everyday questions.",
    costPer1kInput: 0.0001,
    costPer1kOutput: 0.0005,
    maxTokens: 128000,
    contextWindow: 1050000,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    // Dropped by the SDK for reasoning models: see gpt-6.1-sol (v6.10.1).
    supportsTemperature: false,
    openWeight: false,
    tags: ["cost_effective", "coding", "large_context", "vision"],
  },
  // Replaced Opus 5 on 2026-09-22 (v6.5.0), the day Opus 5.5 landed, per the
  // professor's direction: $4/$20 per million against $5/$25, and Anthropic
  // reports roughly Fable 5.1 level performance.
  // https://platform.claude.com/docs/en/models/opus-5-5/overview
  // Same tokenizer as Opus 4.7 onward (about 30 percent more tokens than
  // pre-4.7 Claude models). Breaking changes from Opus 5 that matter here:
  // thinking is always on, and forced tool use returns an error, so any
  // structured-output path that forces a tool call would fail on this model
  // only. verify:models exercises structured output on it before it ships.
  //
  // Kept from the 4.8 entry because the lesson outlives the model: 4.8 was
  // briefly withheld on 2026-07-21 after returning HTTP 500 for every request
  // carrying a system prompt. That was an Anthropic incident, not a defect.
  // The failure reproduced 5/5, which felt conclusive, but an active incident
  // reproduces 100 percent of the time inside its own window; only re-testing
  // after a delay can tell a defect from an outage (ADR-019).
  "claude-opus-5-5": {
    provider: "anthropic",
    displayName: "Claude Opus 5.5",
    description: "Anthropic's most capable model. Strong at careful explanation and code review.",
    costPer1kInput: 0.004,
    costPer1kOutput: 0.02,
    maxTokens: 128000,
    contextWindow: 1000000,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    // Rejected by the provider: see supportsTemperature.
    supportsTemperature: false,
    openWeight: false,
    tags: ["premium", "reasoning", "coding", "large_context", "vision"],
  },
  // Replaced Claude Sonnet 5 on 2026-11-16 (v6.10.1); Sonnet 5.5 shipped
  // 2026-09-28 at Sonnet 5's price, $2/$10 per million, so the three module
  // defaults on Sonnet 5 move here unchanged in cost. Like Opus 5.5 it rejects
  // sampling parameters (see supportsTemperature) and forced tool use, so
  // structured output needs an @ai-sdk/anthropic that knows this id and uses
  // native structured output for it: 4.0.60 did not and would have fallen back
  // to a forced tool call; 4.0.105 does (upgraded in the same release).
  // https://platform.claude.com/docs/en/models/sonnet-5-5/overview
  "claude-sonnet-5-5": {
    provider: "anthropic",
    displayName: "Claude Sonnet 5.5",
    description: "Balanced Anthropic model. Fast, capable, and well suited to tutoring.",
    costPer1kInput: 0.002,
    costPer1kOutput: 0.01,
    maxTokens: 128000,
    contextWindow: 1000000,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    // Rejected by the provider: see supportsTemperature.
    supportsTemperature: false,
    openWeight: false,
    tags: ["premium", "reasoning", "coding", "large_context", "vision"],
  },
  // Added 2026-11-16 (v6.10.1) on the professor's direction. Priced by prompt
  // length: $0.10/$0.50 per million for prompts up to 100,000 tokens and
  // $0.50/$2.50 above. The catalog has one rate per model, so the lower one is
  // recorded: a chat almost never sends 100K tokens, but a long PDF in Ask
  // Anything can, and its cost is then understated fivefold. Rejects sampling
  // parameters like the other 5.5 models. Supports code execution and PDF
  // input (read from Anthropic's Models API), which Ask Anything requires.
  // https://platform.claude.com/docs/en/models/haiku-5-5/overview
  "claude-haiku-5-5": {
    provider: "anthropic",
    displayName: "Claude Haiku 5.5",
    description: "Anthropic's fastest, lowest cost model. Good for quick questions.",
    costPer1kInput: 0.0001,
    costPer1kOutput: 0.0005,
    maxTokens: 128000,
    contextWindow: 1000000,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    // Rejected by the provider: see supportsTemperature.
    supportsTemperature: false,
    openWeight: false,
    tags: ["cost_effective", "coding", "large_context", "vision"],
  },
  // Gemini 4 replaced both Gemini entries on 2026-11-16 (v6.10.1), per the
  // professor's direction: Gemini 3.1 Pro (still a preview a year on) and
  // Gemini 3.8 Flash. Limits read from Google's model listing; prices from
  // Google's pricing page. https://ai.google.dev/gemini-api/docs/pricing
  //
  // Carbon is Google's "most capable reasoning model". Its $2/$10 per million
  // is promotional through 2026-11-30; the standard $4/$20 from 2026-12-01 is
  // recorded, because the promotion ends two weeks after this release.
  // Output is capped at 32,768 tokens, half of what Gemini 3.1 Pro allowed.
  "gemini-4-carbon": {
    provider: "google",
    displayName: "Gemini 4 Carbon",
    description: "Google's most capable reasoning model, with a very large context window.",
    costPer1kInput: 0.004,
    costPer1kOutput: 0.02,
    maxTokens: 32768,
    contextWindow: 1048576,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: false,
    tags: ["premium", "reasoning", "coding", "large_context", "vision"],
  },
  // Argon: "efficient frontier reasoning". Same per-token price as Carbon,
  // including the promotion ending 2026-11-30 ($4/$20 recorded), but Google
  // positions it as spending fewer thinking tokens per answer.
  "gemini-4-argon": {
    provider: "google",
    displayName: "Gemini 4 Argon",
    description: "Google's efficient reasoning model. Built to spend less thinking per answer than Carbon.",
    costPer1kInput: 0.004,
    costPer1kOutput: 0.02,
    maxTokens: 32768,
    contextWindow: 1048576,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: false,
    tags: ["premium", "reasoning", "coding", "large_context", "vision"],
  },
  // Hydrogen: Google's fast tier ("everyday coding and agentic workflows"),
  // taking Gemini 3.8 Flash's place, including as the Job Scout weekly tagger
  // (lib/scout/tag.ts). $1.50/$7.50 per million, no promotion: twice 3.8
  // Flash's promotional price and equal to its price from 2027-01-01.
  "gemini-4-hydrogen": {
    provider: "google",
    displayName: "Gemini 4 Hydrogen",
    description: "Google's fast model. Large context at a lower price than Carbon and Argon.",
    costPer1kInput: 0.0015,
    costPer1kOutput: 0.0075,
    maxTokens: 65536,
    contextWindow: 1048576,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: false,
    tags: ["coding", "large_context", "vision"],
  },
  // Replaced GLM-5.2:deepinfra on 2026-09-22 (v6.5.0). The measurement rule
  // decided the route again: for 5.2, baseten failed a two-field schema six
  // times in a row; for 5.3 it is deepinfra, the cheapest route ($1.2/$4),
  // that passed only 2 of 6 identical structured-output probes ("response did
  // not match schema"), while novita and baseten both passed 6 of 6. novita
  // wins on a typical answer (76 tokens per second with a 1.7s first token,
  // against 60 at 1.1s on baseten); both charge $1.4/$4.4 per million. together
  // is fastest (137) but has no structured output.
  // Repriced 2026-11-16 (v6.10.1): novita cut the route to $0.63/$1.98 per
  // million, so the $1.4/$4.4 recorded since v6.5.0 had overstated cost.
  "zai-org/GLM-5.3:novita": {
    provider: "huggingface_inference",
    displayName: "GLM-5.3",
    description: "Open weight flagship from Z.ai. Strong general reasoning.",
    costPer1kInput: 0.00063,
    costPer1kOutput: 0.00198,
    maxTokens: 8192,
    contextWindow: 1048576,
    supportsVision: false,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "reasoning", "coding", "large_context"],
  },
  // Re-routed from together on 2026-11-16 (v6.10.1): together stopped offering
  // structured output for these weights, so the model failed in every module
  // that needs it (Exam Prep, JobApp, Job Scout, Portfolio Builder). deepinfra
  // is the only other route: 4 of 4 schema probes passed, vision probe
  // answered correctly, 96 tokens per second, $0.95/$4.05 per million.
  "thinkingmachines/Inkling:deepinfra": {
    provider: "huggingface_inference",
    displayName: "Inkling",
    description: "Open weight multimodal model from Thinking Machines.",
    costPer1kInput: 0.00095,
    costPer1kOutput: 0.00405,
    maxTokens: 8192,
    contextWindow: 524288,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "reasoning", "large_context", "vision"],
  },
  // Replaced DeepSeek-V4-Pro:together on 2026-09-22 (v6.5.0): together stopped
  // serving the original V4 Pro weights, so the pinned route was dead, and
  // DeepSeek has since published the 0813 snapshot. baseten is the fastest
  // priced route with structured output (54 tokens per second; deepinfra 36,
  // fireworks 43 and unpriced) at $1.32/$3.96 per million. Its first token is
  // slow (about 2.7s), which is the trade for keeping it in Exam Ally; together
  // is faster (58) but has no structured output. The router now publishes a
  // 1,048,576 context for every route, up from the 512,000 recorded before.
  // Re-routed from baseten on 2026-11-16 (v6.10.1): baseten stopped offering
  // structured output for this snapshot. Of six routes measured, five passed
  // 4 of 4 schema probes; deepinfra gave the shortest typical answer (4.0s,
  // 0.36s first token) and is the cheapest of them at $1.30/$2.60 per million.
  "deepseek-ai/DeepSeek-V4-Pro-0813:deepinfra": {
    provider: "huggingface_inference",
    displayName: "DeepSeek V4 Pro",
    description: "Open weight reasoning model from DeepSeek.",
    costPer1kInput: 0.0013,
    costPer1kOutput: 0.0026,
    maxTokens: 8192,
    contextWindow: 1048576,
    supportsVision: false,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "reasoning", "coding", "large_context"],
  },
  // Replaced DeepSeek-V4-Flash:deepinfra on 2026-09-22 (v6.5.0) with V4.1.
  // The old route had slowed to about 20 tokens per second, which no longer
  // earned the "very fast" description. baseten is the fastest priced route
  // with structured output (108 tokens per second; novita 116 without
  // structured output, deepinfra 31). The price rises from $0.09/$0.18 to
  // $0.30/$1.20 per million; deepinfra is cheaper ($0.2/$0.6) but a third of
  // the speed.
  "deepseek-ai/DeepSeek-V4.1-Flash:baseten": {
    provider: "huggingface_inference",
    displayName: "DeepSeek V4.1 Flash",
    description: "Very cheap, very fast open weight model. Good for quick questions.",
    costPer1kInput: 0.0003,
    costPer1kOutput: 0.0012,
    maxTokens: 8192,
    contextWindow: 1048576,
    supportsVision: false,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "cost_effective", "large_context"],
  },
  // Added 2026-08-20 (v6.3.0). Routed via together: fireworks has a faster
  // first token (0.15s against 0.57s) but lower throughput (63 against 80
  // tokens per second), publishes no price, and has no structured output.
  // Vision verified live on this route on 2026-08-20.
  "meta-models/Muse-Glimmer-30B:together": {
    provider: "huggingface_inference",
    displayName: "Muse Glimmer 30B",
    description: "Small open weight multimodal model. Fast, cheap, and reads images.",
    costPer1kInput: 0.00035,
    costPer1kOutput: 0.0015,
    maxTokens: 8192,
    contextWindow: 131072,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "cost_effective", "vision"],
  },
  // Kimi K3 (v6.3.0, on baseten) replaced on 2026-11-16 (v6.10.1) by Kimi K3.1
  // Pro (released 2026-11-02), per the professor's direction: baseten had
  // stopped offering structured output for K3, and K3.1 Pro is cheaper.
  // inference-net chosen of five routes measured: 4 of 4 schema probes, vision
  // probe answered correctly, the fastest first token (0.26s, which the Sandbox
  // inline completion depends on), 82 tokens per second, and the lowest price
  // of the reliable routes at $1.80/$9 per million. digitalocean streams faster
  // (151) but takes about a second to start.
  "moonshotai/Kimi-K3.1-Pro:inference-net": {
    provider: "huggingface_inference",
    displayName: "Kimi K3.1 Pro",
    description: "Open weight flagship from Moonshot. Strong at coding and reads images.",
    costPer1kInput: 0.0018,
    costPer1kOutput: 0.009,
    maxTokens: 8192,
    contextWindow: 1048576,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "reasoning", "coding", "large_context", "vision"],
  },
  // Added 2026-08-20 (v6.3.0) on fireworks-ai. Re-routed 2026-09-22 (v6.5.0)
  // because fireworks stopped serving these weights. Of the remaining routes,
  // deepinfra is the fastest with structured output (47 tokens per second
  // against 32 on novita) at $2/$6 per million; together is faster (137) but
  // has no structured output. Text only: the card is text-generation, no
  // vision encoder.
  // Chat only since 2026-11-16 (v6.10.1). On the app's structured-output path
  // (plain JSON mode, schema described rather than enforced) the model now
  // returns malformed keys: '{":color": "teal", ":count": 12}', '{"": ""}'.
  // 0 of 3 direct attempts and 2 of 3 verify:models runs failed, so it was
  // failing for students in Exam Prep, JobApp, Job Scout and Portfolio
  // Builder. The router still advertises structured output for the route.
  "Qwen/Qwen3.8-2.4T-A95B:deepinfra": {
    provider: "huggingface_inference",
    displayName: "Qwen3.8 2.4T",
    description: "Alibaba's largest open weight model. Rivals the commercial flagships.",
    costPer1kInput: 0.002,
    costPer1kOutput: 0.006,
    maxTokens: 8192,
    contextWindow: 262144,
    supportsVision: false,
    supportsFunctionCalling: true,
    supportsStructuredOutput: false,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "reasoning", "coding", "large_context"],
  },
  // Added 2026-08-20 (v6.3.0) on the professor's suggestion, NVFP4 build on
  // together. Moved 2026-09-22 (v6.5.0) to the full-precision BF16 weights on
  // deepinfra, per the professor's direction: together stopped serving the
  // NVFP4 build, and its only remaining route (fireworks-ai) rejects our
  // HuggingFace account ("Pay-as-you go is not enabled for provider
  // fireworks-ai yet"), which the account settings do not offer to enable.
  // deepinfra is the only BF16 route: 47 tokens per second, $1/$5 per million,
  // 262,144 context. No structured output (the router says so, and 1 of 6
  // live schema probes passed), so it is chat only and hidden from Exam Ally.
  // Text only.
  // 2026-11-16 (v6.10.1): deepinfra now advertises structured output and 4 of
  // 4 live schema probes passed (77 tokens per second), so the model returns
  // to Exam Prep, JobApp, Job Scout and Portfolio Builder. Repriced to the
  // route's $0.50/$2.20 per million.
  "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:deepinfra": {
    provider: "huggingface_inference",
    displayName: "Nemotron 3 Ultra",
    description: "NVIDIA's large open weight reasoning model.",
    costPer1kInput: 0.0005,
    costPer1kOutput: 0.0022,
    maxTokens: 8192,
    contextWindow: 262144,
    supportsVision: false,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "reasoning", "coding", "large_context"],
  },
  // Replaced Qwen3.6-35B on 2026-08-20 (v6.3.0) per the professor's direction,
  // then on featherless-ai, the only route at the time (no tools, no
  // structured output, unpriced, 10-17s to a short answer). Re-routed
  // 2026-09-22 (v6.5.0) because featherless stopped serving it. ovhcloud is the
  // fastest of the three new routes (67 tokens per second against 29 on novita
  // and 19 on deepinfra), publishes a price ($0.47/$3.19 per million), and
  // advertises tools and structured output, which this model previously
  // lacked; both claims are verified live before shipping (ADR-018). Vision is
  // re-probed on this route. ovhcloud publishes its context (262,144), so the
  // limit is no longer inferred. Not tagged "reasoning", as before.
  "Qwen/Qwen3.8-27B:ovhcloud": {
    provider: "huggingface_inference",
    displayName: "Qwen3.8 27B",
    description: "Compact open weight model from Alibaba that also reads images.",
    costPer1kInput: 0.00047,
    costPer1kOutput: 0.00319,
    maxTokens: 8192,
    contextWindow: 262144,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "cost_effective", "vision"],
  },
  // cerebras kept for speed (about 1096 tokens per second against 45 to 226 on
  // every other route). As of 2026-08-20 the route publishes its price
  // ($0.35/$0.75 per million, up from $0.25/$0.69), publishes its context, and
  // advertises structured output, so this model is no longer hidden from exam
  // generation; the claim was verified live before shipping (ADR-018).
  "openai/gpt-oss-120b:cerebras": {
    provider: "huggingface_inference",
    displayName: "GPT-OSS 120B",
    description: "OpenAI's open weight model, served extremely fast.",
    costPer1kInput: 0.00035,
    costPer1kOutput: 0.00075,
    maxTokens: 8192,
    contextWindow: 131072,
    supportsVision: false,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "reasoning", "coding"],
  },
  // groq kept for speed (about 732 tokens per second). The route advertises
  // structured output, but a live probe on 2026-08-20 was rejected unless the
  // prompt itself contains the word "json" (a groq-side validation rule), which
  // real exam prompts cannot be relied on to satisfy. The measurement wins over
  // the advertised capability, so this stays chat-only.
  // Repriced 2026-11-16 (v6.10.1) to the route's $0.075/$0.30 per million.
  "openai/gpt-oss-20b:groq": {
    provider: "huggingface_inference",
    displayName: "GPT-OSS 20B",
    description: "Small open weight model from OpenAI, served extremely fast.",
    costPer1kInput: 0.000075,
    costPer1kOutput: 0.0003,
    maxTokens: 8192,
    contextWindow: 131072,
    supportsVision: false,
    supportsFunctionCalling: true,
    supportsStructuredOutput: false,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "cost_effective", "coding"],
  },
  // Ternary Bonsai 27B (the catalog's only free model) was removed on
  // 2026-11-16 (v6.10.1): no provider on the router serves its weights any more.
  //
  // The next four were added 2026-11-16 (v6.10.1) on the professor's direction.
  // Each route was chosen by measurement on the router: a ~150-word answer for
  // speed, four probes against a two-field schema, and a 64x64 solid-colour
  // image for vision. One sample per route, so the speeds are indicative.
  //
  // Meta's large open model (released 2026-09-15). together chosen of three
  // measured routes: 4 of 4 schema probes, vision correct, and by far the
  // fastest (0.20s first token, 263 tokens per second). Every route charges
  // $1.25/$4.25 per million.
  "meta-models/Muse-Spark-Open-816B-A42B:together": {
    provider: "huggingface_inference",
    displayName: "Muse Spark Open",
    description: "Meta's large open weight model. Very fast, and reads images.",
    costPer1kInput: 0.00125,
    costPer1kOutput: 0.00425,
    maxTokens: 8192,
    contextWindow: 1048576,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "reasoning", "coding", "large_context", "vision"],
  },
  // Mistral's flagship (released 2026-09-30); Mistral's own route is the only
  // one, $0.70/$2.10 per million. It reasons before answering: a one-line
  // question used 466 to 1,147 output tokens and 7 to 16 seconds. Four schema
  // probes capped at 400 output tokens all failed because the reasoning used
  // the whole budget before any answer; with a realistic budget 6 of 6 passed,
  // and 3 of 3 on the app's own path (plain JSON mode, schema described), so
  // structured output is claimed. Vision probe answered correctly.
  "mistralai/Mistral-Large-4-1T-A52B:mistral": {
    provider: "huggingface_inference",
    displayName: "Mistral Large 4",
    description: "Mistral's open weight flagship. Thinks before answering, so it is slower.",
    costPer1kInput: 0.0007,
    costPer1kOutput: 0.0021,
    maxTokens: 8192,
    contextWindow: 1048576,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "reasoning", "coding", "large_context", "vision"],
  },
  // NVIDIA's 120B (12B active) vision-language model (released 2026-10-12).
  // deepinfra chosen of five measured routes: 4 of 4 schema probes (venice,
  // primeintellect and together each dropped one or two), vision correct, and
  // the shortest typical answer of the reliable routes (2.7s at 406 tokens
  // per second; coreweave also passed 4 of 4 but took 3.4s). $0.15/$0.50 per
  // million, 262,144 context.
  "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16:deepinfra": {
    provider: "huggingface_inference",
    displayName: "Nemotron 3.5 Super VL",
    description: "NVIDIA's fast, inexpensive open weight model that also reads images.",
    costPer1kInput: 0.00015,
    costPer1kOutput: 0.0005,
    maxTokens: 8192,
    contextWindow: 262144,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: true,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "cost_effective", "vision"],
  },
  // MiniMax's small fast model (released 2026-10-17). wafer chosen of five
  // measured routes: 154 tokens per second and the lowest price ($0.135/$0.54
  // per million). digitalocean was faster (301) but answered the vision probe
  // with nothing, as did parasail; wafer and together named the colour
  // correctly. inferact-vllm was rate limited throughout.
  //
  // Chat only. The route passed 4 of 4 probes with the schema enforced
  // (json_schema), but the app's HuggingFace provider asks for plain JSON mode
  // with the schema described rather than enforced, and on that path the
  // model wrapped its answer in keys of its own ("thought", "analysis") in 8
  // of 9 verify:models attempts and 2 of 3 direct ones. The measurement on
  // the path students use wins, so it is hidden from the modules that need
  // structured output.
  "MiniMaxAI/MiniMax-M3.1-Flash:wafer": {
    provider: "huggingface_inference",
    displayName: "MiniMax M3.1 Flash",
    description: "Small, fast open weight model from MiniMax. Very cheap and reads images.",
    costPer1kInput: 0.000135,
    costPer1kOutput: 0.00054,
    maxTokens: 8192,
    contextWindow: 1048576,
    supportsVision: true,
    supportsFunctionCalling: true,
    supportsStructuredOutput: false,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "cost_effective", "large_context", "vision"],
  },
  // Context is only 16,384 tokens, far smaller than everything else here, so it is unsuitable for long documents. Reports no tool support.
  // The router advertises structured output on this route, but a live probe on
  // 2026-07-21 returned an object that did not match a two-field schema. The
  // measurement wins over the advertised capability, so it is recorded as
  // false. Excluded from Exam Ally by context anyway.
  "microsoft/phi-4:deepinfra": {
    provider: "huggingface_inference",
    displayName: "Phi-4",
    description: "Small, inexpensive open weight model from Microsoft.",
    costPer1kInput: 7e-05,
    costPer1kOutput: 0.00014,
    maxTokens: 4096,
    contextWindow: 16384,
    supportsVision: false,
    supportsFunctionCalling: false,
    supportsStructuredOutput: false,
    temperatureRange: [0.0, 2.0],
    defaultTemperature: 0.7,
    openWeight: true,
    tags: ["open_weight", "cost_effective"],
  },
};
export type ModuleKey =
  | "coding_companion"
  | "project_coach"
  | "exam_ally"
  | "interview_mentor"
  | "interview_mentor_transcription"
  | "jobapp_assistant"
  | "ai_sandbox"
  | "sandbox_chat"
  | "ai_comparisons"
  | "ask_anything"
  | "job_scout"
  | "portfolio";

export const DEFAULT_MODELS: Partial<Record<ModuleKey, string>> = {
  coding_companion: "claude-sonnet-5-5",
  // The Sandbox side chat is coding help, so it mirrors the Coding Companion.
  sandbox_chat: "claude-sonnet-5-5",
  // Ask Anything: strong agentic tool use at mid cost (design 2026-07-24).
  ask_anything: "claude-sonnet-5-5",
  project_coach: "gpt-6.1-sol",
  exam_ally: "gpt-6.1-sol",
  // Speech is moving to Deepgram; the previous default,
  // gpt-4o-realtime-preview-2025-06-03, was withdrawn by OpenAI and is no
  // longer served, so nothing here can point at it.
  interview_mentor_transcription: "gpt-6.1-sol",
  jobapp_assistant: "gpt-6.1-sol",
  // Job Scout's student-facing generation (resume skills, project
  // scaffolds); mirrors JobApp. The weekly tagging pipeline pins its own
  // model in lib/scout/tag.ts and does not read this.
  job_scout: "gpt-6.1-sol",
  // Portfolio Builder: structured content for a published site; mirrors Job Scout.
  portfolio: "gpt-6.1-sol",
};

interface PageModelRule {
  includeAll?: boolean;
  excludeTags?: string[];
  tags?: string[];
  includeRealtime?: boolean;
  minContextWindow?: number;
  specificModels?: string[];
  /**
   * Restrict to models that can return schema-conforming JSON. Exam Ally sets
   * this: without structured output it cannot build an exam at all, and
   * offering a model that always fails is worse than offering fewer models.
   */
  requireStructuredOutput?: boolean;
}

const PAGE_MODELS: Record<string, PageModelRule> = {
  coding_companion: { includeAll: true, excludeTags: ["realtime", "speech"] },
  project_coach: {
    tags: ["reasoning", "large_context"],
    minContextWindow: 64000,
  },
  // Exam generation needs structured output, and the document must fit, so the
  // 16k-context model is excluded by minContextWindow rather than by name.
  exam_ally: {
    includeAll: true,
    excludeTags: ["realtime", "speech"],
    requireStructuredOutput: true,
    minContextWindow: 64000,
  },
  // Speech runs through Deepgram, not through a chat model, so the interviewer
  // model is an ordinary chat model. Both keys resolve to the same set: the
  // "realtime", "speech" and "transcription" tags they used to filter on are
  // carried by no model in the current catalog, so these resolved to an empty
  // list and to the five premium models respectively.
  interview_mentor: { includeAll: true, minContextWindow: 64000 },
  // Tailoring needs structured output and enough room for a whole resume.
  jobapp_assistant: {
    includeAll: true,
    requireStructuredOutput: true,
    minContextWindow: 64000,
  },
  interview_mentor_transcription: { includeAll: true, minContextWindow: 64000 },
  // Scaffold generation needs structured output (a file manifest), and gap
  // evidence plus a README needs room; same shape as jobapp_assistant.
  job_scout: {
    includeAll: true,
    requireStructuredOutput: true,
    minContextWindow: 64000,
  },
  portfolio: { includeAll: true, requireStructuredOutput: true, minContextWindow: 64000 },
  ai_sandbox: { specificModels: ["gpt-6.1-sol"] },
  // The Sandbox side chat offers the same models as the Coding Companion.
  sandbox_chat: { includeAll: true, excludeTags: ["realtime", "speech"] },
  ai_comparisons: { includeAll: true, excludeTags: ["realtime", "speech"] },
  // Ask Anything: Anthropic + OpenAI only (professor's decision 2026-07-24,
  // slice C revision). Both providers accept the same native file parts
  // (images, PDFs) and have hosted code execution, so any chat can switch
  // between all roster models mid-conversation, attachments included. Gemini
  // and Kimi remain available in AI Comparison.
  // v6.10.1 adds GPT-6.1 Terra and Claude Haiku 5.5: both providers list
  // hosted code execution and image input for them (Anthropic's Models API
  // also reports PDF input for Haiku 5.5), which the roster rule requires.
  ask_anything: {
    specificModels: [
      "gpt-6.1-sol",
      "gpt-6.1-terra",
      "gpt-6-luna",
      "claude-opus-5-5",
      "claude-sonnet-5-5",
      "claude-haiku-5-5",
    ],
  },
};


export const MODEL_CATEGORIES: Record<
  string,
  { displayName: string; description: string; models: string[] }
> = {
  commercial_api: {
    displayName: "Commercial APIs",
    description: "Hosted models from the large AI labs",
    models: [
      "gpt-6.1-sol",
      "gpt-6.1-terra",
      "gpt-6-luna",
      "claude-opus-5-5",
      "claude-sonnet-5-5",
      "claude-haiku-5-5",
      "gemini-4-carbon",
      "gemini-4-argon",
      "gemini-4-hydrogen",
    ],
  },
  open_weight_large: {
    displayName: "Open weight, large",
    description:
      "Openly published models that rival the commercial labs. Worth trying to see how far open alternatives have come.",
    models: [
      "zai-org/GLM-5.3:novita",
      "deepseek-ai/DeepSeek-V4-Pro-0813:deepinfra",
      "moonshotai/Kimi-K3.1-Pro:inference-net",
      "meta-models/Muse-Spark-Open-816B-A42B:together",
      "mistralai/Mistral-Large-4-1T-A52B:mistral",
      "Qwen/Qwen3.8-2.4T-A95B:deepinfra",
      "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16:deepinfra",
      "thinkingmachines/Inkling:deepinfra",
      "openai/gpt-oss-120b:cerebras",
    ],
  },
  open_weight_small: {
    displayName: "Open weight, small and fast",
    description:
      "Smaller open models. Cheap, quick to respond, and a good way to see the trade-off against the large models.",
    models: [
      "deepseek-ai/DeepSeek-V4.1-Flash:baseten",
      "meta-models/Muse-Glimmer-30B:together",
      "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16:deepinfra",
      "MiniMaxAI/MiniMax-M3.1-Flash:wafer",
      "Qwen/Qwen3.8-27B:ovhcloud",
      "openai/gpt-oss-20b:groq",
      "microsoft/phi-4:deepinfra",
    ],
  },
};


/**
 * Models available to a module. Mirrors legacy get_page_models, including its
 * ordering (provider, then display name) and its fallback for unknown pages.
 */
export function getPageModels(page: string): string[] {
  const rule = PAGE_MODELS[page];
  if (!rule) {
    return Object.keys(MODELS).filter((m) => !MODELS[m].realtimeOnly);
  }
  if (rule.specificModels) return [...rule.specificModels];

  let available: string[];
  if (rule.includeAll) {
    available = Object.keys(MODELS);
  } else if (rule.tags) {
    available = Object.keys(MODELS).filter((m) =>
      rule.tags!.some((tag) => MODELS[m].tags.includes(tag)),
    );
  } else {
    available = Object.keys(MODELS);
  }

  if (rule.excludeTags) {
    available = available.filter(
      (m) => !rule.excludeTags!.some((tag) => MODELS[m].tags.includes(tag)),
    );
  }
  if (!rule.includeRealtime) {
    available = available.filter((m) => !MODELS[m].realtimeOnly);
  }
  if (rule.minContextWindow !== undefined) {
    available = available.filter(
      (m) => MODELS[m].contextWindow >= rule.minContextWindow!,
    );
  }
  if (rule.requireStructuredOutput) {
    available = available.filter((m) => MODELS[m].supportsStructuredOutput);
  }

  // Legacy sorts by (provider, display_name) using Python's stable sort.
  return available.sort((a, b) => {
    const pa = MODELS[a].provider;
    const pb = MODELS[b].provider;
    if (pa !== pb) return pa < pb ? -1 : 1;
    const da = MODELS[a].displayName;
    const db = MODELS[b].displayName;
    if (da !== db) return da < db ? -1 : 1;
    return 0;
  });
}

export function getModelsByTag(...tags: string[]): string[] {
  return Object.keys(MODELS).filter((m) =>
    tags.some((tag) => MODELS[m].tags.includes(tag)),
  );
}

export function getDefaultModelForPage(page: string): string {
  const configured = DEFAULT_MODELS[page as ModuleKey];
  const available = getPageModels(page);
  if (configured && available.includes(configured)) return configured;
  return available[0] ?? Object.keys(MODELS)[0];
}

export interface CostBreakdown {
  inputTokens: number;
  outputTokens: number;
  inputCost: number;
  outputCost: number;
  totalCost: number;
  model: string;
  currency: "USD";
}

/**
 * The temperature to send for this model, or undefined when it must be omitted.
 *
 * Every call that sets a temperature goes through here, because the failure it
 * prevents is total: Claude Opus 5 rejects the parameter outright
 * (`AI_APICallError: temperature is deprecated for this model`), so before this
 * existed the model was offered in five modules and answered in none of them.
 * An unknown model id is treated as accepting temperature, which is the
 * pre-existing behaviour.
 *
 * The AI SDK omits an option set to undefined, so callers can pass the result
 * straight through without branching.
 */
export function temperatureFor(
  modelId: string,
  requested: number,
): number | undefined {
  return MODELS[modelId]?.supportsTemperature === false ? undefined : requested;
}

/** Mirrors legacy calculate_cost, including its 6-decimal rounding. */
export function calculateCost(
  model: string,
  inputTokens: number,
  outputTokens: number,
): CostBreakdown | { error: string } {
  const cfg = MODELS[model];
  if (!cfg) return { error: `Unknown model: ${model}` };
  const inputCost = (inputTokens / 1000) * cfg.costPer1kInput;
  const outputCost = (outputTokens / 1000) * cfg.costPer1kOutput;
  const round6 = (n: number) => Number(n.toFixed(6));
  return {
    inputTokens,
    outputTokens,
    inputCost: round6(inputCost),
    outputCost: round6(outputCost),
    totalCost: round6(inputCost + outputCost),
    model,
    currency: "USD",
  };
}

export interface ModelOption {
  id: string;
  /** Product name only. Badges are derived, never baked into this. */
  name: string;
  /** One student-facing sentence. Already written for every model. */
  description: string;
  /** MODEL_CATEGORIES key this model belongs to. */
  groupId: string;
  groupName: string;
  /** Short factual labels, rendered as text rather than colour. */
  badges: string[];
  recommended: boolean;
}

/** Which category a model sits in. Categories are a total, disjoint partition
 * of the catalog, asserted by tests, so this always resolves. */
export function categoryOf(modelId: string): string | null {
  for (const [key, category] of Object.entries(MODEL_CATEGORIES)) {
    if (category.models.includes(modelId)) return key;
  }
  return null;
}

export function badgesFor(modelId: string): string[] {
  const cfg = MODELS[modelId];
  if (!cfg) return [];
  return [
    cfg.openWeight ? "open weight" : null,
    cfg.costPer1kInput === 0 ? "free" : null,
    cfg.supportsVision ? "reads images" : null,
  ].filter((b): b is string => b !== null);
}

/**
 * Everything a picker needs for one module, in category order.
 *
 * Exists because three pages each built this inline and drifted: one included a
 * vision badge and two did not, and all three appended badges to a displayName
 * that already ended in "(open weight)", producing
 * "Gemma 4 31B (open weight, free) (open weight, free tier)".
 */
export function buildModelOptions(page: string, availableIds?: string[]): {
  options: ModelOption[];
  defaultModelId: string;
} {
  const allowed = getPageModels(page);
  const usable = availableIds
    ? allowed.filter((id) => availableIds.includes(id))
    : allowed;

  const preferred = getDefaultModelForPage(page);
  const defaultModelId = usable.includes(preferred) ? preferred : (usable[0] ?? "");

  const groupOrder = Object.keys(MODEL_CATEGORIES);
  const options = usable
    .map((id): ModelOption => {
      const groupId = categoryOf(id) ?? groupOrder[0];
      return {
        id,
        name: MODELS[id].displayName,
        description: MODELS[id].description,
        groupId,
        groupName: MODEL_CATEGORIES[groupId]?.displayName ?? "Other",
        badges: badgesFor(id),
        recommended: id === defaultModelId,
      };
    })
    .sort((a, b) => {
      const ga = groupOrder.indexOf(a.groupId);
      const gb = groupOrder.indexOf(b.groupId);
      if (ga !== gb) return ga - gb;
      return a.name < b.name ? -1 : a.name > b.name ? 1 : 0;
    });

  return { options, defaultModelId };
}

export function getModelDisplayName(model: string): string {
  return MODELS[model]?.displayName ?? model;
}
