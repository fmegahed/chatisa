import { z } from "zod";

/**
 * Server-side environment validation. Values are never logged or echoed;
 * every message names the variable, never its content.
 *
 * Feature keys are optional at boot (the app runs with a reduced model
 * catalog when a provider key is absent, mirroring the legacy behavior),
 * but each API route that needs a key re-asserts it and fails closed.
 */
/**
 * Treats a blank or whitespace-only value as "not set". `.env` templates ship
 * with empty placeholders, so a half-filled file must degrade gracefully
 * rather than stop the server.
 */
function optional<T extends z.ZodTypeAny>(schema: T) {
  return z.preprocess(
    (value) =>
      typeof value === "string" && value.trim() === "" ? undefined : value,
    schema.optional(),
  );
}

const serverEnvSchema = z.object({
  NODE_ENV: z
    .enum(["development", "test", "production"])
    .default("development"),

  /** Providers: optional at boot; required per-feature at call time. */
  OPENAI_API_KEY: optional(z.string().min(1)),
  ANTHROPIC_API_KEY: optional(z.string().min(1)),
  GOOGLE_API_KEY: optional(z.string().min(1)),
  HF_TOKEN: optional(z.string().min(1)),
  /** Deepgram speech-to-text and text-to-speech. Server only, never sent to a browser. */
  DEEPGRAM_TOKEN: optional(z.string().min(1)),

  /**
   * Ask Anything's literature sources. Not model providers, so they are not in
   * PROVIDER_KEYS: absent keys degrade search quality rather than hiding models.
   * Semantic Scholar's keyless pool is shared across all anonymous callers and
   * answers 429 to nearly every request, so without this key that source is
   * effectively always unavailable. OpenAlex asks for a contact address to put
   * callers in its faster, more reliable pool.
   */
  SEMANTIC_SCHOLAR_API_KEY: optional(z.string().min(1)),
  OPENALEX_MAILTO: optional(z.string().min(3)),

  /**
   * Job Scout's harvest sources (design 2026-07-28). Not model providers,
   * so not in PROVIDER_KEYS: absent keys leave the weekly feed empty (the
   * module says so honestly) rather than hiding models or blocking boot.
   * USAJobs requires the registered account email in the User-Agent header.
   */
  RAPIDAPI_KEY: optional(z.string().min(1)),
  USAJOBS_API_KEY: optional(z.string().min(1)),
  USAJOBS_EMAIL: optional(z.string().min(3)),
  /** Comma-separated emails allowed to trigger a manual harvest. */
  CHATISA_SCOUT_ADMINS: optional(z.string().min(3)),
  /** Per-run tagging cost ceiling in USD; defaults to 10 in lib/scout/tag.ts. */
  CHATISA_SCOUT_MAX_RUN_USD: optional(z.string().min(1)),

  /**
   * Job Scout's GitHub auto-push (v6.3.0). A registered GitHub OAuth App;
   * the server uses these ONLY to exchange the OAuth code for a token, which
   * is then handed to the browser and never stored server-side. Absent pair:
   * the Connect/Push/Publish buttons are hidden and students keep the manual
   * zip-and-CLI flow.
   */
  GITHUB_OAUTH_CLIENT_ID: optional(z.string().min(1)),
  GITHUB_OAUTH_CLIENT_SECRET: optional(z.string().min(1)),

  /** Auth (required in production). */
  AUTH_SECRET: optional(z.string().min(32)),
  AUTH_GOOGLE_ID: optional(z.string().min(1)),
  AUTH_GOOGLE_SECRET: optional(z.string().min(1)),
  AUTH_URL: optional(z.url()),
  /** Test-only Credentials provider toggle. NEVER "1" in production. */
  AUTH_TEST_MODE: optional(z.enum(["0", "1"])),

  /** HTTPS production wrapper (Slice 10). */
  SSL_CERT_FILE_PATH: optional(z.string().min(1)),
  SSL_KEY_FILE_PATH: optional(z.string().min(1)),
});

const productionRules = serverEnvSchema.superRefine((env, ctx) => {
  if (env.NODE_ENV !== "production") return;
  for (const key of [
    "AUTH_SECRET",
    "AUTH_GOOGLE_ID",
    "AUTH_GOOGLE_SECRET",
    "AUTH_URL",
  ] as const) {
    if (!env[key]) {
      ctx.addIssue({
        code: "custom",
        path: [key],
        message: "required in production",
      });
    }
  }
  if (env.AUTH_TEST_MODE === "1") {
    ctx.addIssue({
      code: "custom",
      path: ["AUTH_TEST_MODE"],
      message: "must not be enabled in production",
    });
  }
});

export type ServerEnv = z.infer<typeof serverEnvSchema>;

const PROVIDER_KEYS = [
  "OPENAI_API_KEY",
  "ANTHROPIC_API_KEY",
  "GOOGLE_API_KEY",
  "HF_TOKEN",
  "DEEPGRAM_TOKEN",
] as const;

export interface EnvReport {
  ok: boolean;
  /** Variables that failed validation (names only). */
  invalid: string[];
  /** Optional provider keys that are absent (names only). */
  missingProviders: string[];
  /** True when all auth variables are present. */
  authConfigured: boolean;
}

export function validateEnv(source: NodeJS.ProcessEnv = process.env): {
  env: ServerEnv | null;
  report: EnvReport;
} {
  const parsed = productionRules.safeParse(source);
  if (!parsed.success) {
    const invalid = [
      ...new Set(parsed.error.issues.map((i) => String(i.path[0] ?? "?"))),
    ];
    return {
      env: null,
      report: { ok: false, invalid, missingProviders: [], authConfigured: false },
    };
  }
  const missingProviders = PROVIDER_KEYS.filter((k) => !parsed.data[k]);
  const authConfigured = Boolean(
    parsed.data.AUTH_SECRET &&
      parsed.data.AUTH_GOOGLE_ID &&
      parsed.data.AUTH_GOOGLE_SECRET,
  );
  return {
    env: parsed.data,
    report: { ok: true, invalid: [], missingProviders, authConfigured },
  };
}

let cached: ServerEnv | null = null;

/** Boot-time assertion: throws (fail-fast) on malformed configuration. */
export function assertBootEnv(): EnvReport {
  const { env, report } = validateEnv();
  if (!env) {
    throw new Error(
      `Invalid server configuration for: ${report.invalid.join(", ")}. ` +
        "Fix these environment variables and restart. Values are never printed.",
    );
  }
  cached = env;
  return report;
}

export function serverEnv(): ServerEnv {
  if (!cached) {
    const { env } = validateEnv();
    if (!env) throw new Error("Server environment is invalid.");
    cached = env;
  }
  return cached;
}
