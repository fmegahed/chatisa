import pino from "pino";

/**
 * Structured server-side logger. Redacts anything secret-shaped; prompt and
 * response bodies must never be passed to it (log lengths and ids instead).
 */
export const logger = pino({
  level: process.env.LOG_LEVEL ?? "info",
  redact: {
    paths: [
      "*.authorization",
      "*.cookie",
      "*.apiKey",
      "*.api_key",
      "*.token",
      "*.client_secret",
      "*.email",
      "req.headers.authorization",
      "req.headers.cookie",
    ],
    censor: "[redacted]",
  },
});
