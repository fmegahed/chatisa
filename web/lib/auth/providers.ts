import type { Provider } from "next-auth/providers";
import Google from "next-auth/providers/google";
import Credentials from "next-auth/providers/credentials";
import { evaluateSignIn } from "./domain";
import { evaluateGuestPass } from "./guest";

/**
 * Provider list for Auth.js. The Credentials "test login" exists ONLY when
 * test mode is on (never in production: see isTestModeEnabled). The "guest
 * pass" provider is different in kind: it is MEANT for production, and its
 * scope is bounded by the hashed invite list and its required expiry date
 * (lib/auth/guest.ts), not by NODE_ENV.
 */
export function buildProviders(opts: { testMode: boolean }): Provider[] {
  const providers: Provider[] = [
    Google({
      // Reads AUTH_GOOGLE_ID / AUTH_GOOGLE_SECRET from the environment.
      authorization: {
        params: { hd: "miamioh.edu", prompt: "select_account" },
      },
    }),
    Credentials({
      id: "guest-pass",
      name: "Collaborator invite link",
      credentials: { pass: { label: "Invite pass", type: "password" } },
      authorize(credentials) {
        // Env is read here, per attempt, so rotating passes needs no restart
        // beyond the platform reloading env; a disabled/expired config makes
        // every attempt fail closed.
        const decision = evaluateGuestPass(
          String(credentials?.pass ?? ""),
          process.env,
          Date.now(),
        );
        if (!decision.allowed) return null;
        return { email: decision.email, name: decision.name };
      },
    }),
  ];
  if (opts.testMode) {
    providers.push(
      Credentials({
        id: "test-login",
        name: "Test login (non-production only)",
        credentials: { email: { label: "Email", type: "email" } },
        authorize(credentials) {
          const email = String(credentials?.email ?? "");
          const decision = evaluateSignIn({ email, emailVerified: true });
          if (!decision.allowed) return null;
          return { email: email.toLowerCase(), name: "Test Student" };
        },
      }),
    );
  }
  return providers;
}
