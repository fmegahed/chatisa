import NextAuth from "next-auth";
import { evaluateSignIn, isTestModeEnabled } from "./domain";
import { buildProviders } from "./providers";

/**
 * Auth.js v5 configuration (ADR-004).
 * - Google OAuth with the miamioh.edu policy enforced SERVER-SIDE in the
 *   signIn callback (the `hd` authorization param is advisory only).
 * - JWT sessions in an HTTP-only cookie; no session rows in the DB.
 * - A Credentials "test login" exists ONLY when AUTH_TEST_MODE=1 outside
 *   production, so e2e tests never need real Google credentials.
 */
export const { handlers, auth, signIn, signOut } = NextAuth({
  trustHost: true,
  session: { strategy: "jwt", maxAge: 7 * 24 * 60 * 60 },
  pages: { signIn: "/login", error: "/login" },
  providers: buildProviders({ testMode: isTestModeEnabled(process.env) }),
  callbacks: {
    signIn({ account, profile }) {
      if (account?.provider === "google") {
        const p = profile as
          | { email?: string; email_verified?: boolean; hd?: string }
          | null
          | undefined;
        return evaluateSignIn({
          email: p?.email,
          emailVerified: p?.email_verified === true,
          hostedDomain: p?.hd ?? null,
        }).allowed;
      }
      // test-login and guest-pass: authorize() has already applied each
      // provider's own policy (domain rule, or hashed invite list + expiry).
      return (
        account?.provider === "test-login" ||
        account?.provider === "guest-pass"
      );
    },
    session({ session, token }) {
      if (session.user && typeof token.email === "string") {
        session.user.email = token.email;
      }
      return session;
    },
  },
  events: {
    async signIn({ user }) {
      if (!user?.email) return;
      // Lazy import keeps better-sqlite3 out of the proxy bundle.
      const { upsertUser } = await import("@/lib/db");
      upsertUser(user.email, user.name ?? null);
    },
  },
});
