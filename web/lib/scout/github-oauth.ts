import "server-only";
import { serverEnv } from "@/lib/config/env";

/**
 * GitHub OAuth code exchange (v6.3.0). This is the ONLY server-side GitHub
 * work in the app: the client secret lives here, the resulting token does
 * not. The token goes straight back to the browser, which talks to
 * api.github.com directly; no ChatISA table or log ever holds a GitHub
 * credential (the CareerBridge prototype stored them in plaintext rows,
 * which is exactly what this design refuses to do).
 */

export interface GithubExchange {
  token: string;
  login: string;
}

export function githubOauthConfigured(): boolean {
  if (process.env.CHATISA_MOCK_GITHUB === "1") return true;
  const env = serverEnv();
  return Boolean(env.GITHUB_OAUTH_CLIENT_ID && env.GITHUB_OAUTH_CLIENT_SECRET);
}

/** The authorize URL the student's browser is sent to. public_repo keeps the
 * token unable to see private repositories, which is the approved scope. */
export function githubAuthorizeUrl(state: string, redirectUri: string): string {
  const url = new URL("https://github.com/login/oauth/authorize");
  url.searchParams.set("client_id", serverEnv().GITHUB_OAUTH_CLIENT_ID ?? "");
  url.searchParams.set("redirect_uri", redirectUri);
  url.searchParams.set("scope", "public_repo");
  url.searchParams.set("state", state);
  return url.toString();
}

/**
 * Exchanges the callback code for a token and resolves the account login.
 * Throws on any failure; the callback route maps that to a student-facing
 * error rather than leaking GitHub's response body.
 */
export async function exchangeCodeForToken(
  code: string,
): Promise<GithubExchange> {
  if (process.env.CHATISA_MOCK_GITHUB === "1") {
    // The login is overridable so a demo capture can show a believable name.
    return { token: "gh-mock-token", login: process.env.CHATISA_MOCK_GITHUB_LOGIN || "mockstudent" };
  }

  const env = serverEnv();
  const tokenRes = await fetch("https://github.com/login/oauth/access_token", {
    method: "POST",
    headers: { accept: "application/json", "content-type": "application/json" },
    body: JSON.stringify({
      client_id: env.GITHUB_OAUTH_CLIENT_ID,
      client_secret: env.GITHUB_OAUTH_CLIENT_SECRET,
      code,
    }),
  });
  if (!tokenRes.ok) throw new Error(`token exchange HTTP ${tokenRes.status}`);
  const tokenBody = (await tokenRes.json()) as {
    access_token?: string;
    error?: string;
  };
  if (!tokenBody.access_token) {
    throw new Error(`token exchange refused: ${tokenBody.error ?? "no token"}`);
  }

  const userRes = await fetch("https://api.github.com/user", {
    headers: {
      authorization: `Bearer ${tokenBody.access_token}`,
      accept: "application/vnd.github+json",
      "x-github-api-version": "2022-11-28",
    },
  });
  if (!userRes.ok) throw new Error(`user lookup HTTP ${userRes.status}`);
  const user = (await userRes.json()) as { login?: string };
  if (!user.login) throw new Error("user lookup returned no login");

  return { token: tokenBody.access_token, login: user.login };
}
