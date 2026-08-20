/**
 * The GitHub connection, browser-only (v6.3.0 decision): the token lives in
 * localStorage and is sent to api.github.com directly, never to a ChatISA
 * route. Read this key ONLY from lib/scout/github*.ts; anything else
 * touching it is a review flag.
 */

const GITHUB_KEY = "js-github-v1";

export interface GithubConnection {
  v: 1;
  token: string;
  login: string;
  connectedAt: string;
}

/** Tolerant read, matching the other scout stores: corrupt JSON is null. */
export function loadGithubConnection(): GithubConnection | null {
  try {
    const raw = localStorage.getItem(GITHUB_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<GithubConnection>;
    if (
      parsed.v === 1 &&
      typeof parsed.token === "string" &&
      parsed.token.length > 0 &&
      typeof parsed.login === "string" &&
      parsed.login.length > 0
    ) {
      return {
        v: 1,
        token: parsed.token,
        login: parsed.login,
        connectedAt: typeof parsed.connectedAt === "string" ? parsed.connectedAt : "",
      };
    }
    return null;
  } catch {
    return null;
  }
}

export function saveGithubConnection(value: {
  token: string;
  login: string;
}): GithubConnection {
  const record: GithubConnection = {
    v: 1,
    token: value.token,
    login: value.login,
    connectedAt: new Date().toISOString(),
  };
  localStorage.setItem(GITHUB_KEY, JSON.stringify(record));
  return record;
}

export function clearGithubConnection(): void {
  try {
    localStorage.removeItem(GITHUB_KEY);
  } catch {
    // Removal is best-effort; a blocked storage API leaves nothing to clear.
  }
}
