import "server-only";
import { hostIsSafe } from "@/lib/jobs/fetch-posting";
import {
  PROXY_BODY_MAX,
  PROXY_RESPONSE_MAX,
  PROXY_TIMEOUT_MS,
  proxyCapText,
} from "@/lib/net/proxy-limits";

/**
 * The Python web proxy's core (2026-07-24): gives in-browser Python the same
 * reach R has had through its WebSocket tunnel. The Pyodide worker rewrites
 * cross-origin requests.get/post calls to /api/py-proxy?url=..., which fetches
 * server-side (same SSRF discipline as the JobApp fetcher: every host resolved
 * and checked against private ranges, redirects followed by hand and
 * re-checked) and relays status, content type, and body. Same-origin requests
 * never come here; they go direct in the browser.
 *
 * Deliberate scope:
 * - GET and POST only; response capped at PROXY_RESPONSE_MAX (25 MB), request
 *   bodies at 1 MB.
 * - Forwarded request headers are an allowlist (accept, content-type,
 *   authorization, x-api-key) so a student can call an API with their own
 *   key; nothing else crosses, in either direction (upstream Set-Cookie in
 *   particular is dropped).
 * - Guard failures return a plain-text body starting with "ChatISA proxy:",
 *   so r.text in the student's console says exactly what happened.
 */

// Re-exported so existing importers of this module keep working unchanged.
export { PROXY_RESPONSE_MAX, PROXY_BODY_MAX, PROXY_TIMEOUT_MS, proxyCapText };

const MAX_REDIRECTS = 3;

const FORWARDED_REQUEST_HEADERS = [
  "accept",
  "content-type",
  "authorization",
  "x-api-key",
];

/** Local/private targets are allowed ONLY for the e2e suite (the offline test
 * fetches the test server through the proxy) and never in production. */
export function proxyAllowsLocal(env: {
  CHATISA_PROXY_ALLOW_LOCAL?: string;
  NODE_ENV?: string;
}): boolean {
  return (
    env.CHATISA_PROXY_ALLOW_LOCAL === "1" && env.NODE_ENV !== "production"
  );
}

/** Pure target validation, unit-tested: null when acceptable, else the
 * student-readable refusal. */
export function validateProxyTarget(raw: string): string | null {
  let url: URL;
  try {
    url = new URL(raw);
  } catch {
    return "ChatISA proxy: that is not a valid absolute URL.";
  }
  if (url.protocol !== "https:" && url.protocol !== "http:") {
    return "ChatISA proxy: only http and https URLs can be fetched.";
  }
  if (url.username || url.password) {
    return "ChatISA proxy: URLs with embedded credentials are not fetched.";
  }
  return null;
}

export interface ProxyResult {
  status: number;
  contentType: string;
  body: ArrayBuffer;
}

/**
 * Fetches the target with the guard discipline and returns what the student's
 * requests call should see. Errors come back as synthetic responses (status +
 * explanatory text body), never thrown, so the Python side always gets a
 * Response object it can inspect.
 */
export async function proxyFetch(params: {
  target: string;
  method: "GET" | "POST";
  headers: Headers;
  body: ArrayBuffer | null;
  allowLocal: boolean;
}): Promise<ProxyResult> {
  const refuse = (status: number, message: string): ProxyResult => ({
    status,
    contentType: "text/plain; charset=utf-8",
    body: new TextEncoder().encode(message).buffer as ArrayBuffer,
  });

  const invalid = validateProxyTarget(params.target);
  if (invalid) return refuse(400, invalid);

  const forwarded: Record<string, string> = {
    "user-agent": "ChatISA/1.0 (Miami University; educational use)",
  };
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = params.headers.get(name);
    if (value) forwarded[name] = value;
  }

  let current = new URL(params.target);
  for (let hop = 0; hop <= MAX_REDIRECTS; hop += 1) {
    if (!params.allowLocal && !(await hostIsSafe(current.hostname))) {
      return refuse(
        403,
        "ChatISA proxy: that address cannot be fetched from here (private or internal hosts are blocked).",
      );
    }
    let response: Response;
    try {
      response = await fetch(current.href, {
        method: params.method,
        headers: forwarded,
        body:
          params.method === "POST" && params.body != null
            ? params.body
            : undefined,
        redirect: "manual",
        signal: AbortSignal.timeout(PROXY_TIMEOUT_MS),
      });
    } catch {
      return refuse(
        502,
        "ChatISA proxy: the site could not be reached (timeout or connection failure).",
      );
    }

    // Redirects are followed by hand so each new host is re-checked; a public
    // hostname must not bounce the fetch to an internal one.
    if (response.status >= 300 && response.status < 400) {
      const location = response.headers.get("location");
      if (!location) break;
      try {
        current = new URL(location, current);
      } catch {
        break;
      }
      continue;
    }

    const tooLarge = `ChatISA proxy: the response is larger than ${proxyCapText()}, which is the cap here.`;

    const declared = Number(response.headers.get("content-length") ?? 0);
    if (declared > PROXY_RESPONSE_MAX) return refuse(502, tooLarge);

    // Read incrementally and stop the moment the cap is passed, rather than
    // buffering the whole response and measuring afterwards. With a 25 MB cap
    // that difference matters: a server that does not declare a content-length
    // could otherwise stream an unbounded body into memory before being refused.
    const body = await readCapped(response);
    if (body === null) return refuse(502, tooLarge);

    return {
      status: response.status,
      contentType:
        response.headers.get("content-type") ?? "application/octet-stream",
      body,
    };
  }
  return refuse(502, "ChatISA proxy: the site redirected too many times.");
}

/**
 * Reads a response body, giving up as soon as it exceeds the cap. Returns null
 * when it does, so the caller can refuse with the same message it uses for a
 * declared oversize.
 *
 * The reader is cancelled on the way out, which closes the upstream connection
 * instead of leaving it draining a file nobody will receive.
 */
async function readCapped(response: Response): Promise<ArrayBuffer | null> {
  if (!response.body) return new ArrayBuffer(0);
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let total = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      total += value.byteLength;
      if (total > PROXY_RESPONSE_MAX) {
        await reader.cancel().catch(() => {});
        return null;
      }
      chunks.push(value);
    }
  } catch {
    // A truncated or reset stream. Treated as unreachable by the caller's own
    // catch path would be wrong here, since we are past the fetch; an empty
    // body is the honest result and the student sees a zero-length response.
    return new ArrayBuffer(0);
  } finally {
    reader.releaseLock();
  }

  const out = new Uint8Array(total);
  let at = 0;
  for (const chunk of chunks) {
    out.set(chunk, at);
    at += chunk.byteLength;
  }
  return out.buffer;
}
