import { NextResponse } from "next/server";
import { auth } from "@/lib/auth";
import { checkRateLimit } from "@/lib/ratelimit";
import {
  PROXY_BODY_MAX,
  proxyAllowsLocal,
  proxyFetch,
} from "@/lib/net/py-proxy";

/**
 * The endpoint behind in-browser Python's web access (lib/net/py-proxy.ts has
 * the design). The Pyodide worker rewrites cross-origin requests here; the
 * session cookie rides along on the same-origin call, so only signed-in users
 * (students and guests alike) can fetch through it.
 */

async function handle(req: Request, method: "GET" | "POST") {
  const session = await auth();
  const userEmail = session?.user?.email;
  if (!userEmail) {
    return NextResponse.json({ error: "Sign in to continue." }, { status: 401 });
  }
  const limit = checkRateLimit(`py-proxy:${userEmail}`, {
    limit: 60,
    windowMs: 60_000,
  });
  if (!limit.allowed) {
    return new Response(
      "ChatISA proxy: too many web requests in one minute. Wait a moment.",
      { status: 429, headers: { "content-type": "text/plain; charset=utf-8" } },
    );
  }

  const target = new URL(req.url).searchParams.get("url") ?? "";
  let body: ArrayBuffer | null = null;
  if (method === "POST") {
    body = await req.arrayBuffer();
    if (body.byteLength > PROXY_BODY_MAX) {
      return new Response(
        "ChatISA proxy: request bodies are capped at 1 MB here.",
        { status: 413, headers: { "content-type": "text/plain; charset=utf-8" } },
      );
    }
  }

  const result = await proxyFetch({
    target,
    method,
    headers: req.headers,
    body,
    allowLocal: proxyAllowsLocal(process.env),
  });
  return new Response(result.body, {
    status: result.status,
    headers: {
      "content-type": result.contentType,
      "cache-control": "private, no-store",
    },
  });
}

export async function GET(req: Request) {
  return handle(req, "GET");
}

export async function POST(req: Request) {
  return handle(req, "POST");
}
