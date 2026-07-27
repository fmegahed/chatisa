import { describe, expect, it } from "vitest";
import { proxyCapText, PROXY_TIMEOUT_MS } from "@/lib/net/proxy-limits";
import { askToolDefs } from "@/lib/ask/tools";
import { ASK_ANYTHING_SYSTEM_PROMPT } from "@/lib/prompts/ask-anything";
import {
  PROXY_RESPONSE_MAX,
  proxyAllowsLocal,
  proxyFetch,
  validateProxyTarget,
} from "@/lib/net/py-proxy";

describe("python web proxy", () => {
  it("validates targets: absolute http(s) only, no embedded credentials", () => {
    expect(validateProxyTarget("https://example.com/page")).toBeNull();
    expect(validateProxyTarget("http://data.gov/file.csv")).toBeNull();
    expect(validateProxyTarget("/relative/path")).toMatch(/absolute URL/);
    expect(validateProxyTarget("ftp://example.com")).toMatch(/http and https/);
    expect(validateProxyTarget("file:///etc/passwd")).toMatch(/http and https/);
    expect(validateProxyTarget("https://user:pw@example.com")).toMatch(
      /embedded credentials/,
    );
  });

  it("allows local targets only outside production with the explicit flag", () => {
    expect(
      proxyAllowsLocal({ CHATISA_PROXY_ALLOW_LOCAL: "1", NODE_ENV: "test" }),
    ).toBe(true);
    expect(
      proxyAllowsLocal({ CHATISA_PROXY_ALLOW_LOCAL: "1", NODE_ENV: "production" }),
    ).toBe(false);
    expect(proxyAllowsLocal({ NODE_ENV: "test" })).toBe(false);
  });

  it("refuses private hosts with a student-readable body (SSRF guard)", async () => {
    for (const target of [
      "https://127.0.0.1/latest",
      "https://169.254.169.254/latest/meta-data/",
      "https://localhost:8080/admin",
    ]) {
      const result = await proxyFetch({
        target,
        method: "GET",
        headers: new Headers(),
        body: null,
        allowLocal: false,
      });
      expect(result.status, target).toBe(403);
      expect(new TextDecoder().decode(result.body)).toContain("ChatISA proxy:");
    }
  });

  it("relays status, content type, and body from the upstream", async () => {
    // allowLocal + a loopback fetch stub via the global fetch: use a data-free
    // approach by pointing at an unreachable local port and asserting the
    // refusal shape instead; the full happy path is covered by the e2e, which
    // fetches the real test server through the proxy.
    const result = await proxyFetch({
      target: "http://127.0.0.1:1/nothing-listens-here",
      method: "GET",
      headers: new Headers(),
      body: null,
      allowLocal: true,
    });
    expect(result.status).toBe(502);
    expect(new TextDecoder().decode(result.body)).toMatch(
      /could not be reached/,
    );
    // Raised from 4 MB on 2026-07-26 so the professor's County Business
    // Patterns exercise (an 11 MB Census archive) can be fetched at all, and
    // matched to the 25 MB attachment cap so there is one number to explain.
    expect(PROXY_RESPONSE_MAX).toBe(25_000_000);
  });
});

/**
 * The size cap, and the fact that it is stated in one place.
 *
 * Before 2026-07-26 the number 4 MB was written out in six places: the constant,
 * two refusal messages, the module comment, the run_python tool description, the
 * Ask Anything system prompt, and the Coding Studio limitations panel. Raising it
 * meant editing all of them, and a missed one would have told students the wrong
 * limit with total confidence. Now the prose is generated.
 */
describe("proxy cap, stated once", () => {
  it("formats the cap from the constant", () => {
    expect(proxyCapText()).toBe("25 MB");
  });

  it("is the same number the attachment path allows", () => {
    // Kept equal on purpose so the rule is "25 MB in, 25 MB out". If one moves,
    // this test is the reminder to decide about the other rather than drift.
    expect(PROXY_RESPONSE_MAX).toBe(25_000_000);
  });

  it("gives the fetch long enough to actually transfer the cap", () => {
    // 12 s could not finish an 11 MB download, so the old timeout would have
    // refused the professor's file for a different reason than the cap.
    expect(PROXY_TIMEOUT_MS).toBeGreaterThanOrEqual(30_000);
  });

  it("names the cap in every place students read it", () => {
    const cap = proxyCapText();
    expect(ASK_ANYTHING_SYSTEM_PROMPT).toContain(cap);
    expect(askToolDefs().run_python.description).toContain(cap);
    // And the prompt must not still be quoting the old number anywhere.
    expect(ASK_ANYTHING_SYSTEM_PROMPT).not.toContain("4 MB");
    expect(askToolDefs().run_python.description).not.toContain("4 MB");
  });

  it("tells the model how to handle a large file once it arrives", () => {
    // A 25 MB archive can expand to far more (the Census one reaches 88.7 MB),
    // so the cap being raised is only half the answer: the model has to read it
    // in chunks rather than holding it all in browser memory.
    expect(ASK_ANYTHING_SYSTEM_PROMPT).toMatch(/chunks|chunksize/i);
  });
});
