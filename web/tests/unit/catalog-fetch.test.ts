import { describe, expect, it } from "vitest";
import { fetchText } from "@/scripts/catalog/collect";

/**
 * The first Action run (2026-09-24) died when the Bulletin's server closed
 * the connection mid-response ("fetch failed: other side closed"): only
 * HTTP error codes were retried, not a dropped connection.
 */
const noWait = async () => {};
const ok = (body: string) => ({ ok: true, status: 200, text: async () => body }) as Response;

describe("fetchText", () => {
  it("retries a dropped connection and returns the page", async () => {
    let calls = 0;
    const flaky = async () => {
      calls++;
      if (calls === 1) throw new TypeError("fetch failed");
      return ok("<html>bulletin</html>");
    };
    expect(await fetchText("https://bulletin.miamioh.edu/x", flaky as typeof fetch, noWait)).toBe("<html>bulletin</html>");
    expect(calls).toBe(2);
  });

  it("retries a body that breaks off while reading", async () => {
    let calls = 0;
    const flaky = async () => {
      calls++;
      return calls === 1
        ? ({ ok: true, status: 200, text: async () => { throw new TypeError("terminated"); } } as unknown as Response)
        : ok("done");
    };
    expect(await fetchText("u", flaky as typeof fetch, noWait)).toBe("done");
  });

  it("gives up after four attempts with the URL in the message", async () => {
    const down = async () => { throw new TypeError("fetch failed"); };
    await expect(fetchText("https://bulletin.miamioh.edu/x", down as typeof fetch, noWait)).rejects.toThrow(/bulletin\.miamioh\.edu\/x.*fetch failed/);
  });

  it("still retries HTTP errors and reports the status", async () => {
    const busy = async () => ({ ok: false, status: 503, text: async () => "" }) as Response;
    await expect(fetchText("u", busy as typeof fetch, noWait)).rejects.toThrow(/HTTP 503/);
  });
});
