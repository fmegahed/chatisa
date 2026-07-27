import { describe, expect, it } from "vitest";
import { GET } from "@/app/api/health/route";

describe("GET /api/health", () => {
  it("returns ok with env check details and no secret values", async () => {
    const res = await GET(new Request("http://localhost/api/health"));
    expect(res.status).toBe(200);
    const body = await res.json();
    expect(body.status).toBe("ok");
    expect(body.checks.env).toBe("ok");
    expect(Array.isArray(body.checks.missingProviderKeys)).toBe(true);
    // The shallow check must stay cheap: no deep block unless asked for.
    expect(body.checks.deep).toBeUndefined();
    // Response must only ever contain variable NAMES.
    const text = JSON.stringify(body);
    for (const v of Object.values(process.env)) {
      if (v && v.length > 8) expect(text).not.toContain(v);
    }
  });
});
