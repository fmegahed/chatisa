import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const grant = vi.fn();

vi.mock("@deepgram/sdk", () => ({
  DeepgramClient: class {
    auth = { v1: { tokens: { grant } } };
    speak = { v1: { audio: { generate: vi.fn() } } };
  },
}));

/**
 * The readiness probe behind /api/health?deep=1.
 *
 * It exists because "the interviewer has no voice in production" could not be
 * diagnosed from outside the server. The shallow health check reported whether
 * DEEPGRAM_TOKEN was PRESENT, and presence is not validity: a revoked key, an
 * account out of credit, and blocked outbound HTTPS all report "present" and all
 * produce silence.
 *
 * The distinction these tests pin is the one that matters operationally: absent
 * is a choice, refused is a fault. Collapsing them either turns every
 * speech-less server red or hides every broken one.
 */
describe("probeSpeech", () => {
  beforeEach(() => {
    vi.resetModules();
    grant.mockReset();
  });

  afterEach(() => {
    delete process.env.DEEPGRAM_TOKEN;
  });

  it("reports not-configured, not broken, when no token is set", async () => {
    delete process.env.DEEPGRAM_TOKEN;
    const { probeSpeech } = await import("@/lib/speech/deepgram");
    const probe = await probeSpeech();
    expect(probe.state).toBe("not-configured");
    // A server without speech is a supported configuration, so nothing here
    // should read as an error, and Deepgram must not be called at all.
    expect(grant).not.toHaveBeenCalled();
    expect(probe.detail).toMatch(/type/i);
  });

  it("treats a whitespace-only token as not set", async () => {
    process.env.DEEPGRAM_TOKEN = "   ";
    const { probeSpeech } = await import("@/lib/speech/deepgram");
    expect((await probeSpeech()).state).toBe("not-configured");
    expect(grant).not.toHaveBeenCalled();
  });

  it("reports ok when Deepgram mints a token", async () => {
    process.env.DEEPGRAM_TOKEN = "test-key-not-a-real-credential";
    grant.mockResolvedValue({ access_token: "jwt", expires_in: 60 });
    const { probeSpeech } = await import("@/lib/speech/deepgram");
    const probe = await probeSpeech();
    expect(probe.state).toBe("ok");
    expect(probe.detail).toContain("60s");
  });

  it("reports broken when a configured token is refused", async () => {
    process.env.DEEPGRAM_TOKEN = "revoked-key";
    grant.mockRejectedValue(Object.assign(new Error("Unauthorized"), { status: 401 }));
    const { probeSpeech } = await import("@/lib/speech/deepgram");
    const probe = await probeSpeech();
    expect(probe.state).toBe("broken");
    // The status code is the actionable part; 401 means the key, not the network.
    expect(probe.detail).toContain("401");
  });

  it("reads the SDK's statusCode property, not just status", async () => {
    // The @deepgram/sdk v4+ error classes carry `statusCode`. The probe used
    // to look only for `status`, so a 400 "Invalid credentials" rejection was
    // reported with the NETWORK advice ("could not be reached... check
    // outbound HTTPS"), which misdirected the 2026-07-27 production outage
    // into a firewall hunt when the real fault was the credential value.
    process.env.DEEPGRAM_TOKEN = "test-key-not-a-real-credential";
    grant.mockRejectedValue(
      Object.assign(new Error("BadRequestError"), { statusCode: 400 }),
    );
    const { probeSpeech } = await import("@/lib/speech/deepgram");
    const probe = await probeSpeech();
    expect(probe.state).toBe("broken");
    expect(probe.detail).toContain("400");
    expect(probe.detail).not.toMatch(/outbound/i);
  });

  it("reports broken, with network advice, when Deepgram is unreachable", async () => {
    process.env.DEEPGRAM_TOKEN = "test-key-not-a-real-credential";
    grant.mockRejectedValue(new Error("getaddrinfo ENOTFOUND api.deepgram.com"));
    const { probeSpeech } = await import("@/lib/speech/deepgram");
    const probe = await probeSpeech();
    expect(probe.state).toBe("broken");
    expect(probe.detail).toMatch(/outbound/i);
  });

  it("reports broken when the response carries no token", async () => {
    process.env.DEEPGRAM_TOKEN = "test-key-not-a-real-credential";
    grant.mockResolvedValue({ access_token: "", expires_in: 60 });
    const { probeSpeech } = await import("@/lib/speech/deepgram");
    expect((await probeSpeech()).state).toBe("broken");
  });

  it("never puts the credential or the provider's message in the detail", async () => {
    const secret = "dg-super-secret-credential-value";
    process.env.DEEPGRAM_TOKEN = secret;
    // A provider error that quotes the key back, which is exactly why the
    // detail is composed here rather than forwarded.
    grant.mockRejectedValue(
      Object.assign(new Error(`invalid key ${secret}`), { status: 403 }),
    );
    const { probeSpeech } = await import("@/lib/speech/deepgram");
    const probe = await probeSpeech();
    expect(probe.detail).not.toContain(secret);
    expect(probe.detail).not.toContain("invalid key");
  });
});

describe("describeErrorChain", () => {
  /**
   * Node's fetch wraps every network failure in TypeError("fetch failed") and
   * hides the actionable part (ETIMEDOUT vs ENOTFOUND vs a TLS error) in
   * err.cause. String(err) drops that, which is why chatisa.log said only
   * "fetch failed" while speech was down in production (2026-07-27) and the
   * DNS-vs-firewall question could not be answered from the log.
   */
  it("includes every cause in the chain, not just the top error", async () => {
    const { describeErrorChain } = await import("@/lib/speech/deepgram");
    const inner = Object.assign(new Error("connect ETIMEDOUT 34.86.0.1:443"), {
      code: "ETIMEDOUT",
    });
    const outer = new TypeError("fetch failed", { cause: inner });
    const text = describeErrorChain(outer);
    expect(text).toContain("fetch failed");
    expect(text).toContain("ETIMEDOUT");
  });

  it("survives non-Error values and cycles", async () => {
    const { describeErrorChain } = await import("@/lib/speech/deepgram");
    expect(describeErrorChain("plain string")).toContain("plain string");
    const a = new Error("a");
    a.cause = a;
    expect(describeErrorChain(a)).toContain("a");
  });
});

describe("GET /api/health?deep=1", () => {
  beforeEach(() => {
    vi.resetModules();
    grant.mockReset();
  });

  afterEach(() => {
    delete process.env.DEEPGRAM_TOKEN;
  });

  it("reports speech and stays healthy when speech is simply not set up", async () => {
    delete process.env.DEEPGRAM_TOKEN;
    const { GET } = await import("@/app/api/health/route");
    const res = await GET(new Request("http://localhost/api/health?deep=1"));
    const body = await res.json();
    expect(body.checks.deep.speech).toMatch(/^not-configured/);
    // The rest of the deep block decides health; speech being off does not.
    expect(body.checks.deep.dbRoundtrip).toBe("ok");
    expect(res.status).toBe(200);
  });

  it("fails the server when a configured credential is refused", async () => {
    process.env.DEEPGRAM_TOKEN = "revoked-key";
    grant.mockRejectedValue(Object.assign(new Error("nope"), { status: 401 }));
    const { GET } = await import("@/app/api/health/route");
    const res = await GET(new Request("http://localhost/api/health?deep=1"));
    const body = await res.json();
    expect(body.checks.deep.speech).toMatch(/^broken/);
    // This is the whole point: a server whose voice is silently broken must
    // announce itself, because from the outside it looks perfectly healthy.
    expect(res.status).toBe(503);
    expect(body.status).toBe("degraded");
  });

  it("never leaks a credential value into the deep payload", async () => {
    process.env.DEEPGRAM_TOKEN = "dg-another-secret-credential-value";
    grant.mockResolvedValue({ access_token: "jwt-token-value", expires_in: 60 });
    const { GET } = await import("@/app/api/health/route");
    const res = await GET(new Request("http://localhost/api/health?deep=1"));
    const text = JSON.stringify(await res.json());
    expect(text).not.toContain("dg-another-secret-credential-value");
    // The minted token is a credential too, short-lived or not.
    expect(text).not.toContain("jwt-token-value");
  });
});
