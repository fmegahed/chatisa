import { afterEach, describe, expect, it, vi } from "vitest";

const PUBLIC_DEFAULT = "socks5h://test:yolo@ws.r-universe.dev:443";

afterEach(() => {
  vi.resetModules();
  delete process.env.NEXT_PUBLIC_WS_PROXY;
});

describe("ws-proxy config", () => {
  it("defaults to the public ws-proxy", async () => {
    const { WS_PROXY } = await import("@/lib/sandbox/network");
    expect(WS_PROXY).toBe(PUBLIC_DEFAULT);
  });

  it("honors NEXT_PUBLIC_WS_PROXY when set", async () => {
    process.env.NEXT_PUBLIC_WS_PROXY = "socks5h://u:p@wsproxy.example:443";
    vi.resetModules();
    const { WS_PROXY } = await import("@/lib/sandbox/network");
    expect(WS_PROXY).toBe("socks5h://u:p@wsproxy.example:443");
  });
});
